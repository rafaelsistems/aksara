"""
AksaraModel - Orchestrator utama pipeline AKSARA.

Pipeline lengkap:
  LPS → BSU → LSK → MEB → CorrectnessHead → Skor

Evaluasi kebenaran kalimat bahasa Indonesia:
  text → LPS(tokenize+morph) → BSU(embed) → LSK(KBBI anchor)
       → MEB(evolve) → CorrectnessHead(score)

Output: skor kebenaran 4 dimensi (morph/struct/semantic/lexical)
Bukan generative model — tidak ada prediksi token.
"""

import hashlib
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Versi format checkpoint yang didukung.
AKSARA_VERSION = "3.0"
SUPPORTED_VERSIONS = {"3.0", "2.0"}

from aksara.core.bsu import BSUConfig, BahasaStateUnit
from aksara.core.meb import MEBConfig, MesinEvolusiBahasa
from aksara.core.correctness import CorrectnessConfig, CorrectnessHead
from aksara.linguistic.lps import (
    LPSConfig, LapisanParsingStuktural, AFFIX_TO_ID, ROLE_LABELS, build_root_vocab
)
from aksara.linguistic.lsk import LSKConfig, LapisanSemantikKBBI
from aksara.training.loss import CorrectnessLoss


@dataclass
class AksaraConfig:
    """Konfigurasi lengkap AksaraModel."""
    bsu_config:         BSUConfig         = field(default_factory=BSUConfig)
    meb_config:         MEBConfig         = field(default_factory=MEBConfig)
    correctness_config: CorrectnessConfig = field(default_factory=CorrectnessConfig)
    lps_config:         LPSConfig         = field(default_factory=LPSConfig)
    lsk_config:         LSKConfig         = field(default_factory=LSKConfig)

    def __post_init__(self):
        self.meb_config.bsu_config         = self.bsu_config
        self.correctness_config.bsu_config = self.bsu_config


class AksaraModel(nn.Module):
    """
    Model utama AKSARA — Evaluator Kebenaran Kalimat Bahasa Indonesia.

    Komponen:
    - LPS : Lapisan Parsing Struktural (tokenisasi + morfologi)
    - BSU : Bahasa State Unit (structured embedding per slot linguistik)
    - LSK : Lapisan Semantik KBBI (semantic grounding ke leksikon)
    - MEB : Mesin Evolusi Bahasa (evolusi state antar token)
    - CorrectnessHead : Evaluasi kebenaran 4 dimensi (morph/struct/semantic/lexical)

    Tidak ada GOS. Tidak ada generative component. Tidak ada next-token prediction.
    """

    def __init__(
        self,
        config: AksaraConfig,
        root_vocab: Dict[str, int],
        known_words: Optional[set] = None,
    ):
        super().__init__()
        self.config = config
        self.root_vocab = root_vocab

        vocab_size       = max(root_vocab.values()) + 1
        affix_vocab_size = len(AFFIX_TO_ID)
        role_vocab_size  = len(ROLE_LABELS)

        # ─── Components ───
        self.lps = LapisanParsingStuktural(
            config.lps_config, root_vocab, known_words
        )

        self.bsu = BahasaStateUnit(
            config.bsu_config,
            vocab_size=vocab_size,
            affix_vocab_size=affix_vocab_size,
            role_vocab_size=role_vocab_size,
            kbbi_input_dim=config.lsk_config.kbbi_vector_dim,
        )

        self.lsk = LapisanSemantikKBBI(config.lsk_config, root_vocab)
        self.lsk.set_sem_dim(config.bsu_config.d_semantic)

        self.meb = MesinEvolusiBahasa(config.meb_config, affix_vocab_size)

        self.correctness_head = CorrectnessHead(config.correctness_config)

        self.loss_fn = CorrectnessLoss()

        # Ablation state
        self._disabled_components: set = set()

    @property
    def num_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def forward(
        self,
        lps_output: Dict,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass evaluasi kebenaran kalimat.

        Args:
            lps_output : dict dari LPS.forward() atau manual batch
                         keys: morpheme_ids, affix_ids, dep_masks, attention_mask
            labels     : (B,) float — 1.0 = kalimat benar, 0.0 = kalimat salah
                         Jika diberikan, hitung loss sekaligus.

        Returns:
            dict dengan:
                scores      : dict {morph, struct, semantic, lexical, total} — (B,) tiap key
                losses      : dict losses (hanya jika labels diberikan)
                meb_out     : (B, L, d_total) — state akhir MEB
                kbbi_mask   : (B, L) bool
        """
        morpheme_ids   = lps_output["morpheme_ids"]
        affix_ids      = lps_output["affix_ids"]
        dep_masks      = lps_output.get("dep_masks")
        attention_mask = lps_output.get("attention_mask",
                                        (morpheme_ids != 0).long())
        role_ids       = lps_output.get("role_ids")

        # ─── LSK: KBBI semantic anchors ───
        kbbi_vectors = self.lsk(morpheme_ids, return_raw=True)   # (B, L, 16)

        # ─── BSU: structured embedding ───
        bsu_states, slots = self.bsu(
            morpheme_ids=morpheme_ids,
            affix_ids=affix_ids,
            kbbi_vectors=kbbi_vectors,
            role_ids=role_ids,
        )
        bsu_original = bsu_states.clone()

        # ─── MEB: evolusi state linguistik ───
        kbbi_anchors = self.lsk.get_anchors(morpheme_ids)        # (B, L, 16)
        meb_out, _ = self.meb(
            bsu_states=bsu_states,
            affix_ids=affix_ids,
            kbbi_anchors=kbbi_anchors,
            dep_mask=dep_masks,
        )

        # ─── KBBI mask ───
        max_id    = self.lsk.vocab_kbbi_map.size(0) - 1
        safe_ids  = morpheme_ids.clamp(0, max_id)
        kbbi_mask = self.lsk.vocab_kbbi_map[safe_ids] > 0        # (B, L) bool

        # ─── CorrectnessHead: evaluasi kebenaran ───
        scores = self.correctness_head(
            meb_out=meb_out,
            bsu_original=bsu_original,
            kbbi_mask=kbbi_mask,
            attention_mask=attention_mask.bool(),
        )

        result = {
            "scores":    scores,
            "meb_out":   meb_out,
            "kbbi_mask": kbbi_mask,
        }

        if labels is not None:
            losses = self.loss_fn(scores["total"], scores, labels)
            result["losses"] = losses

        return result

    def score(self, texts: List[str]) -> Dict:
        """
        Evaluasi kebenaran kalimat dari teks mentah.

        Args:
            texts : List[str] — kalimat yang akan dievaluasi

        Returns:
            dict scores {morph, struct, semantic, lexical, total} — nilai float per kalimat
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            lps_out = self.lps(texts, device=device)
            result  = self.forward(lps_out)

        # Konversi ke Python float untuk kemudahan pakai
        return {
            k: v.cpu().tolist() if isinstance(v, torch.Tensor) else v
            for k, v in result["scores"].items()
        }

    # ─── Ablation API ───

    def disable(self, component: str):
        """Disable MEB component for ablation."""
        self.meb.disable(component)
        self._disabled_components.add(component)

    def enable(self, component: str):
        """Re-enable MEB component."""
        self.meb.enable(component)
        self._disabled_components.discard(component)

    def freeze(self, component: str):
        """Freeze MEB component (no gradient)."""
        self.meb.freeze(component)

    def unfreeze(self, component: str):
        """Unfreeze MEB component."""
        self.meb.unfreeze(component)

    # ─── Save/Load ───

    def save(self, path: str, metadata: Optional[Dict] = None):
        """
        Simpan checkpoint lengkap ke direktori path.

        Struktur direktori:
            path/
              model.pt       — state_dict semua parameter nn.Module
              vocab.json     — root_vocab {token: id}
              config.json    — semua config (bsu, meb, gos, lps, lsk)
              checkpoint.json — metadata (versi, timestamp, vocab_size, n_params)

        Args:
            path     : direktori tujuan (dibuat jika belum ada)
            metadata : dict tambahan yang disimpan di checkpoint.json
        """
        import dataclasses
        import time

        os.makedirs(path, exist_ok=True)

        # ── 1. State dict (weights) ──────────────────────────────────────────
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

        # ── 2. Vocab ─────────────────────────────────────────────────────────
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.root_vocab, f, ensure_ascii=False, indent=2)

        # ── 3. Config ────────────────────────────────────────────────────────
        def config_to_dict(cfg) -> dict:
            """Rekursif konversi dataclass config ke dict yang JSON-serializable."""
            if dataclasses.is_dataclass(cfg):
                result = {}
                for field in dataclasses.fields(cfg):
                    val = getattr(cfg, field.name)
                    result[field.name] = config_to_dict(val)
                return result
            elif isinstance(cfg, (str, int, float, bool)) or cfg is None:
                return cfg
            elif isinstance(cfg, (list, tuple)):
                return [config_to_dict(v) for v in cfg]
            else:
                return str(cfg)

        config_dict = config_to_dict(self.config)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        # ── 4. Checkpoint metadata + integrity checksum ──────────────────────
        n_params = self.num_parameters
        model_checksum = self._compute_checksum(os.path.join(path, "model.pt"))
        ckpt = {
            "aksara_version": AKSARA_VERSION,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "vocab_size": len(self.root_vocab),
            "n_params_total": n_params["total"],
            "n_params_trainable": n_params["trainable"],
            "pretrained_kbbi": os.path.exists(
                self.config.lsk_config.pretrained_path
            ),
            "model_sha256": model_checksum,
        }
        if metadata:
            ckpt.update(metadata)
        with open(os.path.join(path, "checkpoint.json"), "w", encoding="utf-8") as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2)

        print(f"[AksaraModel] Checkpoint disimpan ke '{path}'")
        print(f"  vocab_size  : {len(self.root_vocab):,}")
        print(f"  n_params    : {n_params['total']:,} ({n_params['trainable']:,} trainable)")

    def load(self, path: str, strict: bool = False, device: str = "cpu"):
        """
        Load weights dari checkpoint direktori.

        Args:
            path   : direktori checkpoint (dari save())
            strict : jika True, semua key harus cocok persis
            device : device target ("cpu", "cuda", dll)

        Raises:
            FileNotFoundError : jika model.pt tidak ditemukan
            RuntimeError      : jika ada key mismatch kritis (strict=True)
        """
        model_path = os.path.join(path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model.pt tidak ditemukan di '{path}'. "
                f"Pastikan path benar atau jalankan save() dulu."
            )

        # ── Version guard + integrity check ──────────────────────────────────
        ckpt_path = os.path.join(path, "checkpoint.json")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)

            ckpt_ver = ckpt.get("aksara_version", "unknown")
            if ckpt_ver not in SUPPORTED_VERSIONS:
                # Checkpoint dari versi yang tidak dikenal — load tetap dilanjutkan
                # tapi user diperingatkan agar tidak terkejut jika ada ketidakcocokan.
                print(
                    f"[AksaraModel] Warning: checkpoint versi '{ckpt_ver}' "
                    f"tidak ada di SUPPORTED_VERSIONS={SUPPORTED_VERSIONS}. "
                    f"Format mungkin tidak kompatibel."
                )

            # Verifikasi integritas: SHA-256 model.pt harus cocok dengan yang tersimpan
            saved_checksum = ckpt.get("model_sha256")
            if saved_checksum:
                actual_checksum = self._compute_checksum(model_path)
                if actual_checksum != saved_checksum:
                    raise RuntimeError(
                        f"[AksaraModel] Integritas gagal: SHA-256 model.pt tidak cocok. "
                        f"File mungkin rusak atau dimodifikasi.\n"
                        f"  Expected : {saved_checksum}\n"
                        f"  Got      : {actual_checksum}"
                    )

            print(f"[AksaraModel] Loaded checkpoint v{ckpt_ver} "
                  f"({ckpt.get('saved_at', 'unknown')}) "
                  f"[integritas OK]")

        state = torch.load(model_path, map_location=device, weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=strict)

        # Log jika ada key yang tidak cocok
        if missing:
            print(f"[AksaraModel] Warning: {len(missing)} key missing saat load: {missing[:3]}...")
        if unexpected:
            print(f"[AksaraModel] Warning: {len(unexpected)} key unexpected: {unexpected[:3]}...")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config: Optional[AksaraConfig] = None,
        device: str = "cpu",
        **kwargs,
    ):
        """
        Load model lengkap dari direktori checkpoint.

        Jika config tidak diberikan, config di-rebuild dari config.json
        yang disimpan oleh save(). Ini memungkinkan load tanpa harus tahu
        konfigurasi model secara manual.

        Args:
            path   : direktori checkpoint (dari save())
            config : opsional — jika None, di-load dari config.json
            device : device target
            kwargs : diteruskan ke __init__ (mis. known_words)

        Returns:
            AksaraModel yang sudah di-load dan siap dipakai

        Raises:
            FileNotFoundError : jika vocab.json atau model.pt tidak ada
        """
        # ── Load vocab ───────────────────────────────────────────────────────
        vocab_path = os.path.join(path, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json tidak ditemukan di '{path}'")
        with open(vocab_path, "r", encoding="utf-8") as f:
            root_vocab = json.load(f)

        # ── Load atau rebuild config ─────────────────────────────────────────
        if config is None:
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    config_dict = json.load(f)
                config = cls._config_from_dict(config_dict)
            else:
                # Fallback: config default (backward compat dengan checkpoint lama)
                print(f"[AksaraModel] config.json tidak ada — menggunakan AksaraConfig default")
                config = AksaraConfig()

        # ── Bangun model ─────────────────────────────────────────────────────
        model = cls(config, root_vocab, **kwargs)

        # ── Load weights ─────────────────────────────────────────────────────
        model.load(path, device=device)

        return model

    @staticmethod
    def _compute_checksum(filepath: str) -> str:
        """Hitung SHA-256 dari file untuk verifikasi integritas."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _config_from_dict(d: Dict) -> "AksaraConfig":
        """
        Rebuild AksaraConfig dari dict yang tersimpan di config.json.

        Hanya merekonstruksi field-field yang dikenal — field baru
        yang ditambahkan setelah checkpoint dibuat akan pakai nilai default.
        """
        def get(d, *keys, default=None):
            """Navigasi nested dict dengan fallback."""
            for k in keys:
                if not isinstance(d, dict):
                    return default
                d = d.get(k, default)
                if d is None:
                    return default
            return d

        bsu = BSUConfig(
            d_morpheme=get(d, "bsu_config", "d_morpheme", default=64),
            d_semantic=get(d, "bsu_config", "d_semantic", default=64),
            d_role=get(d, "bsu_config", "d_role", default=32),
            d_context=get(d, "bsu_config", "d_context", default=64),
        )
        meb = MEBConfig(
            n_layers=get(d, "meb_config", "n_layers", default=4),
            n_dep_heads=get(d, "meb_config", "n_dep_heads", default=4),
            dropout=get(d, "meb_config", "dropout", default=0.1),
        )
        meb.bsu_config = bsu

        cor = CorrectnessConfig(
            bsu_config=bsu,
            hidden_dim=get(d, "correctness_config", "hidden_dim", default=128),
            dropout=get(d, "correctness_config", "dropout", default=0.1),
            w_morph=get(d, "correctness_config", "w_morph", default=0.25),
            w_struct=get(d, "correctness_config", "w_struct", default=0.30),
            w_semantic=get(d, "correctness_config", "w_semantic", default=0.30),
            w_lexical=get(d, "correctness_config", "w_lexical", default=0.15),
        )

        lps = LPSConfig(
            dep_window=get(d, "lps_config", "dep_window", default=4),
            min_root_length=get(d, "lps_config", "min_root_length", default=3),
        )
        lsk = LSKConfig(
            kbbi_path=get(d, "lsk_config", "kbbi_path", default="kbbi_core_v2.json"),
            kbbi_vector_dim=get(d, "lsk_config", "kbbi_vector_dim", default=16),
            max_lemmas=get(d, "lsk_config", "max_lemmas", default=50000),
            pretrained_path=get(d, "lsk_config", "pretrained_path",
                                default="data/kbbi_pretrained.pt"),
        )

        return AksaraConfig(
            bsu_config=bsu,
            meb_config=meb,
            correctness_config=cor,
            lps_config=lps,
            lsk_config=lsk,
        )
