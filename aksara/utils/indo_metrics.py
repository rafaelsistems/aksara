"""
IndoNativeMetrics - Metrik evaluasi Indo-native untuk AKSARA (Fase 2.2).

Tiga metrik utama yang beroperasi di level STRUKTUR LINGUISTIK, bukan ML generic:

1. MorphologicalConsistencyScore (MCS)
   - Bukan sekadar affix accuracy
   - Mengukur: apakah transform imbuhan konsisten dengan kaidah Indonesia?
   - Contoh: apakah me- diikuti root yang valid? apakah di- selalu kata kerja?
   - Sub-metrik: affix_validity, root_affix_coherence, transform_consistency

2. StructureValidityScore (SVS)
   - Bukan sekadar role accuracy
   - Mengukur: apakah S-P-O-K terpenuhi dan urutan valid?
   - Contoh: kalimat tanpa P (predikat) = invalid, kalimat S-S-S = invalid
   - Sub-metrik: spok_completeness, order_validity, dep_coherence

3. SemanticDriftScore (SDS)
   - Mengukur pergeseran makna semantic slot dari KBBI anchor selama training
   - Bukan hanya cosine sim sesaat, tapi DRIFT over time
   - Jika semantic slot bergerak jauh dari KBBI → model kehilangan grounding
   - Sub-metrik: anchor_distance, drift_velocity, coverage_drift

Semua metrik ini dapat ditrack per-epoch untuk Stability Benchmark (Fase 4.0).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from aksara.linguistic.lps import (
    MorfologiAnalyzer, AFFIX_TO_ID, ROLE_LABELS,
    PREFIXES_ID, SUFFIXES_ID, CONFIXES_ID,
)


# ─── Valid affix → POS rules (morfologi Indonesia) ──────────────────────────

# me-, mem-, men-, meng-, meny- → harus kata kerja (verb)
_ACTIVE_VERB_PREFIXES = {"me", "mem", "men", "meng", "meny"}
# di- → pasif, harus kata kerja
_PASSIVE_PREFIX = {"di"}
# ber-, ter- → bisa verb atau adjective
_BER_TER_PREFIXES = {"ber", "ter"}
# pe-, pem-, peng- → nominalisasi
_NOMINALIZATION_PREFIXES = {"pe", "pem", "pen", "peng", "peny"}
# ke-an, pe-an → nomina abstrak
_ABSTRACT_CONFIXES = {("ke", "an"), ("pe", "an"), ("pem", "an"),
                      ("peng", "an"), ("peny", "an"), ("pen", "an")}

# Role order yang valid dalam kalimat Indonesia
# Format: set of (role_i, role_j) yang valid berurutan
_VALID_ROLE_TRANSITIONS = {
    ("S", "P"), ("P", "O"), ("P", "K"), ("S", "K"),
    ("O", "K"), ("K", "K"), ("S", "P"),
    ("UNK", "S"), ("UNK", "P"), ("UNK", "O"), ("UNK", "K"), ("UNK", "UNK"),
    ("S", "UNK"), ("P", "UNK"), ("O", "UNK"), ("K", "UNK"),
}

_ID_TO_ROLE = {v: k for k, v in ROLE_LABELS.items()}


# ─── 1. MorphologicalConsistencyScore ────────────────────────────────────────

@dataclass
class MCSResult:
    """Hasil Morphological Consistency Score satu batch/epoch."""
    affix_validity: float        # % imbuhan yang valid secara morfologi (0..1)
    root_affix_coherence: float  # % root + affix yang koheren (0..1)
    transform_consistency: float # % transform imbuhan yang konsisten lintas batch (0..1)
    overall: float               # rata-rata ketiga sub-metrik
    n_tokens: int                # total token yang dievaluasi

    def __str__(self) -> str:
        return (
            f"MCS={self.overall:.3f} "
            f"[validity={self.affix_validity:.3f} "
            f"coherence={self.root_affix_coherence:.3f} "
            f"consistency={self.transform_consistency:.3f}]"
            f" n={self.n_tokens}"
        )


class MorphologicalConsistencyScore:
    """
    Morphological Consistency Score (MCS).

    Mengukur apakah model belajar imbuhan Indonesia secara konsisten,
    bukan sekadar menghafal distribusi.

    Cara kerja:
    - Ambil prediksi affix dari GOS
    - Cek validity: apakah imbuhan yang diprediksi merupakan imbuhan valid Indonesia?
    - Cek coherence: apakah kombinasi (root, affix) masuk akal?
    - Cek consistency: apakah transform yang sama untuk kata yang sama konsisten?
    """

    def __init__(self, id_to_affix: Dict[int, str] = None):
        self.id_to_affix = id_to_affix or {v: k for k, v in AFFIX_TO_ID.items()}
        self._valid_affixes = set(PREFIXES_ID) | set(SUFFIXES_ID)
        for p, s in CONFIXES_ID:
            self._valid_affixes.add(f"{p}+{s}")
        self._valid_affixes.add("<NONE>")
        self._valid_affixes.add("<PAD>")
        self._valid_affixes.add("<UNK>")

        # Tracking untuk transform consistency
        self._root_to_predicted_affix: Dict[str, List[str]] = defaultdict(list)

        # Akumulasi
        self._reset()

    def _reset(self):
        self._valid_count = 0
        self._coherent_count = 0
        self._total = 0
        self._root_to_predicted_affix.clear()

    def update(
        self,
        pred_affix_ids: torch.Tensor,   # (B, L) predicted affix ids
        true_affix_ids: torch.Tensor,   # (B, L) ground truth affix ids
        root_texts: Optional[List[List[str]]] = None,   # (B, L) surface roots
        attention_mask: Optional[torch.Tensor] = None,   # (B, L)
    ):
        """Update dengan satu batch."""
        with torch.no_grad():
            B, L = pred_affix_ids.shape

            for b in range(B):
                for l in range(L):
                    if attention_mask is not None and not attention_mask[b, l]:
                        continue

                    pred_id = pred_affix_ids[b, l].item()
                    pred_affix = self.id_to_affix.get(pred_id, "<UNK>")

                    # Validity: apakah imbuhan ini valid?
                    is_valid = pred_affix in self._valid_affixes
                    if is_valid:
                        self._valid_count += 1

                    # Coherence: apakah prefix aktif digunakan wajar?
                    coherent = self._check_coherence(pred_affix, root_texts, b, l)
                    if coherent:
                        self._coherent_count += 1

                    # Track untuk consistency
                    if root_texts is not None:
                        try:
                            root = root_texts[b][l]
                            self._root_to_predicted_affix[root].append(pred_affix)
                        except IndexError:
                            pass

                    self._total += 1

    def _check_coherence(
        self,
        affix: str,
        root_texts: Optional[List[List[str]]],
        b: int, l: int,
    ) -> bool:
        """
        Cek apakah kombinasi (affix, root) koheren.
        Aturan: prefix pasif 'di' tidak boleh dipakai pada kata benda (noun).

        Jika root_texts tidak tersedia, default coherent = True.
        """
        if root_texts is None:
            return True
        try:
            root = root_texts[b][l].lower()
        except IndexError:
            return True

        # Kata benda pendek (< 3 huruf) biasanya tidak berimbuhan aktif
        if affix in _ACTIVE_VERB_PREFIXES and len(root) < 3:
            return False

        # Partikel tidak boleh berimbuhan
        if root in {"di", "ke", "dari", "dan", "atau", "yang", "ini", "itu"} and affix != "<NONE>":
            return False

        return True

    def compute(self) -> MCSResult:
        """Hitung MCS dari akumulasi."""
        if self._total == 0:
            return MCSResult(0.0, 0.0, 0.0, 0.0, 0)

        affix_validity = self._valid_count / self._total
        root_affix_coherence = self._coherent_count / self._total

        # Transform consistency: untuk setiap root, seberapa konsisten imbuhan diprediksi?
        consistency_scores = []
        for root, affixes in self._root_to_predicted_affix.items():
            if len(affixes) < 2:
                continue
            from collections import Counter
            most_common_count = Counter(affixes).most_common(1)[0][1]
            consistency_scores.append(most_common_count / len(affixes))

        transform_consistency = (
            sum(consistency_scores) / len(consistency_scores)
            if consistency_scores else 1.0
        )

        overall = (affix_validity + root_affix_coherence + transform_consistency) / 3

        return MCSResult(
            affix_validity=affix_validity,
            root_affix_coherence=root_affix_coherence,
            transform_consistency=transform_consistency,
            overall=overall,
            n_tokens=self._total,
        )

    def reset(self):
        self._reset()


# ─── 2. StructureValidityScore ───────────────────────────────────────────────

@dataclass
class SVSResult:
    """Hasil Structure Validity Score satu batch/epoch."""
    spok_completeness: float     # % kalimat yang punya minimal S dan P (0..1)
    order_validity: float        # % transisi role yang valid (0..1)
    dep_coherence: float         # % token yang punya minimal 1 dependency edge (0..1)
    overall: float
    n_sentences: int

    def __str__(self) -> str:
        return (
            f"SVS={self.overall:.3f} "
            f"[spok={self.spok_completeness:.3f} "
            f"order={self.order_validity:.3f} "
            f"dep={self.dep_coherence:.3f}]"
            f" n={self.n_sentences}"
        )


class StructureValidityScore:
    """
    Structure Validity Score (SVS).

    Mengukur apakah kalimat yang diprediksi memiliki struktur S-P-O-K yang valid.
    Ini jauh lebih kuat dari sekadar role accuracy karena mengevaluasi
    kalimat sebagai unit, bukan token individual.
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._spok_complete = 0
        self._order_valid = 0
        self._dep_coherent = 0
        self._total_sentences = 0
        self._total_transitions = 0
        self._valid_transitions = 0

    def update(
        self,
        pred_role_ids: torch.Tensor,     # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L)
        dep_masks: Optional[torch.Tensor] = None,       # (B, L, L)
    ):
        """Update dengan satu batch prediksi role."""
        with torch.no_grad():
            B, L = pred_role_ids.shape

            for b in range(B):
                # Extract roles untuk kalimat ini (filter padding)
                if attention_mask is not None:
                    valid_mask = attention_mask[b].bool()
                    roles_seq = pred_role_ids[b][valid_mask].tolist()
                else:
                    roles_seq = pred_role_ids[b].tolist()

                roles_str = [_ID_TO_ROLE.get(r, "UNK") for r in roles_seq]

                # SPOK completeness: minimal ada S dan P
                has_s = "S" in roles_str
                has_p = "P" in roles_str
                if has_s and has_p:
                    self._spok_complete += 1

                # Order validity: cek transisi role
                for i in range(len(roles_str) - 1):
                    r1, r2 = roles_str[i], roles_str[i + 1]
                    self._total_transitions += 1
                    if (r1, r2) in _VALID_ROLE_TRANSITIONS:
                        self._valid_transitions += 1

                # Dep coherence: berapa banyak token yang punya edge
                if dep_masks is not None:
                    dep_mask = dep_masks[b]  # (L, L)
                    # Token yang punya minimal 1 neighbor
                    has_edge = dep_mask.any(dim=-1) | dep_mask.any(dim=0)
                    if attention_mask is not None:
                        valid = attention_mask[b].bool()
                        coherent = has_edge[valid].float().mean().item()
                    else:
                        coherent = has_edge.float().mean().item()
                    self._dep_coherent += coherent
                else:
                    self._dep_coherent += 1.0  # assume coherent if no dep_mask

                self._total_sentences += 1

    def compute(self) -> SVSResult:
        if self._total_sentences == 0:
            return SVSResult(0.0, 0.0, 0.0, 0.0, 0)

        spok = self._spok_complete / self._total_sentences
        order = (self._valid_transitions / max(self._total_transitions, 1))
        dep = self._dep_coherent / self._total_sentences
        overall = (spok + order + dep) / 3

        return SVSResult(
            spok_completeness=spok,
            order_validity=order,
            dep_coherence=dep,
            overall=overall,
            n_sentences=self._total_sentences,
        )

    def reset(self):
        self._reset()


# ─── 3. SemanticDriftScore ───────────────────────────────────────────────────

@dataclass
class SDSSnapshot:
    """Snapshot semantic slots pada satu titik waktu (epoch/step)."""
    epoch: int
    step: int
    mean_anchor_distance: float   # rata-rata jarak cosine ke KBBI anchor (rendah = baik)
    mean_slot_norm: float         # rata-rata norm semantic slot (stabilitas)
    coverage_score: float         # % token yang semantic slot-nya dekat KBBI (< threshold)
    n_tokens: int


@dataclass
class SDSResult:
    """Hasil Semantic Drift Score."""
    current_anchor_distance: float   # jarak saat ini ke KBBI (0 = sempurna aligned)
    drift_velocity: float            # perubahan anchor distance dari snapshot sebelumnya
    coverage_score: float            # % token yang tetap ter-grounded ke KBBI
    overall: float                   # composite score (lebih tinggi = lebih baik)
    n_tokens: int
    snapshots: List[SDSSnapshot] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"SDS={self.overall:.3f} "
            f"[anchor_dist={self.current_anchor_distance:.3f} "
            f"drift_vel={self.drift_velocity:+.4f} "
            f"coverage={self.coverage_score:.3f}]"
            f" n={self.n_tokens}"
        )


class SemanticDriftScore:
    """
    Semantic Drift Score (SDS).

    Mengukur apakah semantic slot BSU tetap ter-grounded ke KBBI
    selama proses training.

    Ini yang memastikan AKSARA tidak kehilangan identitas semantiknya:
    jika semantic slot bergerak jauh dari KBBI anchor → model mulai
    belajar representasi arbitrary, bukan struktur makna Indonesia.

    Tracked per epoch → digunakan oleh Stability Benchmark.
    """

    GROUNDED_THRESHOLD = 0.95  # anchor_distance < ini → dianggap grounded
                                   # 0.95 realistis: cosine dist of random vectors ≈ 1.0

    def __init__(self):
        self._snapshots: List[SDSSnapshot] = []
        self._reset()

    def _reset(self):
        self._anchor_dist_sum = 0.0
        self._slot_norm_sum = 0.0
        self._grounded_count = 0
        self._total = 0

    def update(
        self,
        semantic_slots: torch.Tensor,          # (B, L, d_sem)
        kbbi_anchors: torch.Tensor,            # (B, L, d_sem)
        attention_mask: Optional[torch.Tensor] = None,
        kbbi_mask: Optional[torch.Tensor] = None,  # (B, L) bool — token yang ada di KBBI
    ):
        """Update SDS dengan satu batch semantic slots."""
        with torch.no_grad():
            # Gunakan kbbi_mask (pre-projection) jika tersedia,
            # fallback ke heuristik abs().sum() untuk kompatibilitas mundur.
            # PENTING: _sem_proj mengubah zero-vector OOV menjadi non-zero,
            # sehingga heuristik abs().sum() tidak bisa membedakan OOV vs KBBI.
            if kbbi_mask is not None:
                has_kbbi = kbbi_mask.bool()
            else:
                has_kbbi = kbbi_anchors.abs().sum(dim=-1) > 1e-6  # (B, L)
            if attention_mask is not None:
                has_kbbi = has_kbbi & attention_mask.bool()

            if not has_kbbi.any():
                return

            # Cosine DISTANCE (bukan similarity): 1 - cos_sim ∈ [0, 2]
            cos_sim = F.cosine_similarity(
                semantic_slots, kbbi_anchors, dim=-1, eps=1e-8
            )  # (B, L)
            cos_sim = torch.nan_to_num(cos_sim, nan=0.0)
            anchor_dist = 1.0 - cos_sim  # distance: rendah = baik

            slots_with_kbbi = semantic_slots[has_kbbi]
            dist_with_kbbi = anchor_dist[has_kbbi]

            self._anchor_dist_sum += dist_with_kbbi.sum().item()
            self._slot_norm_sum += slots_with_kbbi.norm(dim=-1).sum().item()
            self._grounded_count += (dist_with_kbbi < self.GROUNDED_THRESHOLD).sum().item()
            self._total += has_kbbi.sum().item()

    def take_snapshot(self, epoch: int, step: int) -> SDSSnapshot:
        """
        Ambil snapshot kondisi saat ini dan simpan ke history.
        Dipanggil di akhir setiap epoch.
        """
        if self._total == 0:
            snap = SDSSnapshot(
                epoch=epoch, step=step,
                mean_anchor_distance=0.0,
                mean_slot_norm=0.0,
                coverage_score=0.0,
                n_tokens=0,
            )
        else:
            snap = SDSSnapshot(
                epoch=epoch,
                step=step,
                mean_anchor_distance=self._anchor_dist_sum / self._total,
                mean_slot_norm=self._slot_norm_sum / self._total,
                coverage_score=self._grounded_count / self._total,
                n_tokens=self._total,
            )
        self._snapshots.append(snap)
        self._reset()
        return snap

    def compute(self) -> SDSResult:
        """Hitung SDS berdasarkan snapshot terakhir dan history drift."""
        if not self._snapshots:
            # Belum ada snapshot — compute dari akumulasi saat ini
            if self._total == 0:
                return SDSResult(0.0, 0.0, 0.0, 0.0, 0)
            dist = self._anchor_dist_sum / self._total
            cov = self._grounded_count / self._total
            # overall: berapa jauh dari baseline (dist=1.0 random)
            # positif = lebih grounded dari random, negatif = lebih buruk dari random
            # dikliping ke [0, 1] untuk tampilan bersih
            overall = max(0.0, 1.0 - dist)  # 0 if dist>=1, positive if dist<1
            return SDSResult(
                current_anchor_distance=dist,
                drift_velocity=0.0,
                coverage_score=cov,
                overall=overall,
                n_tokens=self._total,
            )

        latest = self._snapshots[-1]
        dist = latest.mean_anchor_distance
        cov = latest.coverage_score

        # Drift velocity: perubahan anchor distance dari 2 epoch terakhir
        if len(self._snapshots) >= 2:
            prev = self._snapshots[-2]
            drift_vel = latest.mean_anchor_distance - prev.mean_anchor_distance
        else:
            drift_vel = 0.0

        # Overall: reward jika dist < 1.0 (lebih grounded dari random vectors)
        # Drift penalty hanya saat drift positif signifikan (menjauhi anchor)
        drift_penalty = max(0.0, drift_vel) * 0.5
        overall = max(0.0, (1.0 - dist) - drift_penalty)

        return SDSResult(
            current_anchor_distance=dist,
            drift_velocity=drift_vel,
            coverage_score=cov,
            overall=overall,
            n_tokens=latest.n_tokens,
            snapshots=list(self._snapshots),
        )

    def reset(self):
        self._reset()

    def reset_all(self):
        """Reset termasuk history snapshot."""
        self._snapshots.clear()
        self._reset()

    @property
    def history(self) -> List[SDSSnapshot]:
        return list(self._snapshots)


# ─── IndoNativeMetrics: wrapper semua metrik ─────────────────────────────────

@dataclass
class IndoNativeMetricsResult:
    """Hasil lengkap evaluasi Indo-native per epoch."""
    epoch: int
    mcs: MCSResult
    svs: SVSResult
    sds: SDSResult
    # Metrik klasik (tetap ada untuk referensi)
    morph_accuracy: float
    root_perplexity: float

    def summary(self) -> str:
        lines = [
            f"=== Epoch {self.epoch} — Indo-Native Metrics ===",
            f"  {self.mcs}",
            f"  {self.svs}",
            f"  {self.sds}",
            f"  morph_acc={self.morph_accuracy:.3f}  root_ppl={self.root_perplexity:.1f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "epoch": self.epoch,
            "mcs_overall": self.mcs.overall,
            "mcs_affix_validity": self.mcs.affix_validity,
            "mcs_root_affix_coherence": self.mcs.root_affix_coherence,
            "mcs_transform_consistency": self.mcs.transform_consistency,
            "svs_overall": self.svs.overall,
            "svs_spok_completeness": self.svs.spok_completeness,
            "svs_order_validity": self.svs.order_validity,
            "svs_dep_coherence": self.svs.dep_coherence,
            "sds_overall": self.sds.overall,
            "sds_anchor_distance": self.sds.current_anchor_distance,
            "sds_drift_velocity": self.sds.drift_velocity,
            "sds_coverage": self.sds.coverage_score,
            "morph_accuracy": self.morph_accuracy,
            "root_perplexity": self.root_perplexity,
        }


class IndoNativeMetrics:
    """
    Evaluator utama Indo-native untuk AKSARA.

    Menggabungkan MCS + SVS + SDS menjadi satu interface yang bisa
    dipanggil dari AksaraTrainer.

    Penggunaan:
        metrics = IndoNativeMetrics()

        # Saat training loop
        for batch in dataloader:
            outputs = model(batch, targets=targets)
            metrics.update(outputs, targets, ...)

        # Di akhir epoch
        result = metrics.end_epoch(epoch=1)
        print(result.summary())
        metrics.reset()
    """

    def __init__(self, id_to_affix: Dict[int, str] = None):
        self.mcs = MorphologicalConsistencyScore(id_to_affix)
        self.svs = StructureValidityScore()
        self.sds = SemanticDriftScore()

        # Classic metrics
        self._morph_correct = 0
        self._morph_total = 0
        self._ctx_nll_sum = 0.0
        self._ctx_steps = 0

    def update(
        self,
        gos_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        semantic_slots: torch.Tensor,
        kbbi_anchors: torch.Tensor,
        dep_masks: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        root_texts: Optional[List[List[str]]] = None,
        kbbi_mask: Optional[torch.Tensor] = None,
    ):
        """
        Update semua metrik dari satu batch.

        Args:
            gos_output     : dict logits dari GOS
            targets        : ground truth (root_ids, affix_ids, role_ids)
            semantic_slots : (B, L, d_sem) dari BSU encode
            kbbi_anchors   : (B, L, d_sem) dari LSK (sudah diproyeksikan ke d_sem)
            dep_masks      : (B, L, L) dependency mask opsional
            attention_mask : (B, L) padding mask
            root_texts     : (B, L) teks root per token, untuk MCS coherence check
            kbbi_mask      : (B, L) bool — token yang benar-benar ada di KBBI (pre-projection)
        """
        with torch.no_grad():
            pred_affix = gos_output["affix_logits"].argmax(dim=-1)
            pred_role  = gos_output["role_logits"].argmax(dim=-1)

            # MCS update
            self.mcs.update(pred_affix, targets["affix_ids"],
                            root_texts, attention_mask)

            # SVS update
            self.svs.update(pred_role, attention_mask, dep_masks)

            # SDS update
            self.sds.update(semantic_slots, kbbi_anchors, attention_mask, kbbi_mask)

            # Classic morph accuracy
            true_affix = targets["affix_ids"]
            if attention_mask is not None:
                valid = attention_mask.bool()
                self._morph_correct += (pred_affix[valid] == true_affix[valid]).sum().item()
                self._morph_total += valid.sum().item()
            else:
                self._morph_correct += (pred_affix == true_affix).sum().item()
                self._morph_total += pred_affix.numel()

            # Classic perplexity
            ctx_logits = gos_output["context_logits"]
            true_root  = targets["root_ids"]
            if ctx_logits.shape[1] > 1:
                nll = F.cross_entropy(
                    ctx_logits[:, :-1].contiguous().view(-1, ctx_logits.size(-1)),
                    true_root[:, 1:].contiguous().view(-1),
                    ignore_index=0, reduction="mean",
                )
                if not torch.isnan(nll):
                    self._ctx_nll_sum += nll.item()
                    self._ctx_steps += 1

    def end_epoch(self, epoch: int) -> IndoNativeMetricsResult:
        """
        Tutup epoch: ambil snapshot SDS, compute semua metrik, return result.
        Tidak reset — panggil reset() secara eksplisit setelah ini.
        """
        mcs_result = self.mcs.compute()
        svs_result = self.svs.compute()
        self.sds.take_snapshot(epoch=epoch, step=epoch)
        sds_result = self.sds.compute()

        morph_acc = (self._morph_correct / max(self._morph_total, 1))
        avg_nll = self._ctx_nll_sum / max(self._ctx_steps, 1)
        root_ppl = math.exp(min(avg_nll, 20))

        return IndoNativeMetricsResult(
            epoch=epoch,
            mcs=mcs_result,
            svs=svs_result,
            sds=sds_result,
            morph_accuracy=morph_acc,
            root_perplexity=root_ppl,
        )

    def reset(self):
        """Reset akumulasi per-epoch (tidak reset SDS history)."""
        self.mcs.reset()
        self.svs.reset()
        self.sds.reset()
        self._morph_correct = 0
        self._morph_total = 0
        self._ctx_nll_sum = 0.0
        self._ctx_steps = 0

    def reset_all(self):
        """Reset total termasuk SDS history (untuk run baru)."""
        self.reset()
        self.sds.reset_all()
