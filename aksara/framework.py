"""
AksaraFramework — API publik AKSARA Framework.

Ini adalah entry point utama untuk developer yang membangun model NLP Indonesia.

CARA PAKAI:
    from aksara.framework import AksaraFramework

    fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")
    state = fw.proses("Makanan tradisional khas Dompu sangat lezat.")
    print(state.ringkasan())

OPOSISI PARADIGMA:
  Model statistik token: developer perlu paham detail internal mekanisme prediksi
  AKSARA:               developer hanya perlu tahu: proses(kalimat) → AksaraState yang interpretatif

Framework ini menjamin:
  1. Unit dasar = morfem (bukan token arbitrer)
  2. Representasi = state dinamis di manifold semantik (bukan vektor statis)
  3. Mekanisme = constraint propagation (bukan attention O(n²))
  4. Pengetahuan = eksplisit dari KBBI (bukan implisit di bobot)
  5. Interpretabilitas = setiap keputusan bisa dijelaskan
  6. Update = patch leksikon, tanpa retrain
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from aksara.base.state import (
    AksaraState, ViolationSpan, ConstraintSatisfaction, PelanggaranConstraint
)
from aksara.primitives.lps.parser import LPSParser
from aksara.primitives.lps.morfem import Morfem
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.primitives.sfm.manifold import SemanticManifold
from aksara.primitives.cpe.engine import CPEngine
from aksara.primitives.cmc.composer import CMComposer
from aksara.primitives.tda.analyzer import TDAnalyzer
from aksara.primitives.krl.layer import KRLayer
from aksara.config import AksaraConfig


class AksaraFramework:
    """
    AKSARA Framework — pipeline linguistik native Indonesia.

    Pipeline: Kalimat → LPS → SFM → CPE → CMC → TDA → KRL → AksaraState

    Setiap tahap punya justifikasi linguistik eksplisit:
      LPS: dekomposisi morfem deterministik dari aturan TBBBI
      SFM: representasi semantik dari struktur KBBI
      CPE: evaluasi constraint linguistik Indonesia
      CMC: verifikasi komposisi makna via category theory
      TDA: deteksi anomali topologis multi-skala
      KRL: representasi proposisi + frame + referensi (Primitif 6)

    Semua oposisi terhadap Transformer/Mamba dipertahankan:
      - Tidak ada mekanisme prediksi token berikutnya
      - Tidak ada komposisi makna berbasis weighted-sum global
      - Tidak ada embedding statis di Euclidean space
      - Tidak ada pengetahuan tersembunyi di bobot
    """

    def __init__(
        self,
        leksikon: LexiconLoader,
        device: torch.device = torch.device("cpu"),
        aktif_cmc: bool = True,
        aktif_tda: bool = True,
        aktif_krl: bool = True,
        cpe_max_iter: int = 10,
        threshold_semantik: float = 1.5,
        config: Optional[AksaraConfig] = None,
    ):
        self.leksikon  = leksikon
        self.device    = device
        self._aktif_cmc = aktif_cmc
        self._aktif_tda = aktif_tda
        self._aktif_krl = aktif_krl
        self.config    = config or AksaraConfig.default()

        # Bangun leksikon dict untuk LPS
        leksikon_dict = {k: v.kelas for k, v in leksikon._entri.items()}

        # threshold dari config menang atas parameter langsung
        _threshold = self.config.threshold_semantik if config else threshold_semantik

        # Inisialisasi semua primitif — config diinjeksi ke CPE dan CMC
        self.lps = LPSParser(leksikon=leksikon_dict)
        self.sfm = SemanticManifold(leksikon)
        self.cpe = CPEngine(
            self.sfm,
            max_iter=cpe_max_iter,
            threshold_semantik=_threshold,
            config=self.config,
        )
        self.cmc = CMComposer(leksikon, config=self.config) if aktif_cmc else None
        self.tda = TDAnalyzer(
            self.sfm.geodesic,
            threshold_edge=_threshold,
        ) if aktif_tda else None
        self.krl = KRLayer(leksikon) if aktif_krl else None

    @classmethod
    def dari_kbbi(
        cls,
        kbbi_path: str,
        device: Optional[torch.device] = None,
        config: Optional[AksaraConfig] = None,
        **kwargs,
    ) -> "AksaraFramework":
        """
        Factory method: bangun framework langsung dari file KBBI.

        Args:
            kbbi_path: path ke kbbi_core_v2.json
            device:    torch device (default: cpu)
            config:    AksaraConfig opsional untuk domain khusus
                       (hukum, kesehatan, militer, pertanahan, dll.)
                       Default: AksaraConfig.default() = bahasa Indonesia umum
            **kwargs:  parameter tambahan ke AksaraFramework.__init__

        Contoh:
            # Bahasa Indonesia umum
            fw = AksaraFramework.dari_kbbi('kbbi_core_v2.json')

            # Domain hukum
            from aksara.config import AksaraConfig
            fw = AksaraFramework.dari_kbbi(
                'kbbi_core_v2.json',
                config=AksaraConfig.untuk_domain('hukum')
            )
        """
        if not Path(kbbi_path).exists():
            raise FileNotFoundError(f"KBBI tidak ditemukan: {kbbi_path}")

        leksikon = LexiconLoader()
        n = leksikon.muat_kbbi(kbbi_path)
        if n == 0:
            raise ValueError(f"KBBI gagal dimuat dari: {kbbi_path}")

        device = device or torch.device("cpu")
        fw = cls(leksikon, device=device, config=config, **kwargs)
        return fw

    def proses(self, kalimat: str) -> AksaraState:
        """
        Proses satu kalimat melalui seluruh pipeline AKSARA.

        Pipeline:
          1. LPS  → dekomposisi morfem
          2. SFM  → encode ke tensor semantik
          3. CPE  → evaluasi constraint, hitung energi
          4. CMC  → verifikasi komposisi makna (jika aktif)
          5. TDA  → analisis topologis (jika aktif)
          6. KRL  → representasi proposisi + frame + referensi (jika aktif)
          7. POST → bangun ViolationSpan + ConstraintSatisfaction

        Args:
            kalimat: string kalimat bahasa Indonesia

        Returns:
            AksaraState lengkap dengan skor_linguistik, violation_spans,
            constraint_satisfaction, dan penjelasan per dimensi
        """
        # ── Tahap 1: LPS ─────────────────────────────────────────────────────
        morfem_list = self.lps.parse(kalimat)
        if not morfem_list:
            return self._state_kosong(kalimat)

        # ── Tahap 2: SFM ─────────────────────────────────────────────────────
        sfm_tensor = self.sfm.encode_kalimat(morfem_list, device=self.device)

        # ── Tahap 3: CPE ─────────────────────────────────────────────────────
        state = self.cpe(morfem_list, sfm_tensor=sfm_tensor, device=self.device)

        # ── Tahap 4: CMC ─────────────────────────────────────────────────────
        if self.cmc is not None and self._aktif_cmc:
            state = self.cmc.perkaya_state(state, morfem_list)

        # ── Tahap 5: TDA ─────────────────────────────────────────────────────
        if self.tda is not None and self._aktif_tda:
            state = self.tda.perkaya_state(state, morfem_list)

        # ── Tahap 6: KRL ──────────────────────────────────────────────────────
        if self.krl is not None and self._aktif_krl:
            state.krl_result = self.krl.proses(morfem_list, kalimat)

        # ── Tahap 7: Post-processing — Violation Localization & Satisfaction ──
        state.violation_spans = self._bangun_violation_spans(
            kalimat, morfem_list, state.pelanggaran
        )
        state.constraint_satisfaction = self._bangun_constraint_satisfaction(
            state.energi_per_dimensi, state.pelanggaran,
            n_edge=state.metadata.get("n_edge", max(len(morfem_list) - 1, 1))
        )

        return state

    def _bangun_violation_spans(
        self,
        teks: str,
        morfem_list: List[Morfem],
        pelanggaran: List[PelanggaranConstraint],
    ) -> List[ViolationSpan]:
        """
        Petakan pelanggaran ke posisi karakter di teks asli.

        Strategi: bangun indeks root→(mulai, akhir) dari posisi token di teks,
        lalu untuk setiap pelanggaran, cari token terlibat dan petakan ke span.
        """
        # Bangun peta root → (mulai_char, akhir_char, teks_asli_token)
        root_to_span: Dict[str, tuple] = {}
        cursor = 0
        teks_lower = teks.lower()

        for m in morfem_list:
            token = m.teks_asli
            token_lower = token.lower()
            # Cari posisi token mulai dari cursor
            pos = teks_lower.find(token_lower, cursor)
            if pos == -1:
                # Fallback: cari dari awal (untuk kasus normalisasi)
                pos = teks_lower.find(token_lower, 0)
            if pos != -1:
                root_to_span[m.root.lower()] = (pos, pos + len(token), token)
                root_to_span[token_lower] = (pos, pos + len(token), token)
                cursor = pos + len(token)

        spans: List[ViolationSpan] = []
        seen_spans: set = set()

        for p in pelanggaran:
            for root in p.token_terlibat:
                root_key = root.lower()
                if root_key not in root_to_span:
                    continue
                mulai, akhir, token_asli = root_to_span[root_key]
                span_key = (mulai, akhir, p.dimensi)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)
                spans.append(ViolationSpan(
                    mulai=mulai,
                    akhir=akhir,
                    token=token_asli,
                    root=root,
                    dimensi=p.dimensi,
                    severitas=p.severitas,
                    penjelasan=p.penjelasan,
                ))

        # Urutkan berdasarkan posisi di teks
        spans.sort(key=lambda s: s.mulai)
        return spans

    def _bangun_constraint_satisfaction(
        self,
        energi_per_dimensi: Dict[str, float],
        pelanggaran: List[PelanggaranConstraint],
        n_edge: int,
    ) -> ConstraintSatisfaction:
        """
        Hitung constraint satisfaction per dimensi.

        Formula: satisfaction_d = 1.0 - energi_d
        di mana energi_d sudah dinormalisasi per edge oleh CPEngine (∈ [0,1]).

        Penalti tambahan per pelanggaran berat (severitas > 0.5):
          satisfaction_d -= 0.1 × n_pelang_berat_d / n_edge
        Ini memastikan kalimat dengan banyak pelanggaran mendapat skor lebih rendah
        bahkan jika energi rata-ratanya masih kecil.
        """
        def sat(dim: str) -> float:
            e = energi_per_dimensi.get(dim, 0.0)
            # Hitung penalti dari pelanggaran berat di dimensi ini
            n_berat = sum(
                1 for p in pelanggaran
                if p.dimensi == dim and p.severitas > 0.5
            )
            penalti = 0.1 * n_berat / max(n_edge, 1)
            return max(0.0, min(1.0, 1.0 - e - penalti))

        return ConstraintSatisfaction(
            morfologis=sat("morfologis"),
            sintaktis =sat("sintaktis"),
            semantik  =sat("semantik"),
            leksikal  =sat("leksikal"),
            topologis =sat("topologis"),
            animasi   =sat("animasi"),
        )

    def proses_batch(self, kalimat_list: List[str]) -> List[AksaraState]:
        """Proses sekumpulan kalimat."""
        return [self.proses(k) for k in kalimat_list]

    def tambah_kata(
        self,
        kata: str,
        kelas: str,
        domain: Optional[str] = None,
        sinonim: Optional[List[str]] = None,
        antonim: Optional[List[str]] = None,
    ) -> None:
        """
        Tambah kata baru ke leksikon — update pengetahuan tanpa retrain.

        OPOSISI TRANSFORMER:
        Transformer: menambah pengetahuan = retrain seluruh model
        AKSARA:      menambah pengetahuan = satu baris di sini, langsung berlaku

        Args:
            kata:    lemma kata baru
            kelas:   POS tag ('n', 'v', 'adj', dll.)
            domain:  domain semantik (opsional)
            sinonim: list sinonim (opsional)
            antonim: list antonim (opsional)
        """
        self.leksikon.tambah_entri(
            kata=kata, kelas=kelas, domain=domain,
            sinonim=sinonim, antonim=antonim,
            layer="custom",
        )
        kata_lower = kata.lower()
        # Update leksikon dict di LPS parser
        self.lps.leksikon[kata_lower] = kelas
        # Invalidate SFM cache untuk kata ini dan semua yang berrelasi
        self.sfm._cache.pop(kata_lower, None)
        for s in (sinonim or []):
            self.sfm._cache.pop(s.lower(), None)
        # Invalidate geodesic cache untuk pasangan yang melibatkan kata ini
        keys_to_remove = [
            k for k in self.sfm.geodesic._cache
            if kata_lower in k
        ]
        for k in keys_to_remove:
            self.sfm.geodesic._cache.pop(k, None)
        # Update leksikon_dict di sfm juga
        self.sfm.leksikon_dict[kata_lower] = kelas

    def info(self) -> Dict:
        """Ringkasan konfigurasi framework."""
        return {
            "leksikon_size":    self.leksikon.ukuran,
            "n_domain":         self.leksikon.n_domain,
            "sfm_dim":          self.sfm.d_output,
            "cpe_max_iter":     self.cpe.max_iter,
            "aktif_cmc":        self._aktif_cmc,
            "aktif_tda":        self._aktif_tda,
            "device":           str(self.device),
        }

    def _state_kosong(self, teks: str) -> AksaraState:
        from dataclasses import replace
        return AksaraState(
            teks_asli=teks,
            morfem_states=[],
            energi_total=0.0,
            energi_per_dimensi={},
            pelanggaran=[],
            register="formal",
            kelengkapan_struktur=0.0,
        )
