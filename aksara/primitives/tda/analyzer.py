"""
TDAnalyzer — Topological Dependency Analyzer (Primitif 5 AKSARA Framework).

OPOSISI TRANSFORMER:
  Transformer: tidak ada analisis multi-skala — depth = jumlah layer (tetap)
  TDA:         persistent homology mendeteksi struktur di berbagai skala jarak

Apa yang dideteksi TDA yang tidak bisa dideteksi CPE/CMC:
  1. Isolated cluster: kelompok kata yang semantically terisolasi dari kalimat
  2. Domain fragmentation: kalimat dengan ≥2 cluster domain berbeda tanpa jembatan
  3. Multi-scale anomaly: kata yang cocok di skala lokal tapi anomali di skala global
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from aksara.primitives.tda.simplex import SimplicialComplex
from aksara.primitives.tda.homology import PersistentHomology, IntervalPersisten
from aksara.primitives.lps.morfem import Morfem
from aksara.primitives.sfm.geodesic import GeodesicDistance
from aksara.base.state import AksaraState, PelanggaranConstraint


class TDAnalyzer:
    """
    Topological Dependency Analyzer — deteksi anomali struktur kalimat.

    OPOSISI TRANSFORMER:
    Transformer tidak punya mekanisme untuk mendeteksi apakah suatu kata
    "terisolasi" dari konteks kalimat secara global — hanya lokal via attention.

    TDAnalyzer membangun simplicial complex dari kalimat dan menghitung
    invariant topologis (Betti numbers) yang tidak bergantung pada representasi
    vektor — ini adalah properti geometris yang lebih fundamental.

    Deteksi:
    - β₀ > 1: ada kata terisolasi (tidak terhubung ke cluster utama)
    - β₁ > 0: ada "lubang" semantik (cycle tanpa makna)
    - Isolated vertices: kata yang tidak punya tetangga dalam threshold
    """

    def __init__(
        self,
        geodesic: GeodesicDistance,
        threshold_edge: float = 1.5,
        n_threshold_filtration: int = 15,
    ):
        self.geodesic   = geodesic
        self.threshold  = threshold_edge
        self.homology   = PersistentHomology(
            n_threshold=n_threshold_filtration,
            max_distance=3.0,
        )

    def analisis(self, morfem_list: List[Morfem]) -> Dict:
        """
        Analisis topologis kalimat lengkap.

        Returns:
            dict dengan:
              - betti_0:          jumlah komponen terhubung
              - betti_1:          approksimasi jumlah cycle
              - isolated:         list root morfem yang terisolasi
              - koheren:          bool — apakah kalimat topologis koheren
              - anomali:          bool — apakah ada anomali topologis signifikan
              - pelanggaran_tda:  list PelanggaranConstraint
              - intervals:        barcode diagram
              - matriks_adj:      tensor adjacency
        """
        if len(morfem_list) <= 1:
            return self._hasil_trivial(morfem_list)

        # Bangun simplicial complex
        sc = SimplicialComplex.dari_morfem_dan_jarak(
            morfem_list,
            jarak_fn=self.geodesic.hitung,
            threshold=self.threshold,
        )

        # Hitung Betti numbers
        b0 = sc.betti_0()
        b1 = sc.betti_1_approx()
        isolated = sc.isolated_vertices()

        # Hitung persistent homology
        roots = [m.root for m in morfem_list]
        intervals = self.homology.hitung(roots, self.geodesic.hitung)
        ringkasan = self.homology.ringkas(intervals)

        # β₁ baseline: kalimat normal dengan n morfem dan window-2 adjacency
        # akan punya sekitar max(0, n_edges - n_vertices + 1) cycles secara natural.
        # Hanya anomali jika β₁ JAUH di atas baseline yang diharapkan.
        b1_baseline = max(0, sc.n_edges - sc.n_vertices)
        b1_anomali  = max(0, b1 - b1_baseline)

        # Deteksi anomali
        anomali = (b0 > 1) or (b1_anomali > 2) or (len(isolated) > 0)

        # Bangun pelanggaran — hanya cycle yang anomali (jauh di atas baseline)
        pelanggaran = self._bangun_pelanggaran(
            sc, b0, b1_anomali, isolated, ringkasan, morfem_list
        )

        return {
            "betti_0":          b0,
            "betti_1":          b1,
            "isolated":         isolated,
            "koheren":          not anomali,
            "anomali":          anomali,
            "pelanggaran_tda":  pelanggaran,
            "intervals":        intervals,
            "ringkasan":        ringkasan,
            "n_vertices":       sc.n_vertices,
            "n_edges":          sc.n_edges,
            "n_triangles":      sc.n_triangles,
            "euler":            sc.euler_characteristic(),
            "matriks_adj":      sc.matriks_adjacency(),
        }

    def perkaya_state(
        self,
        state: AksaraState,
        morfem_list: List[Morfem],
    ) -> AksaraState:
        """Perkaya AksaraState dengan hasil analisis TDA."""
        hasil = self.analisis(morfem_list)

        pelanggaran_baru = list(state.pelanggaran) + hasil["pelanggaran_tda"]
        # Hanya hitung energi dari anomali yang benar-benar signifikan
        b1_raw      = float(hasil["betti_1"])
        n_vertices  = float(max(hasil["n_vertices"], 1))
        n_edges     = float(hasil["n_edges"])
        b1_baseline = max(0.0, n_edges - n_vertices)
        b1_anomali  = max(0.0, b1_raw - b1_baseline)
        energi_tda  = b1_anomali * 0.2 + float(len(hasil["isolated"])) * 0.3
        energi_baru = state.energi_total + energi_tda * 0.15

        meta_baru = dict(state.metadata)
        meta_baru["betti_0"]   = hasil["betti_0"]
        meta_baru["betti_1"]   = hasil["betti_1"]
        meta_baru["n_isolated"] = len(hasil["isolated"])
        meta_baru["koheren_tda"] = hasil["koheren"]

        from dataclasses import replace
        return replace(
            state,
            pelanggaran=pelanggaran_baru,
            energi_total=energi_baru,
            energi_per_dimensi={
                **state.energi_per_dimensi,
                "topologis": energi_tda,
            },
            anomali_topologis=hasil["anomali"],
            fitur_topologis=hasil["matriks_adj"].flatten(),
            metadata=meta_baru,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _bangun_pelanggaran(
        self,
        sc: SimplicialComplex,
        b0: int,
        b1: int,
        isolated: List[str],
        ringkasan: Dict,
        morfem_list: List[Morfem],
    ) -> List[PelanggaranConstraint]:
        hasil = []

        # Pelanggaran: kata terisolasi
        for kata in isolated:
            morfem_terlibat = next(
                (m for m in morfem_list if m.root == kata), None
            )
            hasil.append(PelanggaranConstraint(
                tipe="topologis",
                token_terlibat=[kata],
                dimensi="topologis",
                severitas=0.7,
                penjelasan=(
                    f"Kata '{kata}' terisolasi secara semantik — "
                    f"tidak ada morfem lain dalam kalimat dengan jarak ≤ {self.threshold:.1f}. "
                    f"Kemungkinan kata ini tidak cocok dengan konteks kalimat."
                ),
            ))

        # Pelanggaran: fragmentation (β₀ > 1)
        if b0 > 1:
            komponen = ringkasan.get("komponen_akhir", [])
            hasil.append(PelanggaranConstraint(
                tipe="topologis",
                token_terlibat=komponen[:4],
                dimensi="topologis",
                severitas=0.8,
                penjelasan=(
                    f"Kalimat terfragmentasi: {b0} cluster semantik terpisah "
                    f"(komponen: {', '.join(str(c) for c in komponen[:3])}...). "
                    f"Kalimat yang koheren harus punya satu cluster terhubung."
                ),
            ))

        # Pelanggaran: cycle (β₁ > 0)
        if b1 > 0:
            hasil.append(PelanggaranConstraint(
                tipe="topologis",
                token_terlibat=[m.root for m in morfem_list[:3]],
                dimensi="topologis",
                severitas=0.5,
                penjelasan=(
                    f"Terdeteksi {b1} siklus semantik dalam kalimat. "
                    f"Siklus yang tidak bermakna bisa mengindikasikan redundansi "
                    f"atau inkonsistensi makna."
                ),
            ))

        return hasil

    def _hasil_trivial(self, morfem_list: List[Morfem]) -> Dict:
        """Hasil trivial untuk kalimat dengan ≤1 morfem."""
        return {
            "betti_0": 1, "betti_1": 0,
            "isolated": [], "koheren": True, "anomali": False,
            "pelanggaran_tda": [], "intervals": [],
            "ringkasan": {"n_komponen_akhir": 1, "koheren": True},
            "n_vertices": len(morfem_list), "n_edges": 0, "n_triangles": 0,
            "euler": len(morfem_list),
            "matriks_adj": torch.zeros(len(morfem_list), len(morfem_list)),
        }
