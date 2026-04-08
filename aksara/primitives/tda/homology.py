"""
PersistentHomology — komputasi homologi persisten untuk deteksi anomali semantik.

OPOSISI TRANSFORMER:
  Transformer: tidak ada analisis topologis — tidak bisa mendeteksi "lubang" semantik
  TDA:         persistent homology mengukur kapan fitur topologis "lahir" dan "mati"
               saat threshold distance dinaikkan secara bertahap

Barcode diagram:
  - Tiap interval [birth, death] = satu fitur topologis
  - Interval panjang = fitur persisten = struktur semantik nyata
  - Interval pendek  = noise
  - β₀ persistences = cluster semantik kalimat
  - β₁ persistences = inkoherensi / "lubang" semantik (anomali)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from aksara.primitives.tda.simplex import SimplicialComplex


class IntervalPersisten:
    """Satu interval dalam barcode diagram."""
    __slots__ = ["dimensi", "lahir", "mati", "panjang", "generator"]

    def __init__(
        self,
        dimensi: int,
        lahir: float,
        mati: float,
        generator: Optional[str] = None,
    ):
        self.dimensi   = dimensi
        self.lahir     = lahir
        self.mati      = mati                        # inf = fitur yang tidak pernah mati
        self.panjang   = (mati - lahir) if not math.isinf(mati) else float("inf")
        self.generator = generator

    @property
    def persisten(self) -> bool:
        """Fitur persisten jika panjang > threshold noise."""
        return self.panjang > 0.5 or math.isinf(self.panjang)

    def __repr__(self):
        mati_str = "∞" if math.isinf(self.mati) else f"{self.mati:.2f}"
        return f"H{self.dimensi}[{self.lahir:.2f}, {mati_str})"


class PersistentHomology:
    """
    Komputasi homologi persisten untuk kalimat bahasa Indonesia.

    Algoritma:
    1. Bangun simplicial complex pada threshold ε = 0
    2. Naikkan ε secara bertahap (dari 0 ke max_distance)
    3. Catat kapan komponen terhubung (β₀) dan cycle (β₁) lahir dan mati
    4. Output: barcode diagram = list IntervalPersisten

    Interpretasi untuk NLP:
    - β₀ interval panjang = komponen terisolasi semantically (kata tidak related)
    - β₁ interval > 0 = ada siklus semantik (inkoherensi)
    - Kalimat benar: satu komponen besar, tidak ada cycle
    - Kalimat salah: multiple komponen, mungkin ada cycle
    """

    def __init__(
        self,
        n_threshold: int = 20,
        max_distance: float = 3.0,
    ):
        self.n_threshold  = n_threshold
        self.max_distance = max_distance

    def hitung(
        self,
        vertices: List[str],
        jarak_fn,
    ) -> List[IntervalPersisten]:
        """
        Hitung persistent homology via Vietoris-Rips filtration.

        Args:
            vertices: list nama vertex (root morfem)
            jarak_fn: callable(a, b) -> float

        Returns:
            list IntervalPersisten (barcode diagram)
        """
        if len(vertices) <= 1:
            return [IntervalPersisten(0, 0.0, float("inf"), vertices[0] if vertices else "")]

        # Pre-hitung semua jarak
        jarak_matrix: Dict[Tuple[str, str], float] = {}
        semua_jarak = []
        for i, a in enumerate(vertices):
            for j, b in enumerate(vertices):
                if i < j:
                    d = jarak_fn(a, b)
                    key = (a, b)
                    jarak_matrix[key] = d
                    semua_jarak.append(d)

        # Threshold steps
        thresholds = sorted(set(semua_jarak))
        thresholds = [0.0] + thresholds + [self.max_distance]

        # State β₀: track komponen terhubung via union-find
        intervals: List[IntervalPersisten] = []
        parent: Dict[str, str] = {v: v for v in vertices}
        komp_lahir: Dict[str, float] = {v: 0.0 for v in vertices}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str, eps: float) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            # Komponen yang lahir lebih terlambat "mati" (merge ke yang lebih awal)
            lahir_a = komp_lahir.get(ra, 0.0)
            lahir_b = komp_lahir.get(rb, 0.0)
            if lahir_a <= lahir_b:
                # rb merge ke ra — catat kematian rb
                intervals.append(IntervalPersisten(0, lahir_b, eps, rb))
                parent[rb] = ra
            else:
                intervals.append(IntervalPersisten(0, lahir_a, eps, ra))
                parent[ra] = rb
                komp_lahir[rb] = min(lahir_a, lahir_b)
            return True

        # Filtration: naikkan threshold, tambah edge satu per satu
        sudah_ditambah: set = set()
        for eps in thresholds:
            for i, a in enumerate(vertices):
                for j, b in enumerate(vertices):
                    if i >= j:
                        continue
                    key = (a, b)
                    d = jarak_matrix.get(key, float("inf"))
                    if d <= eps and key not in sudah_ditambah:
                        sudah_ditambah.add(key)
                        union(a, b, eps)

        # Komponen yang masih hidup sampai akhir → interval [lahir, ∞)
        komponen_akhir: Dict[str, str] = {}
        for v in vertices:
            r = find(v)
            if r not in komponen_akhir:
                komponen_akhir[r] = v

        for r in komponen_akhir:
            lahir = komp_lahir.get(r, 0.0)
            intervals.append(IntervalPersisten(0, lahir, float("inf"), r))

        return sorted(intervals, key=lambda x: x.lahir)

    def ringkas(self, intervals: List[IntervalPersisten]) -> Dict:
        """
        Ringkas barcode diagram menjadi metrik yang mudah diinterpretasi.
        """
        h0 = [iv for iv in intervals if iv.dimensi == 0]
        h1 = [iv for iv in intervals if iv.dimensi == 1]

        # Komponen terhubung akhir = yang tidak pernah mati
        komponen_akhir = [iv for iv in h0 if math.isinf(iv.mati)]
        n_komponen     = len(komponen_akhir)

        # Komponen terisolasi = yang pernah mati tapi lahir terlambat
        terisolasi = [iv for iv in h0
                      if not math.isinf(iv.mati) and iv.lahir > 0.5]

        # Fitur H1 persisten = anomali inkoherensi
        anomali_h1 = [iv for iv in h1 if iv.persisten]

        return {
            "n_komponen_akhir": n_komponen,
            "n_terisolasi":     len(terisolasi),
            "n_anomali_h1":     len(anomali_h1),
            "koheren":          n_komponen == 1 and len(anomali_h1) == 0,
            "komponen_akhir":   [iv.generator for iv in komponen_akhir],
            "terisolasi":       [iv.generator for iv in terisolasi],
        }
