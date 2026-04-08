"""
SimplicialComplex — representasi topologis dari kalimat bahasa Indonesia.

OPOSISI TRANSFORMER:
  Transformer: relasi antar token = weighted sum (tidak punya struktur topologis)
  TDA:         relasi antar morfem = simplicial complex (punya invariant topologis)

Simplicial Complex dari kalimat:
  - 0-simplex: tiap morfem (vertex)
  - 1-simplex: pasangan morfem yang "dekat" secara semantik (edge)
  - 2-simplex: triple morfem yang semua pasangannya dekat (triangle)
  - k-simplex: (k+1) morfem yang semua pasangannya dekat

Invariant topologis yang dihitung:
  - β₀ (Betti number 0): jumlah komponen terhubung (harusnya 1 jika kalimat koheren)
  - β₁ (Betti number 1): jumlah "lubang" / isolated cluster (anomali jika > 0)
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch


class SimplicialComplex:
    """
    Simplicial complex dari kalimat — representasi topologis.

    OPOSISI TRANSFORMER:
    Transformer merepresentasikan relasi token sebagai matriks dense.
    SimplicialComplex merepresentasikan relasi morfem sebagai struktur
    kombinatorial yang punya invariant topologis yang dapat dihitung.

    Properti:
    - Bersifat sparse — hanya pasangan yang "dekat" yang dihubungkan
    - Invariant topologis (Betti numbers) tidak bergantung pada
      pilihan representasi vektor — ini adalah properti geometris murni
    - Anomali topologis terdeteksi sebagai β₁ > 0 (ada "lubang")
    """

    def __init__(self, threshold: float = 1.5):
        """
        Args:
            threshold: batas jarak semantik untuk membuat edge (1-simplex)
        """
        self.threshold = threshold
        self._vertices: List[str]                   = []
        self._edges: Set[FrozenSet[str]]            = set()
        self._triangles: Set[FrozenSet[str]]        = set()
        self._jarak: Dict[Tuple[str, str], float]   = {}

    @classmethod
    def dari_morfem_dan_jarak(
        cls,
        morfem_list: List,
        jarak_fn,
        threshold: float = 1.5,
    ) -> "SimplicialComplex":
        """
        Bangun simplicial complex dari morfem list dan fungsi jarak.

        Args:
            morfem_list: list Morfem dari LPS
            jarak_fn:    callable(root_a, root_b) -> float (dari GeodesicDistance)
            threshold:   batas jarak untuk membuat edge
        """
        sc = cls(threshold=threshold)
        roots = [m.root for m in morfem_list]
        sc._vertices = roots

        # Hitung semua jarak pasangan
        n = len(roots)
        for i in range(n):
            for j in range(i + 1, n):
                d = jarak_fn(roots[i], roots[j])
                key = (min(roots[i], roots[j]), max(roots[i], roots[j]))
                sc._jarak[key] = d

                # 1-simplex: edge jika jarak <= threshold
                if d <= threshold:
                    sc._edges.add(frozenset([roots[i], roots[j]]))

        # 2-simplex: triangle jika semua 3 pasangan dekat
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    ab = sc._jarak.get(
                        (min(roots[i], roots[j]), max(roots[i], roots[j])), float("inf"))
                    ac = sc._jarak.get(
                        (min(roots[i], roots[k]), max(roots[i], roots[k])), float("inf"))
                    bc = sc._jarak.get(
                        (min(roots[j], roots[k]), max(roots[j], roots[k])), float("inf"))
                    if max(ab, ac, bc) <= threshold:
                        sc._triangles.add(frozenset([roots[i], roots[j], roots[k]]))

        return sc

    @property
    def n_vertices(self) -> int:
        return len(self._vertices)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def n_triangles(self) -> int:
        return len(self._triangles)

    def euler_characteristic(self) -> int:
        """
        Karakteristik Euler: χ = V - E + F
        Untuk kalimat yang benar dan koheren, χ seharusnya konsisten.
        """
        return self.n_vertices - self.n_edges + self.n_triangles

    def komponen_terhubung(self) -> List[Set[str]]:
        """
        Hitung komponen terhubung via union-find.
        β₀ = jumlah komponen terhubung.
        β₀ > 1 = kalimat punya kata yang terisolasi semantically.
        """
        parent: Dict[str, str] = {v: v for v in self._vertices}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in self._edges:
            vertices = list(edge)
            if len(vertices) == 2:
                union(vertices[0], vertices[1])

        komponen: Dict[str, Set[str]] = {}
        for v in self._vertices:
            root = find(v)
            if root not in komponen:
                komponen[root] = set()
            komponen[root].add(v)

        return list(komponen.values())

    def betti_0(self) -> int:
        """β₀ = jumlah komponen terhubung."""
        return len(self.komponen_terhubung())

    def betti_1_approx(self) -> int:
        """
        Approksimasi β₁ = jumlah independent cycles.
        Untuk graf: β₁ = E - V + β₀
        """
        return max(0, self.n_edges - self.n_vertices + self.betti_0())

    def isolated_vertices(self) -> List[str]:
        """Vertex yang tidak punya edge — kata terisolasi semantically."""
        dalam_edge: Set[str] = set()
        for edge in self._edges:
            dalam_edge.update(edge)
        return [v for v in self._vertices if v not in dalam_edge]

    def jarak(self, a: str, b: str) -> float:
        """Kembalikan jarak antara dua vertex."""
        key = (min(a, b), max(a, b))
        return self._jarak.get(key, float("inf"))

    def matriks_adjacency(self) -> torch.Tensor:
        """
        Matriks adjacency dari simplicial complex.
        Shape (n_vertices, n_vertices).
        """
        n = self.n_vertices
        mat = torch.zeros(n, n)
        idx = {v: i for i, v in enumerate(self._vertices)}
        for edge in self._edges:
            vertices = list(edge)
            if len(vertices) == 2:
                i, j = idx[vertices[0]], idx[vertices[1]]
                d = self.jarak(vertices[0], vertices[1])
                # Bobot edge = 1 - (jarak / threshold) ∈ [0, 1]
                w = max(0.0, 1.0 - d / self.threshold)
                mat[i, j] = w
                mat[j, i] = w
        return mat
