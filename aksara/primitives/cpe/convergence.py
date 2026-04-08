"""
ConvergenceChecker — pengecekan konvergensi untuk CPE.

OPOSISI TRANSFORMER:
  Transformer: fixed depth (jumlah layer tetap)
  CPE:         iterasi dinamis sampai konvergen (energi tidak berubah signifikan)

Ini seperti belief propagation — berhenti saat sistem mencapai kesetimbangan,
bukan saat mencapai jumlah iterasi tertentu.
"""

from __future__ import annotations
from typing import List


class ConvergenceChecker:
    """
    Cek konvergensi energi sistem antar iterasi.

    Sistem konvergen jika perubahan energi total antar iterasi < delta.
    Ini berbeda dari Transformer yang selalu jalan N layer terlepas dari kondisi.
    """

    def __init__(self, delta: float = 1e-4, window: int = 3):
        """
        Args:
            delta:  threshold perubahan energi yang dianggap konvergen
            window: jumlah iterasi terakhir yang dicek stabilitas-nya
        """
        self.delta   = delta
        self.window  = window
        self._riwayat: List[float] = []

    def reset(self) -> None:
        self._riwayat.clear()

    def update(self, energi: float) -> None:
        self._riwayat.append(energi)

    def konvergen(self) -> bool:
        """
        Apakah sistem sudah konvergen?

        Konvergen jika dalam `window` iterasi terakhir,
        variasi energi < delta.
        """
        if len(self._riwayat) < self.window:
            return False
        recent = self._riwayat[-self.window:]
        return (max(recent) - min(recent)) < self.delta

    def energi_terakhir(self) -> float:
        if not self._riwayat:
            return float("inf")
        return self._riwayat[-1]

    def n_iterasi(self) -> int:
        return len(self._riwayat)

    def ringkasan(self) -> str:
        if not self._riwayat:
            return "belum ada iterasi"
        return (
            f"iterasi={self.n_iterasi()} "
            f"energi_awal={self._riwayat[0]:.4f} "
            f"energi_akhir={self._riwayat[-1]:.4f} "
            f"delta={self._riwayat[0]-self._riwayat[-1]:.4f} "
            f"konvergen={self.konvergen()}"
        )
