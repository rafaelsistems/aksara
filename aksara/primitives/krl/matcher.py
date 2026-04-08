"""
matcher.py — FrameMatcher: cocokkan proposisi ke frame yang paling sesuai.

OPOSISI TRANSFORMER:
  Transformer: frame matching implisit via similarity di embedding space
  FrameMatcher: matching eksplisit via skor verba + slot coverage

Strategi matching:
  1. Cek verba pemicu — apakah verba proposisi ada di frame.verba_pemicu?
  2. Hitung slot coverage — berapa slot frame yang terpenuhi proposisi?
  3. Skor = bobot_verba * match_verba + bobot_slot * coverage_slot
  4. Kembalikan frame dengan skor tertinggi

Skor [0,1]:
  1.0 = verba tepat + semua slot wajib terpenuhi
  0.5 = verba tepat tapi slot tidak lengkap / slot lengkap tapi verba tidak tepat
  0.0 = tidak ada kecocokan sama sekali
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from aksara.primitives.krl.proposition import Proposisi, TipeSlot
from aksara.primitives.krl.frame import Frame, FrameBank


# Mapping TipeSlot proposisi → nama slot frame
# Digunakan untuk mencocokkan slot proposisi dengan slot frame
SLOT_PROPOSISI_KE_FRAME: Dict[TipeSlot, List[str]] = {
    TipeSlot.AGEN:      ["pembeli", "pelaku", "pengirim", "pengajar", "pembuat",
                         "pemilik", "penuntut", "pihak_1", "otoritas", "tenaga_medis"],
    TipeSlot.PASIEN:    ["barang", "pasien", "pelajar", "terdakwa", "objek"],
    TipeSlot.PENERIMA:  ["penerima", "pihak_2"],
    TipeSlot.LOKASI:    ["lokasi", "tempat"],
    TipeSlot.TUJUAN:    ["tujuan", "tujuan_final"],
    TipeSlot.ASAL:      ["asal", "sumber"],
    TipeSlot.CARA:      ["cara", "moda", "media"],
    TipeSlot.SEBAB:     ["sebab"],
    TipeSlot.WAKTU:     ["waktu"],
    TipeSlot.TEMA:      ["materi", "pesan", "kebijakan", "perbuatan", "hasil"],
    TipeSlot.HASIL:     ["hasil", "putusan"],
    TipeSlot.ATRIBUT:   ["atribut", "entitas"],
}


@dataclass
class HasilMatch:
    """Hasil pencocokan proposisi ke satu frame."""
    frame:          Frame
    skor:           float        # [0,1] kecocokan keseluruhan
    match_verba:    bool         # apakah verba ada di pemicu frame
    coverage_wajib: float        # fraksi slot wajib yang terpenuhi [0,1]
    slot_terpenuhi: List[str]    # nama slot frame yang terpenuhi
    slot_kosong:    List[str]    # nama slot wajib yang kosong

    def __repr__(self) -> str:
        return (f"HasilMatch(frame={self.frame.nama}, "
                f"skor={self.skor:.2f}, "
                f"verba={'V' if self.match_verba else 'X'}, "
                f"wajib={self.coverage_wajib:.0%})")


class FrameMatcher:
    """
    Mencocokkan Proposisi ke Frame yang paling sesuai.

    Deterministik — tidak ada similarity embedding, hanya aturan eksplisit.
    """

    BOBOT_VERBA = 0.6   # verba pemicu lebih penting dari slot
    BOBOT_SLOT  = 0.4   # slot coverage sebagai penyempurna

    def __init__(self, frame_bank: FrameBank) -> None:
        self.frame_bank = frame_bank

    def cocokkan(self, proposisi: Proposisi) -> Optional[HasilMatch]:
        """
        Cocokkan satu proposisi ke frame terbaik.
        Returns None jika tidak ada frame yang cocok (skor < 0.3).
        """
        semua = self.cocokkan_semua(proposisi)
        if not semua:
            return None
        best = semua[0]
        return best if best.skor >= 0.3 else None

    def cocokkan_semua(self, proposisi: Proposisi) -> List[HasilMatch]:
        """
        Cocokkan proposisi ke semua frame, urutkan dari terbaik.
        """
        hasil = []
        for frame in self.frame_bank.semua_frame():
            h = self._hitung_skor(proposisi, frame)
            hasil.append(h)

        # Urutkan: skor tertinggi dulu, tie-break: match_verba
        hasil.sort(key=lambda h: (h.skor, h.match_verba), reverse=True)
        return hasil

    def _hitung_skor(self, prop: Proposisi, frame: Frame) -> HasilMatch:
        """Hitung skor kecocokan proposisi terhadap satu frame."""

        # ── Cek verba pemicu ──────────────────────────────────────────────
        aksi = prop.aksi.lower()

        def _strip_prefiks_verba(v: str) -> str:
            """Strip prefiks umum verba Indonesia untuk matching."""
            for p in ("menge", "meng", "meny", "mem", "men", "me",
                      "ber", "ter", "di", "ke"):
                if v.startswith(p) and len(v) > len(p) + 2:
                    return v[len(p):]
            return v

        aksi_stripped = _strip_prefiks_verba(aksi)

        match_verba = (
            aksi in frame.verba_pemicu
            or aksi_stripped in frame.verba_pemicu
            or any(aksi.endswith(v) and len(aksi) > len(v)
                   for v in frame.verba_pemicu)
            or any(aksi_stripped.endswith(v) and len(aksi_stripped) > len(v)
                   for v in frame.verba_pemicu)
        )

        # ── Hitung coverage slot ──────────────────────────────────────────
        # Kumpulkan nama slot frame yang bisa dipenuhi oleh slot proposisi
        slot_frame_tersedia: set = set(frame.slot.keys())
        slot_terpenuhi: List[str] = []

        for tipe_prop, slot_list in SLOT_PROPOSISI_KE_FRAME.items():
            if tipe_prop not in prop.slot:
                continue
            for nama_slot in slot_list:
                if nama_slot in slot_frame_tersedia and nama_slot not in slot_terpenuhi:
                    slot_terpenuhi.append(nama_slot)
                    break

        # Coverage slot wajib
        slot_wajib = set(frame.slot_wajib)
        slot_wajib_terpenuhi = slot_wajib & set(slot_terpenuhi)
        coverage_wajib = (
            len(slot_wajib_terpenuhi) / len(slot_wajib)
            if slot_wajib else 1.0
        )
        slot_kosong = list(slot_wajib - slot_wajib_terpenuhi)

        # ── Hitung skor gabungan ──────────────────────────────────────────
        skor_verba = 1.0 if match_verba else 0.0
        skor_slot  = coverage_wajib

        # Bonus kecil jika domain aksi cocok dengan domain frame
        bonus_domain = 0.1 if (
            prop.aksi_domain and prop.aksi_domain == frame.domain_utama
        ) else 0.0

        skor = min(1.0,
                   self.BOBOT_VERBA * skor_verba
                   + self.BOBOT_SLOT  * skor_slot
                   + bonus_domain)

        return HasilMatch(
            frame=frame,
            skor=skor,
            match_verba=match_verba,
            coverage_wajib=coverage_wajib,
            slot_terpenuhi=slot_terpenuhi,
            slot_kosong=slot_kosong,
        )
