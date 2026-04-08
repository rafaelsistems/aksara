"""
proposition.py — Representasi proporsional kalimat bahasa Indonesia.

OPOSISI TRANSFORMER:
  Transformer: makna tersimpan implisit di vektor hidden state
  KRL:         makna direpresentasikan eksplisit sebagai proposisi terstruktur

Proposisi adalah unit makna terkecil yang bisa di-reasoning:
  AKSI(agen=X, pasien=Y, lokasi=Z, waktu=W, ...)

Contoh:
  "Budi membeli beras di pasar kemarin."
  → BELI(agen=Budi, pasien=beras, lokasi=pasar, waktu=kemarin)

  "Hakim menjatuhkan hukuman kepada terdakwa."
  → JATUHI_HUKUMAN(agen=hakim, pasien=terdakwa, instrumen=hukuman)

Justifikasi linguistik:
  Slot proposisi langsung dari peran gramatikal TBBBI:
    S (Subjek)   → agen atau pasien (tergantung struktur aktif/pasif)
    P (Predikat) → aksi atau atribut
    O (Objek)    → pasien atau tema
    K (Ket)      → lokasi, waktu, cara, sebab, tujuan
    Pel (Pel.)   → komplemen atribut
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set


class TipeSlot(str, Enum):
    """
    Tipe peran tematik (thematic role) dalam proposisi.
    Berbasis teori peran tematik linguistik — bukan arbitrary label.
    """
    # Peran inti (dari struktur S-P-O)
    AGEN        = "agen"        # pelaku sadar: Budi membeli...
    PASIEN      = "pasien"      # yang dikenai aksi: ...membeli beras
    TEMA        = "tema"        # yang dipindahkan/dibicarakan
    PENERIMA    = "penerima"    # target transfer: memberikan kepada X
    SUMBER      = "sumber"      # asal entitas: mengambil dari X
    HASIL       = "hasil"       # yang dihasilkan: membuat X

    # Peran keterangan (dari K dalam S-P-O-K)
    LOKASI      = "lokasi"      # di mana aksi terjadi
    ASAL        = "asal"        # dari mana
    TUJUAN      = "tujuan"      # ke mana / untuk apa
    WAKTU       = "waktu"       # kapan
    CARA        = "cara"        # bagaimana
    SEBAB       = "sebab"       # mengapa / karena
    SYARAT      = "syarat"      # jika / apabila

    # Peran atribut (dari kalimat nominal/predikatif)
    ATRIBUT     = "atribut"     # X adalah Y / X sangat Y
    EKSISTENSI  = "eksistensi"  # ada/tidak ada

    # Aksi itu sendiri
    AKSI        = "aksi"        # verba inti proposisi

    TIDAK_DIKETAHUI = "?"


# Preposisi bahasa Indonesia → TipeSlot
PREPOSISI_KE_SLOT: Dict[str, TipeSlot] = {
    # Lokasi
    "di":        TipeSlot.LOKASI,
    "pada":      TipeSlot.LOKASI,
    "dalam":     TipeSlot.LOKASI,

    # Asal
    "dari":      TipeSlot.ASAL,
    "asal":      TipeSlot.ASAL,

    # Tujuan
    "ke":        TipeSlot.TUJUAN,
    "kepada":    TipeSlot.PENERIMA,
    "untuk":     TipeSlot.TUJUAN,
    "bagi":      TipeSlot.PENERIMA,
    "demi":      TipeSlot.TUJUAN,

    # Waktu
    "ketika":    TipeSlot.WAKTU,
    "saat":      TipeSlot.WAKTU,
    "sejak":     TipeSlot.WAKTU,
    "sampai":    TipeSlot.WAKTU,
    "selama":    TipeSlot.WAKTU,

    # Cara
    "dengan":    TipeSlot.CARA,
    "secara":    TipeSlot.CARA,
    "melalui":   TipeSlot.CARA,

    # Sebab
    "karena":    TipeSlot.SEBAB,
    "akibat":    TipeSlot.SEBAB,
    "lantaran":  TipeSlot.SEBAB,

    # Syarat
    "jika":      TipeSlot.SYARAT,
    "apabila":   TipeSlot.SYARAT,
    "bila":      TipeSlot.SYARAT,
    "kalau":     TipeSlot.SYARAT,
}


@dataclass
class SlotProposisi:
    """Satu slot yang terisi dalam proposisi."""
    tipe:    TipeSlot
    nilai:   str            # teks token/frasa pengisi slot
    root:    str            # root morfem
    domain:  Optional[str]  # domain semantik dari SFM
    indeks:  int            # posisi dalam kalimat
    keyakinan: float = 1.0  # seberapa yakin slot ini benar [0,1]


@dataclass
class Proposisi:
    """
    Representasi proposisi satu kalimat.

    Proposisi adalah unit makna yang bisa di-reasoning:
      AKSI(slot1=val1, slot2=val2, ...)

    Berbeda dari string kalimat: proposisi bisa dibandingkan,
    dikomposisi, dan digunakan dalam inferensi.
    """
    aksi:        str                         # verba inti (root)
    aksi_domain: Optional[str]               # domain semantik verba
    slot:        Dict[TipeSlot, SlotProposisi] = field(default_factory=dict)
    polaritas:   bool = True                 # True = positif, False = negasi
    modalitas:   Optional[str] = None        # "harus", "boleh", "mungkin"
    sumber_kalimat: str = ""                 # teks kalimat asli

    # ── Properti ──────────────────────────────────────────────────────

    @property
    def agen(self) -> Optional[str]:
        s = self.slot.get(TipeSlot.AGEN)
        return s.nilai if s else None

    @property
    def pasien(self) -> Optional[str]:
        s = self.slot.get(TipeSlot.PASIEN)
        return s.nilai if s else None

    @property
    def lokasi(self) -> Optional[str]:
        s = self.slot.get(TipeSlot.LOKASI)
        return s.nilai if s else None

    @property
    def waktu(self) -> Optional[str]:
        s = self.slot.get(TipeSlot.WAKTU)
        return s.nilai if s else None

    @property
    def slot_terisi(self) -> List[TipeSlot]:
        return list(self.slot.keys())

    @property
    def kelengkapan(self) -> float:
        """
        Seberapa lengkap proposisi ini [0,1].
        Proposisi minimal: aksi + agen = 0.5
        Proposisi lengkap: aksi + agen + pasien = 0.75
        Proposisi penuh:   aksi + agen + pasien + 1 keterangan = 1.0
        """
        skor = 0.0
        if self.aksi:
            skor += 0.25
        if TipeSlot.AGEN in self.slot:
            skor += 0.25
        if TipeSlot.PASIEN in self.slot or TipeSlot.TEMA in self.slot:
            skor += 0.25
        if any(t in self.slot for t in (
            TipeSlot.LOKASI, TipeSlot.WAKTU, TipeSlot.TUJUAN,
            TipeSlot.CARA, TipeSlot.SEBAB
        )):
            skor += 0.25
        return skor

    def __str__(self) -> str:
        neg = "TIDAK_" if not self.polaritas else ""
        modal = f"[{self.modalitas}] " if self.modalitas else ""
        slot_str = ", ".join(
            f"{t.value}={s.nilai}"
            for t, s in self.slot.items()
        )
        return f"{modal}{neg}{self.aksi.upper()}({slot_str})"

    def ke_dict(self) -> dict:
        return {
            "aksi":       self.aksi,
            "polaritas":  self.polaritas,
            "modalitas":  self.modalitas,
            "slot":       {
                t.value: {"nilai": s.nilai, "root": s.root, "domain": s.domain}
                for t, s in self.slot.items()
            },
            "kelengkapan": self.kelengkapan,
        }
