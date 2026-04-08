"""
Morfem — unit dasar linguistik AKSARA.

OPOSISI TRANSFORMER:
  Transformer: unit dasar = subword token (arbitrer, statistik)
  AKSARA LPS:  unit dasar = morfem (unit bermakna, deterministik)

Setiap morfem adalah unit linguistik yang membawa makna — bukan
potongan string yang muncul karena frekuensi statistik.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class KelasKata(str, Enum):
    """Kelas kata bahasa Indonesia — deterministik, bukan probabilistik."""
    NOMINA          = "N"
    VERBA           = "V"
    ADJEKTIVA       = "Adj"
    ADVERBIA        = "Adv"
    PRONOMINA       = "Pron"
    NUMERALIA       = "Num"
    PREPOSISI       = "Prep"
    KONJUNGSI       = "Konj"
    INTERJEKSI      = "Interj"
    ARTIKULA        = "Art"
    PARTIKEL        = "Part"
    NOMINA_PROPER   = "N_proper"
    NOMINA_SERAPAN  = "N_serapan"
    VERBA_SERAPAN   = "V_serapan"
    TIDAK_DIKETAHUI = "?"


class PeranGramatikal(str, Enum):
    """Peran gramatikal dalam kalimat — slot linguistik Indonesia."""
    SUBJEK      = "S"
    PREDIKAT    = "P"
    OBJEK       = "O"
    KETERANGAN  = "K"
    PELENGKAP   = "Pel"
    MODIFIER    = "Mod"
    DETERMINER  = "Det"
    TIDAK_DIKETAHUI = "?"


class TipeAfiks(str, Enum):
    """Tipe afiks bahasa Indonesia."""
    PREFIKS  = "prefiks"
    SUFIKS   = "sufiks"
    KONFIKS  = "konfiks"
    INFIKS   = "infiks"
    REDUPLIKASI_PENUH   = "redup_penuh"
    REDUPLIKASI_PARSIAL = "redup_parsial"
    REDUPLIKASI_BERUBAH = "redup_berubah"


@dataclass
class AfiksAktif:
    """Satu afiks yang aktif pada morfem."""
    bentuk: str
    tipe: TipeAfiks
    fungsi: str
    valid: bool = True


@dataclass
class Morfem:
    """
    Unit dasar linguistik AKSARA — morfem dengan metadata lengkap.

    Ini bukan token. Token adalah potongan string arbitrer.
    Morfem adalah unit bahasa yang membawa makna dan peran linguistik.

    Setiap field punya justifikasi linguistik — tidak ada dimensi tersembunyi.
    """

    indeks: int
    teks_asli: str
    root: str

    kelas_kata: KelasKata = KelasKata.TIDAK_DIKETAHUI
    peran_gramatikal: PeranGramatikal = PeranGramatikal.TIDAK_DIKETAHUI

    afiks_aktif: List[AfiksAktif] = field(default_factory=list)

    adalah_reduplikasi: bool = False
    tipe_reduplikasi: Optional[str] = None
    base_reduplikasi: Optional[str] = None

    adalah_serapan: bool = False
    bahasa_asal: Optional[str] = None

    adalah_informal: bool = False
    adalah_proper: bool = False

    ada_di_kbbi: bool = False
    ada_di_wiktionary: bool = False

    @property
    def semua_afiks_valid(self) -> bool:
        return all(a.valid for a in self.afiks_aktif)

    @property
    def punya_afiks(self) -> bool:
        return len(self.afiks_aktif) > 0

    @property
    def teks_normalisasi(self) -> str:
        return self.root.lower()

    def ringkasan(self) -> str:
        afiks_str = "+".join(a.bentuk for a in self.afiks_aktif) or "-"
        return (
            f"Morfem({self.indeks}, '{self.teks_asli}', "
            f"root='{self.root}', kelas={self.kelas_kata.value}, "
            f"peran={self.peran_gramatikal.value}, afiks=[{afiks_str}])"
        )
