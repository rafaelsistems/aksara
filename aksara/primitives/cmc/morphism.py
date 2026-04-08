"""
Morfisme — fungsi komposisi makna dalam Category Theory untuk bahasa Indonesia.

OPOSISI TRANSFORMER:
  Transformer: komposisi makna implisit via stacking layers (blackbox)
  CMC:         komposisi makna eksplisit via morfisme kategori (provable)

Hukum kategori yang harus dipenuhi:
  1. Asosiativitas: (f ∘ g) ∘ h = f ∘ (g ∘ h)
  2. Identitas: id ∘ f = f = f ∘ id
  3. Non-commutativity: f ∘ g ≠ g ∘ f (untuk bahasa Indonesia)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class TipeMorfisme(str, Enum):
    """Tipe morfisme dalam kategori makna bahasa Indonesia."""
    NOMINALISASI   = "nominalisasi"    # N → NP (kata → frasa nominal)
    VERBALISASI    = "verbalisasi"     # V + Args → Klausa
    MODIFIKASI     = "modifikasi"      # Adj/Adv memodifikasi head
    KOMPOSISI      = "komposisi"       # komponen → unit yang lebih besar
    NEGASI         = "negasi"          # membalik makna
    KAUSATIF       = "kausatif"        # V → V yang menyebabkan
    PASIVISASI     = "pasivisasi"      # V aktif → V pasif
    IDENTITAS      = "identitas"       # tidak mengubah makna


@dataclass
class DomainMakna:
    """
    Domain makna suatu satuan linguistik — objek dalam kategori.

    Dalam category theory: objek adalah tipe/domain.
    Morfisme adalah fungsi antara dua objek (domain ke domain).
    """
    kelas_kata: str           # n, v, adj, adv, dll.
    domain_semantik: Optional[str]  # kuliner, senjata, dll.
    register: str             # formal, informal, netral
    animasi: Optional[bool]   # bernyawa atau tidak (untuk agreement)
    abstrak: bool = False     # abstrak atau konkret

    def __hash__(self):
        return hash((self.kelas_kata, self.domain_semantik, self.register))

    def __eq__(self, other):
        if not isinstance(other, DomainMakna):
            return False
        return (self.kelas_kata == other.kelas_kata and
                self.domain_semantik == other.domain_semantik)


@dataclass
class Morfisme:
    """
    Morfisme dalam kategori makna — fungsi yang memetakan domain ke domain.

    OPOSISI TRANSFORMER:
    Transformer: tidak ada morfisme eksplisit, semua di bobot
    CMC: setiap perubahan makna dikodekan sebagai morfisme yang bisa
         diinspeksi, diverifikasi, dan dibuktikan konsistensinya.

    Hukum yang harus dipenuhi:
    - Komposisi f ∘ g valid hanya jika domain_target(g) == domain_source(f)
    - Morfisme identitas ada untuk setiap domain
    """
    nama: str
    tipe: TipeMorfisme
    domain_source: DomainMakna   # objek asal
    domain_target: DomainMakna   # objek tujuan
    valid: bool = True
    penjelasan: str = ""

    def dapat_komposisi_dengan(self, lain: "Morfisme") -> bool:
        """
        Apakah morfisme ini bisa dikomposi dengan morfisme lain?
        f ∘ g valid jika domain_target(g) == domain_source(f).
        """
        return self.domain_source == lain.domain_target

    def komposisi(self, lain: "Morfisme") -> Optional["Morfisme"]:
        """
        Hitung komposisi f ∘ g (self ∘ lain).
        Returns None jika komposisi tidak valid.
        """
        if not self.dapat_komposisi_dengan(lain):
            return None
        return Morfisme(
            nama=f"({self.nama} ∘ {lain.nama})",
            tipe=TipeMorfisme.KOMPOSISI,
            domain_source=lain.domain_source,
            domain_target=self.domain_target,
            valid=self.valid and lain.valid,
            penjelasan=f"{lain.penjelasan} → {self.penjelasan}",
        )

    @classmethod
    def identitas(cls, domain: DomainMakna) -> "Morfisme":
        """Morfisme identitas — tidak mengubah makna."""
        return cls(
            nama=f"id({domain.kelas_kata})",
            tipe=TipeMorfisme.IDENTITAS,
            domain_source=domain,
            domain_target=domain,
            valid=True,
            penjelasan="identitas — tidak ada perubahan makna",
        )


# ── Definisi morfisme per kelas kata bahasa Indonesia ────────────────────────
# Setiap kelas kata punya morfisme yang valid dalam komposisi makna.
# Ini adalah hukum linguistik, bukan statistik.

def buat_morfisme_adjektiva(
    domain_adj: DomainMakna,
    domain_nomina: DomainMakna,
    nama_adj: str,
    nama_nomina: str,
) -> Optional[Morfisme]:
    """
    Adjektiva memodifikasi nomina — hanya valid jika domain compatible.
    "lezat" (kuliner) + "makanan" (kuliner) = valid
    "meriam" (senjata) sebagai modifier "makanan" (kuliner) = TIDAK valid
    """
    # Cek domain compatibility untuk modifier
    if (domain_adj.domain_semantik and domain_nomina.domain_semantik and
            domain_adj.domain_semantik != domain_nomina.domain_semantik):
        # Domain berbeda — modifikasi tidak natural
        return Morfisme(
            nama=f"mod({nama_adj}→{nama_nomina})",
            tipe=TipeMorfisme.MODIFIKASI,
            domain_source=domain_adj,
            domain_target=domain_nomina,
            valid=False,
            penjelasan=(
                f"Modifikasi tidak valid: '{nama_adj}' [{domain_adj.domain_semantik}] "
                f"tidak bisa memodifikasi '{nama_nomina}' [{domain_nomina.domain_semantik}]"
            ),
        )

    return Morfisme(
        nama=f"mod({nama_adj}→{nama_nomina})",
        tipe=TipeMorfisme.MODIFIKASI,
        domain_source=domain_adj,
        domain_target=domain_nomina,
        valid=True,
        penjelasan=f"'{nama_adj}' memodifikasi '{nama_nomina}' secara valid",
    )


def buat_morfisme_verba(
    domain_subj: DomainMakna,
    domain_verba: DomainMakna,
    domain_obj: Optional[DomainMakna],
    nama_subj: str,
    nama_verba: str,
    nama_obj: Optional[str] = None,
) -> Optional[Morfisme]:
    """
    Verba mengikat subjek dan objek — komposisi menjadi klausa.
    Constraint: subjek harus animate untuk verba fisik.
    """
    return Morfisme(
        nama=f"pred({nama_subj}+{nama_verba}+{nama_obj or ''})",
        tipe=TipeMorfisme.VERBALISASI,
        domain_source=domain_subj,
        domain_target=DomainMakna(
            kelas_kata="klausa",
            domain_semantik=domain_verba.domain_semantik,
            register=domain_subj.register,
            animasi=domain_subj.animasi,
        ),
        valid=True,
        penjelasan=f"Klausa: {nama_subj} + {nama_verba}" + (f" + {nama_obj}" if nama_obj else ""),
    )
