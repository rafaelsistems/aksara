"""
resolver.py — ReferenceResolver: resolusi anafor lintas kalimat.

OPOSISI TRANSFORMER:
  Transformer: resolusi anafor via attention score (implisit, tidak bisa dijelaskan)
  ReferenceResolver: resolusi via aturan kompatibilitas morfem + domain (eksplisit)

Strategi resolusi (Hobbs Algorithm adaptasi bahasa Indonesia):
  1. Kumpulkan kandidat anteseden dari proposisi sebelumnya
  2. Filter berdasarkan kompatibilitas gender/jumlah (dari morfologi)
  3. Filter berdasarkan kompatibilitas domain semantik
  4. Pilih anteseden paling baru yang kompatibel (recency bias)

Tipe anafor yang ditangani:
  - Pronomina persona: dia, mereka, ia, beliau, -nya
  - Demonstrativa: ini, itu, tersebut, tadi
  - Ellipsis subjek (pro-drop): kalimat tanpa subjek eksplisit

Contoh:
  Kalimat 1: "Budi membeli beras di pasar."
  Kalimat 2: "Dia membayar dengan tunai."
             → "Dia" = Budi (pronomina → anteseden PERSONA terakhir)

  Kalimat 1: "Pemerintah mengeluarkan kebijakan baru."
  Kalimat 2: "Kebijakan tersebut mulai berlaku besok."
             → "tersebut" = kebijakan baru (demonstrativa → NP terakhir)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from aksara.primitives.lps.morfem import Morfem, KelasKata
from aksara.primitives.krl.proposition import Proposisi, TipeSlot, SlotProposisi


# Pronomina persona bahasa Indonesia
PRONOMINA_PERSONA_TUNGGAL: Set[str] = {
    "dia", "ia", "beliau", "anda", "kamu", "engkau",
    "-nya", "nya",
}
PRONOMINA_PERSONA_JAMAK: Set[str] = {
    "mereka", "kalian", "kami", "kita",
}
PRONOMINA_PERSONA: Set[str] = PRONOMINA_PERSONA_TUNGGAL | PRONOMINA_PERSONA_JAMAK

# Demonstrativa yang menunjuk anaforis (ke belakang)
DEMONSTRATIVA_ANAFORIS: Set[str] = {
    "itu", "tersebut", "tadi", "dimaksud", "yang dimaksud",
}

# Tipe entitas yang kompatibel dengan pronomina persona
ENTITAS_PERSONA: Set[str] = {"PERSONA", "ORGANISASI", "INSTITUSI"}


@dataclass
class IkatanReferensi:
    """Satu ikatan anafor → anteseden."""
    anafor:      str           # teks pronomina/demonstrativa
    anteseden:   str           # teks anteseden yang dirujuk
    root_anteseden: str        # root morfem anteseden
    domain:      Optional[str] # domain semantik anteseden
    indeks_kalimat_anafor:    int
    indeks_kalimat_anteseden: int
    keyakinan:   float = 1.0

    def __str__(self) -> str:
        return (f"'{self.anafor}' → '{self.anteseden}' "
                f"(kal.{self.indeks_kalimat_anteseden}→{self.indeks_kalimat_anafor})")


@dataclass
class KonteksWacana:
    """
    Konteks wacana lintas kalimat — menyimpan entitas yang disebutkan.

    Diperbarui setiap kali kalimat baru diproses.
    """
    entitas_aktif: List[SlotProposisi] = field(default_factory=list)
    # Stack: entitas terbaru di akhir (untuk recency bias)
    proposisi_history: List[Proposisi] = field(default_factory=list)
    ikatan: List[IkatanReferensi] = field(default_factory=list)

    def tambah_proposisi(self, prop: Proposisi, idx_kalimat: int) -> None:
        """Perbarui konteks dengan proposisi kalimat baru."""
        self.proposisi_history.append(prop)
        # Tambahkan semua slot berisi entitas ke stack
        for tipe, slot in prop.slot.items():
            self.entitas_aktif.append(slot)
        # Batasi window ke 5 kalimat terakhir
        if len(self.proposisi_history) > 5:
            # Hapus entitas dari proposisi tertua
            oldest = self.proposisi_history.pop(0)
            oldest_roots = {s.root for s in oldest.slot.values()}
            self.entitas_aktif = [
                e for e in self.entitas_aktif
                if e.root not in oldest_roots
            ]

    @property
    def entitas_persona_terakhir(self) -> Optional[SlotProposisi]:
        """Entitas persona (manusia/organisasi) yang paling baru disebutkan."""
        for e in reversed(self.entitas_aktif):
            if e.domain in ("sosial", "hukum") or e.tipe == TipeSlot.AGEN:
                return e
        return None

    @property
    def entitas_terakhir(self) -> Optional[SlotProposisi]:
        """Entitas apapun yang paling baru disebutkan."""
        if self.entitas_aktif:
            return self.entitas_aktif[-1]
        return None


class ReferenceResolver:
    """
    Resolver anafor deterministik untuk bahasa Indonesia.

    Memproses kalimat secara berurutan dan membangun ikatan referensi
    berdasarkan aturan linguistik — bukan attention weight.
    """

    def __init__(self) -> None:
        self.konteks = KonteksWacana()
        self._idx_kalimat = 0

    def reset(self) -> None:
        """Reset konteks — mulai wacana baru."""
        self.konteks = KonteksWacana()
        self._idx_kalimat = 0

    def proses(
        self,
        morfem_list: List[Morfem],
        proposisi: Optional[Proposisi],
    ) -> List[IkatanReferensi]:
        """
        Proses satu kalimat:
        1. Deteksi anafor dalam kalimat
        2. Resolusi setiap anafor ke anteseden dari konteks
        3. Perbarui konteks dengan proposisi baru

        Returns: list ikatan referensi yang ditemukan dalam kalimat ini.
        """
        ikatan_baru: List[IkatanReferensi] = []

        # ── Deteksi dan resolusi anafor ───────────────────────────────────
        for m in morfem_list:
            root = m.root.lower()

            # Pronomina persona → cari anteseden persona
            if root in PRONOMINA_PERSONA or m.kelas_kata == KelasKata.PRONOMINA:
                anteseden = self.konteks.entitas_persona_terakhir
                if anteseden:
                    ikatan = IkatanReferensi(
                        anafor=m.teks_asli,
                        anteseden=anteseden.nilai,
                        root_anteseden=anteseden.root,
                        domain=anteseden.domain,
                        indeks_kalimat_anafor=self._idx_kalimat,
                        indeks_kalimat_anteseden=self._idx_kalimat - 1,
                        keyakinan=0.85 if root in PRONOMINA_PERSONA_TUNGGAL else 0.70,
                    )
                    ikatan_baru.append(ikatan)
                    self.konteks.ikatan.append(ikatan)

            # Demonstrativa anaforis → cari anteseden NP terakhir
            elif root in DEMONSTRATIVA_ANAFORIS:
                anteseden = self.konteks.entitas_terakhir
                if anteseden:
                    ikatan = IkatanReferensi(
                        anafor=m.teks_asli,
                        anteseden=anteseden.nilai,
                        root_anteseden=anteseden.root,
                        domain=anteseden.domain,
                        indeks_kalimat_anafor=self._idx_kalimat,
                        indeks_kalimat_anteseden=self._idx_kalimat - 1,
                        keyakinan=0.80,
                    )
                    ikatan_baru.append(ikatan)
                    self.konteks.ikatan.append(ikatan)

        # ── Perbarui konteks dengan proposisi kalimat ini ─────────────────
        if proposisi:
            self.konteks.tambah_proposisi(proposisi, self._idx_kalimat)

        self._idx_kalimat += 1
        return ikatan_baru

    @property
    def semua_ikatan(self) -> List[IkatanReferensi]:
        return self.konteks.ikatan
