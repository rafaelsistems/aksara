"""
layer.py — KRLayer: orkestrator Knowledge Representation Layer.

Mengintegrasikan PropositionalEncoder + FrameMatcher + ReferenceResolver
menjadi satu unit yang dipanggil oleh AksaraFramework.

Input:  morfem_list (dari LPS), AksaraState (dari CPE/CMC)
Output: KRLResult — representasi pemahaman kalimat yang lengkap

OPOSISI TRANSFORMER:
  Transformer: hidden state di setiap layer tidak bisa diinterpretasi
  KRLayer:     setiap output bisa dibaca, dijelaskan, dan diverifikasi
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aksara.primitives.lps.morfem import Morfem
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.primitives.krl.proposition import Proposisi, TipeSlot
from aksara.primitives.krl.frame import Frame, FrameBank
from aksara.primitives.krl.encoder import PropositionalEncoder
from aksara.primitives.krl.matcher import FrameMatcher, HasilMatch
from aksara.primitives.krl.resolver import ReferenceResolver, IkatanReferensi
from aksara.primitives.krl.kb import KnowledgeBase
from aksara.primitives.krl.inference import InferenceEngine, HasilInferensi, Inferensi


@dataclass
class KRLResult:
    """
    Hasil lengkap Knowledge Representation Layer untuk satu kalimat.

    Ini adalah representasi "pemahaman" kalimat — bukan sekadar validasi.
    Setiap field bisa dibaca, dijelaskan, dan digunakan untuk reasoning.
    """
    # ── Inti representasi ────────────────────────────────────────────────
    proposisi:       Optional[Proposisi]       # Struktur proporsional kalimat
    frame_cocok:     Optional[HasilMatch]       # Frame semantik yang paling sesuai
    frame_alternatif: List[HasilMatch]          # Frame lain yang mungkin

    # ── Inferensi ─────────────────────────────────────────────────────────
    hasil_inferensi: Optional[HasilInferensi] = None  # Kesimpulan dari proposisi + KB

    # ── Referensi ────────────────────────────────────────────────────────
    ikatan_referensi: List[IkatanReferensi] = field(default_factory=list)    # Anafor yang berhasil diselesaikan

    # ── Metadata ─────────────────────────────────────────────────────────
    kalimat_asli:    str = ""
    kelengkapan_pemahaman: float = 0.0         # [0,1] seberapa lengkap representasi

    # ── Properti ─────────────────────────────────────────────────────────

    @property
    def frame_nama(self) -> Optional[str]:
        return self.frame_cocok.frame.nama if self.frame_cocok else None

    @property
    def agen(self) -> Optional[str]:
        if not self.proposisi:
            return None
        return self.proposisi.agen

    @property
    def aksi(self) -> Optional[str]:
        if not self.proposisi:
            return None
        return self.proposisi.aksi

    @property
    def pasien(self) -> Optional[str]:
        if not self.proposisi:
            return None
        return self.proposisi.pasien

    @property
    def slot_kosong_wajib(self) -> List[str]:
        """Slot wajib frame yang tidak terisi — sinyal ketidaklengkapan."""
        if not self.frame_cocok:
            return []
        return self.frame_cocok.slot_kosong

    @property
    def inferensi(self) -> List[Inferensi]:
        """Semua inferensi yang berhasil ditarik dari proposisi + KB."""
        if self.hasil_inferensi:
            return self.hasil_inferensi.inferensi
        return []

    def tanya(self, relasi: str, subjek: Optional[str] = None) -> List[Inferensi]:
        """
        Query inferensial: apa yang diketahui tentang subjek dengan relasi tertentu?

        Contoh:
          krl.tanya('STATUS_MENJADI')  → [terdakwa STATUS_MENJADI terpidana]
          krl.tanya('MEMILIKI', 'terdakwa')  → semua yang dimiliki terdakwa

        Ini adalah kemampuan yang tidak dimiliki Transformer:
        sistem bisa menjawab pertanyaan terstruktur dari kalimat yang dipahami.
        """
        if not self.hasil_inferensi:
            return []
        hasil = self.hasil_inferensi.inferensi_per_relasi(relasi)
        if subjek:
            hasil = [i for i in hasil if i.subjek.lower() == subjek.lower()]
        return hasil

    def jelaskan(self) -> str:
        """Penjelasan bahasa Indonesia tentang representasi KRL."""
        baris = []
        baris.append("── Knowledge Representation ──────────────────────")

        if self.proposisi:
            baris.append(f"  Proposisi  : {self.proposisi}")
            baris.append(f"  Kelengkapan: {self.proposisi.kelengkapan:.0%}")
            if self.proposisi.modalitas:
                baris.append(f"  Modalitas  : {self.proposisi.modalitas}")
            if not self.proposisi.polaritas:
                baris.append("  Polaritas  : NEGATIF")
        else:
            baris.append("  Proposisi  : (tidak terdeteksi)")

        if self.frame_cocok:
            fc = self.frame_cocok
            baris.append(f"  Frame      : {fc.frame.nama} (skor={fc.skor:.2f})")
            baris.append(f"  Deskripsi  : {fc.frame.deskripsi}")
            if fc.slot_kosong:
                baris.append(f"  Slot kosong: {', '.join(fc.slot_kosong)}")
            if fc.slot_terpenuhi:
                baris.append(f"  Slot terisi: {', '.join(fc.slot_terpenuhi)}")
        else:
            baris.append("  Frame      : (tidak cocok dengan frame yang dikenal)")

        if self.hasil_inferensi and self.hasil_inferensi.inferensi:
            baris.append("  Inferensi  :")
            for inf in self.hasil_inferensi.inferensi:
                baris.append(f"    → {inf}")

        if self.ikatan_referensi:
            baris.append("  Referensi  :")
            for ik in self.ikatan_referensi:
                baris.append(f"    {ik}")

        baris.append(f"  Pemahaman  : {self.kelengkapan_pemahaman:.0%}")
        return "\n".join(baris)

    def ke_dict(self) -> dict:
        return {
            "proposisi":    self.proposisi.ke_dict() if self.proposisi else None,
            "frame":        self.frame_nama,
            "frame_skor":   self.frame_cocok.skor if self.frame_cocok else 0.0,
            "slot_kosong":  self.slot_kosong_wajib,
            "referensi":    [str(ik) for ik in self.ikatan_referensi],
            "pemahaman":    self.kelengkapan_pemahaman,
            "inferensi":    [
                {
                    "pernyataan": i.pernyataan,
                    "relasi":     i.relasi,
                    "subjek":     i.subjek,
                    "objek":      i.objek,
                    "aturan":     i.aturan_sumber,
                    "keyakinan":  round(i.keyakinan, 3),
                }
                for i in self.inferensi
            ],
        }


class KRLayer:
    """
    Knowledge Representation Layer — Primitif 6 AKSARA.

    Mengubah output primitif LPS/SFM/CPE/CMC menjadi representasi makna
    yang bisa di-reasoning: proposisi + frame + referensi.

    OPOSISI TRANSFORMER:
      Transformer: tidak bisa menjawab "siapa melakukan apa kepada siapa?"
                   tanpa probing atau extraction head khusus
      KRLayer:     selalu bisa menjawab — tersimpan eksplisit di KRLResult
    """

    def __init__(self, leksikon: LexiconLoader) -> None:
        self.leksikon   = leksikon
        self.encoder    = PropositionalEncoder(leksikon)
        self.frame_bank = FrameBank()
        self.matcher    = FrameMatcher(self.frame_bank)
        self.resolver   = ReferenceResolver()
        self.kb         = KnowledgeBase()
        self.inference  = InferenceEngine(self.kb)

    def proses(
        self,
        morfem_list: List[Morfem],
        kalimat_asli: str = "",
    ) -> KRLResult:
        """
        Proses satu kalimat → KRLResult.

        Dipanggil oleh AksaraFramework setelah semua primitif lain selesai.
        """
        # ── 1. Encode ke proposisi ────────────────────────────────────────
        proposisi = self.encoder.encode(morfem_list, kalimat_asli)

        # ── 2. Cocokkan ke frame ──────────────────────────────────────────
        frame_cocok = None
        frame_alternatif = []
        if proposisi:
            semua_match = self.matcher.cocokkan_semua(proposisi)
            kandidat = [h for h in semua_match if h.skor >= 0.25]
            if kandidat:
                frame_cocok      = kandidat[0]
                frame_alternatif = kandidat[1:3]

        # ── 3. Inference Engine — forward chaining dari proposisi + KB ─────
        # Ini yang membedakan AKSARA dari pattern matching:
        # sistem MENYIMPULKAN konsekuensi dari kalimat, bukan hanya mendeteksi.
        hasil_inferensi = None
        if proposisi:
            hasil_inferensi = self.inference.inferensi(proposisi)

        # ── 4. Resolusi referensi ─────────────────────────────────────────
        ikatan = self.resolver.proses(morfem_list, proposisi)

        # ── 5. Hitung kelengkapan pemahaman ───────────────────────────────
        kelengkapan = self._hitung_kelengkapan(proposisi, frame_cocok, hasil_inferensi)

        return KRLResult(
            proposisi=proposisi,
            frame_cocok=frame_cocok,
            frame_alternatif=frame_alternatif,
            hasil_inferensi=hasil_inferensi,
            ikatan_referensi=ikatan,
            kalimat_asli=kalimat_asli,
            kelengkapan_pemahaman=kelengkapan,
        )

    def reset_konteks(self) -> None:
        """Reset konteks wacana — mulai dokumen/paragraf baru."""
        self.resolver.reset()

    def _hitung_kelengkapan(
        self,
        proposisi:      Optional[Proposisi],
        frame_cocok:    Optional[HasilMatch],
        hasil_inferensi: Optional[HasilInferensi] = None,
    ) -> float:
        """
        Kelengkapan pemahaman [0,1].

        Formula:
          - Ada proposisi:                          +0.35
          - Proposisi lengkap (kelengkapan >= 0.75): +0.15
          - Frame cocok:                            +0.25
          - Frame wajib slot terpenuhi penuh:       +0.10
          - Ada inferensi yang berhasil ditarik:    +0.15
        """
        skor = 0.0
        if proposisi:
            skor += 0.35
            if proposisi.kelengkapan >= 0.75:
                skor += 0.15
        if frame_cocok:
            skor += 0.25
            if frame_cocok.coverage_wajib >= 1.0:
                skor += 0.10
        if hasil_inferensi and hasil_inferensi.ada_inferensi:
            skor += 0.15
        return min(1.0, skor)
