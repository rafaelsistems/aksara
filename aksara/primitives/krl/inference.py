"""
inference.py — Inference Engine AKSARA.

OPOSISI TRANSFORMER:
  Transformer: "reasoning" = prediksi token berikutnya yang paling probable.
               Model tidak TAHU apa yang disimpulkan — hanya mereproduksi pola.
  InferenceEngine: reasoning = forward chaining eksplisit dari:
               Proposisi (KRL) + Aturan Dunia (KB) → Kesimpulan baru
               Setiap kesimpulan bisa ditelusuri ke aturan yang menghasilkannya.

Mekanisme:
  1. Terima Proposisi dari KRL encoder
  2. Klasifikasikan agen dan pasien ke TipeEntitas via KB
  3. Identifikasi tipe aksi dari verba proposisi
  4. Match aturan yang kondisinya terpenuhi
  5. Generate kesimpulan dari aturan yang match
  6. Output: list Inferensi terstruktur + alasan linguistik

Ini bukan statistik. Setiap inferensi punya:
  - Aturan sumber yang bisa dikutip
  - Tipe entitas yang diverifikasi
  - Tingkat keyakinan berdasarkan kelengkapan proposisi

Referensi:
  - Forward Chaining: Russell & Norvig, AI: A Modern Approach
  - Event Semantics: Davidson (1967), "The Logical Form of Action Sentences"
  - Frame Semantics: Fillmore (1976), "Frame semantics and the nature of language"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aksara.primitives.krl.proposition import Proposisi, TipeSlot
from aksara.primitives.krl.kb import (
    KnowledgeBase, TipeEntitas, AturanDunia,
    adalah_subtipe, tipe_entitas, tipe_aksi,
)


# ── Struktur Output Inferensi ─────────────────────────────────────────────────

@dataclass
class Inferensi:
    """
    Satu unit inferensi yang dihasilkan dari proposisi + aturan.

    Setiap inferensi bisa ditelusuri ke:
      - proposisi sumber (kalimat asli)
      - aturan dunia yang digunakan
      - entitas yang terlibat

    Berbeda dari hidden state Transformer: ini bisa dibaca, diaudit, dikoreksi.
    """
    pernyataan:     str              # "hakim TELAH_MEMUTUS kasus"
    relasi:         str              # "TELAH_MEMUTUS", "MEMILIKI", "BERADA_DI", dll.
    subjek:         str              # entitas subjek inferensi
    objek:          str              # entitas objek inferensi
    aturan_sumber:  str              # nama aturan yang menghasilkan ini
    proposisi_asal: str              # kalimat sumber (teks asli)
    keyakinan:      float = 1.0      # [0,1]: dipengaruhi kelengkapan proposisi
    domain:         str = "universal"

    def __str__(self) -> str:
        return f"{self.subjek} {self.relasi} {self.objek} [{self.aturan_sumber}, yakin={self.keyakinan:.2f}]"


@dataclass
class HasilInferensi:
    """Kumpulan inferensi dari satu kalimat."""
    kalimat:        str
    proposisi:      Proposisi
    inferensi:      List[Inferensi] = field(default_factory=list)
    tipe_agen:      Optional[str] = None
    tipe_pasien:    Optional[str] = None
    tipe_aksi_:     Optional[str] = None

    @property
    def ada_inferensi(self) -> bool:
        return len(self.inferensi) > 0

    @property
    def n_inferensi(self) -> int:
        return len(self.inferensi)

    def inferensi_per_relasi(self, relasi: str) -> List[Inferensi]:
        return [i for i in self.inferensi if i.relasi == relasi]

    def ke_dict(self) -> dict:
        return {
            "kalimat":   self.kalimat,
            "tipe_agen":  self.tipe_agen,
            "tipe_pasien": self.tipe_pasien,
            "tipe_aksi":  self.tipe_aksi_,
            "inferensi": [
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

    def __str__(self) -> str:
        if not self.inferensi:
            return f"[{self.kalimat}] → (tidak ada inferensi)"
        baris = [f"[{self.kalimat}]"]
        for inf in self.inferensi:
            baris.append(f"  → {inf}")
        return "\n".join(baris)


# ── Inference Engine ──────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Forward chaining inference engine untuk AKSARA.

    Proses:
      1. Klasifikasikan entitas proposisi via KB
      2. Identifikasi tipe aksi dari verba
      3. Cari aturan yang kondisinya match
      4. Generate kesimpulan dari aturan yang match
      5. Resolusi template: ganti placeholder dengan nilai nyata

    Tidak ada statistik. Tidak ada bobot hidden. Setiap langkah deterministik
    dan bisa dijelaskan dalam bahasa Indonesia.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def inferensi(self, proposisi: Proposisi) -> HasilInferensi:
        """
        Jalankan forward chaining dari satu proposisi.

        Returns:
            HasilInferensi dengan semua inferensi yang bisa ditarik.
        """
        hasil = HasilInferensi(
            kalimat=proposisi.sumber_kalimat,
            proposisi=proposisi,
        )

        # ── Langkah 1: Klasifikasikan entitas ─────────────────────────────
        agen_slot   = proposisi.slot.get(TipeSlot.AGEN)
        pasien_slot = proposisi.slot.get(TipeSlot.PASIEN)
        penerima_slot = proposisi.slot.get(TipeSlot.PENERIMA)

        agen_kata   = agen_slot.root   if agen_slot   else None
        pasien_kata = pasien_slot.root if pasien_slot else None

        # Gunakan nilai (teks asli) untuk tipe_entitas agar fallback kapital→PERSONA bekerja.
        # root selalu lowercase dari LPS, nilai menjaga kapitalisasi asli.
        tipe_ag  = self.kb.tipe_entitas(agen_slot.nilai)   if agen_slot   else None
        tipe_pa  = self.kb.tipe_entitas(pasien_slot.nilai) if pasien_slot else None

        hasil.tipe_agen   = tipe_ag
        hasil.tipe_pasien = tipe_pa

        # ── Langkah 2: Identifikasi tipe aksi ─────────────────────────────
        ta = self.kb.tipe_aksi(proposisi.aksi)
        hasil.tipe_aksi_ = ta

        if ta is None:
            return hasil  # verba tidak dikenal — tidak ada aturan yang bisa diapply

        # ── Langkah 3: Match aturan ────────────────────────────────────────
        aturan_kandidat = self.kb.aturan_untuk_aksi(ta)

        for aturan in sorted(aturan_kandidat, key=lambda a: -a.prioritas):
            if not self._kondisi_match(aturan, tipe_ag, tipe_pa, proposisi):
                continue

            # ── Langkah 4: Generate kesimpulan ────────────────────────────
            keyakinan = proposisi.kelengkapan
            if not aturan.tipe_agen:   # aturan universal → sedikit lebih rendah
                keyakinan *= 0.9

            for template in aturan.kesimpulan:
                inf = self._resolve_template(
                    template, aturan, proposisi,
                    agen_kata, pasien_kata, penerima_slot,
                    keyakinan,
                )
                if inf:
                    hasil.inferensi.append(inf)

        return hasil

    def inferensi_batch(self, proposisi_list: List[Proposisi]) -> List[HasilInferensi]:
        """Jalankan inferensi untuk setiap proposisi dalam list."""
        return [self.inferensi(p) for p in proposisi_list]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _kondisi_match(
        self,
        aturan:    AturanDunia,
        tipe_ag:   Optional[str],
        tipe_pa:   Optional[str],
        proposisi: Proposisi,
    ) -> bool:
        """
        Cek apakah kondisi aturan terpenuhi oleh proposisi.

        Logika:
        - Jika aturan.tipe_agen != None → agen harus subtipe dari tipe_agen
        - Jika aturan.tipe_pasien != None → pasien harus subtipe dari tipe_pasien
        - Jika proposisi negasi → aturan kausalitas dibatalkan
        """
        # Negasi membatalkan kausalitas
        if not proposisi.polaritas:
            return False

        # Cek tipe agen
        if aturan.tipe_agen is not None:
            if tipe_ag is None:
                return False
            if not adalah_subtipe(tipe_ag, aturan.tipe_agen):
                return False

        # Cek tipe pasien
        if aturan.tipe_pasien is not None:
            if tipe_pa is None:
                return False
            if not adalah_subtipe(tipe_pa, aturan.tipe_pasien):
                return False

        return True

    def _resolve_template(
        self,
        template:     str,
        aturan:       AturanDunia,
        proposisi:    Proposisi,
        agen_kata:    Optional[str],
        pasien_kata:  Optional[str],
        penerima_slot,
        keyakinan:    float,
    ) -> Optional[Inferensi]:
        """
        Resolusi template kesimpulan dengan nilai nyata dari proposisi.

        Template: "penerima STATUS_MENJADI terpidana"
        Hasil:    "terdakwa STATUS_MENJADI terpidana"
        """
        penerima_kata = penerima_slot.root if penerima_slot else None

        # Resolusi placeholder
        resolved = template
        resolved = resolved.replace("agen",    agen_kata   or "agen_tidak_diketahui")
        resolved = resolved.replace("pasien",  pasien_kata or "pasien_tidak_diketahui")
        resolved = resolved.replace("penerima", penerima_kata or (pasien_kata or "penerima_tidak_diketahui"))

        # Ekstrak subjek–relasi–objek dari template terselesaikan
        bagian = resolved.split(" ", 2)
        if len(bagian) < 3:
            return None

        subjek, relasi, objek = bagian[0], bagian[1], bagian[2]

        # Jangan output inferensi dengan placeholder — tidak informatif
        if "tidak_diketahui" in subjek or "tidak_diketahui" in objek:
            return None

        return Inferensi(
            pernyataan=resolved,
            relasi=relasi,
            subjek=subjek,
            objek=objek,
            aturan_sumber=aturan.nama,
            proposisi_asal=proposisi.sumber_kalimat,
            keyakinan=round(keyakinan, 3),
            domain=aturan.domain,
        )

    def tanya(
        self,
        hasil: HasilInferensi,
        relasi: str,
        subjek: Optional[str] = None,
    ) -> List[Inferensi]:
        """
        Query: "apa yang diketahui tentang subjek dengan relasi tertentu?"

        Contoh:
          engine.tanya(hasil, "STATUS_MENJADI")
          → [terdakwa STATUS_MENJADI terpidana]

          engine.tanya(hasil, "MEMILIKI", subjek="terdakwa")
          → semua yang dimiliki terdakwa

        Ini adalah kemampuan yang tidak dimiliki Transformer:
        sistem bisa menjawab pertanyaan terstruktur tentang konten kalimat.
        """
        kandidat = hasil.inferensi_per_relasi(relasi)
        if subjek:
            kandidat = [i for i in kandidat if i.subjek.lower() == subjek.lower()]
        return kandidat
