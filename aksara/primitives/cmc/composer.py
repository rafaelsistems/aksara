"""
CMComposer — mesin komposisi makna via category theory.

OPOSISI TRANSFORMER:
  Transformer: komposisi implisit via feed-forward layers (blackbox)
  CMComposer:  komposisi eksplisit via morfisme kategori (verifiable)

Output CMComposer memperkaya AksaraState dengan:
  - Daftar morfisme aktif dan validitasnya
  - Pelanggaran komposisi makna (type mismatch)
  - Struktur kategori kalimat
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from aksara.primitives.cmc.morphism import Morfisme, TipeMorfisme, DomainMakna
from aksara.primitives.cmc.category import KategoriMakna
from aksara.primitives.lps.morfem import Morfem
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.base.state import AksaraState, PelanggaranConstraint
from aksara.config import AksaraConfig


class CMComposer:
    """
    Categorical Meaning Composer — komposisi makna via hukum kategori.

    OPOSISI TRANSFORMER:
    Transformer: tidak bisa menjelaskan mengapa "sangat meriam" salah secara
                 compositional — semua tersimpan di bobot.
    CMComposer:  bisa menjelaskan: "morfisme modifikasi 'sangat'→'meriam' tidak valid
                 karena 'sangat' (adv) hanya bisa memodifikasi adj/v, bukan n/senjata"

    Output berupa pelanggaran yang bisa dibaca manusia — bukan angka probabilitas.
    """

    def __init__(self, leksikon: LexiconLoader,
                 config: Optional[AksaraConfig] = None):
        self.leksikon  = leksikon
        self.config    = config or AksaraConfig.default()
        self.kategori  = KategoriMakna(leksikon, config=self.config)

    def analisis(
        self,
        morfem_list: List[Morfem],
        state: Optional[AksaraState] = None,
    ) -> Dict:
        """
        Analisis komposisi makna kalimat via category theory.

        Returns:
            dict dengan:
              - morfisme_list: semua morfisme yang terdeteksi
              - morfisme_invalid: morfisme yang melanggar hukum kategori
              - pelanggaran_cmc: list PelanggaranConstraint dari CMC
              - energi_komposisi: skor ketidakvalidan komposisi makna
        """
        objek_list, morfisme_list = self.kategori.bangun_dari_kalimat(morfem_list)
        verifikasi = self.kategori.verifikasi_hukum_kategori(morfisme_list)

        morfisme_invalid = [m for m in morfisme_list if not m.valid]
        pelanggaran_cmc  = self._ke_pelanggaran(morfisme_invalid)

        n_total   = max(len(morfisme_list), 1)
        n_invalid = len(morfisme_invalid)
        energi    = n_invalid / n_total

        return {
            "morfisme_list":      morfisme_list,
            "morfisme_invalid":   morfisme_invalid,
            "pelanggaran_cmc":    pelanggaran_cmc,
            "energi_komposisi":   energi,
            "n_morfisme":         len(morfisme_list),
            "n_invalid":          n_invalid,
            "verifikasi":         verifikasi,
        }

    def perkaya_state(
        self,
        state: AksaraState,
        morfem_list: List[Morfem],
    ) -> AksaraState:
        """
        Perkaya AksaraState dengan hasil analisis CMC.
        Tambahkan pelanggaran komposisi ke state yang sudah ada dari CPE.
        """
        hasil = self.analisis(morfem_list, state)

        # Tambahkan pelanggaran CMC ke pelanggaran yang sudah ada
        pelanggaran_baru = list(state.pelanggaran) + hasil["pelanggaran_cmc"]

        # Update energi: gabung dengan energi_komposisi dari CMC
        energi_baru = state.energi_total + hasil["energi_komposisi"] * 0.2

        # Update metadata
        meta_baru = dict(state.metadata)
        meta_baru["n_morfisme_cmc"]   = hasil["n_morfisme"]
        meta_baru["n_invalid_cmc"]    = hasil["n_invalid"]
        meta_baru["energi_komposisi"] = hasil["energi_komposisi"]

        from dataclasses import replace
        return replace(
            state,
            pelanggaran=pelanggaran_baru,
            energi_total=energi_baru,
            energi_per_dimensi={
                **state.energi_per_dimensi,
                "komposisi": hasil["energi_komposisi"],
            },
            metadata=meta_baru,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _ke_pelanggaran(
        self, morfisme_invalid: List[Morfisme]
    ) -> List[PelanggaranConstraint]:
        """Konversi morfisme invalid ke format PelanggaranConstraint."""
        hasil = []
        for m in morfisme_invalid:
            nama = m.nama
            bagian_nama = (nama
                           .replace("mod(", "").replace("pred(", "")
                           .replace("menggantung(", "").replace("inkoherensi_vo(", "")
                           .replace(")", ""))
            bagian = bagian_nama.split("→") if "→" in bagian_nama else bagian_nama.split("+")
            token_terlibat = [b.strip() for b in bagian if b.strip()]

            # Petakan tipe morfisme ke dimensi AKSARA
            if nama.startswith("menggantung("):
                dimensi   = "semantik"
                severitas = 0.90
            elif nama.startswith("inkoherensi_vo("):
                dimensi   = "semantik"
                severitas = 0.70
            elif m.tipe.value == "modifikasi":
                dimensi   = "sintaktis"
                severitas = 0.65
            else:
                dimensi   = "semantik"
                severitas = 0.60

            hasil.append(PelanggaranConstraint(
                tipe="komposisi",
                token_terlibat=token_terlibat,
                dimensi=dimensi,
                severitas=severitas,
                penjelasan=m.penjelasan,
            ))
        return hasil
