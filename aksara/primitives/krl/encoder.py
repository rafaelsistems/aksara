"""
encoder.py — PropositionalEncoder: kalimat → proposisi terstruktur.

OPOSISI TRANSFORMER:
  Transformer: encode kalimat ke vektor yang tidak bisa diinterpretasi
  PropositionalEncoder: encode kalimat ke proposisi yang bisa dibaca dan di-reasoning

Strategi encoding:
  1. Gunakan output LPS (peran gramatikal S/P/O/K) sebagai peta slot
  2. Gunakan output SFM (domain semantik) untuk mengisi tipe entitas
  3. Gunakan output CPE (pelanggaran constraint) untuk mendeteksi polaritas/modalitas
  4. Hasilkan Proposisi dengan slot yang terisi

Mapping peran gramatikal → slot proposisi:
  S (Subjek)   → AGEN (kalimat aktif) atau PASIEN (kalimat pasif di-)
  P (Predikat) → AKSI (verba) atau ATRIBUT (nomina/adjektiva)
  O (Objek)    → PASIEN (kalimat aktif)
  K+di/pada    → LOKASI
  K+ke         → TUJUAN
  K+dari       → ASAL
  K+dengan     → CARA
  K+karena     → SEBAB
  K+untuk      → TUJUAN
  K+kepada     → PENERIMA
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from aksara.primitives.lps.morfem import Morfem, KelasKata, PeranGramatikal
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.primitives.krl.proposition import (
    Proposisi, SlotProposisi, TipeSlot, PREPOSISI_KE_SLOT
)


# Kata negasi bahasa Indonesia
KATA_NEGASI: Set[str] = {
    "tidak", "bukan", "tak", "tiada", "jangan", "belum", "tanpa",
}

# Kata modalitas
KATA_MODALITAS: Dict[str, str] = {
    "harus": "harus", "wajib": "harus", "perlu": "harus",
    "boleh": "boleh", "dapat": "boleh", "bisa": "boleh",
    "mungkin": "mungkin", "barangkali": "mungkin", "kiranya": "mungkin",
    "pasti": "pasti", "tentu": "pasti",
}

# Kelas kata yang bisa menjadi agen/pasien
KELAS_NOMINAL: Set[KelasKata] = {
    KelasKata.NOMINA, KelasKata.PRONOMINA,
    KelasKata.NOMINA_PROPER, KelasKata.NOMINA_SERAPAN,
    KelasKata.NUMERALIA,
}


class PropositionalEncoder:
    """
    Encoder kalimat → Proposisi.

    Menggunakan output LPS (morfem + peran gramatikal) dan SFM (domain)
    untuk membangun representasi proposisi secara deterministik.

    TIDAK menggunakan embedding atau attention — hanya aturan linguistik.
    """

    def __init__(self, leksikon: LexiconLoader) -> None:
        self.leksikon = leksikon

    def encode(
        self,
        morfem_list: List[Morfem],
        kalimat_asli: str = "",
    ) -> Optional[Proposisi]:
        """
        Encode list morfem → Proposisi.

        Returns None jika kalimat tidak memiliki verba (non-propositional).
        """
        if not morfem_list:
            return None

        # ── Langkah 1: Deteksi polaritas dan modalitas ────────────────────
        polaritas = True
        modalitas = None
        for m in morfem_list:
            root = m.root.lower()
            if root in KATA_NEGASI:
                polaritas = False
            if root in KATA_MODALITAS:
                modalitas = KATA_MODALITAS[root]

        # ── Langkah 2: Temukan verba predikat utama ───────────────────────
        verba_utama = self._cari_verba_utama(morfem_list)
        if verba_utama is None:
            # Kalimat nominal predikatif: N adalah N / N sangat Adj
            return self._encode_nominal(morfem_list, polaritas, modalitas, kalimat_asli)

        aksi_root = verba_utama.root.lower()
        aksi_domain = self.leksikon.domain_kata(aksi_root)

        # ── Langkah 3: Deteksi struktur aktif/pasif ───────────────────────
        adalah_pasif = aksi_root.startswith("di") or any(
            a.bentuk.startswith("di") for a in (verba_utama.afiks_aktif or [])
        )

        # ── Langkah 4: Isi slot dari peran gramatikal ─────────────────────
        slot: Dict[TipeSlot, SlotProposisi] = {}

        # Kelompokkan morfem berdasarkan peran
        subjek    = [m for m in morfem_list if m.peran_gramatikal == PeranGramatikal.SUBJEK]
        objek     = [m for m in morfem_list if m.peran_gramatikal == PeranGramatikal.OBJEK]
        keterangan = [m for m in morfem_list if m.peran_gramatikal == PeranGramatikal.KETERANGAN]
        pelengkap = [m for m in morfem_list if m.peran_gramatikal == PeranGramatikal.PELENGKAP]

        # Subjek → AGEN (aktif) atau PASIEN (pasif)
        if subjek:
            m = subjek[0]
            tipe = TipeSlot.PASIEN if adalah_pasif else TipeSlot.AGEN
            slot[tipe] = self._buat_slot(tipe, m)

        # Objek → PASIEN (aktif) atau AGEN (pasif)
        if objek:
            m = objek[0]
            tipe = TipeSlot.AGEN if adalah_pasif else TipeSlot.PASIEN
            slot[tipe] = self._buat_slot(tipe, m)

        # Pelengkap → TEMA atau HASIL
        if pelengkap:
            m = pelengkap[0]
            slot[TipeSlot.TEMA] = self._buat_slot(TipeSlot.TEMA, m)

        # Keterangan → slot berdasarkan preposisi sebelumnya
        self._isi_slot_keterangan(morfem_list, slot)

        # ── Langkah 5: Fallback jika agen/pasien tidak terdeteksi ─────────
        # Cari nomina terdekat ke kiri verba sebagai agen
        if TipeSlot.AGEN not in slot:
            idx_v = morfem_list.index(verba_utama)
            for m in reversed(morfem_list[:idx_v]):
                if m.kelas_kata in KELAS_NOMINAL:
                    slot[TipeSlot.AGEN] = self._buat_slot(TipeSlot.AGEN, m)
                    break

        # Cari nomina terdekat ke kanan verba sebagai pasien
        if TipeSlot.PASIEN not in slot:
            idx_v = morfem_list.index(verba_utama)
            for m in morfem_list[idx_v + 1:]:
                if m.kelas_kata in KELAS_NOMINAL and m.root.lower() not in KATA_NEGASI:
                    slot[TipeSlot.PASIEN] = self._buat_slot(TipeSlot.PASIEN, m)
                    break

        return Proposisi(
            aksi=aksi_root,
            aksi_domain=aksi_domain,
            slot=slot,
            polaritas=polaritas,
            modalitas=modalitas,
            sumber_kalimat=kalimat_asli,
        )

    def _encode_nominal(
        self,
        morfem_list: List[Morfem],
        polaritas: bool,
        modalitas: Optional[str],
        kalimat_asli: str,
    ) -> Optional[Proposisi]:
        """
        Encode kalimat nominal predikatif:
          "X adalah Y" → ATRIBUSI(entitas=X, atribut=Y)
          "X sangat Y" → ATRIBUSI(entitas=X, atribut=Y)
        """
        nomina = [m for m in morfem_list if m.kelas_kata in KELAS_NOMINAL]
        adj    = [m for m in morfem_list if m.kelas_kata == KelasKata.ADJEKTIVA]

        if not nomina:
            return None

        slot: Dict[TipeSlot, SlotProposisi] = {}
        slot[TipeSlot.ATRIBUT] = self._buat_slot(TipeSlot.ATRIBUT, nomina[0])

        if len(nomina) >= 2:
            slot[TipeSlot.TEMA] = self._buat_slot(TipeSlot.TEMA, nomina[1])
        elif adj:
            slot[TipeSlot.TEMA] = self._buat_slot(TipeSlot.TEMA, adj[0])

        return Proposisi(
            aksi="adalah",
            aksi_domain="deskriptif",
            slot=slot,
            polaritas=polaritas,
            modalitas=modalitas,
            sumber_kalimat=kalimat_asli,
        )

    # Prefiks verba aktif Indonesia — digunakan untuk fallback deteksi verba
    _PREFIKS_VERBA_AKTIF = ("me", "ber", "ter", "per", "memper", "diper")

    def _cari_verba_utama(self, morfem_list: List[Morfem]) -> Optional[Morfem]:
        """Cari verba predikat utama — bukan verba bantu atau modalitas."""
        kandidat = []
        for m in morfem_list:
            is_verba = m.kelas_kata in (KelasKata.VERBA, KelasKata.VERBA_SERAPAN)

            # Fallback: token TIDAK_DIKETAHUI yang punya pola prefiks verba aktif
            # Contoh: 'memeriksa', 'meresepkan', 'mengoperasi'
            if not is_verba and m.kelas_kata == KelasKata.TIDAK_DIKETAHUI:
                t = m.teks_asli.lower()
                if any(t.startswith(p) and len(t) > len(p) + 3
                       for p in self._PREFIKS_VERBA_AKTIF):
                    is_verba = True

            if not is_verba:
                continue
            if m.root.lower() in KATA_MODALITAS:
                continue
            kandidat.append(m)

        if not kandidat:
            return None
        # Prioritas: verba dengan peran PREDIKAT, fallback verba pertama
        for m in kandidat:
            if m.peran_gramatikal == PeranGramatikal.PREDIKAT:
                return m
        return kandidat[0]

    def _isi_slot_keterangan(
        self,
        morfem_list: List[Morfem],
        slot: Dict[TipeSlot, SlotProposisi],
    ) -> None:
        """
        Isi slot keterangan berdasarkan preposisi sebelum nomina.
        Scan kiri ke kanan: jika ketemu preposisi → nomina berikutnya adalah slot.
        """
        i = 0
        while i < len(morfem_list):
            m = morfem_list[i]
            if m.kelas_kata == KelasKata.PREPOSISI:
                prep = m.root.lower()
                tipe = PREPOSISI_KE_SLOT.get(prep)
                if tipe and i + 1 < len(morfem_list):
                    m_next = morfem_list[i + 1]
                    if m_next.kelas_kata in KELAS_NOMINAL:
                        # Jangan overwrite slot yang sudah ada kecuali lebih spesifik
                        if tipe not in slot:
                            slot[tipe] = self._buat_slot(tipe, m_next)
            i += 1

    def _buat_slot(self, tipe: TipeSlot, m: Morfem) -> SlotProposisi:
        root = m.root.lower()
        domain = self.leksikon.domain_kata(root)
        return SlotProposisi(
            tipe=tipe,
            nilai=m.teks_asli,
            root=root,
            domain=domain,
            indeks=m.indeks,
        )
