"""
KategoriMakna — sistem kategori untuk komposisi makna bahasa Indonesia.

OPOSISI TRANSFORMER:
  Transformer: komposisi = multiply(Q, K^T) / sqrt(d) — tidak punya struktur algebraik
  CMC:         komposisi = f ∘ g dengan hukum asosiativitas dan identitas yang provable

Kategori C terdiri dari:
  - Objek: DomainMakna (kelas kata × domain semantik × register)
  - Morfisme: fungsi antara objek yang merepresentasikan perubahan makna
  - Komposisi: f ∘ g yang associative
  - Identitas: id_A untuk setiap objek A
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from aksara.primitives.cmc.morphism import (
    DomainMakna, Morfisme, TipeMorfisme,
    buat_morfisme_adjektiva, buat_morfisme_verba,
)
from aksara.primitives.lps.morfem import Morfem, KelasKata, PeranGramatikal
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.config import AksaraConfig

# Kata penanda komplemen yang wajib diikuti komplemen bermakna
# Dibagi menjadi frasa multi-kata dan kata tunggal agar matching lebih akurat
KATA_PENANDA_KOMPLEMEN_FRASA = [
    "antara lain", "terdiri dari", "terdiri atas", "sebagai berikut",
    "antara lain:", "yaitu:", "yakni:", "adalah:",
]
KATA_PENANDA_KOMPLEMEN_TUNGGAL = [
    "yaitu", "yakni", "ialah", "meliputi",
    "misalnya", "contohnya",
]

# Pasangan verba → domain objek yang TIDAK kompatibel
# Format: domain_verba → set domain_objek yang tidak lazim
# CATATAN: domain "ekonomi" TIDAK ada di sini karena verba ekonomi
# seperti "beli/jual/bayar" adalah domain-neutral — bisa berelasi
# dengan objek dari domain apapun (membeli senapan, menjual makanan, dll.)
VERBA_OBJEK_INCOMPATIBLE = {
    "kuliner":    {"senjata", "kendaraan", "alat_musik"},
    "pendidikan": {"senjata", "kendaraan"},
    "kesehatan":  {"senjata", "alat_musik"},
    "senjata":    {"kuliner", "pendidikan", "kesehatan"},
}

# Root verba yang bersifat domain-neutral — dikecualikan dari cek inkoherensi
# Verba ini secara leksikal bisa berelasi dengan objek dari domain manapun
VERBA_DOMAIN_NEUTRAL_CMC = {
    "beli", "jual", "dagang", "niaga", "bayar", "sewa", "pinjam",
    "beri", "berikan", "hibah", "hadiahkan", "kirim", "antar",
    "miliki", "punya", "simpan", "ambil", "bawa", "dapat", "terima",
    "buat", "bangun", "produksi", "ciptakan", "hasilkan", "kembangkan",
    "pakai", "gunakan", "manfaatkan", "butuhkan",
    "lihat", "tahu", "kenal", "ingat", "cari", "temukan",
}


class KategoriMakna:
    """
    Kategori makna untuk kalimat bahasa Indonesia.

    Mengimplementasikan struktur kategori C = (Obj, Mor, ∘, id) di mana:
    - Obj = himpunan DomainMakna (kelas kata × semantik × register)
    - Mor = himpunan Morfisme antara objek
    - ∘   = komposisi morfisme (associative)
    - id  = morfisme identitas per objek

    OPOSISI TRANSFORMER:
    Transformer tidak punya struktur algebraik yang provable.
    KategoriMakna memastikan setiap komposisi makna bisa diverifikasi
    apakah memenuhi hukum kategori atau tidak.
    """

    def __init__(self, leksikon: LexiconLoader,
                 config: Optional[AksaraConfig] = None):
        self.leksikon = leksikon
        self.config   = config or AksaraConfig.default()
        self._objek: Dict[str, DomainMakna] = {}
        self._morfisme: List[Morfisme] = []

    def dari_morfem(self, morfem: Morfem) -> DomainMakna:
        """
        Konversi Morfem ke DomainMakna — objek dalam kategori.
        """
        kata = morfem.root.lower()
        domain_sem = self.leksikon.domain_kata(kata)
        entri = self.leksikon.cari(kata)

        kelas = morfem.kelas_kata.value.lower()
        mapping = {
            "n": "n", "v": "v", "adj": "adj", "adv": "adv",
            "adjektiva": "adj", "adverbia": "adv",
            "n_proper": "n", "n_serapan": "n", "v_serapan": "v",
            "pron": "pron", "prep": "prep", "konj": "konj",
            "num": "num", "part": "part", "interj": "interj",
        }
        kelas_canonical = mapping.get(kelas, kelas)

        domain = DomainMakna(
            kelas_kata=kelas_canonical,
            domain_semantik=domain_sem,
            register="informal" if morfem.adalah_informal else
                     ("netral" if morfem.adalah_serapan else "formal"),
            animasi=self._inferensi_animasi(kata, kelas_canonical),
            abstrak=self._inferensi_abstrak(kata, domain_sem),
        )
        self._objek[kata] = domain
        return domain

    def bangun_dari_kalimat(
        self, morfem_list: List[Morfem]
    ) -> Tuple[List[DomainMakna], List[Morfisme]]:
        """
        Bangun sistem kategori dari seluruh kalimat.

        Returns:
            (objek_list, morfisme_list) — dua komponen kategori
        """
        objek_list = [self.dari_morfem(m) for m in morfem_list]
        morfisme_list = self._buat_morfisme_kalimat(morfem_list, objek_list)
        self._morfisme = morfisme_list
        return objek_list, morfisme_list

    def verifikasi_hukum_kategori(
        self, morfisme_list: List[Morfisme]
    ) -> Dict[str, bool]:
        """
        Verifikasi apakah morfisme memenuhi hukum kategori.

        Properti yang dicek:
        1. Identitas: ada morfisme id untuk setiap objek
        2. Closure: komposisi valid antara morfisme compatible
        3. Validitas: setiap morfisme individually valid
        """
        semua_valid = all(m.valid for m in morfisme_list)
        ada_invalid = [m for m in morfisme_list if not m.valid]

        return {
            "semua_morfisme_valid": semua_valid,
            "n_pelanggaran": len(ada_invalid),
            "pelanggaran": [m.penjelasan for m in ada_invalid],
        }

    def cek_kompatibilitas_makna(
        self,
        morfem_a: Morfem,
        morfem_b: Morfem,
    ) -> Tuple[bool, str]:
        """
        Cek apakah dua morfem bisa membentuk morfisme valid.
        Ini adalah cek kompatibilitas tipe dalam category theory.
        """
        domain_a = self.dari_morfem(morfem_a)
        domain_b = self.dari_morfem(morfem_b)

        peran_a = morfem_a.peran_gramatikal
        peran_b = morfem_b.peran_gramatikal

        # Adj + Nomina (MODIFIER + HEAD)
        if (morfem_a.kelas_kata == KelasKata.ADJEKTIVA and
                morfem_b.kelas_kata in (KelasKata.NOMINA, KelasKata.NOMINA_PROPER)):
            m = buat_morfisme_adjektiva(domain_a, domain_b,
                                        morfem_a.root, morfem_b.root)
            if m:
                return m.valid, m.penjelasan

        # Nomina/Pronomina (SUBJEK) + Verba (PREDIKAT)
        if (peran_a == PeranGramatikal.SUBJEK and
                peran_b == PeranGramatikal.PREDIKAT):
            m = buat_morfisme_verba(domain_a, domain_b, None,
                                    morfem_a.root, morfem_b.root)
            if m:
                return m.valid, m.penjelasan

        return True, "Tidak ada morfisme spesifik yang dicek"

    # ── Private ───────────────────────────────────────────────────────────────

    def _buat_morfisme_kalimat(
        self,
        morfem_list: List[Morfem],
        objek_list: List[DomainMakna],
    ) -> List[Morfisme]:
        """Buat morfisme untuk setiap pasangan yang berrelasi dalam kalimat."""
        morfisme_list: List[Morfisme] = []
        n = len(morfem_list)

        for i in range(n):
            # Morfisme identitas untuk setiap morfem
            morfisme_list.append(Morfisme.identitas(objek_list[i]))

            for j in range(i + 1, min(i + 3, n)):
                ma, mb   = morfem_list[i], morfem_list[j]
                da, db   = objek_list[i], objek_list[j]

                # Adj → Nomina: morfisme modifikasi
                if (ma.kelas_kata == KelasKata.ADJEKTIVA and
                        mb.kelas_kata in (KelasKata.NOMINA, KelasKata.NOMINA_PROPER,
                                          KelasKata.NOMINA_SERAPAN)):
                    m = buat_morfisme_adjektiva(da, db, ma.root, mb.root)
                    if m:
                        morfisme_list.append(m)

                # Subj + Pred: morfisme verbalisasi
                elif (ma.peran_gramatikal == PeranGramatikal.SUBJEK and
                      mb.peran_gramatikal == PeranGramatikal.PREDIKAT):
                    m = buat_morfisme_verba(da, db, None, ma.root, mb.root)
                    if m:
                        morfisme_list.append(m)

                # Pred + Obj: morfisme verbalisasi dengan objek
                elif (ma.peran_gramatikal == PeranGramatikal.PREDIKAT and
                      mb.peran_gramatikal == PeranGramatikal.OBJEK):
                    m = buat_morfisme_verba(da, db, db, ma.root, mb.root, mb.root)
                    if m:
                        morfisme_list.append(m)

        # ── Deteksi koherensi kalimat tingkat kalimat ────────────────────────
        morfisme_list.extend(
            self._deteksi_kalimat_menggantung(morfem_list)
        )
        morfisme_list.extend(
            self._deteksi_verb_obj_inkoherensi(morfem_list, objek_list)
        )

        return morfisme_list

    def _deteksi_kalimat_menggantung(
        self, morfem_list: List[Morfem]
    ) -> List[Morfisme]:
        """
        Deteksi kalimat menggantung: ada kata penanda komplemen tapi tidak
        ada komplemen yang mengikutinya, atau kalimat berakhir dengan titik dua.

        Contoh yang salah:
          'Jenis-jenis tarian yang terdapat di umum ini antara lain:'
          → berakhir titik dua setelah 'antara lain', komplemen tidak ada

        Contoh yang benar:
          'Jenis-jenis tarian antara lain tari saman dan tari kecak.'
          → 'antara lain' diikuti minimal 2 morfem bermakna
        """
        hasil = []
        if not morfem_list:
            return hasil

        # Rekonstruksi teks dari morfem (tanpa tanda baca)
        teks_list  = [m.teks_asli.lower().strip(".,;:!?") for m in morfem_list]
        teks_gabung = " ".join(teks_list)  # mis. "antara lain" akan muncul di sini

        # Teks asli lengkap (untuk cek titik dua di akhir)
        teks_asli_full = " ".join(m.teks_asli for m in morfem_list).rstrip()
        berakhir_titik_dua = teks_asli_full.endswith(":")

        kelas_bermakna = {
            KelasKata.NOMINA, KelasKata.VERBA, KelasKata.ADJEKTIVA,
            KelasKata.NOMINA_PROPER, KelasKata.NOMINA_SERAPAN,
            KelasKata.VERBA_SERAPAN,
        }

        penanda_ditemukan = None
        idx_penanda = -1

        # Cek frasa multi-kata dulu (lebih spesifik)
        for penanda in KATA_PENANDA_KOMPLEMEN_FRASA:
            penanda_bersih = penanda.strip(":")
            if penanda_bersih in teks_gabung:
                # Cari indeks morfem pertama dari frasa penanda
                kata_pertama = penanda_bersih.split()[0]
                for i, teks in enumerate(teks_list):
                    if teks == kata_pertama:
                        # Verifikasi kata kedua (jika frasa dua kata)
                        kata_penanda = penanda_bersih.split()
                        if len(kata_penanda) == 2:
                            if (i + 1 < len(teks_list) and
                                    teks_list[i + 1] == kata_penanda[1]):
                                idx_penanda = i + 1  # indeks setelah frasa
                                penanda_ditemukan = penanda_bersih
                                break
                        else:
                            idx_penanda = i
                            penanda_ditemukan = penanda_bersih
                            break
                if penanda_ditemukan:
                    break

        # Cek kata tunggal jika frasa tidak ditemukan
        if not penanda_ditemukan:
            for penanda in KATA_PENANDA_KOMPLEMEN_TUNGGAL:
                if penanda in teks_list:
                    idx_penanda = teks_list.index(penanda)
                    penanda_ditemukan = penanda
                    break

        if not penanda_ditemukan:
            return hasil

        # Hitung morfem bermakna setelah penanda
        n_komplemen = sum(
            1 for m in morfem_list[idx_penanda + 1:]
            if m.kelas_kata in kelas_bermakna
        )

        # Pelanggaran: tidak ada komplemen ATAU kalimat berakhir titik dua
        # (titik dua di akhir = kalimat menggantung secara eksplisit)
        if n_komplemen < 2 or berakhir_titik_dua:
            dm_dummy = DomainMakna(
                kelas_kata="kalimat",
                domain_semantik=None,
                register="netral",
                animasi=None,
            )
            alasan = "berakhir dengan titik dua tanpa komplemen" if berakhir_titik_dua \
                     else "hanya %d morfem bermakna setelah penanda" % n_komplemen
            hasil.append(Morfisme(
                nama="menggantung(%s)" % penanda_ditemukan,
                tipe=TipeMorfisme.KOMPOSISI,
                domain_source=dm_dummy,
                domain_target=dm_dummy,
                valid=False,
                penjelasan=(
                    "Kalimat tidak lengkap makna: penanda '%s' %s. "
                    "Kalimat ini tidak bisa berdiri sendiri."
                ) % (penanda_ditemukan, alasan),
            ))

        return hasil

    def _deteksi_verb_obj_inkoherensi(
        self,
        morfem_list: List[Morfem],
        objek_list: List[DomainMakna],
    ) -> List[Morfisme]:
        """
        Deteksi relasi verb-objek yang tidak koheren secara semantik.

        Verba dengan domain tertentu memiliki ekspektasi domain objeknya.
        Contoh yang salah:
          'mengirim desain untuk menemui orang' — 'mengirim' (tindak fisik)
          diikuti tujuan 'menemui orang' — relasi kausal tidak koheren.

          'batu mulia dalam pemilihan' — domain senjata/mineral dalam
          konteks ekonomi/politik — tidak ada relasi semantik yang valid.
        """
        hasil = []
        n = len(morfem_list)

        # Ambil konfigurasi Layer 2 dari config (bisa dikustomisasi per domain)
        verba_neutral_cmc = self.config.verba_domain_neutral
        verba_incompatible = self.config.verba_objek_incompatible

        for i, ma in enumerate(morfem_list):
            if ma.kelas_kata not in (KelasKata.VERBA, KelasKata.VERBA_SERAPAN):
                continue
            # Verba domain-neutral (dari config) tidak dicek inkoherensi
            root_lower = ma.root.lower()
            is_neutral = root_lower in verba_neutral_cmc or any(
                root_lower.endswith(b) and len(root_lower) > len(b)
                for b in verba_neutral_cmc
            )
            if is_neutral:
                continue
            da = objek_list[i]
            if not da.domain_semantik:
                continue

            # Cek objek langsung (dalam window 3 ke kanan)
            for j in range(i + 1, min(i + 4, n)):
                mb = morfem_list[j]
                db = objek_list[j]

                if mb.kelas_kata not in (
                    KelasKata.NOMINA, KelasKata.NOMINA_SERAPAN,
                    KelasKata.NOMINA_PROPER
                ):
                    continue
                if not db.domain_semantik:
                    continue

                # Cek apakah domain verba dan domain objek inkompatibel
                domain_v = da.domain_semantik
                domain_o = db.domain_semantik

                incompatible = verba_incompatible.get(domain_v, set())
                if domain_o in incompatible:
                    dm_v = da
                    dm_o = db
                    hasil.append(Morfisme(
                        nama="inkoherensi_vo(%s→%s)" % (ma.root, mb.root),
                        tipe=TipeMorfisme.VERBALISASI,
                        domain_source=dm_v,
                        domain_target=dm_o,
                        valid=False,
                        penjelasan=(
                            "Relasi verb-objek tidak koheren: "
                            "'%s' [%s] tidak lazim berelasi dengan "
                            "'%s' [%s] — domain tidak compatible."
                        ) % (ma.root, domain_v, mb.root, domain_o),
                    ))
                    break  # Satu pelanggaran per verba

        return hasil

    def _inferensi_animasi(self, kata: str, kelas: str) -> Optional[bool]:
        """Apakah kata ini merujuk ke entitas bernyawa?"""
        KATA_ANIMATE = {
            "orang", "manusia", "anak", "ibu", "bapak", "guru", "dokter",
            "dia", "mereka", "kita", "saya", "aku", "kamu", "anda",
            "hewan", "binatang", "kucing", "anjing", "burung", "ikan",
        }
        if kata.lower() in KATA_ANIMATE:
            return True
        if kelas == "pron":
            return True
        return None

    def _inferensi_abstrak(self, kata: str, domain: Optional[str]) -> bool:
        """Apakah kata ini merujuk ke konsep abstrak?"""
        KATA_ABSTRAK = {
            "keadilan", "kebebasan", "kebenaran", "kesadaran", "pikiran",
            "perasaan", "cinta", "harapan", "impian", "kebahagiaan",
            "demokrasi", "ideologi", "konsep", "teori", "ide",
        }
        return kata.lower() in KATA_ABSTRAK
