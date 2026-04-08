"""
LexiconLoader — pemuat leksikon multi-layer untuk SFM.

OPOSISI TRANSFORMER:
  Transformer: pengetahuan semantik tersimpan implisit di bobot (tidak bisa diupdate)
  SFM LexiconLoader: pengetahuan semantik tersimpan eksplisit di graf leksikon
                     (bisa di-patch realtime tanpa retrain)

Layer leksikon:
  1. KBBI Core     — ~127k lemma formal
  2. Wiktionary-ID — ~300k+ termasuk informal (opsional)
  3. Corpus-derived — pola konsisten dari korpus (opsional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


class EntriLeksikon:
    """Satu entri kata dalam leksikon multi-layer."""
    __slots__ = ["kata", "kelas", "domain", "sinonim", "antonim",
                 "hiponim", "hipernim", "layer", "frekuensi"]

    def __init__(
        self,
        kata: str,
        kelas: str = "?",
        domain: Optional[str] = None,
        sinonim: Optional[List[str]] = None,
        antonim: Optional[List[str]] = None,
        hiponim: Optional[List[str]] = None,
        hipernim: Optional[str] = None,
        layer: str = "kbbi",
        frekuensi: int = 1,
    ):
        self.kata     = kata
        self.kelas    = kelas
        self.domain   = domain
        self.sinonim  = sinonim or []
        self.antonim  = antonim or []
        self.hiponim  = hiponim or []
        self.hipernim = hipernim
        self.layer    = layer
        self.frekuensi = frekuensi


class LexiconLoader:
    """
    Pemuat dan pengelola leksikon multi-layer untuk SFM.

    OPOSISI TRANSFORMER:
    Transformer menyimpan pengetahuan di bobot — tidak bisa diupdate tanpa retrain.
    LexiconLoader menyimpan pengetahuan di struktur data eksplisit — bisa di-patch
    kapanpun tanpa menyentuh bobot model apapun.

    Ini adalah implementasi dari prinsip:
    "Update pengetahuan = patch knowledge graph, bobot tetap"
    """

    # Override domain untuk kata yang sense KBBI-nya menyesatkan.
    # Kunci = lemma lowercase, nilai = domain yang benar (None = hapus domain).
    # Justifikasi: sense primer yang lazim tidak selalu muncul pertama di kbbi_core_v2.json.
    _DOMAIN_KOREKSI: Dict[str, Optional[str]] = {
        # 'rumah' sense-1 di file = rumah kaca (greenhouse) → domain kuliner (salah)
        # Sense primer = bangunan tempat tinggal
        "rumah":     "bangunan",
        # 'sayuran' = sayur-mayur → primer domain kuliner
        "sayuran":   "kuliner",
        # Kata kerja gerak umum tidak punya domain spesifik
        "pergi":     None,
        "datang":    None,
        "berlari":   None,
        "berjalan":  None,
        "bergerak":  None,
        # Kata bantu/aspektual tidak punya domain semantik
        "akan":      None,
        "sudah":     None,
        "telah":     None,
        "sedang":    None,
        "belum":     None,
        "pernah":    None,
        # ── Koreksi false-match keyword di definisi KBBI ──────────────────────
        # 'budi' sense-1: definisi mengandung "pendidikan untuk..." sebagai kalimat
        # contoh, bukan sebagai penanda domain. Makna primer = akal budi (universal).
        "budi":      None,
        # 'sakit' sense-1: definisi mengandung "(demam, sakit perut)" → Tier-2
        # kuliner salah cocok. Makna primer = kondisi tubuh tidak sehat (universal).
        "sakit":     None,
        # 'toko' sense-1: definisi mengandung "makanan kecil" sebagai contoh barang.
        # Makna primer = tempat berjualan umum, bukan domain kuliner.
        "toko":      None,
        # 'baru' sense-1: berbagai definisi bisa memicu false match. Baru = adjektiva umum.
        "baru":      None,
        # 'lama': adjektiva umum (waktu), tidak ada domain spesifik.
        "lama":      None,
        # Kata-kata yang menjadi nama orang umum di Indonesia tapi ada di KBBI
        # dengan domain karena makna non-nama mereka — tidak boleh kena mismatch.
        # Justifikasi: dalam kalimat dengan konteks NLP, kata-kata ini hampir selalu
        # dipakai sebagai nama proper, bukan sebagai kata bermaknaan domain tersebut.
        "indah":     None,  # KBBI: keindahan seni → bisa dipicu domain
        "sari":      None,  # KBBI: inti/sari → bisa dipicu domain kuliner/kesehatan
        "dewi":      None,  # KBBI: bidadari → bisa dipicu domain
        "melati":    None,  # KBBI: tanaman bunga → bisa dipicu kuliner/alat_musik
    }

    # Kata nomina yang merujuk entitas bernyawa [+animate] — agen valid verba psikologis.
    # Justifikasi linguistik (theta-role selection / GB Theory):
    #   Verba emosi dan volitional memilih argumen [+animate] sebagai agen.
    #   Melanggar selektivitas-θ ini adalah categorical violation, bukan degree violation.
    # Dasar: tata bahasa universal (Chomsky 1981) + tata bahasa Indonesia (Alwi dkk. 2003).
    # Daftar dibagi dua tier:
    #   Tier-1: nomina manusia — selalu animate (orang, dokter, hakim, ...)
    #   Tier-2: nomina hewan — animate (anjing, burung, ikan, ...)
    # Benda, abstraksi, institusi = TIDAK animate → melanggar jika jadi agen verba emosi.
    KATA_ANIMATE_MANUSIA: frozenset = frozenset({
        # Manusia generik
        "orang", "manusia", "seseorang", "individu", "masyarakat",
        "penduduk", "warga", "rakyat", "umat",
        # Peran sosial / profesi
        "dokter", "guru", "hakim", "jaksa", "polisi", "tentara", "prajurit",
        "petani", "nelayan", "pedagang", "pegawai", "karyawan", "buruh",
        "mahasiswa", "siswa", "pelajar", "murid", "dosen",
        "presiden", "menteri", "gubernur", "bupati", "walikota",
        "direktur", "manajer", "atasan", "bawahan", "kolega",
        "pasien", "terdakwa", "saksi", "tersangka", "narapidana",
        "anak", "ayah", "ibu", "kakak", "adik", "saudara", "keluarga",
        "suami", "istri", "teman", "sahabat", "musuh", "lawan", "kawan",
        "pemimpin", "ketua", "anggota", "peserta", "pengunjung",
        "penumpang", "pengemudi", "sopir", "pilot",
        "penulis", "pembicara", "pendengar", "penonton",
        "kurir", "direktur", "komisaris", "sekretaris", "bendahara",
        "konsultan", "auditor", "notaris", "pengacara", "advokat",
        "kepala", "wakil", "staf", "relawan", "petugas", "pejabat",
        "wali", "narasumber", "perwakilan", "delegasi",
        # Pronomina (ditangani terpisah oleh LPS, tapi untuk konsistensi lookup)
        "saya", "aku", "kamu", "anda", "dia", "ia", "kami", "kita", "mereka",
        "beliau", "kalian",
    })
    KATA_ANIMATE_HEWAN: frozenset = frozenset({
        "anjing", "kucing", "burung", "ikan", "ayam", "sapi", "kambing",
        "kuda", "harimau", "singa", "gajah", "monyet", "kera", "tikus",
        "kelinci", "bebek", "angsa", "ular", "buaya", "kura-kura",
    })

    def __init__(self):
        self._entri: Dict[str, EntriLeksikon] = {}
        self._domain_index: Dict[str, Set[str]] = {}
        self._sinonim_index: Dict[str, Set[str]] = {}
        self._antonim_index: Dict[str, Set[str]] = {}
        self._hiponim_index: Dict[str, Set[str]] = {}

    def muat_kbbi(self, path: str) -> int:
        """
        Muat KBBI dari file JSON.

        Format KBBI yang didukung:
          {lemma: {pos: str, domain: str, ...}}
          atau format kbbi_core_v2.json yang sudah ada
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return 0

        data = json.loads(path_obj.read_text(encoding="utf-8"))
        count = 0

        # Format kbbi_core_v2.json: {entries: [{lemma, pos, ...}, ...]}
        # Format alternatif: {lemma: {pos, domain, ...}, ...}
        if isinstance(data, dict) and "entries" in data:
            entri_list = data["entries"]
        elif isinstance(data, list):
            entri_list = data
        elif isinstance(data, dict):
            # Format lama: dict langsung
            entri_list = [
                {"lemma": k, **v} if isinstance(v, dict) else {"lemma": k, "pos": str(v)}
                for k, v in data.items()
            ]
        else:
            return 0

        for item in entri_list:
            if not isinstance(item, dict):
                continue

            lemma  = item.get("lemma", "")
            if not lemma:
                continue

            pos    = item.get("pos", "?")
            # domain bisa dari field 'domain', 'bidang', atau diekstrak dari definisi
            domain = item.get("domain", item.get("bidang", None))
            if isinstance(domain, list):
                domain = domain[0] if domain else None

            # Kata fungsi tidak punya domain semantik
            POS_TANPA_DOMAIN = {"bt", "p", "pron", "adv", "konj", "part", "num", "ark", "kl", "kp"}
            if str(pos).lower() in POS_TANPA_DOMAIN:
                domain = None
            elif domain is None:
                defn = item.get("clean_definition", item.get("definition", ""))
                # Hanya ekstrak domain dari bagian definisi utama (sebelum ';' atau '(2)')
                defn_utama = str(defn).split(";")[0].split("(2)")[0] if defn else ""
                # Definisi harus cukup pendek/informatif (bukan kalimat contoh panjang)
                if defn_utama and len(defn_utama) < 200:
                    domain = self._ekstrak_domain_dari_definisi(defn_utama)
                else:
                    domain = None

            # Domain correction layer: override kata-kata yang sense KBBI-nya menyesatkan
            # Sense primer yang umum tidak selalu muncul pertama di kbbi_core_v2.json
            domain = self._DOMAIN_KOREKSI.get(lemma.lower(), domain)

            kata_lower = lemma.lower()

            # Prioritaskan POS yang lebih substantif (n, v, adj) daripada ref/bt/ark
            POS_PRIORITAS = {"n": 3, "v": 3, "adj": 2, "a": 2, "adv": 1}
            if kata_lower in self._entri:
                existing_pos = self._entri[kata_lower].kelas.lower()
                current_pos  = str(pos).lower()
                # Hanya override jika POS baru lebih substantif
                if POS_PRIORITAS.get(current_pos, 0) <= POS_PRIORITAS.get(existing_pos, 0):
                    continue
                # Override dengan entri yang lebih baik (hapus dari domain index dulu)
                old_domain = self._entri[kata_lower].domain
                if old_domain and kata_lower in self._domain_index.get(old_domain.lower(), set()):
                    self._domain_index[old_domain.lower()].discard(kata_lower)

            entri_obj = EntriLeksikon(
                kata=kata_lower,
                kelas=str(pos) if pos else "?",
                domain=str(domain) if domain else None,
                layer="kbbi",
            )
            self._entri[kata_lower] = entri_obj

            if domain:
                dom_key = str(domain).lower()
                if dom_key not in self._domain_index:
                    self._domain_index[dom_key] = set()
                self._domain_index[dom_key].add(kata_lower)

            count += 1

        return count

    def _ekstrak_domain_dari_definisi(self, defn: str) -> Optional[str]:
        """
        Ekstrak domain dari definisi KBBI berdasarkan kata kunci dalam teks definisi.

        Strategi dua-tier:
          Tier-1: keyword sangat spesifik → 1 match sudah cukup
          Tier-2: keyword lebih umum → butuh ≥ 2 match
        Ini mencegah 'pergi' dapat domain bangunan hanya karena definisi
        mengandung 'kamar', tapi 'senapan' tetap mendapat domain senjata
        karena definisinya mengandung 'senjata' (tier-1).
        """
        # Tier-1: keyword sangat spesifik, 1 match = cukup
        DOMAIN_T1: Dict[str, List[str]] = {
            "kuliner": [
                "makanan", "masakan", "minuman", "bumbu", "kuliner",
                "dimasak", "direbus", "dikukus", "lauk", "hidangan", "rempah",
            ],
            "senjata": [
                "senjata", "peluru", "senapan", "meriam", "pistol", "pedang",
                "keris", "tombak", "panah", "granat", "amunisi", "bedil",
                "senjata api", "senapan api",
            ],
            "kendaraan": [
                "kendaraan", "berkendara", "dikendarai", "mesin kendaraan",
                "transportasi", "angkutan umum",
            ],
            "kesehatan": [
                "penyakit", "medis", "klinis", "farmasi", "rumah sakit",
                "infeksi", "bakteri", "virus", "pengobatan", "terapi",
            ],
            "hukum": [
                "hukum", "undang-undang", "pidana", "perdata", "pengadilan",
                "jaksa", "hakim", "terdakwa", "putusan", "vonis", "sanksi",
            ],
            "ekonomi": [
                "ekonomi", "keuangan", "bisnis", "perdagangan", "transaksi",
                "investasi", "kredit", "modal usaha",
            ],
            "pendidikan": [
                "pendidikan", "akademik", "kurikulum", "mahasiswa",
                "universitas", "diploma", "mata pelajaran",
            ],
            "bangunan": [
                "bangunan", "arsitektur", "konstruksi bangunan", "arsitek",
                "pondasi", "beton", "batu bata", "struktur bangunan", "denah",
                "gedung bertingkat",
            ],
            "busana": [
                "pakaian", "busana", "tekstil", "menjahit", "seragam",
                "kostum", "lengan baju",
            ],
            "alat_musik": [
                "alat musik", "melodi", "harmoni", "orkestra", "musisi",
                "instrumen musik",
            ],
        }
        # Tier-2: keyword lebih umum, butuh ≥ 2 match
        DOMAIN_T2: Dict[str, List[str]] = {
            "kuliner": ["makan", "nasi", "sayur", "buah", "daging", "ikan",
                        "resep", "rasa", "enak", "sedap", "lezat", "menu",
                        "beras", "sayuran"],
            "senjata": ["perang", "pertempuran", "militer", "tentara", "menyerang"],
            "kendaraan": ["mobil", "motor", "kereta", "pesawat", "kapal",
                          "sepeda", "bus", "truk", "roda"],
            "kesehatan": ["obat", "dokter", "pasien", "sakit", "sembuh", "gejala"],
            "hukum": ["pasal", "peraturan", "regulasi", "denda", "hukuman"],
            "ekonomi": ["dagang", "pasar", "harga", "jual", "beli", "uang",
                        "untung", "rugi", "bank"],
            "pendidikan": ["sekolah", "belajar", "ilmu", "guru", "siswa",
                           "pelajaran", "ujian", "nilai"],
            "bangunan": ["gedung", "konstruksi", "renovasi", "tembok", "semen"],
            "busana": ["baju", "celana", "rok", "kemeja", "kain", "mode",
                       "fashion", "kancing", "kerah"],
            "alat_musik": ["musik", "nada", "gitar", "piano", "biola",
                           "seruling", "drum", "gamelan", "lagu", "bernyanyi"],
        }

        defn_lower = defn.lower()
        skor: Dict[str, float] = {}

        # Tier-1: 1 match = score 2.0 (lebih tinggi agar tier-1 menang atas tier-2)
        for domain, kw_list in DOMAIN_T1.items():
            count = sum(1 for kk in kw_list if kk in defn_lower)
            if count >= 1:
                skor[domain] = skor.get(domain, 0) + count * 2.0

        # Tier-2: butuh ≥ 2 match, score 1.0 per match
        for domain, kw_list in DOMAIN_T2.items():
            count = sum(1 for kk in kw_list if kk in defn_lower)
            if count >= 2:
                skor[domain] = skor.get(domain, 0) + count * 1.0

        if not skor:
            return None
        return max(skor, key=lambda d: skor[d])

    def tambah_entri(
        self,
        kata: str,
        kelas: str = "?",
        domain: Optional[str] = None,
        sinonim: Optional[List[str]] = None,
        antonim: Optional[List[str]] = None,
        layer: str = "custom",
    ) -> None:
        """
        Tambah satu entri ke leksikon — patch knowledge graph tanpa retrain.
        Inilah keunggulan utama SFM vs embedding Transformer.
        """
        kata = kata.lower()
        entri = EntriLeksikon(
            kata=kata, kelas=kelas, domain=domain,
            sinonim=sinonim or [], antonim=antonim or [],
            layer=layer,
        )
        self._entri[kata] = entri

        if domain:
            dom_key = domain.lower()
            if dom_key not in self._domain_index:
                self._domain_index[dom_key] = set()
            self._domain_index[dom_key].add(kata)

        for s in (sinonim or []):
            if s not in self._sinonim_index:
                self._sinonim_index[s] = set()
            self._sinonim_index[s].add(kata)

        for a in (antonim or []):
            if a not in self._antonim_index:
                self._antonim_index[a] = set()
            self._antonim_index[a].add(kata)

    def cari(self, kata: str) -> Optional[EntriLeksikon]:
        return self._entri.get(kata.lower())

    def ada(self, kata: str) -> bool:
        return kata.lower() in self._entri

    def adalah_animate(self, kata: str) -> bool:
        """
        Apakah kata ini merujuk entitas bernyawa [+animate]?

        Digunakan oleh CPE untuk theta-role selection constraint:
        verba emosi/psikologis/volitional membutuhkan agen [+animate].

        Hierarki keputusan:
          1. NOMINA_PROPER (nama orang) → selalu animate
          2. PRONOMINA persona → selalu animate
          3. Ada di KATA_ANIMATE_MANUSIA → animate
          4. Ada di KATA_ANIMATE_HEWAN → animate
          5. Lainnya → tidak animate (benda, abstraksi, institusi)
        """
        k = kata.lower()
        if k in self.KATA_ANIMATE_MANUSIA:
            return True
        if k in self.KATA_ANIMATE_HEWAN:
            return True
        # Cek dari leksikon: kata dengan domain sosial/profesi bisa animate
        entri = self._entri.get(k)
        if entri and entri.kelas in ("pron", "Pron"):
            return True
        return False

    def domain_kata(self, kata: str) -> Optional[str]:
        entri = self.cari(kata)
        return entri.domain if entri else None

    def kata_per_domain(self, domain: str) -> Set[str]:
        return self._domain_index.get(domain.lower(), set())

    def semua_domain(self) -> List[str]:
        return sorted(self._domain_index.keys())

    def relasi(self, kata: str) -> Dict[str, List[str]]:
        """Kembalikan semua relasi kata dalam leksikon."""
        entri = self.cari(kata)
        if not entri:
            return {}
        return {
            "sinonim":  entri.sinonim,
            "antonim":  entri.antonim,
            "hiponim":  entri.hiponim,
            "hipernim": [entri.hipernim] if entri.hipernim else [],
            "domain":   [entri.domain] if entri.domain else [],
        }

    @property
    def ukuran(self) -> int:
        return len(self._entri)

    @property
    def n_domain(self) -> int:
        return len(self._domain_index)
