"""
kb.py — Knowledge Base Indonesia untuk AKSARA.

OPOSISI TRANSFORMER:
  Transformer: pengetahuan tersimpan implisit di parameter — tidak bisa diaudit.
  KB AKSARA:   pengetahuan eksplisit sebagai ontologi — bisa diaudit, dikoreksi,
               diperluas tanpa retrain, setiap fakta bisa ditelusuri.

Isi KB:
  1. TipeEntitas   — hierarki tipe entitas dunia nyata Indonesia
  2. AturanDunia   — aturan kausalitas dan perubahan state dunia
  3. KATA_KE_TIPE  — peta kata Indonesia ke tipe entitas
  4. VERBA_KE_TIPE_AKSI — peta verba ke jenis aksi untuk matching aturan
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Hierarki Tipe Entitas ─────────────────────────────────────────────────────

class TipeEntitas:
    ENTITAS              = "ENTITAS"
    PERSONA              = "PERSONA"
    ORGANISASI           = "ORGANISASI"
    LOKASI               = "LOKASI"
    BENDA                = "BENDA"
    ABSTRAK              = "ABSTRAK"
    WAKTU                = "WAKTU"
    PROFESI_MEDIS        = "PROFESI_MEDIS"
    PROFESI_HUKUM        = "PROFESI_HUKUM"
    PROFESI_DIDIK        = "PROFESI_DIDIK"
    PEJABAT              = "PEJABAT"
    WARGA                = "WARGA"
    INSTITUSI_HUKUM      = "INSTITUSI_HUKUM"
    INSTITUSI_MEDIS      = "INSTITUSI_MEDIS"
    INSTITUSI_DIDIK      = "INSTITUSI_DIDIK"
    INSTITUSI_PEMERINTAH = "INSTITUSI_PEMERINTAH"
    PERUSAHAAN           = "PERUSAHAAN"
    LOKASI_EKONOMI       = "LOKASI_EKONOMI"
    LOKASI_ALAM          = "LOKASI_ALAM"
    HUKUM_NORMA          = "HUKUM_NORMA"
    PENYAKIT             = "PENYAKIT"
    PENGETAHUAN          = "PENGETAHUAN"
    HAK                  = "HAK"


_SUPERTIPE: Dict[str, str] = {
    TipeEntitas.PERSONA:              TipeEntitas.ENTITAS,
    TipeEntitas.ORGANISASI:           TipeEntitas.ENTITAS,
    TipeEntitas.LOKASI:               TipeEntitas.ENTITAS,
    TipeEntitas.BENDA:                TipeEntitas.ENTITAS,
    TipeEntitas.ABSTRAK:              TipeEntitas.ENTITAS,
    TipeEntitas.WAKTU:                TipeEntitas.ENTITAS,
    TipeEntitas.PROFESI_MEDIS:        TipeEntitas.PERSONA,
    TipeEntitas.PROFESI_HUKUM:        TipeEntitas.PERSONA,
    TipeEntitas.PROFESI_DIDIK:        TipeEntitas.PERSONA,
    TipeEntitas.PEJABAT:              TipeEntitas.PERSONA,
    TipeEntitas.WARGA:                TipeEntitas.PERSONA,
    TipeEntitas.INSTITUSI_HUKUM:      TipeEntitas.ORGANISASI,
    TipeEntitas.INSTITUSI_MEDIS:      TipeEntitas.ORGANISASI,
    TipeEntitas.INSTITUSI_DIDIK:      TipeEntitas.ORGANISASI,
    TipeEntitas.INSTITUSI_PEMERINTAH: TipeEntitas.ORGANISASI,
    TipeEntitas.PERUSAHAAN:           TipeEntitas.ORGANISASI,
    TipeEntitas.LOKASI_EKONOMI:       TipeEntitas.LOKASI,
    TipeEntitas.LOKASI_ALAM:          TipeEntitas.LOKASI,
    TipeEntitas.HUKUM_NORMA:          TipeEntitas.ABSTRAK,
    TipeEntitas.PENYAKIT:             TipeEntitas.ABSTRAK,
    TipeEntitas.PENGETAHUAN:          TipeEntitas.ABSTRAK,
    TipeEntitas.HAK:                  TipeEntitas.ABSTRAK,
}


def adalah_subtipe(tipe: str, supertipe: str) -> bool:
    """Cek apakah tipe adalah subtipe dari supertipe (termasuk diri sendiri)."""
    if tipe == supertipe:
        return True
    parent = _SUPERTIPE.get(tipe)
    if parent is None:
        return False
    return adalah_subtipe(parent, supertipe)


# ── Peta Kata → Tipe Entitas ──────────────────────────────────────────────────

KATA_KE_TIPE: Dict[str, str] = {
    "dokter": TipeEntitas.PROFESI_MEDIS,
    "perawat": TipeEntitas.PROFESI_MEDIS,
    "bidan": TipeEntitas.PROFESI_MEDIS,
    "apoteker": TipeEntitas.PROFESI_MEDIS,
    "hakim": TipeEntitas.PROFESI_HUKUM,
    "jaksa": TipeEntitas.PROFESI_HUKUM,
    "pengacara": TipeEntitas.PROFESI_HUKUM,
    "notaris": TipeEntitas.PROFESI_HUKUM,
    "polisi": TipeEntitas.PROFESI_HUKUM,
    "penyidik": TipeEntitas.PROFESI_HUKUM,
    "guru": TipeEntitas.PROFESI_DIDIK,
    "dosen": TipeEntitas.PROFESI_DIDIK,
    "pengajar": TipeEntitas.PROFESI_DIDIK,
    "profesor": TipeEntitas.PROFESI_DIDIK,
    "presiden": TipeEntitas.PEJABAT,
    "menteri": TipeEntitas.PEJABAT,
    "gubernur": TipeEntitas.PEJABAT,
    "bupati": TipeEntitas.PEJABAT,
    "direktur": TipeEntitas.PEJABAT,
    "siswa": TipeEntitas.WARGA,
    "mahasiswa": TipeEntitas.WARGA,
    "pasien": TipeEntitas.WARGA,
    "terdakwa": TipeEntitas.WARGA,
    "tersangka": TipeEntitas.WARGA,
    "korban": TipeEntitas.WARGA,
    "warga": TipeEntitas.WARGA,
    "murid": TipeEntitas.WARGA,
    "pengadilan": TipeEntitas.INSTITUSI_HUKUM,
    "kejaksaan": TipeEntitas.INSTITUSI_HUKUM,
    "kepolisian": TipeEntitas.INSTITUSI_HUKUM,
    "kpk": TipeEntitas.INSTITUSI_HUKUM,
    "mahkamah": TipeEntitas.INSTITUSI_HUKUM,
    "rumah_sakit": TipeEntitas.INSTITUSI_MEDIS,
    "puskesmas": TipeEntitas.INSTITUSI_MEDIS,
    "klinik": TipeEntitas.INSTITUSI_MEDIS,
    "apotek": TipeEntitas.INSTITUSI_MEDIS,
    "sekolah": TipeEntitas.INSTITUSI_DIDIK,
    "universitas": TipeEntitas.INSTITUSI_DIDIK,
    "kampus": TipeEntitas.INSTITUSI_DIDIK,
    "pesantren": TipeEntitas.INSTITUSI_DIDIK,
    "pemerintah": TipeEntitas.INSTITUSI_PEMERINTAH,
    "kementerian": TipeEntitas.INSTITUSI_PEMERINTAH,
    "dpr": TipeEntitas.INSTITUSI_PEMERINTAH,
    "pasar": TipeEntitas.LOKASI_EKONOMI,
    "toko": TipeEntitas.LOKASI_EKONOMI,
    "bank": TipeEntitas.LOKASI_EKONOMI,
    "sungai": TipeEntitas.LOKASI_ALAM,
    "gunung": TipeEntitas.LOKASI_ALAM,
    "hutan": TipeEntitas.LOKASI_ALAM,
    "pantai": TipeEntitas.LOKASI_ALAM,
    "sawah": TipeEntitas.LOKASI_ALAM,
    "hukuman": TipeEntitas.HUKUM_NORMA,
    "putusan": TipeEntitas.HUKUM_NORMA,
    "vonis": TipeEntitas.HUKUM_NORMA,
    "peraturan": TipeEntitas.HUKUM_NORMA,
    "dakwaan": TipeEntitas.HUKUM_NORMA,
    "tuntutan": TipeEntitas.HUKUM_NORMA,
    "covid": TipeEntitas.PENYAKIT,
    "diabetes": TipeEntitas.PENYAKIT,
    "hipertensi": TipeEntitas.PENYAKIT,
    "kanker": TipeEntitas.PENYAKIT,
    "demam": TipeEntitas.PENYAKIT,
}


def tipe_entitas(kata: str) -> str:
    """
    Inferensi tipe entitas dari kata.

    Strategi:
    1. Cari di KATA_KE_TIPE (eksplisit)
    2. Jika awalan kapital (NOMINA_PROPER) → fallback ke PERSONA
       Justifikasi: nama orang selalu PERSONA; kita tidak bisa tahu lebih spesifik
       tanpa Named Entity Recognition, tapi PERSONA lebih baik dari ENTITAS generik.
    3. Default: ENTITAS
    """
    hit = KATA_KE_TIPE.get(kata.lower())
    if hit:
        return hit
    if kata and kata[0].isupper():
        return TipeEntitas.PERSONA
    return TipeEntitas.ENTITAS


# ── Peta Verba → Tipe Aksi ────────────────────────────────────────────────────

VERBA_KE_TIPE_AKSI: Dict[str, str] = {
    # Transfer kepemilikan
    "beri": "transfer", "berikan": "transfer", "jual": "transfer",
    "beli": "transfer", "serahkan": "transfer", "kirim": "transfer",
    "bayar": "transfer", "hibah": "transfer",
    # Perjalanan
    "pergi": "perjalanan", "berangkat": "perjalanan", "pulang": "perjalanan",
    "tiba": "perjalanan", "datang": "perjalanan", "menuju": "perjalanan",
    "pindah": "perjalanan", "bawa": "perjalanan",
    # Komunikasi
    "bicara": "komunikasi", "katakan": "komunikasi", "sampaikan": "komunikasi",
    "beritahu": "komunikasi", "lapor": "komunikasi", "umumkan": "komunikasi",
    "nyatakan": "komunikasi", "tulis": "komunikasi",
    # Hukum
    "jatuhkan": "hukum", "vonis": "hukum", "dakwa": "hukum",
    "tuntut": "hukum", "tangkap": "hukum", "tahan": "hukum",
    "bebaskan": "hukum", "hukum": "hukum", "putuskan": "hukum", "adili": "hukum",
    # Medis
    "periksa": "medis", "diagnosa": "medis", "obati": "medis",
    "rawat": "medis", "operasi": "medis", "bedah": "medis",
    "suntik": "medis", "terapi": "medis", "vaksinasi": "medis",
    # Pendidikan
    "ajar": "didik", "didik": "didik", "latih": "didik",
    "bimbing": "didik", "belajar": "didik", "lulus": "didik",
    # Pembuatan
    "buat": "buat", "bangun": "buat", "ciptakan": "buat",
    "produksi": "buat", "rancang": "buat", "masak": "buat",
    # Perusakan
    "hancur": "rusak", "rusak": "rusak", "bakar": "rusak",
    "bunuh": "rusak", "serang": "rusak",

    # Bentuk berimbuhan yang tidak bisa di-strip via prefiks biasa (nasalisasi)
    # Hukum
    "menangkap": "hukum", "menahan": "hukum", "mendakwa": "hukum",
    "menuntut": "hukum", "memvonis": "hukum", "mengadili": "hukum",
    "memutuskan": "hukum", "menjatuhkan": "hukum", "membebaskan": "hukum",
    # Medis
    "memeriksa": "medis", "mendiagnosis": "medis", "mengobati": "medis",
    "merawat": "medis", "mengoperasi": "medis", "menyuntik": "medis",
    "meresepkan": "medis", "memvaksinasi": "medis", "merehabilitasi": "medis",
    # Didik
    "mengajar": "didik", "mendidik": "didik", "melatih": "didik",
    "membimbing": "didik", "mempelajari": "didik",
    # Transfer
    "membeli": "transfer", "menjual": "transfer", "memberi": "transfer",
    "menyerahkan": "transfer", "mengirim": "transfer", "membayar": "transfer",
    # Komunikasi
    "berbicara": "komunikasi", "melaporkan": "komunikasi",
    "mengumumkan": "komunikasi", "menyampaikan": "komunikasi",
    "memberitahu": "komunikasi",
    # Perjalanan
    "berangkat": "perjalanan", "berpindah": "perjalanan",
    "membawa": "perjalanan", "menuju": "perjalanan",
    # Pemerintahan (tipe baru — mapped ke "hukum" karena regulatory effect)
    "mengeluarkan": "hukum", "menerbitkan": "hukum", "mengesahkan": "hukum",
    "memberlakukan": "hukum", "mencabut": "hukum", "mengumumkan": "hukum",
    "melantik": "hukum", "memecat": "hukum", "meresmikan": "hukum",
    "lantik": "hukum", "pecat": "hukum", "resmikan": "hukum",

    # Root alternatif yang muncul dari LPS (afiks tidak selalu terstrip penuh)
    # Hukum
    "jatuh": "hukum",         # jatuhkan → jatuh
    "voniskan": "hukum",
    "dakwakan": "hukum",
    "tuntutkan": "hukum",
    "tangkap": "hukum",       # menangkap → tangkap
    "tahan": "hukum",
    "adil": "hukum",
    "bebas": "hukum",         # membebaskan → bebas
    # Medis
    "periksa": "medis",       # memeriksa → periksa
    "diagnos": "medis",
    "obat": "medis",
    "rawat": "medis",
    "bedah": "medis",
    "suntik": "medis",
    "terapi": "medis",
    # Didik
    "ajar": "didik",
    "didik": "didik",
    "latih": "didik",
    "bimbing": "didik",
    "lulus": "didik",
    # Transfer
    "beli": "transfer",
    "jual": "transfer",
    "beri": "transfer",
    "serah": "transfer",      # serahkan → serah
    "kirim": "transfer",
    "bayar": "transfer",
    # Perjalanan
    "pergi": "perjalanan",
    "tiba": "perjalanan",
    "datang": "perjalanan",
    "pindah": "perjalanan",
    "bawa": "perjalanan",
    # Komunikasi
    "bicara": "komunikasi",
    "lapor": "komunikasi",
    "umumkan": "komunikasi",  # bisa muncul utuh
    # Buat
    "buat": "buat",
    "bangun": "buat",
    "ciptakan": "buat",
    "masak": "buat",
}


_PREFIKS_VERBA = (
    "memper", "diper",
    "meng", "meny", "mem", "men", "me",
    "ber", "ter", "per", "di",
)


def _strip_prefiks(verba: str) -> str:
    """Strip prefiks verba Indonesia untuk normalisasi ke root."""
    v = verba.lower()
    for pref in _PREFIKS_VERBA:
        if v.startswith(pref) and len(v) > len(pref) + 2:
            return v[len(pref):]
    return v


def tipe_aksi(verba_root: str) -> Optional[str]:
    """
    Inferensi tipe aksi dari root verba.
    Strategi: (1) lookup langsung, (2) strip prefiks lalu lookup lagi.
    Justifikasi: LPS kadang mengembalikan root berimbuhan seperti 'memeriksa'.
    """
    v = verba_root.lower()
    hit = VERBA_KE_TIPE_AKSI.get(v)
    if hit:
        return hit
    stripped = _strip_prefiks(v)
    if stripped != v:
        return VERBA_KE_TIPE_AKSI.get(stripped)
    return None


# ── Aturan Dunia ──────────────────────────────────────────────────────────────

@dataclass
class AturanDunia:
    """
    Satu aturan kausalitas dunia nyata.
    JIKA kondisi_tipe_aksi + tipe_agen + tipe_pasien MAKA kesimpulan.

    Berbasis Event Semantics (Davidson 1967) dan Frame Semantics (Fillmore 1976).
    """
    nama:             str
    tipe_aksi:        str            # tipe aksi yang memicu aturan
    tipe_agen:        Optional[str]  # tipe agen yang diperlukan (None = semua)
    tipe_pasien:      Optional[str]  # tipe pasien yang diperlukan (None = semua)
    kesimpulan:       List[str]      # pernyataan inferensi yang dihasilkan
    domain:           str
    prioritas:        int = 1


SEMUA_ATURAN: List[AturanDunia] = [

    # ── Universal ────────────────────────────────────────────────────────
    AturanDunia(
        nama="TRANSFER_KEPEMILIKAN",
        tipe_aksi="transfer",
        tipe_agen=TipeEntitas.PERSONA,
        tipe_pasien=None,
        kesimpulan=[
            "penerima MEMILIKI pasien",
            "agen TIDAK_LAGI_MEMILIKI pasien",
        ],
        domain="universal", prioritas=3,
    ),
    AturanDunia(
        nama="PERUBAHAN_LOKASI",
        tipe_aksi="perjalanan",
        tipe_agen=TipeEntitas.PERSONA,
        tipe_pasien=None,
        kesimpulan=[
            "agen BERADA_DI tujuan",
            "agen TIDAK_LAGI_DI asal",
        ],
        domain="universal", prioritas=3,
    ),
    AturanDunia(
        nama="KOMUNIKASI_IMPLISIT",
        tipe_aksi="komunikasi",
        tipe_agen=TipeEntitas.PERSONA,
        tipe_pasien=None,
        kesimpulan=[
            "penerima MENGETAHUI pesan",
        ],
        domain="universal", prioritas=2,
    ),

    # ── Hukum ────────────────────────────────────────────────────────────
    AturanDunia(
        nama="VONIS_MENGUBAH_STATUS",
        tipe_aksi="hukum",
        tipe_agen=TipeEntitas.PROFESI_HUKUM,
        tipe_pasien=TipeEntitas.HUKUM_NORMA,
        kesimpulan=[
            "penerima STATUS_MENJADI terpidana",
            "penerima WAJIB menjalani putusan",
            "agen TELAH_MEMUTUS kasus",
        ],
        domain="hukum", prioritas=3,
    ),
    AturanDunia(
        nama="PENANGKAPAN_MEMBATASI_KEBEBASAN",
        tipe_aksi="hukum",
        tipe_agen=TipeEntitas.PROFESI_HUKUM,
        tipe_pasien=TipeEntitas.WARGA,
        kesimpulan=[
            "pasien KEBEBASAN_DIBATASI",
            "pasien BERADA_DI tahanan",
        ],
        domain="hukum", prioritas=3,
    ),

    # ── Kesehatan ─────────────────────────────────────────────────────────
    AturanDunia(
        nama="DIAGNOSIS_MENETAPKAN_KONDISI",
        tipe_aksi="medis",
        tipe_agen=TipeEntitas.PROFESI_MEDIS,
        tipe_pasien=TipeEntitas.WARGA,
        kesimpulan=[
            "pasien MEMILIKI_KONDISI penyakit",
            "pasien MEMBUTUHKAN pengobatan",
            "agen BERTANGGUNG_JAWAB atas kondisi pasien",
        ],
        domain="kesehatan", prioritas=3,
    ),

    # ── Pendidikan ────────────────────────────────────────────────────────
    AturanDunia(
        nama="MENGAJAR_TRANSFER_PENGETAHUAN",
        tipe_aksi="didik",
        tipe_agen=TipeEntitas.PROFESI_DIDIK,
        tipe_pasien=TipeEntitas.WARGA,
        kesimpulan=[
            "pasien MEMPELAJARI materi",
            "pasien PENGETAHUAN_BERTAMBAH",
        ],
        domain="pendidikan", prioritas=2,
    ),

    # ── Pembuatan ─────────────────────────────────────────────────────────
    AturanDunia(
        nama="PEMBUATAN_MENGHASILKAN_ENTITAS",
        tipe_aksi="buat",
        tipe_agen=None,
        tipe_pasien=None,
        kesimpulan=[
            "hasil ADA setelah aksi",
            "agen MEMILIKI hasil",
        ],
        domain="universal", prioritas=2,
    ),

    # ── Perusakan ─────────────────────────────────────────────────────────
    AturanDunia(
        nama="PERUSAKAN_MENGUBAH_STATE",
        tipe_aksi="rusak",
        tipe_agen=None,
        tipe_pasien=None,
        kesimpulan=[
            "pasien KONDISI_MENJADI rusak_atau_tidak_ada",
        ],
        domain="universal", prioritas=3,
    ),
]


class KnowledgeBase:
    """
    Knowledge Base Indonesia — akses terpusat ke semua pengetahuan eksplisit.

    Dipakai oleh InferenceEngine untuk reasoning dari proposisi KRL.
    """

    def __init__(self) -> None:
        self._aturan = SEMUA_ATURAN
        self._kata_ke_tipe = KATA_KE_TIPE
        self._verba_ke_aksi = VERBA_KE_TIPE_AKSI

    def tipe_entitas(self, kata: str) -> str:
        return tipe_entitas(kata)

    def tipe_aksi(self, verba_root: str) -> Optional[str]:
        return tipe_aksi(verba_root)

    def aturan_untuk_aksi(self, tipe_aksi_: str) -> List[AturanDunia]:
        """Ambil semua aturan yang relevan untuk tipe aksi tertentu."""
        return [a for a in self._aturan if a.tipe_aksi == tipe_aksi_]

    def adalah_subtipe(self, tipe: str, supertipe: str) -> bool:
        return adalah_subtipe(tipe, supertipe)

    def tambah_kata(self, kata: str, tipe: str) -> None:
        """Tambah kata baru ke KB tanpa retrain."""
        self._kata_ke_tipe[kata.lower()] = tipe

    def tambah_aturan(self, aturan: AturanDunia) -> None:
        """Tambah aturan baru ke KB tanpa retrain."""
        self._aturan.append(aturan)

    @property
    def n_aturan(self) -> int:
        return len(self._aturan)

    @property
    def n_entitas(self) -> int:
        return len(self._kata_ke_tipe)
