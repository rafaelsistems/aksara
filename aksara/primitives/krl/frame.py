"""
frame.py — FrameBank: katalog frame semantik bahasa Indonesia.

OPOSISI TRANSFORMER:
  Transformer: tidak punya konsep frame — semua implisit di distribusi token
  KRL FrameBank: setiap situasi punya template slot yang eksplisit dan verifiable

Frame adalah "template situasi" — mendefinisikan:
  - Slot wajib: harus ada agar situasi "lengkap" secara semantik
  - Slot opsional: memperkaya pemahaman tapi tidak wajib
  - Verba pemicu: verba-verba yang mengaktifkan frame ini
  - Entitas yang kompatibel: tipe entitas yang bisa mengisi setiap slot

Contoh Frame JUAL_BELI:
  Slot wajib:   pembeli, barang
  Slot opsional: penjual, harga, lokasi, waktu
  Pemicu:       beli, jual, dagang, bayar, barter, transaksi
  Pembeli:      [PERSONA, ORGANISASI]
  Barang:       [BENDA, JASA]

Justifikasi linguistik:
  Berbasis FrameNet (Fillmore 1976) yang diadaptasi untuk bahasa Indonesia.
  Setiap frame mencerminkan skenario kognitif yang universal dalam budaya Indonesia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class SlotFrame:
    """Definisi satu slot dalam frame."""
    nama:         str
    wajib:        bool
    tipe_entitas: Set[str]          # tipe entitas yang valid: PERSONA, BENDA, LOKASI, dll.
    deskripsi:    str = ""


@dataclass
class Frame:
    """
    Satu frame semantik — template situasi dalam bahasa Indonesia.

    Frame merepresentasikan pemahaman tentang "jenis situasi" —
    bukan hanya string kata, tapi skema kognitif yang bisa di-reasoning.
    """
    nama:         str                       # nama frame: "JUAL_BELI", "PERJALANAN", dll.
    deskripsi:    str
    verba_pemicu: Set[str]                  # root verba yang mengaktifkan frame ini
    slot:         Dict[str, SlotFrame]      # nama_slot → SlotFrame
    domain_utama: str                       # domain semantik utama frame ini
    sub_frame:    List[str] = field(default_factory=list)  # frame yang bisa menjadi bagian ini

    @property
    def slot_wajib(self) -> List[str]:
        return [n for n, s in self.slot.items() if s.wajib]

    @property
    def slot_opsional(self) -> List[str]:
        return [n for n, s in self.slot.items() if not s.wajib]

    def kelengkapan_proposisi(self, slot_terisi: Set[str]) -> float:
        """
        Seberapa lengkap proposisi mengisi slot frame ini [0,1].
        Berbasis slot wajib yang terpenuhi.
        """
        if not self.slot_wajib:
            return 1.0
        n_terpenuhi = sum(1 for s in self.slot_wajib if s in slot_terisi)
        return n_terpenuhi / len(self.slot_wajib)

    def __repr__(self) -> str:
        return f"Frame({self.nama}, wajib={self.slot_wajib})"


# ── FrameBank: 12 Frame Inti Bahasa Indonesia ─────────────────────────────────

def _buat_frame_bank() -> Dict[str, Frame]:
    """
    Bangun katalog 12 frame inti bahasa Indonesia secara manual.

    Frame dipilih berdasarkan:
    1. Frekuensi tinggi dalam teks Indonesia umum
    2. Relevansi lintas domain (hukum, kesehatan, pendidikan, dll.)
    3. Kemampuan representasi yang kaya untuk reasoning

    Referensi: FrameNet (Fillmore), KBBI domain ontology, tata bahasa TBBBI.
    """
    return {

        # ── F1: JUAL_BELI ──────────────────────────────────────────────────
        "JUAL_BELI": Frame(
            nama="JUAL_BELI",
            deskripsi="Transaksi pertukaran barang/jasa dengan nilai ekonomi",
            verba_pemicu={
                "beli", "jual", "dagang", "bayar", "barter", "transaksi",
                "sewa", "lelang", "tenda", "niaga", "perdagangkan",
            },
            slot={
                "pembeli":  SlotFrame("pembeli",  wajib=True,
                                      tipe_entitas={"PERSONA", "ORGANISASI"},
                                      deskripsi="pihak yang membeli"),
                "barang":   SlotFrame("barang",   wajib=True,
                                      tipe_entitas={"BENDA", "JASA", "PROPERTI"},
                                      deskripsi="objek yang diperjualbelikan"),
                "penjual":  SlotFrame("penjual",  wajib=False,
                                      tipe_entitas={"PERSONA", "ORGANISASI"},
                                      deskripsi="pihak yang menjual"),
                "harga":    SlotFrame("harga",    wajib=False,
                                      tipe_entitas={"NUMERALIA", "MATA_UANG"},
                                      deskripsi="nilai tukar"),
                "lokasi":   SlotFrame("lokasi",   wajib=False,
                                      tipe_entitas={"LOKASI", "TEMPAT"},
                                      deskripsi="tempat transaksi"),
                "waktu":    SlotFrame("waktu",    wajib=False,
                                      tipe_entitas={"WAKTU"},
                                      deskripsi="waktu transaksi"),
            },
            domain_utama="ekonomi",
        ),

        # ── F2: PERJALANAN ────────────────────────────────────────────────
        "PERJALANAN": Frame(
            nama="PERJALANAN",
            deskripsi="Perpindahan entitas dari satu lokasi ke lokasi lain",
            verba_pemicu={
                "pergi", "berangkat", "tiba", "datang", "pulang", "kembali",
                "lewat", "melintasi", "melewati", "menuju", "menempuh",
                "berlayar", "terbang", "berkendara", "berjalan",
                "bawa",  # root dari 'membawa'
            },
            slot={
                "pelaku":   SlotFrame("pelaku",  wajib=True,
                                      tipe_entitas={"PERSONA", "KENDARAAN", "ORGANISASI"},
                                      deskripsi="entitas yang berpindah"),
                "tujuan":   SlotFrame("tujuan",  wajib=True,
                                      tipe_entitas={"LOKASI", "TEMPAT"},
                                      deskripsi="lokasi tujuan"),
                "asal":     SlotFrame("asal",    wajib=False,
                                      tipe_entitas={"LOKASI", "TEMPAT"},
                                      deskripsi="lokasi asal"),
                "moda":     SlotFrame("moda",    wajib=False,
                                      tipe_entitas={"KENDARAAN", "CARA"},
                                      deskripsi="alat/cara perjalanan"),
                "waktu":    SlotFrame("waktu",   wajib=False,
                                      tipe_entitas={"WAKTU"},
                                      deskripsi="waktu perjalanan"),
                "tujuan_final": SlotFrame("tujuan_final", wajib=False,
                                          tipe_entitas={"TUJUAN", "AKTIVITAS"},
                                          deskripsi="maksud/tujuan perjalanan"),
            },
            domain_utama="aktivitas",
        ),

        # ── F3: KOMUNIKASI ────────────────────────────────────────────────
        "KOMUNIKASI": Frame(
            nama="KOMUNIKASI",
            deskripsi="Transfer informasi dari pengirim ke penerima",
            verba_pemicu={
                "bicara", "katakan", "beritahu", "sampaikan", "lapor",
                "ceritakan", "jelaskan", "tanya", "minta", "perintah",
                "surat", "kirim", "tulis", "baca", "umumkan", "nyatakan",
            },
            slot={
                "pengirim":  SlotFrame("pengirim",  wajib=True,
                                       tipe_entitas={"PERSONA", "ORGANISASI"},
                                       deskripsi="pihak yang menyampaikan"),
                "pesan":     SlotFrame("pesan",     wajib=True,
                                       tipe_entitas={"INFORMASI", "PROPOSISI"},
                                       deskripsi="isi komunikasi"),
                "penerima":  SlotFrame("penerima",  wajib=False,
                                       tipe_entitas={"PERSONA", "ORGANISASI"},
                                       deskripsi="pihak yang menerima"),
                "media":     SlotFrame("media",     wajib=False,
                                       tipe_entitas={"BENDA", "PLATFORM"},
                                       deskripsi="media komunikasi"),
                "waktu":     SlotFrame("waktu",     wajib=False,
                                       tipe_entitas={"WAKTU"},
                                       deskripsi="waktu komunikasi"),
            },
            domain_utama="aktivitas",
        ),

        # ── F4: PENDIDIKAN ────────────────────────────────────────────────
        "PENDIDIKAN": Frame(
            nama="PENDIDIKAN",
            deskripsi="Transfer pengetahuan/keterampilan dari pengajar ke pelajar",
            verba_pemicu={
                "ajar", "belajar", "didik", "latih", "bimbing", "mentor",
                "kursus", "sekolah", "kuliah", "studi", "ujian", "nilai",
                "lulus", "wisuda", "dosen", "guru",
                "pelajar", "pelajari",  # bentuk lain
            },
            slot={
                "pengajar":  SlotFrame("pengajar",  wajib=False,
                                       tipe_entitas={"PERSONA", "INSTITUSI"},
                                       deskripsi="pihak yang mengajar"),
                "pelajar":   SlotFrame("pelajar",   wajib=True,
                                       tipe_entitas={"PERSONA"},
                                       deskripsi="pihak yang belajar"),
                "materi":    SlotFrame("materi",    wajib=True,
                                       tipe_entitas={"PENGETAHUAN", "KETERAMPILAN"},
                                       deskripsi="isi pembelajaran"),
                "tempat":    SlotFrame("tempat",    wajib=False,
                                       tipe_entitas={"LOKASI", "INSTITUSI"},
                                       deskripsi="tempat pembelajaran"),
                "waktu":     SlotFrame("waktu",     wajib=False,
                                       tipe_entitas={"WAKTU"},
                                       deskripsi="durasi/waktu pembelajaran"),
            },
            domain_utama="pendidikan",
        ),

        # ── F5: KESEHATAN ─────────────────────────────────────────────────
        "KESEHATAN": Frame(
            nama="KESEHATAN",
            deskripsi="Aksi medis: diagnosis, pengobatan, perawatan",
            verba_pemicu={
                "rawat", "obat", "periksa", "diagnosa", "operasi",
                "sembuh", "sakit", "resep", "konsultasi", "vaksin",
                "imunisasi", "suntik", "infus", "bedah", "terapi",
                "rehabilitasi", "luka", "patah", "demam",
                # bentuk berimbuhan (LPS tidak dekomposisi kata serapan)
                "meresepkan", "memeriksa", "mengoperasi", "mendiagnosis",
                "merawat", "mengobati", "memvaksinasi", "merehabilitasi",
            },
            slot={
                "tenaga_medis": SlotFrame("tenaga_medis", wajib=False,
                                          tipe_entitas={"PERSONA", "PROFESI"},
                                          deskripsi="dokter/perawat/tenaga medis"),
                "pasien":       SlotFrame("pasien",  wajib=True,
                                          tipe_entitas={"PERSONA"},
                                          deskripsi="pihak yang dirawat"),
                "kondisi":      SlotFrame("kondisi", wajib=False,
                                          tipe_entitas={"PENYAKIT", "KONDISI"},
                                          deskripsi="kondisi medis"),
                "tindakan":     SlotFrame("tindakan", wajib=True,
                                          tipe_entitas={"PROSEDUR", "OBAT"},
                                          deskripsi="tindakan medis"),
                "tempat":       SlotFrame("tempat",  wajib=False,
                                          tipe_entitas={"LOKASI", "FASILITAS"},
                                          deskripsi="fasilitas kesehatan"),
            },
            domain_utama="kesehatan",
        ),

        # ── F6: HUKUM_PIDANA ─────────────────────────────────────────────
        "HUKUM_PIDANA": Frame(
            nama="HUKUM_PIDANA",
            deskripsi="Proses hukum pidana: tuntutan, persidangan, putusan",
            verba_pemicu={
                "dakwa", "tuntut", "adili", "vonis", "hukum", "tangkap",
                "tahan", "sidang", "jaksa", "hakim", "gugat", "banding",
                "kasasi", "eksekusi", "bebaskan", "rehabilitasi",
                "jatuhkan", "putuskan", "tetapkan", "ancam",
                "jatuh",    # root dari 'menjatuhkan'
            },
            slot={
                "terdakwa":   SlotFrame("terdakwa",  wajib=True,
                                        tipe_entitas={"PERSONA", "ORGANISASI"},
                                        deskripsi="pihak yang dituntut"),
                "penuntut":   SlotFrame("penuntut",  wajib=False,
                                        tipe_entitas={"PERSONA", "INSTITUSI"},
                                        deskripsi="jaksa/penuntut umum"),
                "hakim":      SlotFrame("hakim",     wajib=False,
                                        tipe_entitas={"PERSONA"},
                                        deskripsi="hakim yang memutus"),
                "perbuatan":  SlotFrame("perbuatan", wajib=True,
                                        tipe_entitas={"TINDAK_PIDANA"},
                                        deskripsi="perbuatan yang didakwakan"),
                "pasal":      SlotFrame("pasal",     wajib=False,
                                        tipe_entitas={"REGULASI"},
                                        deskripsi="pasal yang dilanggar"),
                "putusan":    SlotFrame("putusan",   wajib=False,
                                        tipe_entitas={"HUKUMAN", "VONIS"},
                                        deskripsi="putusan pengadilan"),
            },
            domain_utama="hukum",
        ),

        # ── F7: PEMBUATAN ─────────────────────────────────────────────────
        "PEMBUATAN": Frame(
            nama="PEMBUATAN",
            deskripsi="Proses menciptakan, membangun, atau memproduksi sesuatu",
            verba_pemicu={
                "buat", "bangun", "ciptakan", "produksi", "rancang", "desain",
                "dirikan", "konstruksi", "fabrikasi", "rakit", "tulis",
                "lukis", "pahat", "jahit", "masak",
            },
            slot={
                "pembuat":  SlotFrame("pembuat",  wajib=True,
                                      tipe_entitas={"PERSONA", "ORGANISASI", "MESIN"},
                                      deskripsi="entitas yang membuat"),
                "hasil":    SlotFrame("hasil",    wajib=True,
                                      tipe_entitas={"BENDA", "KARYA", "BANGUNAN"},
                                      deskripsi="objek yang dibuat"),
                "bahan":    SlotFrame("bahan",    wajib=False,
                                      tipe_entitas={"BAHAN", "MATERIAL"},
                                      deskripsi="bahan yang digunakan"),
                "cara":     SlotFrame("cara",     wajib=False,
                                      tipe_entitas={"METODE", "ALAT"},
                                      deskripsi="cara/metode pembuatan"),
                "tujuan":   SlotFrame("tujuan",   wajib=False,
                                      tipe_entitas={"TUJUAN", "FUNGSI"},
                                      deskripsi="tujuan pembuatan"),
            },
            domain_utama="aktivitas",
        ),

        # ── F8: KEPEMILIKAN ───────────────────────────────────────────────
        "KEPEMILIKAN": Frame(
            nama="KEPEMILIKAN",
            deskripsi="Relasi kepemilikan atau penguasaan atas entitas",
            verba_pemicu={
                "miliki", "punya", "dapat", "terima", "kehilangan",
                "mewarisi", "menyerahkan", "menghibahkan", "mengalihkan",
            },
            slot={
                "pemilik":  SlotFrame("pemilik",  wajib=True,
                                      tipe_entitas={"PERSONA", "ORGANISASI"},
                                      deskripsi="pihak yang memiliki"),
                "objek":    SlotFrame("objek",    wajib=True,
                                      tipe_entitas={"BENDA", "PROPERTI", "HAK"},
                                      deskripsi="yang dimiliki"),
                "sumber":   SlotFrame("sumber",   wajib=False,
                                      tipe_entitas={"PERSONA", "ORGANISASI"},
                                      deskripsi="asal kepemilikan"),
                "waktu":    SlotFrame("waktu",    wajib=False,
                                      tipe_entitas={"WAKTU"},
                                      deskripsi="kapan kepemilikan dimulai"),
            },
            domain_utama="ekonomi",
        ),

        # ── F9: ATRIBUSI ──────────────────────────────────────────────────
        "ATRIBUSI": Frame(
            nama="ATRIBUSI",
            deskripsi="Penugasan atribut/sifat kepada entitas (kalimat predikatif)",
            verba_pemicu={
                "adalah", "ialah", "merupakan", "menjadi", "tampak",
                "terlihat", "terasa", "terdengar",
            },
            slot={
                "entitas":   SlotFrame("entitas",  wajib=True,
                                       tipe_entitas={"SEMUA"},
                                       deskripsi="entitas yang diberi atribut"),
                "atribut":   SlotFrame("atribut",  wajib=True,
                                       tipe_entitas={"SIFAT", "KATEGORI", "PERAN"},
                                       deskripsi="sifat/kategori yang diatribusikan"),
                "kondisi":   SlotFrame("kondisi",  wajib=False,
                                       tipe_entitas={"WAKTU", "SITUASI"},
                                       deskripsi="kondisi/konteks atribusi"),
            },
            domain_utama="deskriptif",
        ),

        # ── F10: KONFLIK ──────────────────────────────────────────────────
        "KONFLIK": Frame(
            nama="KONFLIK",
            deskripsi="Situasi pertentangan, pertikaian, atau kompetisi",
            verba_pemicu={
                "lawan", "perang", "bertarung", "bersaing", "berkompetisi",
                "menyerang", "mempertahankan", "menentang", "melawan",
                "protes", "demonstrasi", "mogok",
            },
            slot={
                "pihak_1":  SlotFrame("pihak_1",  wajib=True,
                                      tipe_entitas={"PERSONA", "ORGANISASI", "NEGARA"},
                                      deskripsi="pihak pertama dalam konflik"),
                "pihak_2":  SlotFrame("pihak_2",  wajib=True,
                                      tipe_entitas={"PERSONA", "ORGANISASI", "NEGARA"},
                                      deskripsi="pihak kedua dalam konflik"),
                "sebab":    SlotFrame("sebab",    wajib=False,
                                      tipe_entitas={"INFORMASI", "BENDA", "HAK"},
                                      deskripsi="penyebab konflik"),
                "lokasi":   SlotFrame("lokasi",   wajib=False,
                                      tipe_entitas={"LOKASI"},
                                      deskripsi="lokasi konflik"),
                "hasil":    SlotFrame("hasil",    wajib=False,
                                      tipe_entitas={"KONDISI", "KEPUTUSAN"},
                                      deskripsi="hasil/resolusi konflik"),
            },
            domain_utama="sosial",
        ),

        # ── F11: PEMERINTAHAN ─────────────────────────────────────────────
        "PEMERINTAHAN": Frame(
            nama="PEMERINTAHAN",
            deskripsi="Aksi pemerintah: regulasi, kebijakan, pembangunan",
            verba_pemicu={
                "atur", "terapkan", "keluarkan", "terbitkan", "sahkan",
                "cabut", "amandemen", "bangun", "anggarkan", "alokasikan",
                "resmikan", "lantik", "copot", "pecat",
                "tetapkan", "putuskan", "berlakukan", "implementasikan",
                "umumkan", "luncurkan", "reformasi",
                "mengeluarkan", "menerbitkan",  # teks_asli dari LPS
            },
            slot={
                "otoritas":  SlotFrame("otoritas",  wajib=True,
                                       tipe_entitas={"INSTITUSI", "PEJABAT"},
                                       deskripsi="lembaga/pejabat yang bertindak"),
                "kebijakan": SlotFrame("kebijakan", wajib=True,
                                       tipe_entitas={"REGULASI", "PROGRAM", "KEPUTUSAN"},
                                       deskripsi="kebijakan yang diambil"),
                "sasaran":   SlotFrame("sasaran",   wajib=False,
                                       tipe_entitas={"PERSONA", "KELOMPOK", "WILAYAH"},
                                       deskripsi="pihak yang terdampak"),
                "tujuan":    SlotFrame("tujuan",    wajib=False,
                                       tipe_entitas={"TUJUAN"},
                                       deskripsi="tujuan kebijakan"),
                "waktu":     SlotFrame("waktu",     wajib=False,
                                       tipe_entitas={"WAKTU"},
                                       deskripsi="waktu berlaku"),
            },
            domain_utama="hukum",
        ),

        # ── F12: EKSISTENSI ───────────────────────────────────────────────
        "EKSISTENSI": Frame(
            nama="EKSISTENSI",
            deskripsi="Keberadaan atau ketiadaan entitas",
            verba_pemicu={
                "ada", "terdapat", "hadir", "muncul", "terletak",
                "berlokasi", "berada", "tinggal", "berdomisili",
            },
            slot={
                "entitas":  SlotFrame("entitas",  wajib=True,
                                      tipe_entitas={"SEMUA"},
                                      deskripsi="entitas yang ada/tidak ada"),
                "lokasi":   SlotFrame("lokasi",   wajib=False,
                                      tipe_entitas={"LOKASI", "TEMPAT"},
                                      deskripsi="di mana entitas berada"),
                "waktu":    SlotFrame("waktu",    wajib=False,
                                      tipe_entitas={"WAKTU"},
                                      deskripsi="kapan entitas ada"),
                "kondisi":  SlotFrame("kondisi",  wajib=False,
                                      tipe_entitas={"KONDISI"},
                                      deskripsi="kondisi keberadaan"),
            },
            domain_utama="deskriptif",
        ),
    }


class FrameBank:
    """
    Katalog frame semantik bahasa Indonesia.

    Menyediakan:
    1. Lookup frame berdasarkan nama
    2. Lookup frame berdasarkan verba pemicu
    3. Daftar semua frame yang tersedia
    """

    def __init__(self) -> None:
        self._frame: Dict[str, Frame] = _buat_frame_bank()
        # Index: verba_root → nama_frame
        self._verba_index: Dict[str, List[str]] = {}
        for nama, frame in self._frame.items():
            for verba in frame.verba_pemicu:
                if verba not in self._verba_index:
                    self._verba_index[verba] = []
                self._verba_index[verba].append(nama)

    def cari_nama(self, nama: str) -> Optional[Frame]:
        """Cari frame berdasarkan nama eksak."""
        return self._frame.get(nama.upper())

    def cari_verba(self, verba_root: str) -> List[Frame]:
        """Cari semua frame yang dipicu verba ini."""
        nama_list = self._verba_index.get(verba_root.lower(), [])
        return [self._frame[n] for n in nama_list if n in self._frame]

    def semua_frame(self) -> List[Frame]:
        return list(self._frame.values())

    @property
    def n_frame(self) -> int:
        return len(self._frame)

    @property
    def n_verba_pemicu(self) -> int:
        return len(self._verba_index)

    def __repr__(self) -> str:
        return f"FrameBank({self.n_frame} frame, {self.n_verba_pemicu} verba pemicu)"
