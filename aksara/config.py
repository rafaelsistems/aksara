"""
AksaraConfig — konfigurasi framework yang bisa diinjeksi per use-case.

PRINSIP DESAIN:
  Layer 1 (universal) — hardcode di primitif, tidak bisa diubah:
    - Aturan afiksasi TBBBI (di-, me-, ber-, dll.)
    - Kompatibilitas POS sebagai modifier
    - Deteksi kalimat menggantung

  Layer 2 (domain-spesifik) — dikonfigurasi via AksaraConfig:
    - Verba domain-neutral (berbeda per domain: "mengancam" valid di hukum)
    - Jarak domain semantik (hukum↔senjata lebih dekat di domain militer)
    - Pasangan verb-objek yang tidak kompatibel
    - Leksikon tambahan

Contoh penggunaan:

  # Bahasa Indonesia umum (default)
  fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")

  # Domain hukum
  from aksara.config import AksaraConfig
  config = AksaraConfig.untuk_domain("hukum")
  fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json", config=config)

  # Domain custom
  config = AksaraConfig(
      verba_domain_neutral_tambahan={"ancam", "gugat", "dakwa"},
      domain_distance_override={"hukum↔senjata": 1.2},
      leksikon_path_tambahan="kbbi_hukum.json",
  )
  fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json", config=config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Set


# ── Layer 2: Default untuk bahasa Indonesia umum ─────────────────────────────
# Verba yang secara semantik bisa berelasi dengan objek dari domain apapun.
# Ini adalah properti leksikal bahasa Indonesia umum — bukan universal.
# Domain khusus (hukum, militer, dll.) bisa menambah atau mengganti daftar ini.

VERBA_DOMAIN_NEUTRAL_DEFAULT: Set[str] = {
    # Transaksional
    "beli", "jual", "dagang", "niaga", "bayar", "sewa",
    "pinjam", "pinjamkan", "beri", "berikan", "hibah", "hadiahkan",
    # Kepemilikan/eksistensi
    "miliki", "punya", "simpan", "pegang", "dapat", "terima",
    "ambil", "bawa", "taruh", "letakkan", "tempatkan",
    # Transfer fisik
    "kirim", "antar", "angkut", "muat", "pindahkan",
    # Kognitif/umum
    "lihat", "tahu", "kenal", "ingat", "cari", "temukan", "pelajari",
    # Produksi/kreasi
    "buat", "bangun", "produksi", "ciptakan", "hasilkan", "kembangkan",
    # Penggunaan
    "pakai", "gunakan", "manfaatkan", "butuhkan", "perlukan",
}

# Pasangan domain verba → domain objek yang TIDAK kompatibel (default umum)
# TIDAK termasuk "ekonomi" karena verba ekonomi (beli/jual) adalah domain-neutral
VERBA_OBJEK_INCOMPATIBLE_DEFAULT: Dict[str, Set[str]] = {
    "kuliner":    {"senjata", "kendaraan", "alat_musik"},
    "pendidikan": {"senjata", "kendaraan"},
    "kesehatan":  {"senjata", "alat_musik"},
    "senjata":    {"kuliner", "pendidikan", "kesehatan"},
}


@dataclass
class AksaraConfig:
    """
    Konfigurasi Layer 2 AKSARA — domain-spesifik, bisa diinjeksi per use-case.

    Semua field opsional. Default = perilaku bahasa Indonesia umum.
    """

    # ── Verba domain-neutral ─────────────────────────────────────────────────
    # Verba tambahan yang dianggap domain-neutral di domain ini.
    # Digabung (union) dengan VERBA_DOMAIN_NEUTRAL_DEFAULT.
    # Contoh domain hukum: {"ancam", "gugat", "dakwa", "adili", "vonis"}
    verba_domain_neutral_tambahan: Set[str] = field(default_factory=set)

    # Set penuh verba domain-neutral (menggantikan default sepenuhnya).
    # Gunakan ini jika domain sangat khusus dan default tidak relevan.
    # None = gunakan default + tambahan
    verba_domain_neutral_override: Optional[Set[str]] = None

    # ── Verb-objek inkompatibilitas ──────────────────────────────────────────
    # Override pasangan domain yang tidak kompatibel.
    # None = gunakan default
    verba_objek_incompatible_override: Optional[Dict[str, Set[str]]] = None

    # Pasangan tambahan yang inkompatibel di domain ini.
    # Digabung dengan default.
    verba_objek_incompatible_tambahan: Dict[str, Set[str]] = field(
        default_factory=dict
    )

    # ── Domain distance override ─────────────────────────────────────────────
    # Override jarak geodesic antara pasangan domain tertentu.
    # Format key: "domain_a↔domain_b" (urutan tidak penting)
    # Contoh: {"hukum↔senjata": 1.2} — lebih dekat dari default (2.8)
    domain_distance_override: Dict[str, float] = field(default_factory=dict)

    # ── Threshold semantik ───────────────────────────────────────────────────
    # Jarak geodesic maksimal sebelum dianggap domain mismatch.
    # Default: 1.5 (bahasa Indonesia umum)
    # Domain militer/keamanan: bisa dinaikkan ke 2.0
    threshold_semantik: float = 1.5

    # ── Leksikon tambahan ────────────────────────────────────────────────────
    # Path ke file JSON leksikon tambahan (format sama dengan kbbi_core_v2.json)
    # Contoh: "data/kbbi_hukum.json", "data/kbbi_medis.json"
    leksikon_path_tambahan: Optional[str] = None

    # ── Nama domain ini (untuk logging/debugging) ────────────────────────────
    nama_domain: str = "umum"

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def verba_domain_neutral(self) -> Set[str]:
        """Set lengkap verba domain-neutral yang aktif."""
        if self.verba_domain_neutral_override is not None:
            return self.verba_domain_neutral_override
        return VERBA_DOMAIN_NEUTRAL_DEFAULT | self.verba_domain_neutral_tambahan

    @property
    def verba_objek_incompatible(self) -> Dict[str, Set[str]]:
        """Dict lengkap inkompatibilitas verb-objek yang aktif."""
        if self.verba_objek_incompatible_override is not None:
            base = dict(self.verba_objek_incompatible_override)
        else:
            base = {k: set(v) for k, v in VERBA_OBJEK_INCOMPATIBLE_DEFAULT.items()}

        # Gabungkan dengan tambahan
        for domain, incompatible in self.verba_objek_incompatible_tambahan.items():
            if domain in base:
                base[domain] = base[domain] | incompatible
            else:
                base[domain] = set(incompatible)
        return base

    # ── Factory methods untuk domain umum ────────────────────────────────────

    @classmethod
    def default(cls) -> "AksaraConfig":
        """Konfigurasi default — bahasa Indonesia umum."""
        return cls(nama_domain="umum")

    @classmethod
    def untuk_domain(cls, domain: str) -> "AksaraConfig":
        """
        Factory method untuk domain yang sudah dikenal.

        Domain yang didukung:
          "hukum"      — teks hukum, perundang-undangan, putusan pengadilan
          "kesehatan"  — teks medis, catatan klinis, jurnal kesehatan
          "militer"    — teks kemiliteran, keamanan, pertahanan
          "pertanahan" — teks agraria, pertanahan, properti
          "pendidikan" — teks akademik, kurikulum, penelitian
          "bisnis"     — teks bisnis, keuangan, perbankan
          "umum"       — bahasa Indonesia umum (default)
        """
        domain_lower = domain.lower()

        if domain_lower == "hukum":
            return cls(
                nama_domain="hukum",
                verba_domain_neutral_tambahan={
                    "ancam", "gugat", "dakwa", "adili", "vonis", "hukum",
                    "tuntut", "tuduh", "sidang", "periksa", "tangkap",
                    "tahan", "bebaskan", "rehabilitasi", "eksekusi",
                },
                domain_distance_override={
                    "hukum↔senjata": 1.5,      # lebih dekat di konteks hukum pidana
                    "hukum↔penyakit": 1.8,      # relevan di hukum kesehatan
                },
                threshold_semantik=1.8,         # lebih toleran di teks hukum
            )

        if domain_lower in ("kesehatan", "medis"):
            return cls(
                nama_domain="kesehatan",
                verba_domain_neutral_tambahan={
                    "infus", "operasi", "diagnosis", "rawat", "obati",
                    "periksa", "injeksi", "vaksinasi", "sterilisasi",
                    "amputasi", "transplantasi", "resep", "dosis",
                },
                domain_distance_override={
                    "kesehatan↔penyakit": 0.3,  # sangat dekat
                    "kesehatan↔kimia": 0.8,
                },
            )

        if domain_lower in ("militer", "pertahanan", "keamanan"):
            return cls(
                nama_domain="militer",
                verba_domain_neutral_tambahan={
                    "operasi", "misi", "serang", "pertahan", "lindungi",
                    "patroli", "kawal", "evakuasi", "infiltrasi",
                },
                domain_distance_override={
                    "senjata\u2194kendaraan": 0.5,   # kendaraan militer
                    "senjata\u2194ekonomi": 1.5,      # logistik militer
                },
                threshold_semantik=2.0,
                # Militer tetap tidak valid: memasak meriam, memasak granat
                # Hanya relasi senjata↔kendaraan dan senjata↔pendidikan yang dilonggarkan
                verba_objek_incompatible_override={
                    "kuliner": {"senjata", "alat_musik"},  # memasak senjata tetap absurd
                    "kesehatan": {"alat_musik"},
                },
            )

        if domain_lower in ("pertanahan", "agraria", "properti"):
            return cls(
                nama_domain="pertanahan",
                verba_domain_neutral_tambahan={
                    "sertifikat", "ukur", "peta", "kavling", "hak",
                    "sengketa", "ganti rugi", "pembebasan", "konsolidasi",
                },
                threshold_semantik=1.8,
            )

        if domain_lower in ("pendidikan", "akademik"):
            return cls(
                nama_domain="pendidikan",
                verba_domain_neutral_tambahan={
                    "ajar", "latih", "didik", "bimbing", "mentor",
                    "evaluasi", "assesmen", "ujian", "nilai",
                },
                # Di domain pendidikan, "mengajarkan senjata" mungkin valid
                # (mis. sejarah perang, pelajaran bela diri)
                verba_objek_incompatible_override={
                    "kuliner": {"kendaraan", "alat_musik"},
                    "kesehatan": {"alat_musik"},
                },
            )

        if domain_lower in ("bisnis", "keuangan", "perbankan"):
            return cls(
                nama_domain="bisnis",
                verba_domain_neutral_tambahan={
                    "investasi", "modal", "profit", "dividen", "akuisisi",
                    "merger", "likuidasi", "portfolio", "aset", "liabilitas",
                },
                threshold_semantik=2.0,  # bisnis sering cross-domain
            )

        # Default untuk domain tidak dikenal
        return cls(nama_domain=domain_lower)

    def __repr__(self) -> str:
        n_neutral = len(self.verba_domain_neutral)
        n_incomp  = sum(len(v) for v in self.verba_objek_incompatible.values())
        return (
            f"AksaraConfig(domain='{self.nama_domain}', "
            f"n_verba_neutral={n_neutral}, "
            f"n_incompatible={n_incomp}, "
            f"threshold={self.threshold_semantik})"
        )
