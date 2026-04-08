"""
AfiksRules — aturan afiks bahasa Indonesia yang dikodekan secara deterministik.

OPOSISI TRANSFORMER:
  Transformer: belajar validitas afiks dari statistik korpus (implisit, bisa salah)
  AKSARA LPS:  validitas afiks dikodekan sebagai finite state rules (deterministik, hard)

Sumber: Tata Bahasa Baku Bahasa Indonesia (TBBBI) Edisi Keempat.
Ini bukan learned parameter — ini hukum linguistik yang tidak bisa dilanggar data.
"""

from __future__ import annotations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from aksara.primitives.lps.morfem import KelasKata, TipeAfiks, AfiksAktif


# ── Definisi Afiks Bahasa Indonesia ──────────────────────────────────────────

PREFIKS_VALID: Dict[str, Dict] = {
    "me-": {
        "fungsi": "verba_aktif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
        "alomorf": ["me-", "mem-", "men-", "meng-", "meny-", "menge-"],
    },
    "di-": {
        "fungsi": "verba_pasif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.VERBA,
        "alomorf": ["di-"],
        "catatan": "juga valid untuk kata serapan: di-ghosting, di-cancel",
    },
    "ber-": {
        "fungsi": "verba_intransitif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
        "alomorf": ["ber-", "be-", "bel-"],
    },
    "ter-": {
        "fungsi": "verba_pasif_spontan",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
        "alomorf": ["ter-", "te-"],
    },
    "per-": {
        "fungsi": "kausatif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
        "alomorf": ["per-", "pel-"],
    },
    "ke-": {
        "fungsi": "numeralia_ordinal",
        "kelas_input_valid": {KelasKata.NUMERALIA},
        "kelas_output": KelasKata.NUMERALIA,
        "alomorf": ["ke-"],
    },
    "se-": {
        "fungsi": "kuantifikasi",
        "kelas_input_valid": {KelasKata.NOMINA, KelasKata.ADJEKTIVA, KelasKata.NUMERALIA},
        "kelas_output": KelasKata.ADJEKTIVA,
        "alomorf": ["se-"],
    },
    "pe-": {
        "fungsi": "nominalisasi_agen",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.NOMINA,
        "alomorf": ["pe-", "pem-", "pen-", "peng-", "peny-", "penge-"],
    },
}

SUFIKS_VALID: Dict[str, Dict] = {
    "-kan": {
        "fungsi": "kausatif_transitif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
    },
    "-i": {
        "fungsi": "verba_lokatif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.VERBA,
    },
    "-an": {
        "fungsi": "nominalisasi",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.NOMINA,
    },
    "-nya": {
        "fungsi": "referensi_definit",
        "kelas_input_valid": {KelasKata.NOMINA, KelasKata.VERBA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.NOMINA,
    },
    "-lah": {
        "fungsi": "penegas",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.ADJEKTIVA, KelasKata.NOMINA},
        "kelas_output": None,
    },
    "-kah": {
        "fungsi": "interogatif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": None,
    },
    "-wan": {
        "fungsi": "nominalisasi_profesi",
        "kelas_input_valid": {KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.NOMINA,
    },
    "-wati": {
        "fungsi": "nominalisasi_profesi_fem",
        "kelas_input_valid": {KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.NOMINA,
    },
}

KONFIKS_VALID: Dict[str, Dict] = {
    "ke-an": {
        "fungsi": "nominalisasi_abstrak",
        "kelas_input_valid": {KelasKata.ADJEKTIVA, KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.NOMINA,
    },
    "pe-an": {
        "fungsi": "nominalisasi_proses",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.NOMINA,
    },
    "per-an": {
        "fungsi": "nominalisasi_hasil",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.NOMINA,
    },
    "ber-an": {
        "fungsi": "verba_resiprokal",
        "kelas_input_valid": {KelasKata.VERBA},
        "kelas_output": KelasKata.VERBA,
    },
    "me-kan": {
        "fungsi": "kausatif_aktif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
    },
    "me-i": {
        "fungsi": "verba_aktif_lokatif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.VERBA,
    },
    "di-kan": {
        "fungsi": "pasif_kausatif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA, KelasKata.ADJEKTIVA},
        "kelas_output": KelasKata.VERBA,
    },
    "di-i": {
        "fungsi": "pasif_lokatif",
        "kelas_input_valid": {KelasKata.VERBA, KelasKata.NOMINA},
        "kelas_output": KelasKata.VERBA,
    },
}

# ── Afiks Informal/Serapan (valid tapi tidak di KBBI formal) ─────────────────

AFIKS_INFORMAL: Dict[str, Dict] = {
    "di-": {
        "fungsi": "pasif_kata_serapan",
        "contoh": ["di-ghosting", "di-cancel", "di-PHP-in", "di-upload"],
        "valid": True,
        "catatan": "prefiks di- produktif untuk kata serapan verba",
    },
    "-in": {
        "fungsi": "sufiks_informal_transitif",
        "contoh": ["suruh-in", "kasih-in", "PHP-in"],
        "valid": True,
        "catatan": "variasi informal dari -kan/-i",
    },
    "ke-": {
        "fungsi": "prefiks_informal_pasif",
        "contoh": ["ke-ghosting", "ke-baper"],
        "valid": True,
        "catatan": "konstruksi informal untuk keadaan tidak disengaja",
    },
    "ng-": {
        "fungsi": "prefiks_informal_me",
        "contoh": ["ngomong", "ngaku", "ngeliat"],
        "valid": True,
        "catatan": "reduksi informal dari me- dalam percakapan",
    },
}


# Tabel alomorf → (panjang strip, root restoration)
# Alomorf me- menyebabkan perubahan fonem awal root:
#   mem+beli → beli (strip 3 karakter 'mem', root='beli')
#   men+tulis → tulis (strip 3, root='tulis')
#   meng+ambil → ambil (strip 4, root='ambil')
#   meny+sapu → sapu (strip 4, root='sapu', 's' direstore)
#   me+rasa → rasa (strip 2, root='rasa')
ALOMORF_STRIP: Dict[str, int] = {
    # me- group
    "me-":    2,   # me+rasa → rasa
    "mem-":   3,   # mem+beli → beli
    "men-":   3,   # men+tulis → tulis
    "meng-":  4,   # meng+ambil → ambil
    "meny-":  4,   # meny+apu → sapu (perlu restore 's')
    "menge-": 5,   # menge+cat → cat
    # pe- group
    "pe-":    2,
    "pem-":   3,
    "pen-":   3,
    "peng-":  4,
    "peny-":  4,
    "penge-": 5,
    # ber- group
    "ber-":   3,
    "be-":    2,
    "bel-":   3,
    # ter- group
    "ter-":   3,
    "te-":    2,
    # per- group
    "per-":   3,
    "pel-":   3,
    # lainnya
    "di-":    2,
    "ke-":    2,
    "se-":    2,
}

# Alomorf yang menyebabkan luluh (fonem pertama root hilang saat diimbuhkan)
# meny+sapu → menyapu (s luluh), perlu restore 's' saat dekomposisi
ALOMORF_RESTORE: Dict[str, str] = {
    "meny-": "s",  # menyapu → sapu (restore 's')
    "peny-": "s",  # penyapu → sapu
    "menge-": "",  # menge+cat → cat (tidak ada restore)
}


class AfiksRules:
    """
    Engine aturan afiks bahasa Indonesia — deterministik, bukan statistik.

    OPOSISI TRANSFORMER:
    Transformer mempelajari validitas afiks dari data — bisa salah pada kasus langka.
    AfiksRules mengimplementasikan aturan sebagai finite state rules — tidak bisa salah
    untuk kasus yang terdefinisi, dan jelas melaporkan "tidak diketahui" untuk kasus lain.
    """

    def __init__(self, termasuk_informal: bool = True):
        self.termasuk_informal = termasuk_informal
        self._prefiks = PREFIKS_VALID
        self._sufiks = SUFIKS_VALID
        self._konfiks = KONFIKS_VALID
        self._informal = AFIKS_INFORMAL if termasuk_informal else {}
        self._alomorf_strip = ALOMORF_STRIP
        self._alomorf_restore = ALOMORF_RESTORE

        self._semua_prefiks_bentuk: Set[str] = set()
        for info in self._prefiks.values():
            self._semua_prefiks_bentuk.update(info.get("alomorf", []))

    def validasi_afiks(
        self,
        afiks: str,
        kelas_root: KelasKata,
        tipe: TipeAfiks,
    ) -> Tuple[bool, str]:
        """
        Validasi satu afiks terhadap kelas kata root-nya.

        Returns:
            (valid, penjelasan)
        """
        if tipe == TipeAfiks.PREFIKS:
            for bentuk, info in self._prefiks.items():
                if afiks in info.get("alomorf", [bentuk]):
                    valid = kelas_root in info["kelas_input_valid"]
                    if valid:
                        return True, f"Prefiks '{afiks}' valid untuk {kelas_root.value}"
                    else:
                        return False, (
                            f"Prefiks '{afiks}' tidak valid untuk kelas {kelas_root.value}. "
                            f"Hanya valid untuk: "
                            f"{[k.value for k in info['kelas_input_valid']]}"
                        )

        elif tipe == TipeAfiks.SUFIKS:
            if afiks in self._sufiks:
                info = self._sufiks[afiks]
                valid = kelas_root in info["kelas_input_valid"]
                if valid:
                    return True, f"Sufiks '{afiks}' valid untuk {kelas_root.value}"
                else:
                    return False, (
                        f"Sufiks '{afiks}' tidak valid untuk kelas {kelas_root.value}. "
                        f"Hanya valid untuk: "
                        f"{[k.value for k in info['kelas_input_valid']]}"
                    )

        elif tipe == TipeAfiks.KONFIKS:
            if afiks in self._konfiks:
                info = self._konfiks[afiks]
                valid = kelas_root in info["kelas_input_valid"]
                if valid:
                    return True, f"Konfiks '{afiks}' valid untuk {kelas_root.value}"
                else:
                    return False, (
                        f"Konfiks '{afiks}' tidak valid untuk kelas {kelas_root.value}."
                    )

        return True, "afiks tidak diketahui — dianggap valid (open world assumption)"

    def kelas_output(self, afiks: str, tipe: TipeAfiks) -> Optional[KelasKata]:
        """Kembalikan kelas kata output setelah afiksasi."""
        if tipe == TipeAfiks.PREFIKS:
            for bentuk, info in self._prefiks.items():
                if afiks in info.get("alomorf", [bentuk]):
                    return info.get("kelas_output")
        elif tipe == TipeAfiks.SUFIKS:
            if afiks in self._sufiks:
                return self._sufiks[afiks].get("kelas_output")
        elif tipe == TipeAfiks.KONFIKS:
            if afiks in self._konfiks:
                return self._konfiks[afiks].get("kelas_output")
        return None

    def deteksi_prefiks(self, kata: str) -> List[Tuple[str, str]]:
        """
        Deteksi prefiks yang mungkin ada di awal kata.
        Returns: list (prefiks_canonical, alomorf_ditemukan)

        Menggunakan ALOMORF_STRIP untuk menentukan panjang strip yang tepat
        per alomorf — bukan sekedar len(alomorf.rstrip('-')).
        """
        hasil = []
        kata_lower = kata.lower()
        for canonical, info in self._prefiks.items():
            for alomorf in info.get("alomorf", [canonical]):
                n_strip = self._alomorf_strip.get(alomorf, len(alomorf.rstrip("-")))
                if kata_lower.startswith(alomorf.rstrip("-")) and len(kata_lower) > n_strip + 2:
                    hasil.append((canonical, alomorf))
        # Urutkan dari alomorf terpanjang ke terpendek untuk match greedy
        hasil.sort(key=lambda x: self._alomorf_strip.get(x[1], 0), reverse=True)
        return hasil

    def strip_prefiks(self, kata: str, alomorf: str) -> str:
        """
        Strip prefiks dari kata dengan benar, termasuk restore fonem yang luluh.

        Contoh:
          strip_prefiks('membeli', 'mem-') → 'beli'
          strip_prefiks('menyapu', 'meny-') → 'sapu'  (restore 's')
          strip_prefiks('mengambil', 'meng-') → 'ambil'
        """
        n_strip = self._alomorf_strip.get(alomorf, len(alomorf.rstrip("-")))
        root = kata[n_strip:]
        restore = self._alomorf_restore.get(alomorf, "")
        if restore and not root.startswith(restore):
            root = restore + root
        return root

    def deteksi_sufiks(self, kata: str) -> List[str]:
        """Deteksi sufiks yang mungkin ada di akhir kata."""
        hasil = []
        kata_lower = kata.lower()
        for sufiks in self._sufiks:
            bersih = sufiks.lstrip("-")
            if kata_lower.endswith(bersih) and len(kata_lower) > len(bersih) + 2:
                hasil.append(sufiks)
        return hasil

    def adalah_kata_informal(self, kata: str) -> bool:
        """Deteksi apakah kata menggunakan afiks informal."""
        kata_lower = kata.lower()
        for afiks_info in self._informal.values():
            contoh = afiks_info.get("contoh", [])
            for c in contoh:
                if kata_lower in c.lower():
                    return True
        return False
