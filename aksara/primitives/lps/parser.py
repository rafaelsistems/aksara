"""
LPSParser — Parser linguistik deterministik untuk bahasa Indonesia.

OPOSISI TRANSFORMER:
  Transformer tokenizer: potong string berdasarkan frekuensi statistik BPE/WordPiece
  LPSParser: dekomposisi morfem berdasarkan aturan linguistik Indonesia (deterministik)

Unit output adalah morfem dengan metadata linguistik lengkap — bukan token arbitrer.
Setiap keputusan parsing punya justifikasi linguistik eksplisit.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Set, Tuple

from aksara.primitives.lps.morfem import (
    AfiksAktif, KelasKata, Morfem, PeranGramatikal, TipeAfiks,
)
from aksara.primitives.lps.afiks import AfiksRules


# ── Pola Reduplikasi ──────────────────────────────────────────────────────────

_POLA_REDUP_PENUH    = re.compile(r"^(\w+)-\1$", re.IGNORECASE)
_POLA_REDUP_PARSIAL  = re.compile(r"^(\w{2,3})-(\w+)$", re.IGNORECASE)

# ── Kata Fungsi (tidak didekomposisi) ─────────────────────────────────────────

KATA_PREPOSISI: Set[str] = {
    "di", "ke", "dari", "pada", "dalam", "antara", "atas", "bawah",
    "depan", "belakang", "samping", "untuk", "bagi", "demi", "sejak",
    "hingga", "sampai", "oleh", "dengan", "tanpa", "tentang", "mengenai",
    "terhadap", "melalui", "melintasi", "sepanjang", "sekitar",
}

KATA_KONJUNGSI: Set[str] = {
    "dan", "atau", "tetapi", "namun", "akan tetapi", "melainkan",
    "sedangkan", "sementara", "meskipun", "walaupun", "karena",
    "sebab", "sehingga", "agar", "supaya", "jika", "kalau", "bila",
    "bahwa", "yang", "saat", "ketika", "setelah", "sebelum", "selama",
}

KATA_PRONOMINA: Set[str] = {
    "saya", "aku", "kamu", "anda", "dia", "ia", "mereka", "kita",
    "kami", "beliau", "kalian", "engkau", "hamba", "-ku", "-mu", "-nya",
    "ini", "itu", "tersebut", "sini", "sana", "situ",
}

KATA_PARTIKEL: Set[str] = {
    "pun", "lah", "kah", "tah", "deh", "dong", "sih", "nih", "kok",
    "ya", "yah", "kan", "lho", "lo", "dah",
}

# Kata yang TIDAK boleh didekomposisi afiks (false positive tinggi)
KATA_TIDAK_DEKOMPOSISI: Set[str] = {
    "pergi", "segar", "dapat", "pesan", "benar", "besar", "dalam",
    "depan", "belakang", "bersama", "selain", "sekitar", "setelah",
    "sebelum", "sesudah", "setidaknya", "seharusnya", "semua", "setiap",
    "pernah", "penting", "penuh", "pendek", "panjang", "pertama",
    "terasa", "terlihat", "tersebut", "termasuk", "terdapat",
    "berbeda", "berupa", "bersifat", "memang", "baru", "lama",
    "keras", "kecil", "besar", "indah", "bagus", "jelek",
    "cepat", "lambat", "tinggi", "rendah", "jauh", "dekat",
    "kira", "mana", "sana", "sini", "situ",
    "kemarin", "besok", "lusa", "lantai", "dinding", "langit",
    "pantai", "sawah", "ladang", "kebun", "hutan", "gunung",
    "sungai", "danau", "laut", "pulau", "negara", "bangsa",
    "makan", "minum", "tidur", "bangun", "duduk",
    "pergi", "pulang", "datang", "tiba", "tinggal",
}

KATA_ADVERBIA_UMUM: Set[str] = {
    "sangat", "amat", "sekali", "terlalu", "agak", "cukup", "paling",
    "lebih", "kurang", "tidak", "bukan", "belum", "sudah", "telah",
    "sedang", "akan", "masih", "lagi", "juga", "pun", "hanya", "saja",
    "justru", "bahkan", "malah", "memang", "tentu", "pasti", "mungkin",
    "hampir", "segera", "selalu", "sering", "jarang", "kadang", "selama",
}

# ── Pola Kata Serapan dan Informal ────────────────────────────────────────────

_POLA_SERAPAN = re.compile(
    r"\b(meeting|upload|download|ghosting|cancel|update|install|login|logout|"
    r"online|offline|streaming|trending|posting|sharing|like|comment|follow|"
    r"unfollow|block|chat|video|foto|story|status|repost|screenshot)\b",
    re.IGNORECASE,
)

_POLA_INFORMAL_PASIF = re.compile(
    r"^(di|ke)-([a-z]+(?:ing|ed|kan|in)?)$", re.IGNORECASE
)


class LPSParser:
    """
    Parser morfem bahasa Indonesia — deterministik, bukan statistik.

    OPOSISI TRANSFORMER:
    - Transformer tokenizer: BPE/WordPiece berdasarkan frekuensi statistik
    - LPSParser: dekomposisi berdasarkan aturan morfologi Indonesia yang dikodekan

    Properti AKSARA yang dijamin:
    - Setiap morfem punya kelas kata dan peran gramatikal yang eksplisit
    - Validitas afiks dicek deterministik, bukan probabilistik
    - Reduplikasi dideteksi dan dikategorikan secara linguistik
    - Kata serapan dan informal ditandai tapi tetap diproses
    """

    def __init__(
        self,
        leksikon: Optional[Dict[str, str]] = None,
        termasuk_informal: bool = True,
    ):
        """
        Args:
            leksikon: dict {kata: kelas_kata} dari KBBI atau sumber lain
            termasuk_informal: apakah parsing kata informal diaktifkan
        """
        self.leksikon: Dict[str, str] = leksikon or {}
        self.afiks_rules = AfiksRules(termasuk_informal=termasuk_informal)
        self.termasuk_informal = termasuk_informal

    def parse(self, kalimat: str) -> List[Morfem]:
        """
        Parse kalimat menjadi list morfem dengan metadata linguistik lengkap.

        Ini bukan tokenisasi — ini dekomposisi linguistik.

        Args:
            kalimat: string kalimat bahasa Indonesia

        Returns:
            list Morfem dengan metadata lengkap
        """
        kalimat_bersih = self._normalisasi(kalimat)
        kata_kata = self._segmentasi(kalimat_bersih)

        morfem_list: List[Morfem] = []
        for idx, kata in enumerate(kata_kata):
            if not kata.strip():
                continue
            morfem = self._parse_kata(kata, idx)
            morfem_list.append(morfem)

        morfem_list = self._assign_peran_gramatikal(morfem_list)
        return morfem_list

    def parse_batch(self, kalimat_list: List[str]) -> List[List[Morfem]]:
        """Parse sekumpulan kalimat."""
        return [self.parse(k) for k in kalimat_list]

    # ── Private Methods ───────────────────────────────────────────────────────

    def _normalisasi(self, teks: str) -> str:
        """Normalisasi unicode dan karakter tidak standar."""
        teks = unicodedata.normalize("NFC", teks)
        teks = teks.replace("\u2019", "'").replace("\u2018", "'")
        teks = teks.replace("\u201c", '"').replace("\u201d", '"')
        teks = re.sub(r"\s+", " ", teks)
        return teks.strip()

    def _segmentasi(self, kalimat: str) -> List[str]:
        """
        Segmentasi kalimat menjadi kata-kata.
        Berbeda dari tokenisasi Transformer — mempertahankan tanda hubung
        untuk reduplikasi dan tidak memotong di batas subword statistik.
        """
        kalimat = re.sub(r"[.,!?;:\"()[\]{}]", " ", kalimat)
        kata_kata = kalimat.split()
        hasil = []
        for kata in kata_kata:
            if re.match(r"^\w+-\w+$", kata):
                hasil.append(kata)
            else:
                kata_bersih = re.sub(r"[^a-zA-Z0-9\-']", "", kata)
                if kata_bersih:
                    hasil.append(kata_bersih)
        return hasil

    def _parse_kata(self, kata: str, indeks: int) -> Morfem:
        """
        Parse satu kata menjadi Morfem dengan metadata lengkap.
        Urutan: reduplikasi → kata fungsi → leksikon → dekomposisi afiks
        """
        kata_lower = kata.lower()

        redup = self._cek_reduplikasi(kata)
        if redup:
            return dc_replace(redup, indeks=indeks)

        if kata_lower in KATA_PREPOSISI:
            return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                          kelas_kata=KelasKata.PREPOSISI,
                          peran_gramatikal=PeranGramatikal.KETERANGAN,
                          ada_di_kbbi=True)

        if kata_lower in KATA_KONJUNGSI:
            return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                          kelas_kata=KelasKata.KONJUNGSI,
                          peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                          ada_di_kbbi=True)

        if kata_lower in KATA_PRONOMINA:
            return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                          kelas_kata=KelasKata.PRONOMINA,
                          peran_gramatikal=PeranGramatikal.SUBJEK,
                          ada_di_kbbi=True)

        if kata_lower in KATA_ADVERBIA_UMUM:
            return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                          kelas_kata=KelasKata.ADVERBIA,
                          peran_gramatikal=PeranGramatikal.MODIFIER,
                          ada_di_kbbi=True)

        if kata_lower in KATA_PARTIKEL:
            return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                          kelas_kata=KelasKata.PARTIKEL,
                          peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                          ada_di_kbbi=True)

        if kata[0].isupper() and indeks > 0:
            return Morfem(indeks=indeks, teks_asli=kata, root=kata,
                          kelas_kata=KelasKata.NOMINA_PROPER,
                          peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                          adalah_proper=True)

        if kata_lower in KATA_TIDAK_DEKOMPOSISI:
            kelas = self._inferensi_kelas(kata_lower)
            if kelas == KelasKata.TIDAK_DIKETAHUI:
                kelas = KelasKata.ADJEKTIVA
            return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                          kelas_kata=kelas,
                          peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                          ada_di_kbbi=kata_lower in self.leksikon)

        if _POLA_SERAPAN.match(kata_lower):
            return self._parse_kata_serapan(kata, indeks)

        if kata_lower in self.leksikon:
            # Cek apakah kata ini perlu didekomposisi meski ada di leksikon.
            # Kriteria: ada prefiks produktif (me-, men-, mem-, meng-, ber-, pe-, per-, ter-)
            # DAN root setelah strip prefiks lebih pendek (berarti kata ini turunan).
            # Contoh: 'membersihkan' ada di leksikon, tapi 'bersih' (root) lebih pendek
            # Contoh counter: 'berlari' ada di leksikon, tapi 'lari' juga di leksikon
            # → berlari lebih baik sebagai kata dasar (tidak ada konfiks)
            perlu_dekomposisi = False
            prefiks_list = self.afiks_rules.deteksi_prefiks(kata_lower)
            for canonical, alomorf in prefiks_list[:1]:
                root_cand = self.afiks_rules.strip_prefiks(kata_lower, alomorf)
                # Hanya dekomposisi jika ada konfiks (sufiks juga) — kata murni berimbuhan
                sufiks_setelah_strip = self.afiks_rules.deteksi_sufiks(root_cand)
                if sufiks_setelah_strip:
                    for sufiks in sufiks_setelah_strip[:1]:
                        bersih = sufiks.lstrip("-")
                        root_konfiks = root_cand[:-len(bersih)] if root_cand.endswith(bersih) else ""
                        if len(root_konfiks) >= 3 and root_konfiks in self.leksikon:
                            # Jangan dekomposisi jika kata utuh sudah ada di leksikon
                            # dengan kelas kata yang BERBEDA dari rootnya.
                            # Contoh: 'pengamanan' (N) vs 'aman' (Adj) — berbeda kelas,
                            # dekomposisi akan kehilangan info kelas kata yang tepat.
                            # Sebaliknya: 'membersihkan' tidak ada di leksikon sebagai
                            # kata dasar → dekomposisi ke 'bersih' lebih informatif.
                            kelas_utuh = self.leksikon.get(kata_lower, "")
                            kelas_root = self.leksikon.get(root_konfiks, "")
                            if kelas_utuh and kelas_root and kelas_utuh != kelas_root:
                                # Kelas berbeda — kata utuh punya identitas sendiri
                                perlu_dekomposisi = False
                            else:
                                perlu_dekomposisi = True
                            break

            if not perlu_dekomposisi:
                kelas = self._str_ke_kelas(self.leksikon[kata_lower])
                return Morfem(indeks=indeks, teks_asli=kata, root=kata_lower,
                              kelas_kata=kelas,
                              peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                              ada_di_kbbi=True)
            # perlu_dekomposisi=True → lanjut ke _dekomposisi_afiks

        return self._dekomposisi_afiks(kata, indeks)

    def _cek_reduplikasi(self, kata: str) -> Optional[Morfem]:
        """Deteksi dan parse reduplikasi bahasa Indonesia."""
        if _POLA_REDUP_PENUH.match(kata):
            base = kata.split("-")[0].lower()
            return Morfem(
                indeks=0, teks_asli=kata, root=base,
                kelas_kata=KelasKata.NOMINA,
                peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                adalah_reduplikasi=True,
                tipe_reduplikasi="penuh",
                base_reduplikasi=base,
            )

        m = _POLA_REDUP_PARSIAL.match(kata)
        if m:
            prefix_redup, base = m.group(1), m.group(2)
            if base.startswith(prefix_redup[:2]):
                return Morfem(
                    indeks=0, teks_asli=kata, root=base.lower(),
                    kelas_kata=KelasKata.NOMINA,
                    peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                    adalah_reduplikasi=True,
                    tipe_reduplikasi="parsial",
                    base_reduplikasi=base.lower(),
                )
        return None

    def _dekomposisi_afiks(self, kata: str, indeks: int) -> Morfem:
        """
        Dekomposisi kata menjadi root + afiks secara deterministik.
        Strategi: kumpulkan semua kandidat dekomposisi, pilih root terpanjang
        yang ada di leksikon. Ini mencegah 'membeli' → 'bel' (terlalu pendek)
        dan memilih 'beli' (lebih panjang, ada di leksikon).
        """
        kata_lower = kata.lower()
        afiks_list: List[AfiksAktif] = []
        kelas_output = KelasKata.TIDAK_DIKETAHUI

        # ── Kumpulkan semua kandidat (konfiks dan prefiks-saja) ───────────────
        # Kandidat: (root, afiks_list, kelas_output, tipe)
        kandidat_list: List[tuple] = []

        for konfiks, info in self.afiks_rules._konfiks.items():
            p, s = konfiks.split("-")
            s_bersih = s.rstrip()
            p_alomorf = self._cari_alomorf_prefiks(kata_lower, p.lstrip())
            if not (p_alomorf and kata_lower.endswith(s_bersih)):
                continue
            root_tanpa_prefiks = self.afiks_rules.strip_prefiks(kata_lower, p_alomorf + "-")
            root = root_tanpa_prefiks[:-len(s_bersih)] if root_tanpa_prefiks.endswith(s_bersih) else ""
            if len(root) < 3:
                continue
            kelas_root = self._inferensi_kelas(root)
            valid, _ = self.afiks_rules.validasi_afiks(konfiks, kelas_root, TipeAfiks.KONFIKS)
            afiks_k = [AfiksAktif(bentuk=konfiks, tipe=TipeAfiks.KONFIKS,
                                  fungsi=info["fungsi"], valid=valid)]
            kls = info.get("kelas_output") or kelas_root
            kandidat_list.append((root, afiks_k, kls, "konfiks"))

        # ── Jika ada kandidat konfiks, pilih yang rootnya terpanjang di leksikon ──
        if kandidat_list and self.leksikon:
            # Tambahkan kandidat prefiks-saja sebagai pembanding
            for canonical, alomorf in self.afiks_rules.deteksi_prefiks(kata_lower)[:1]:
                rtp = self.afiks_rules.strip_prefiks(kata_lower, alomorf)
                if len(rtp) >= 3:
                    info_p = self.afiks_rules._prefiks[canonical]
                    kelas_root = self._inferensi_kelas(rtp)
                    valid, _ = self.afiks_rules.validasi_afiks(alomorf, kelas_root, TipeAfiks.PREFIKS)
                    afiks_p = [AfiksAktif(bentuk=canonical, tipe=TipeAfiks.PREFIKS,
                                          fungsi=info_p["fungsi"], valid=valid)]
                    kls_p = info_p.get("kelas_output") or kelas_root
                    kandidat_list.append((rtp, afiks_p, kls_p, "prefiks"))

            # Prioritas 1: root ada di leksikon, pilih yang terpanjang
            in_lex = [(r, af, kl, tp) for r, af, kl, tp in kandidat_list if r in self.leksikon]
            if in_lex:
                best = max(in_lex, key=lambda x: len(x[0]))
                root, afiks_list, kelas_output, _ = best
                return Morfem(
                    indeks=indeks, teks_asli=kata, root=root,
                    kelas_kata=kelas_output,
                    peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                    afiks_aktif=afiks_list,
                    ada_di_kbbi=True,
                )
            # Prioritas 2: tidak ada di leksikon, pilih konfiks terpendek root (paling agresif strip)
            konfiks_only = [(r, af, kl, tp) for r, af, kl, tp in kandidat_list if tp == "konfiks"]
            if konfiks_only:
                root, afiks_list, kelas_output, _ = konfiks_only[0]
                return Morfem(
                    indeks=indeks, teks_asli=kata, root=root,
                    kelas_kata=kelas_output,
                    peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                    afiks_aktif=afiks_list,
                    ada_di_kbbi=False,
                )
        elif kandidat_list:
            # Tidak ada leksikon — pakai kandidat konfiks pertama
            root, afiks_list, kelas_output, _ = kandidat_list[0]
            return Morfem(
                indeks=indeks, teks_asli=kata, root=root,
                kelas_kata=kelas_output,
                peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
                afiks_aktif=afiks_list,
                ada_di_kbbi=False,
            )

        prefiks_list = self.afiks_rules.deteksi_prefiks(kata_lower)
        root_setelah_prefiks = kata_lower
        for canonical, alomorf in prefiks_list[:1]:
            kandidat_root = self.afiks_rules.strip_prefiks(kata_lower, alomorf)
            # Root harus minimal 3 karakter setelah strip prefiks
            if len(kandidat_root) < 3:
                continue
            root_setelah_prefiks = kandidat_root
            info = self.afiks_rules._prefiks[canonical]
            kelas_root = self._inferensi_kelas(root_setelah_prefiks)
            valid, _ = self.afiks_rules.validasi_afiks(
                alomorf, kelas_root, TipeAfiks.PREFIKS
            )
            afiks_list.append(AfiksAktif(
                bentuk=canonical, tipe=TipeAfiks.PREFIKS,
                fungsi=info["fungsi"], valid=valid,
            ))
            kelas_output = info.get("kelas_output") or kelas_root

        sufiks_list = self.afiks_rules.deteksi_sufiks(root_setelah_prefiks)
        root_akhir = root_setelah_prefiks
        for sufiks in sufiks_list[:1]:
            bersih = sufiks.lstrip("-")
            kandidat_root = root_setelah_prefiks[:-len(bersih)]
            # Root harus minimal 3 karakter setelah strip sufiks
            if len(kandidat_root) < 3:
                continue
            # Validasi: kandidat root harus ada di leksikon ATAU
            # kata asli tidak ada di leksikon (baru benar-benar berimbuhan)
            kata_asli_ada = root_setelah_prefiks in self.leksikon
            kandidat_ada  = kandidat_root in self.leksikon
            if kata_asli_ada and not kandidat_ada:
                # "nasi" ada di leksikon, "nas" tidak → jangan strip
                continue
            root_akhir = kandidat_root
            info = self.afiks_rules._sufiks[sufiks]
            kelas_root = self._inferensi_kelas(root_akhir)
            valid, _ = self.afiks_rules.validasi_afiks(
                sufiks, kelas_root, TipeAfiks.SUFIKS
            )
            afiks_list.append(AfiksAktif(
                bentuk=sufiks, tipe=TipeAfiks.SUFIKS,
                fungsi=info["fungsi"], valid=valid,
            ))
            if info.get("kelas_output"):
                kelas_output = info["kelas_output"]

        if not root_akhir or len(root_akhir) < 2:
            root_akhir = kata_lower

        if kelas_output == KelasKata.TIDAK_DIKETAHUI:
            kelas_output = self._inferensi_kelas(root_akhir)

        return Morfem(
            indeks=indeks, teks_asli=kata, root=root_akhir,
            kelas_kata=kelas_output,
            peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
            afiks_aktif=afiks_list,
            ada_di_kbbi=root_akhir in self.leksikon,
        )

    def _parse_kata_serapan(self, kata: str, indeks: int) -> Morfem:
        """Parse kata serapan Inggris-Indonesia."""
        kata_lower = kata.lower()
        m = _POLA_INFORMAL_PASIF.match(kata_lower)
        if m:
            prefiks, root = m.group(1), m.group(2)
            return Morfem(
                indeks=indeks, teks_asli=kata, root=root,
                kelas_kata=KelasKata.VERBA_SERAPAN,
                peran_gramatikal=PeranGramatikal.PREDIKAT,
                afiks_aktif=[AfiksAktif(
                    bentuk=f"{prefiks}-", tipe=TipeAfiks.PREFIKS,
                    fungsi="pasif_serapan", valid=True,
                )],
                adalah_serapan=True, bahasa_asal="en",
            )
        return Morfem(
            indeks=indeks, teks_asli=kata, root=kata_lower,
            kelas_kata=KelasKata.NOMINA_SERAPAN,
            peran_gramatikal=PeranGramatikal.TIDAK_DIKETAHUI,
            adalah_serapan=True, bahasa_asal="en",
        )

    def _cari_alomorf_prefiks(self, kata: str, prefiks_base: str) -> Optional[str]:
        """
        Cari alomorf prefiks yang cocok dengan awal kata.
        Mengembalikan bentuk alomorf DENGAN tanda hubung (contoh: 'mem-')
        agar bisa dipakai dengan strip_prefiks.
        """
        kata_lower = kata.lower()
        cocok: List[Tuple[str, int]] = []  # (alomorf_dengan_dash, n_strip)
        for canonical, info in self.afiks_rules._prefiks.items():
            if not (canonical.startswith(prefiks_base) or prefiks_base in canonical):
                continue
            for alomorf in info.get("alomorf", []):
                n_strip = self.afiks_rules._alomorf_strip.get(alomorf, len(alomorf.rstrip("-")))
                bersih = alomorf.rstrip("-")
                if kata_lower.startswith(bersih) and len(kata_lower) > n_strip + 2:
                    cocok.append((alomorf, n_strip))
        if not cocok:
            return None
        # Ambil yang terpanjang (greedy match)
        cocok.sort(key=lambda x: x[1], reverse=True)
        return cocok[0][0].rstrip("-")  # kembalikan tanpa dash untuk backward compat

    def _inferensi_kelas(self, root: str) -> KelasKata:
        """Inferensi kelas kata dari leksikon, fallback ke TIDAK_DIKETAHUI."""
        if root in self.leksikon:
            return self._str_ke_kelas(self.leksikon[root])
        return KelasKata.TIDAK_DIKETAHUI

    def _str_ke_kelas(self, s: str) -> KelasKata:
        """Konversi string POS tag ke KelasKata."""
        mapping = {
            "n": KelasKata.NOMINA, "v": KelasKata.VERBA,
            "a": KelasKata.ADJEKTIVA, "adv": KelasKata.ADVERBIA,
            "adj": KelasKata.ADJEKTIVA, "num": KelasKata.NUMERALIA,
            "pron": KelasKata.PRONOMINA, "prep": KelasKata.PREPOSISI,
            "konj": KelasKata.KONJUNGSI, "part": KelasKata.PARTIKEL,
            "p": KelasKata.PREPOSISI,
        }
        return mapping.get(s.lower(), KelasKata.TIDAK_DIKETAHUI)

    def _assign_peran_gramatikal(self, morfem_list: List[Morfem]) -> List[Morfem]:
        """
        Assign peran gramatikal berdasarkan posisi dan kelas kata.
        Heuristik dasar tata bahasa Indonesia:
          - Nomina/Pronomina pertama sebelum verba → Subjek
          - Verba pertama → Predikat
          - Nomina/Pronomina setelah verba → Objek
          - Preposisi + nomina → Keterangan
        """
        posisi_verba: Optional[int] = None
        for i, m in enumerate(morfem_list):
            if m.kelas_kata == KelasKata.VERBA and posisi_verba is None:
                posisi_verba = i
                object.__setattr__(morfem_list[i], "peran_gramatikal",
                                   PeranGramatikal.PREDIKAT) \
                    if hasattr(morfem_list[i], "__dataclass_fields__") else None
                morfem_list[i] = Morfem(
                    **{**morfem_list[i].__dict__,
                       "peran_gramatikal": PeranGramatikal.PREDIKAT}
                )

        subjek_assigned = False
        objek_assigned = False
        dalam_prep = False

        for i, m in enumerate(morfem_list):
            if m.peran_gramatikal != PeranGramatikal.TIDAK_DIKETAHUI:
                continue

            if m.kelas_kata == KelasKata.PREPOSISI:
                dalam_prep = True
                continue

            if dalam_prep:
                morfem_list[i] = Morfem(
                    **{**m.__dict__, "peran_gramatikal": PeranGramatikal.KETERANGAN}
                )
                dalam_prep = False
                continue

            if m.kelas_kata in (KelasKata.NOMINA, KelasKata.PRONOMINA,
                                 KelasKata.NOMINA_PROPER, KelasKata.NOMINA_SERAPAN):
                if posisi_verba is None or i < posisi_verba:
                    if not subjek_assigned:
                        morfem_list[i] = Morfem(
                            **{**m.__dict__,
                               "peran_gramatikal": PeranGramatikal.SUBJEK}
                        )
                        subjek_assigned = True
                elif posisi_verba is not None and i > posisi_verba:
                    if not objek_assigned:
                        morfem_list[i] = Morfem(
                            **{**m.__dict__,
                               "peran_gramatikal": PeranGramatikal.OBJEK}
                        )
                        objek_assigned = True
                    else:
                        morfem_list[i] = Morfem(
                            **{**m.__dict__,
                               "peran_gramatikal": PeranGramatikal.PELENGKAP}
                        )

            elif m.kelas_kata == KelasKata.ADJEKTIVA:
                morfem_list[i] = Morfem(
                    **{**m.__dict__, "peran_gramatikal": PeranGramatikal.MODIFIER}
                )

        return morfem_list
