"""
ConstraintSet — definisi dan evaluasi constraint linguistik untuk CPE.

OPOSISI TRANSFORMER:
  Transformer: tidak ada constraint eksplisit — semua implisit di bobot
  CPE:         constraint didefinisikan secara eksplisit dan dievaluasi deterministik

Jenis constraint:
  1. Morfologis  — validitas afiks terhadap kelas root
  2. Sintaktis   — kompatibilitas peran gramatikal antar morfem
  3. Semantik    — kompatibilitas domain antara kata yang berrelasi
  4. Leksikal    — konsistensi register dalam satu kalimat
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aksara.primitives.lps.morfem import Morfem, KelasKata, PeranGramatikal
from aksara.primitives.sfm.geodesic import GeodesicDistance
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.config import AksaraConfig


@dataclass
class ConstraintResult:
    """
    Hasil evaluasi satu constraint antara dua morfem.
    Setiap hasil punya penjelasan linguistik eksplisit — bukan angka blackbox.
    """
    tipe: str
    morfem_a: str
    morfem_b: str
    ketegangan: float          # 0.0 = tidak ada ketegangan, 1.0 = maksimal
    dilanggar: bool
    penjelasan: str

    @property
    def valid(self) -> bool:
        return not self.dilanggar


# ── Bobot constraint per dimensi ─────────────────────────────────────────────
# Menentukan kontribusi tiap jenis constraint ke energi total.
# Bukan learned parameter — ini keputusan linguistik eksplisit.

BOBOT_CONSTRAINT: Dict[str, float] = {
    "morfologis": 0.28,
    "sintaktis":  0.23,
    "semantik":   0.32,
    "leksikal":   0.09,
    "animasi":    0.08,
}

# ── Verba dengan selektivitas [+animate] pada argumen agen ───────────────────
# Justifikasi (Alwi dkk. 2003 + Levin 1993 — theta-role selection):
#   Verba psikologis, emosi, volitional, dan persepsi mensyaratkan agen
#   yang memiliki kapasitas mental/rasa ([+animate, +sentient]).
#   Melanggar ini = categorical animacy violation — bukan sekadar domain mismatch.
# Format: root verba (LPS akan strip prefiks me-/ber-/ter- sebelum lookup)
VERBA_SELEKTIVITAS_ANIMATE: frozenset = frozenset({
    # Verba emosi
    "tangis", "menangis", "menangisi",
    "tawa", "tertawa", "menertawakan",
    "sedih", "bersedih",
    "marah", "murka", "gusar",
    "takut", "khawatir", "cemas",
    "senang", "gembira", "bahagia",
    "rindu", "kangen",
    "malu", "sungkan",
    "benci", "dendam",
    "cinta", "sayang", "mencintai", "menyayangi",
    # Verba psikologis / kognitif
    "pikir", "berpikir", "memikir", "memikirkan",
    "ingat", "mengingat", "mengingat",
    "lupa", "melupakan",
    "paham", "mengerti", "memahami",
    "tahu", "mengetahui",
    "percaya", "mempercayai",
    "ragu", "meragukan",
    "sadar", "menyadari",
    "harap", "berharap",
    "ingin", "menginginkan",
    "bermimpi", "bermimpi",
    # Verba volitional (aksi berdasarkan kehendak)
    "memutuskan", "putus",
    "memilih", "pilih",
    "memerintah", "perintah",
    "bersidang", "sidang",
    "berunding", "runding",
    "bernegosiasi",
    # Verba bicara / komunikasi aktif
    "berkata", "kata",
    "berteriak", "meneriakkan",
    "berbisik", "membisikkan",
    "berdoa", "memohon",
    "mengeluh", "keluh",
    "mengaku", "aku",
    "berjanji", "janji",
})

# ── Kompatibilitas peran gramatikal ──────────────────────────────────────────
# Pasangan (peran_a, peran_b) yang compatible dalam bahasa Indonesia

RELASI_PERAN_VALID = {
    (PeranGramatikal.SUBJEK,    PeranGramatikal.PREDIKAT),
    (PeranGramatikal.PREDIKAT,  PeranGramatikal.OBJEK),
    (PeranGramatikal.PREDIKAT,  PeranGramatikal.PELENGKAP),
    (PeranGramatikal.SUBJEK,    PeranGramatikal.MODIFIER),
    (PeranGramatikal.OBJEK,     PeranGramatikal.MODIFIER),
    (PeranGramatikal.SUBJEK,    PeranGramatikal.KETERANGAN),
    (PeranGramatikal.PREDIKAT,  PeranGramatikal.KETERANGAN),
    (PeranGramatikal.OBJEK,     PeranGramatikal.KETERANGAN),
}

# ── Kompatibilitas kelas kata sebagai modifier ────────────────────────────────

MODIFIER_VALID: Dict[KelasKata, set] = {
    KelasKata.ADJEKTIVA: {KelasKata.NOMINA, KelasKata.NOMINA_PROPER,
                          KelasKata.NOMINA_SERAPAN},
    KelasKata.ADVERBIA:  {KelasKata.VERBA, KelasKata.ADJEKTIVA,
                          KelasKata.VERBA_SERAPAN},
    KelasKata.NUMERALIA: {KelasKata.NOMINA, KelasKata.NOMINA_PROPER},
}


class ConstraintSet:
    """
    Kumpulan constraint linguistik yang dievaluasi oleh CPE.

    OPOSISI TRANSFORMER:
    Transformer tidak punya constraint eksplisit — semua di bobot.
    ConstraintSet mendefinisikan aturan linguistik Indonesia secara eksplisit.
    Tiap constraint menghasilkan nilai ketegangan ∈ [0, 1] dengan penjelasan.

    Ini bukan heuristic — ini formalisasi tata bahasa Indonesia ke dalam
    fungsi ketegangan yang bisa dihitung secara deterministik.
    """

    def __init__(
        self,
        leksikon: LexiconLoader,
        geodesic: GeodesicDistance,
        threshold_semantik: float = 1.5,
        config: Optional[AksaraConfig] = None,
    ):
        self.leksikon           = leksikon
        self.geodesic           = geodesic
        self.config             = config or AksaraConfig.default()
        # threshold dari config menang atas parameter langsung
        self.threshold_semantik = self.config.threshold_semantik if config else threshold_semantik

    def evaluasi_pasangan(
        self,
        ma: Morfem,
        mb: Morfem,
        relasi: str = "adjacent",
    ) -> List[ConstraintResult]:
        """
        Evaluasi semua constraint untuk pasangan morfem (ma, mb).

        Args:
            ma, mb:  dua morfem yang berrelasi
            relasi:  tipe relasi ('adjacent', 'subj-pred', 'modifier', dll.)

        Returns:
            list ConstraintResult — satu per jenis constraint
        """
        hasil = []
        hasil.append(self._cek_morfologis(ma, mb))
        hasil.append(self._cek_sintaktis(ma, mb, relasi))
        hasil.append(self._cek_semantik(ma, mb))
        hasil.append(self._cek_leksikal(ma, mb))
        # Animacy: hanya relevan untuk relasi subjek-predikat
        if relasi == "subj-pred":
            hasil.append(self._cek_animasi(ma, mb))
        return hasil

    def ketegangan_total(
        self,
        ma: Morfem,
        mb: Morfem,
        relasi: str = "adjacent",
    ) -> float:
        """
        Hitung ketegangan total antara dua morfem — weighted sum per jenis.

        Returns:
            float ∈ [0, 1] — 0 = tidak ada ketegangan, 1 = ketegangan maksimal
        """
        hasil = self.evaluasi_pasangan(ma, mb, relasi)
        total = 0.0
        for r in hasil:
            bobot = BOBOT_CONSTRAINT.get(r.tipe, 0.25)
            total += bobot * r.ketegangan
        return min(1.0, total)

    # ── Private: evaluasi per jenis constraint ────────────────────────────────

    def _cek_animasi(self, ma: Morfem, mb: Morfem) -> ConstraintResult:
        """
        Constraint animacy (theta-role selection).

        Justifikasi linguistik:
          Verba psikologis, emosi, dan volitional memilih argumen agen [+animate].
          Ini adalah property verba itu sendiri (selektivitas-θ), bukan preferensi.
          Melanggar = categorical animacy violation — bukan degree mismatch.

        Referensi:
          Chomsky (1981) Lectures on Government and Binding — theta-role
          Levin (1993) English Verb Classes and Alternations — psych-verbs
          Alwi dkk. (2003) Tata Bahasa Baku Indonesia — verba keadaan mental

        Hanya dievaluasi untuk relasi subj-pred (dipanggil dari evaluasi_pasangan).
        """
        # Identifikasi subjek dan predikat dari pasangan
        subj = mb if mb.peran_gramatikal == PeranGramatikal.SUBJEK else (
               ma if ma.peran_gramatikal == PeranGramatikal.SUBJEK else None)
        pred = ma if ma.peran_gramatikal == PeranGramatikal.PREDIKAT else (
               mb if mb.peran_gramatikal == PeranGramatikal.PREDIKAT else None)

        if subj is None or pred is None:
            return ConstraintResult(
                tipe="animasi",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=0.0, dilanggar=False,
                penjelasan="Bukan pasangan subj-pred — animacy tidak dievaluasi",
            )

        # Cek apakah predikat termasuk verba yang mensyaratkan agen animate
        root_pred = pred.root.lower()
        # Coba cocokkan root langsung, lalu strip prefiks verba umum
        PREFIKS_VERBA = ("me", "ber", "ter", "di", "ke", "pe")
        root_tanpa_prefiks = root_pred
        for pref in PREFIKS_VERBA:
            if root_pred.startswith(pref) and len(root_pred) > len(pref) + 2:
                root_tanpa_prefiks = root_pred[len(pref):]
                break

        butuh_animate = (
            root_pred in VERBA_SELEKTIVITAS_ANIMATE
            or root_tanpa_prefiks in VERBA_SELEKTIVITAS_ANIMATE
            or pred.teks_asli.lower() in VERBA_SELEKTIVITAS_ANIMATE
        )

        if not butuh_animate:
            return ConstraintResult(
                tipe="animasi",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=0.0, dilanggar=False,
                penjelasan="Verba tidak mensyaratkan agen animate",
            )

        # Cek apakah subjek animate
        root_subj = subj.root.lower()
        # NOMINA_PROPER dan PRONOMINA selalu animate
        subj_animate = (
            subj.kelas_kata in (KelasKata.NOMINA_PROPER, KelasKata.PRONOMINA)
            or self.leksikon.adalah_animate(root_subj)
        )

        if subj_animate:
            return ConstraintResult(
                tipe="animasi",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=0.0, dilanggar=False,
                penjelasan=f"'{subj.root}' [+animate] — kompatibel dengan verba '{pred.root}'",
            )
        else:
            return ConstraintResult(
                tipe="animasi",
                morfem_a=subj.root, morfem_b=pred.root,
                ketegangan=0.85,
                dilanggar=True,
                penjelasan=(
                    f"Animacy violation: '{subj.root}' [−animate] — "
                    f"verba '{pred.root}' mensyaratkan agen bernyawa. "
                    f"Benda mati/abstraksi tidak bisa {pred.teks_asli}."
                ),
            )

    def _cek_morfologis(self, ma: Morfem, mb: Morfem) -> ConstraintResult:
        """
        Constraint morfologis: apakah afiks valid terhadap kelas kata?
        Ini adalah constraint HARD — dikodekan dari TBBBI, bukan statistik.
        """
        pelanggaran = []

        for morfem in [ma, mb]:
            for afiks in morfem.afiks_aktif:
                if not afiks.valid:
                    pelanggaran.append(
                        f"Afiks '{afiks.bentuk}' tidak valid untuk "
                        f"'{morfem.root}' ({morfem.kelas_kata.value})"
                    )

        if pelanggaran:
            return ConstraintResult(
                tipe="morfologis",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=1.0,
                dilanggar=True,
                penjelasan=" | ".join(pelanggaran),
            )

        return ConstraintResult(
            tipe="morfologis",
            morfem_a=ma.root, morfem_b=mb.root,
            ketegangan=0.0, dilanggar=False,
            penjelasan="Morfologi valid",
        )

    def _cek_sintaktis(
        self, ma: Morfem, mb: Morfem, relasi: str
    ) -> ConstraintResult:
        """
        Constraint sintaktis: apakah peran gramatikal konsisten?
        Berdasarkan tata bahasa Indonesia — deterministik.

        Catatan penting: constraint modifier hanya dievaluasi jika relasi
        adalah 'modifier' (bukan 'adjacent'). Edge adjacent tidak selalu berarti
        salah satu memodifikasi yang lain — mereka bisa berada dalam relasi
        struktural berbeda (mis. adverbia temporal 'sedang' adjacent ke subjek
        tapi sebenarnya memodifikasi verba di posisi lain).
        """
        peran_a = ma.peran_gramatikal
        peran_b = mb.peran_gramatikal

        # Cek modifier compatibility — HANYA untuk relasi eksplisit 'modifier'
        # Ini mencegah false positive: 'sedang'(Adv) adjacent 'Pemerintah'(N)
        # sebenarnya bukan relasi modifier, tapi constraint akan salah mena
        if relasi == "modifier" and (
            peran_a == PeranGramatikal.MODIFIER or peran_b == PeranGramatikal.MODIFIER
        ):
            modifier = ma if peran_a == PeranGramatikal.MODIFIER else mb
            head     = mb if peran_a == PeranGramatikal.MODIFIER else ma

            kelas_modifier = modifier.kelas_kata
            kelas_head     = head.kelas_kata

            if kelas_modifier in MODIFIER_VALID:
                valid_targets = MODIFIER_VALID[kelas_modifier]
                if kelas_head not in valid_targets:
                    return ConstraintResult(
                        tipe="sintaktis",
                        morfem_a=ma.root, morfem_b=mb.root,
                        ketegangan=0.7, dilanggar=True,
                        penjelasan=(
                            f"Modifier '{modifier.root}' ({kelas_modifier.value}) "
                            f"tidak bisa memodifikasi '{head.root}' ({kelas_head.value})"
                        ),
                    )

        # Cek subjek-predikat agreement dasar
        if {peran_a, peran_b} == {PeranGramatikal.SUBJEK, PeranGramatikal.PREDIKAT}:
            subj = ma if peran_a == PeranGramatikal.SUBJEK else mb
            pred = mb if peran_a == PeranGramatikal.SUBJEK else ma
            # Subjek abstrak + predikat fisik = ketegangan
            if (subj.kelas_kata == KelasKata.NOMINA and
                    pred.kelas_kata == KelasKata.VERBA):
                domain_subj = self.leksikon.domain_kata(subj.root)
                domain_pred = self.leksikon.domain_kata(pred.root)
                # Khusus: subjek manusia/animate wajib untuk verba fisik
                # (heuristik sederhana — bisa diperluas)

        return ConstraintResult(
            tipe="sintaktis",
            morfem_a=ma.root, morfem_b=mb.root,
            ketegangan=0.0, dilanggar=False,
            penjelasan="Sintaksis valid",
        )

    def _cek_semantik(self, ma: Morfem, mb: Morfem) -> ConstraintResult:
        """
        Constraint semantik: apakah domain kata compatible?

        Layer 1 (universal): selalu dievaluasi
        Layer 2 (domain-spesifik): dikonfigurasi via AksaraConfig
          - verba_domain_neutral: verba yang bisa berelasi dengan objek apapun
          - threshold_semantik: batas jarak geodesic
        """
        neutral = self.config.verba_domain_neutral

        root_a = ma.root.lower()
        root_b = mb.root.lower()

        # Guard 1: NOMINA_PROPER (nama orang/tempat) tidak punya domain semantik
        # Membandingkan nama orang dengan kata apapun tidak valid secara linguistik.
        if (ma.kelas_kata == KelasKata.NOMINA_PROPER or
                mb.kelas_kata == KelasKata.NOMINA_PROPER):
            return ConstraintResult(
                tipe="semantik",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=0.0, dilanggar=False,
                penjelasan="Nomina proper — domain tidak dievaluasi",
            )

        # Layer 2: cek verba domain-neutral dari config (bisa dikustomisasi)
        def _is_neutral(root: str) -> bool:
            if root in neutral:
                return True
            # Cek bentuk berimbuhan: root='menjual' mengandung basis 'jual'
            for basis in neutral:
                if root.endswith(basis) and len(root) > len(basis):
                    return True
            return False

        if _is_neutral(root_a) or _is_neutral(root_b):
            return ConstraintResult(
                tipe="semantik",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=0.0, dilanggar=False,
                penjelasan="Verba domain-neutral — relasi dengan objek apapun valid",
            )

        d = self.geodesic.hitung(ma.root, mb.root)

        if d <= self.threshold_semantik:
            return ConstraintResult(
                tipe="semantik",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=d / self.threshold_semantik * 0.3,
                dilanggar=False,
                penjelasan=f"Domain compatible (jarak={d:.2f})",
            )
        else:
            domain_a = self.leksikon.domain_kata(ma.root) or "?"
            domain_b = self.leksikon.domain_kata(mb.root) or "?"
            path = self.geodesic.path_semantik(ma.root, mb.root)
            path_str = " → ".join(path)
            ketegangan = min(1.0, (d - self.threshold_semantik) / 2.0)
            return ConstraintResult(
                tipe="semantik",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=ketegangan,
                dilanggar=True,
                penjelasan=(
                    f"Domain mismatch: '{ma.root}'=[{domain_a}] "
                    f"≠ '{mb.root}'=[{domain_b}] "
                    f"jarak={d:.2f} | path: {path_str}"
                ),
            )

    def _cek_leksikal(self, ma: Morfem, mb: Morfem) -> ConstraintResult:
        """
        Constraint leksikal: apakah register konsisten?
        Pencampuran register formal/informal dalam satu kalimat = ketegangan.
        """
        reg_a = 0.0 if ma.adalah_informal else (0.5 if ma.adalah_serapan else 1.0)
        reg_b = 0.0 if mb.adalah_informal else (0.5 if mb.adalah_serapan else 1.0)

        selisih = abs(reg_a - reg_b)
        if selisih > 0.5:
            return ConstraintResult(
                tipe="leksikal",
                morfem_a=ma.root, morfem_b=mb.root,
                ketegangan=selisih,
                dilanggar=True,
                penjelasan=(
                    f"Register tidak konsisten: '{ma.root}' "
                    f"({'informal' if ma.adalah_informal else 'formal'}) "
                    f"vs '{mb.root}' "
                    f"({'informal' if mb.adalah_informal else 'formal'})"
                ),
            )

        return ConstraintResult(
            tipe="leksikal",
            morfem_a=ma.root, morfem_b=mb.root,
            ketegangan=selisih * 0.3,
            dilanggar=False,
            penjelasan="Register konsisten",
        )
