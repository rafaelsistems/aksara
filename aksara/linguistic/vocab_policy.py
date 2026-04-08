"""
vocab_policy.py — AksaraVocabPolicy & AksaraVocabValidator
===========================================================
Komponen resmi framework AKSARA untuk mendefinisikan dan memvalidasi
vocabulary yang sehat.

Filosofi:
    corpus → dunia yang diketahui model
    KBBI   → makna yang dipegang model
    vocab  → jembatan antara keduanya

    Framework yang buruk membiarkan kesalahan.
    Framework yang baik menjelaskan kesalahan.
    Framework yang hebat membuat kesalahan sulit diabaikan.

Desain tiga lapisan:

    🔴 HARD CONSTRAINT  — selalu raise Exception (crash prevention, dimension mismatch)
    🟡 SOFT CONSTRAINT  — tidak memblokir, tapi menurunkan quality tier
    🟢 GUIDELINE        — rekomendasi, dicatat tapi tidak menurunkan tier

Quality Tier (dari terbaik ke terburuk):
    OPTIMAL      → semua constraint terpenuhi, mendekati ideal
    VALID        → constraint utama terpenuhi, mungkin ada warning kecil
    DEGRADED     → satu atau lebih constraint gagal, output berisiko
    EXPERIMENTAL → banyak constraint gagal, hanya untuk eksperimen

Aturan dasar:
    1. Vocab size     : 5K–15K, proporsional terhadap corpus
    2. Komposisi      : 70–80% corpus-driven + 20–30% KBBI-driven
    3. Coverage       : ≥ 75% kemunculan token corpus terwakili
    4. OOV            : ≤ 15% dari top-K token corpus jadi <UNK>
    5. Domain sanity  : kata kunci per domain wajib hadir di vocab

Penggunaan:
    from aksara.linguistic.vocab_policy import AksaraVocabPolicy, AksaraVocabValidator

    policy = AksaraVocabPolicy.from_corpus_size(n_sentences=100_000)
    validator = AksaraVocabValidator(policy)
    result = validator.validate(vocab, corpus_token_freq, kbbi_set)
    result.print_report()          # selalu bisa dipanggil
    result.assert_hard_constraints()  # hanya raise jika hard constraint gagal
"""

from __future__ import annotations

import re
import math
import collections
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ─── Domain sanity check seeds ────────────────────────────────────────────────
# Kata kunci minimum per domain yang HARUS ada di vocab.
# Framework akan FAIL jika domain wajib tidak punya representasi.
DOMAIN_SANITY_SEEDS: Dict[str, List[str]] = {
    "makan":      ["makan", "minum", "nasi", "makanan"],
    "membaca":    ["baca", "buku", "kata", "teks"],
    "berjalan":   ["jalan", "kaki", "gerak", "pergi"],
    "bekerja":    ["kerja", "kantor", "tugas", "laporan"],
    "pemerintah": ["pemerintah", "hukum", "negara", "kebijakan"],
    "pendidikan": ["sekolah", "belajar", "guru", "ilmu"],
    "sejarah":    ["sejarah", "tahun", "masa", "perang"],
    "sains":      ["penelitian", "data", "hasil", "ilmu"],
}


# ─── AksaraVocabPolicy ────────────────────────────────────────────────────────

@dataclass
class AksaraVocabPolicy:
    """
    Aturan resmi AKSARA untuk vocabulary.

    Policy ini adaptif: angka absolut disesuaikan dengan ukuran corpus,
    tapi batas kewarasan (sanity bounds) tetap dijaga.

    Attributes:
        min_vocab        : batas bawah vocab size
        max_vocab        : batas atas vocab size
        target_vocab     : target aktual (dihitung dari corpus size)
        corpus_ratio_min : minimum proporsi token dari corpus (0.70)
        corpus_ratio_max : maximum proporsi token dari corpus (0.80)
        kbbi_ratio_min   : minimum proporsi lemma KBBI (0.20)
        kbbi_ratio_max   : maximum proporsi lemma KBBI (0.30)
        min_coverage     : minimum coverage corpus di vocab (0.75)
        max_oov_rate     : maximum OOV rate yang diizinkan (0.15)
        domain_seeds     : kata kunci domain yang wajib ada
        corpus_n_sentences: ukuran corpus untuk adaptive sizing
    """
    min_vocab: int = 5_000
    max_vocab: int = 15_000
    target_vocab: int = 10_000

    corpus_ratio_min: float = 0.70
    corpus_ratio_max: float = 0.80
    kbbi_ratio_min: float = 0.20
    kbbi_ratio_max: float = 0.30

    min_coverage: float = 0.75
    max_oov_rate: float = 0.15

    domain_seeds: Dict[str, List[str]] = field(
        default_factory=lambda: DOMAIN_SANITY_SEEDS
    )
    corpus_n_sentences: int = 0

    @classmethod
    def from_corpus_size(cls, n_sentences: int) -> "AksaraVocabPolicy":
        """
        Buat policy dengan target vocab yang disesuaikan ukuran corpus.

        Skala proporsional:
            10K  kalimat → 5K–8K  vocab
            50K  kalimat → 7K–11K vocab
            100K kalimat → 9K–13K vocab
            150K kalimat → 10K–14K vocab
            200K kalimat → 12K–15K vocab (saturasi)

        Insight: vocabulary growth mengikuti Heap's Law — sublinear
        terhadap ukuran corpus. Kita gunakan aproksimasi log-linear.
        """
        # Heap's Law approximation: V ≈ K * N^beta
        # Kalibrasi untuk Indonesian Wikipedia:
        #   K=50, beta=0.5 (konservatif)
        k = 50
        beta = 0.5
        heaps_estimate = int(k * (n_sentences ** beta))

        # Clamp ke range yang waras
        target = max(5_000, min(15_000, heaps_estimate))

        # Round ke ribuan terdekat
        target = round(target / 1000) * 1000

        return cls(
            target_vocab=target,
            corpus_n_sentences=n_sentences,
        )

    @property
    def corpus_ratio_center(self) -> float:
        return (self.corpus_ratio_min + self.corpus_ratio_max) / 2

    @property
    def kbbi_ratio_center(self) -> float:
        return (self.kbbi_ratio_min + self.kbbi_ratio_max) / 2

    def corpus_slots(self) -> int:
        """Jumlah slot untuk corpus tokens."""
        return int(self.target_vocab * self.corpus_ratio_center)

    def kbbi_slots(self) -> int:
        """Jumlah slot untuk KBBI tokens."""
        return self.target_vocab - self.corpus_slots()

    def __repr__(self) -> str:
        return (
            f"AksaraVocabPolicy("
            f"target={self.target_vocab:,}, "
            f"corpus={self.corpus_ratio_min:.0%}–{self.corpus_ratio_max:.0%}, "
            f"kbbi={self.kbbi_ratio_min:.0%}–{self.kbbi_ratio_max:.0%}, "
            f"coverage≥{self.min_coverage:.0%}, "
            f"oov≤{self.max_oov_rate:.0%})"
        )


# ─── Quality Tier ─────────────────────────────────────────────────────────────

class QualityTier:
    """
    4-tier quality status untuk vocab AKSARA.

    Tier tidak memblokir eksekusi — hanya menjelaskan kualitas.
    Hard constraints (crash prevention) ditangani terpisah.
    """
    OPTIMAL      = "OPTIMAL"       # semua constraint terpenuhi, mendekati ideal
    VALID        = "VALID"         # constraint utama terpenuhi, warning kecil
    DEGRADED     = "DEGRADED"      # ≥1 constraint gagal, output berisiko
    EXPERIMENTAL = "EXPERIMENTAL"  # ≥2 constraint gagal, hanya untuk eksperimen

    # Deskripsi prediksi perilaku per tier
    BEHAVIOR: Dict[str, List[str]] = {
        OPTIMAL: [
            "output coherence tinggi",
            "domain coverage baik",
            "OOV minimal saat inferensi",
            "siap training skala penuh",
        ],
        VALID: [
            "output coherence cukup baik",
            "beberapa domain mungkin kurang representasi",
            "OOV terkontrol",
            "bisa training, tapi pantau loss domain",
        ],
        DEGRADED: [
            "output coherence berisiko rendah",
            "domain gaps mungkin muncul",
            "OOV tinggi → banyak token jadi <UNK>",
            "training bisa jalan tapi hasil tidak optimal",
            "pertimbangkan perbesar vocab atau perbaiki corpus",
        ],
        EXPERIMENTAL: [
            "output kemungkinan tidak coherent",
            "banyak domain tidak terwakili",
            "OOV sangat tinggi → model sulit belajar",
            "hanya gunakan untuk eksplorasi arsitektur",
            "JANGAN gunakan untuk evaluasi atau produksi",
        ],
    }

    @staticmethod
    def rank(tier: str) -> int:
        """Nilai numerik tier (lebih tinggi = lebih baik)."""
        return {
            QualityTier.OPTIMAL:      4,
            QualityTier.VALID:        3,
            QualityTier.DEGRADED:     2,
            QualityTier.EXPERIMENTAL: 1,
        }.get(tier, 0)

    @staticmethod
    def symbol(tier: str) -> str:
        return {
            QualityTier.OPTIMAL:      "🟢",
            QualityTier.VALID:        "🟡",
            QualityTier.DEGRADED:     "🟠",
            QualityTier.EXPERIMENTAL: "🔴",
        }.get(tier, "⚪")


# ─── Validation result ────────────────────────────────────────────────────────

@dataclass
class VocabValidationResult:
    """
    Hasil validasi vocab terhadap policy.

    Design:
    - Setiap check punya status: PASS / WARN / FAIL / SKIP
    - Status individual TIDAK memblokir — hanya mempengaruhi quality tier
    - quality_tier adalah ringkasan akhir: OPTIMAL / VALID / DEGRADED / EXPERIMENTAL
    - Hard constraints divalidasi terpisah via assert_hard_constraints()
    """
    policy: AksaraVocabPolicy

    # Check results: (status, measured_value, detail_string)
    # status: "PASS" | "WARN" | "FAIL" | "SKIP"
    check_size:        Tuple[str, int, str]   = ("SKIP", 0,   "")
    check_composition: Tuple[str, float, str] = ("SKIP", 0.0, "")
    check_coverage:    Tuple[str, float, str] = ("SKIP", 0.0, "")
    check_oov:         Tuple[str, float, str] = ("SKIP", 0.0, "")
    check_domain:      Tuple[str, float, str] = ("SKIP", 0.0, "")

    domain_details: Dict[str, List[str]] = field(default_factory=dict)

    # Hard constraint violations (raise-worthy issues)
    hard_violations: List[str] = field(default_factory=list)

    @property
    def quality_tier(self) -> str:
        """
        Hitung quality tier dari semua check results.

        Logika agregasi:
          FAIL count = 0, WARN count = 0  → OPTIMAL
          FAIL count = 0, WARN count ≥ 1  → VALID
          FAIL count = 1                  → DEGRADED
          FAIL count ≥ 2                  → EXPERIMENTAL
        """
        statuses = [
            self.check_size[0],
            self.check_composition[0],
            self.check_coverage[0],
            self.check_oov[0],
            self.check_domain[0],
        ]
        fail_count = sum(1 for s in statuses if s == "FAIL")
        warn_count = sum(1 for s in statuses if s == "WARN")

        if fail_count >= 2:
            return QualityTier.EXPERIMENTAL
        elif fail_count == 1:
            return QualityTier.DEGRADED
        elif warn_count >= 1:
            return QualityTier.VALID
        else:
            return QualityTier.OPTIMAL

    @property
    def passed(self) -> bool:
        """True jika tier VALID atau lebih baik (backward compat)."""
        return QualityTier.rank(self.quality_tier) >= QualityTier.rank(QualityTier.VALID)

    @property
    def all_pass(self) -> bool:
        """True jika tier OPTIMAL (backward compat)."""
        return self.quality_tier == QualityTier.OPTIMAL

    def print_report(self, title: str = "AKSARA Vocab Validation Report") -> None:
        """
        Cetak laporan lengkap ke stdout.
        Selalu bisa dipanggil — tidak pernah raise.
        """
        W = 68
        tier = self.quality_tier
        sym  = QualityTier.symbol(tier)

        print(f"\n{'='*W}")
        print(f"  {title}")
        print(f"{'='*W}")
        print(f"  Policy  : {self.policy}")
        print()

        # ── Check results ──
        checks = [
            ("Vocab Size",               self.check_size),
            ("Composition (corpus/KBBI)", self.check_composition),
            ("Corpus Coverage",          self.check_coverage),
            ("OOV Rate",                 self.check_oov),
            ("Domain Sanity",            self.check_domain),
        ]
        status_sym = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "SKIP": "—"}
        for name, (status, value, detail) in checks:
            s = status_sym.get(status, "?")
            print(f"  {s} {name:<30} {status:<5}  {detail}")

        # ── Domain details ──
        if self.domain_details:
            print()
            print("  Domain word coverage:")
            for domain, info in self.domain_details.items():
                present = [w for w in info if not w.startswith("MISSING:")]
                missing = [w[8:] for w in info if w.startswith("MISSING:")]
                ds = "✅" if not missing else "❌"
                print(f"    {ds} [{domain:<12}]  present={present}  missing={missing}")

        # ── Quality tier verdict ──
        print()
        print(f"  {'─'*64}")
        print(f"  {sym} Overall Quality  : {tier}")
        print(f"  {'─'*64}")

        behaviors = QualityTier.BEHAVIOR.get(tier, [])
        print(f"  Expected behavior:")
        for b in behaviors:
            marker = "  ✓" if tier in (QualityTier.OPTIMAL, QualityTier.VALID) else "  ⚠"
            print(f"  {marker} {b}")

        # ── Hard violations ──
        if self.hard_violations:
            print()
            print("  🔴 HARD CONSTRAINT VIOLATIONS (memerlukan perbaikan segera):")
            for v in self.hard_violations:
                print(f"    ✗ {v}")

        print(f"{'='*W}\n")

    def assert_hard_constraints(self) -> None:
        """
        Raise VocabHardConstraintError jika ada hard constraint yang dilanggar.
        Soft constraint TIDAK pernah raise dari sini.
        """
        if self.hard_violations:
            raise VocabHardConstraintError(
                f"Vocab hard constraint violated: {self.hard_violations}"
            )

    def assert_valid(self, strict: bool = False) -> None:
        """
        Backward-compatible method.
        strict=True → raise jika tier bukan OPTIMAL
        strict=False → raise hanya jika tier EXPERIMENTAL
        """
        if strict and self.quality_tier != QualityTier.OPTIMAL:
            raise ValueError(
                f"Vocab quality tier is {self.quality_tier}, expected OPTIMAL. "
                f"Checks: size={self.check_size[0]}, "
                f"composition={self.check_composition[0]}, "
                f"coverage={self.check_coverage[0]}, "
                f"oov={self.check_oov[0]}, "
                f"domain={self.check_domain[0]}"
            )
        elif not strict and self.quality_tier == QualityTier.EXPERIMENTAL:
            raise ValueError(
                f"Vocab quality tier is EXPERIMENTAL — too many constraints failed. "
                f"Use result.print_report() for details."
            )


class VocabHardConstraintError(Exception):
    """Raised when a hard (crash-preventing) constraint is violated."""
    pass


# ─── AksaraVocabValidator ─────────────────────────────────────────────────────

class AksaraVocabValidator:
    """
    Validator resmi untuk vocabulary AKSARA.

    Tiga lapisan validasi:
        🔴 HARD   — dicek pertama, dicatat ke result.hard_violations
                    Contoh: vocab kosong, token ID duplikat, special tokens hilang
        🟡 SOFT   — dicek semua, menurunkan quality_tier
                    Contoh: OOV > 15%, coverage < 75%, komposisi di luar range
        🟢 INFO   — dicatat tapi tidak menurunkan tier
                    Contoh: corpus sangat kecil (dimaklumi), slang rendah

    Tidak pernah memblokir eksekusi.
    Gunakan result.assert_hard_constraints() jika ingin raise pada hard violations.
    """

    SPECIAL_TOKENS  = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"}
    REQUIRED_TOKENS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}

    def __init__(self, policy: Optional[AksaraVocabPolicy] = None):
        self.policy = policy or AksaraVocabPolicy()

    def validate(
        self,
        vocab: Dict[str, int],
        corpus_token_freq: Optional[Dict[str, int]] = None,
        kbbi_set: Optional[Set[str]] = None,
        n_corpus_tokens_from_corpus_slot: Optional[int] = None,
        n_kbbi_tokens_from_kbbi_slot: Optional[int] = None,
    ) -> VocabValidationResult:
        """
        Validasi vocab lengkap. Tidak pernah raise — semua disimpan di result.

        Args:
            vocab                            : vocab dict {token: id}
            corpus_token_freq                : frekuensi token di corpus
            kbbi_set                         : semua lemma KBBI
            n_corpus_tokens_from_corpus_slot : token dari corpus slot
            n_kbbi_tokens_from_kbbi_slot     : token dari KBBI-only slot

        Returns:
            VocabValidationResult dengan quality_tier dan semua detail checks
        """
        result = VocabValidationResult(policy=self.policy)
        p = self.policy

        # ── HARD CONSTRAINT CHECKS (tidak memblokir, hanya dicatat) ──────────
        # 1. Vocab kosong
        if not vocab:
            result.hard_violations.append("vocab kosong — tidak ada token sama sekali")

        # 2. Special tokens wajib hadir
        for tok in self.REQUIRED_TOKENS:
            if tok not in vocab:
                result.hard_violations.append(
                    f"special token wajib '{tok}' tidak ada di vocab"
                )

        # 3. Token ID harus unik
        id_values = list(vocab.values())
        if len(id_values) != len(set(id_values)):
            dupes = len(id_values) - len(set(id_values))
            result.hard_violations.append(
                f"{dupes} token ID duplikat — akan menyebabkan embedding collision"
            )

        # 4. Special token IDs harus di posisi benar (0,1,2,3)
        expected_ids = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for tok, expected in expected_ids.items():
            if tok in vocab and vocab[tok] != expected:
                result.hard_violations.append(
                    f"'{tok}' ada di id={vocab[tok]}, seharusnya id={expected} "
                    f"— menyebabkan embedding mismatch"
                )

        real_tokens = {t for t in vocab if t not in self.SPECIAL_TOKENS}
        vocab_size = len(real_tokens)

        # ── Check 1: Size ──────────────────────────────────────────────────
        result.check_size = self._check_size(vocab_size, p)

        # ── Check 2: Composition ───────────────────────────────────────────
        result.check_composition = self._check_composition(
            vocab, real_tokens, kbbi_set,
            n_corpus_tokens_from_corpus_slot,
            n_kbbi_tokens_from_kbbi_slot,
            p,
        )

        # ── Check 3 & 4: Coverage + OOV ───────────────────────────────────
        if corpus_token_freq is not None:
            result.check_coverage = self._check_coverage(vocab, corpus_token_freq, p)
            result.check_oov = self._check_oov(vocab, corpus_token_freq, p)
        else:
            result.check_coverage = ("SKIP", 0.0, "corpus_token_freq tidak disediakan")
            result.check_oov      = ("SKIP", 0.0, "corpus_token_freq tidak disediakan")

        # ── Check 5: Domain sanity ─────────────────────────────────────────
        result.check_domain, result.domain_details = self._check_domain(vocab, p)

        return result

    # ── Internal check methods ──────────────────────────────────────────────

    def _check_size(self, vocab_size: int, p: AksaraVocabPolicy) -> Tuple[str, int, str]:
        target = p.target_vocab
        tolerance_low  = max(p.min_vocab, int(target * 0.85))
        tolerance_high = min(p.max_vocab, int(target * 1.15))

        if p.min_vocab <= vocab_size <= p.max_vocab:
            if tolerance_low <= vocab_size <= tolerance_high:
                status = "PASS"
                detail = f"{vocab_size:,} token (target={target:,}, range={p.min_vocab:,}–{p.max_vocab:,})"
            else:
                status = "WARN"
                detail = (
                    f"{vocab_size:,} token di luar target ±15% "
                    f"(target={target:,}, acceptable={tolerance_low:,}–{tolerance_high:,})"
                )
        elif vocab_size < p.min_vocab:
            # Adaptive: jika corpus kecil, WARN bukan FAIL
            if p.corpus_n_sentences > 0 and p.corpus_n_sentences < 20_000:
                status = "WARN"
                detail = (
                    f"{vocab_size:,} < min {p.min_vocab:,} "
                    f"(corpus kecil={p.corpus_n_sentences:,} kalimat, diterima)"
                )
            else:
                status = "FAIL"
                detail = f"{vocab_size:,} < min {p.min_vocab:,}"
        else:
            status = "WARN"
            detail = f"{vocab_size:,} > max {p.max_vocab:,} (mungkin noisy)"

        return status, vocab_size, detail

    def _check_composition(
        self,
        vocab: Dict[str, int],
        real_tokens: Set[str],
        kbbi_set: Optional[Set[str]],
        n_corpus: Optional[int],
        n_kbbi: Optional[int],
        p: AksaraVocabPolicy,
    ) -> Tuple[str, float, str]:
        total = len(real_tokens)
        if total == 0:
            return "FAIL", 0.0, "Vocab kosong"

        # Jika angka slot langsung disediakan
        if n_corpus is not None and n_kbbi is not None:
            corpus_ratio = n_corpus / total
            kbbi_only_ratio = n_kbbi / total
            detail = (
                f"corpus={corpus_ratio:.0%} (target {p.corpus_ratio_min:.0%}–{p.corpus_ratio_max:.0%}), "
                f"kbbi_only={kbbi_only_ratio:.0%} (target {p.kbbi_ratio_min:.0%}–{p.kbbi_ratio_max:.0%})"
            )
            in_range = (
                p.corpus_ratio_min <= corpus_ratio <= p.corpus_ratio_max
                and p.kbbi_ratio_min <= kbbi_only_ratio <= p.kbbi_ratio_max
            )
            status = "PASS" if in_range else "WARN"
            return status, corpus_ratio, detail

        # Fallback: estimasi dari kbbi_set
        if kbbi_set is not None:
            in_kbbi = sum(1 for t in real_tokens if t in kbbi_set)
            kbbi_ratio = in_kbbi / total
            corpus_ratio = 1 - kbbi_ratio
            detail = (
                f"in_KBBI={kbbi_ratio:.0%} — "
                f"NOTE: ini overlap, bukan KBBI-only slot. "
                f"Gunakan n_corpus/n_kbbi untuk akurasi."
            )
            return "WARN", corpus_ratio, detail

        return "SKIP", 0.0, "kbbi_set dan n_corpus/n_kbbi tidak disediakan"

    def _check_coverage(
        self,
        vocab: Dict[str, int],
        corpus_freq: Dict[str, int],
        p: AksaraVocabPolicy,
    ) -> Tuple[str, float, str]:
        """
        Coverage = proporsi kemunculan token di corpus yang terwakili di vocab.
        Ini weighted coverage — token frekuensi tinggi lebih penting.
        """
        total_occurrences = sum(corpus_freq.values())
        if total_occurrences == 0:
            return "SKIP", 0.0, "corpus_freq kosong"

        covered_occurrences = sum(
            freq for token, freq in corpus_freq.items()
            if token in vocab
        )
        coverage = covered_occurrences / total_occurrences

        if coverage >= p.min_coverage:
            status = "PASS"
        elif coverage >= p.min_coverage * 0.90:
            status = "WARN"
        else:
            status = "FAIL"

        detail = (
            f"{coverage:.1%} kemunculan terwakili "
            f"(min={p.min_coverage:.0%}, "
            f"covered={covered_occurrences:,}/{total_occurrences:,} token occurrences)"
        )
        return status, coverage, detail

    def _check_oov(
        self,
        vocab: Dict[str, int],
        corpus_freq: Dict[str, int],
        p: AksaraVocabPolicy,
    ) -> Tuple[str, float, str]:
        """
        OOV check: proporsi token SIGNIFICANT di corpus yang tidak ada di vocab.

        "Significant" = muncul ≥ freq_threshold kali di corpus.
        Threshold dipilih adaptif: ambil percentile ke-50 dari distribusi frekuensi
        (median), tapi minimal 3.

        Ini ukuran OOV yang paling bermakna untuk inferensi:
        - Token yang muncul 1-2x hampir pasti noise/typo/proper noun langka → wajar OOV
        - Token yang muncul ≥ threshold kali adalah kata-kata yang benar-benar dipakai
        - Model perlu bisa memahami kata-kata ini, bukan jadi <UNK>

        max_oov_rate ≤ 15% berarti: dari semua kata yang sering dipakai,
        maksimal 15% tidak dikenali model.
        """
        if not corpus_freq:
            return "SKIP", 0.0, "corpus_freq kosong"

        # Adaptive threshold: min(median_freq, 10), minimal 3
        all_freqs = sorted(corpus_freq.values())
        if all_freqs:
            median_freq = all_freqs[len(all_freqs) // 2]
            freq_threshold = max(3, min(median_freq, 10))
        else:
            freq_threshold = 3

        significant = {t for t, f in corpus_freq.items() if f >= freq_threshold}
        if not significant:
            return "SKIP", 0.0, f"tidak ada token dengan freq≥{freq_threshold}"

        oov_significant = {t for t in significant if t not in vocab}
        oov_rate = len(oov_significant) / len(significant)

        if oov_rate <= p.max_oov_rate:
            status = "PASS"
        elif oov_rate <= p.max_oov_rate * 1.5:
            status = "WARN"
        else:
            status = "FAIL"

        detail = (
            f"{oov_rate:.1%} token significant OOV "
            f"(freq≥{freq_threshold}, max={p.max_oov_rate:.0%}, "
            f"oov={len(oov_significant):,}/{len(significant):,} unique tokens)"
        )
        return status, oov_rate, detail

    def _check_domain(
        self,
        vocab: Dict[str, int],
        p: AksaraVocabPolicy,
    ) -> Tuple[Tuple[str, float, str], Dict[str, List[str]]]:
        """
        Domain sanity: setiap domain wajib punya minimal 50% seed words di vocab.
        FAIL jika ada domain wajib yang coverage < 25%.
        """
        domain_details: Dict[str, List[str]] = {}
        domain_scores: Dict[str, float] = {}

        for domain, seeds in p.domain_seeds.items():
            info = []
            present = 0
            for word in seeds:
                if word in vocab:
                    info.append(word)
                    present += 1
                else:
                    info.append(f"MISSING:{word}")
            score = present / len(seeds) if seeds else 1.0
            domain_details[domain] = info
            domain_scores[domain] = score

        if not domain_scores:
            return ("SKIP", 0.0, "domain_seeds kosong"), {}

        avg_score = sum(domain_scores.values()) / len(domain_scores)
        failing_domains = [d for d, s in domain_scores.items() if s < 0.25]
        warn_domains = [d for d, s in domain_scores.items() if 0.25 <= s < 0.50]

        if failing_domains:
            status = "FAIL"
            detail = (
                f"avg={avg_score:.0%}, "
                f"FAIL domains (<25%): {failing_domains}"
            )
        elif warn_domains:
            status = "WARN"
            detail = (
                f"avg={avg_score:.0%}, "
                f"WARN domains (<50%): {warn_domains}"
            )
        else:
            status = "PASS"
            detail = f"avg coverage={avg_score:.0%}, semua domain ≥50%"

        return (status, avg_score, detail), domain_details


# ─── Convenience function ─────────────────────────────────────────────────────

def validate_vocab(
    vocab: Dict[str, int],
    corpus_token_freq: Optional[Dict[str, int]] = None,
    kbbi_set: Optional[Set[str]] = None,
    n_corpus_slot: Optional[int] = None,
    n_kbbi_slot: Optional[int] = None,
    n_corpus_sentences: int = 0,
    print_report: bool = True,
    strict: bool = False,
) -> VocabValidationResult:
    """
    Shortcut untuk validasi vocab dengan satu fungsi.

    Args:
        vocab               : vocab dict yang ingin divalidasi
        corpus_token_freq   : frekuensi token dari corpus
        kbbi_set            : set semua lemma KBBI
        n_corpus_slot       : jumlah token dari corpus slot
        n_kbbi_slot         : jumlah token dari KBBI-only slot
        n_corpus_sentences  : ukuran corpus (untuk adaptive sizing)
        print_report        : cetak laporan ke stdout
        strict              : raise exception jika ada WARN

    Returns:
        VocabValidationResult
    """
    policy = AksaraVocabPolicy.from_corpus_size(n_corpus_sentences)
    validator = AksaraVocabValidator(policy)
    result = validator.validate(
        vocab=vocab,
        corpus_token_freq=corpus_token_freq,
        kbbi_set=kbbi_set,
        n_corpus_tokens_from_corpus_slot=n_corpus_slot,
        n_kbbi_tokens_from_kbbi_slot=n_kbbi_slot,
    )
    if print_report:
        result.print_report()
    if strict:
        result.assert_valid(strict=True)
    return result
