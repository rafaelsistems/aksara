"""
test_vocab_policy.py — Unit tests untuk AksaraVocabPolicy dan AksaraVocabValidator.
"""
import pytest
from aksara.linguistic.vocab_policy import (
    AksaraVocabPolicy,
    AksaraVocabValidator,
    VocabValidationResult,
    VocabHardConstraintError,
    QualityTier,
    validate_vocab,
    DOMAIN_SANITY_SEEDS,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

def make_good_vocab(size: int = 8000) -> dict:
    """Vocab sehat: special tokens + kata-kata domain."""
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4}
    # Tambahkan domain words wajib
    domain_words = [
        "makan", "minum", "nasi", "makanan",
        "baca", "buku", "kata", "teks",
        "jalan", "kaki", "gerak", "pergi",
        "kerja", "kantor", "tugas", "laporan",
        "pemerintah", "hukum", "negara", "kebijakan",
        "sekolah", "belajar", "guru", "ilmu",
        "sejarah", "tahun", "masa", "perang",
        "penelitian", "data", "hasil",
    ]
    for w in domain_words:
        if w not in vocab:
            vocab[w] = len(vocab)
    # Isi sisanya dengan kata generik
    i = 0
    while len(vocab) < size:
        vocab[f"kata_{i}"] = len(vocab)
        i += 1
    return vocab


def make_corpus_freq(vocab: dict, total_tokens: int = 500_000) -> dict:
    """Simulasi corpus freq — semua kata dalam vocab punya frekuensi."""
    freq = {}
    real_tokens = [t for t in vocab if not t.startswith("<")]
    per_token = total_tokens // max(len(real_tokens), 1)
    for tok in real_tokens:
        freq[tok] = per_token
    return freq


# ─── AksaraVocabPolicy tests ───────────────────────────────────────────────────

class TestAksaraVocabPolicy:

    def test_default_policy(self):
        p = AksaraVocabPolicy()
        assert p.min_vocab == 5_000
        assert p.max_vocab == 15_000
        assert p.target_vocab == 10_000
        assert p.corpus_ratio_min == 0.70
        assert p.corpus_ratio_max == 0.80
        assert p.kbbi_ratio_min == 0.20
        assert p.kbbi_ratio_max == 0.30
        assert p.min_coverage == 0.75
        assert p.max_oov_rate == 0.15

    def test_from_corpus_size_small(self):
        p = AksaraVocabPolicy.from_corpus_size(10_000)
        assert p.min_vocab <= p.target_vocab <= p.max_vocab
        assert p.target_vocab <= 8_000  # corpus kecil → vocab kecil

    def test_from_corpus_size_medium(self):
        # Heap's Law: k=50, beta=0.5 → 50*√100000 ≈ 15811 → clamps to max 15000
        p = AksaraVocabPolicy.from_corpus_size(100_000)
        assert p.min_vocab <= p.target_vocab <= p.max_vocab

    def test_from_corpus_size_large(self):
        p = AksaraVocabPolicy.from_corpus_size(500_000)
        assert p.target_vocab == p.max_vocab  # saturasi di max

    def test_from_corpus_size_clamped(self):
        p_small = AksaraVocabPolicy.from_corpus_size(100)
        assert p_small.target_vocab >= p_small.min_vocab
        p_huge = AksaraVocabPolicy.from_corpus_size(10_000_000)
        assert p_huge.target_vocab <= p_huge.max_vocab

    def test_corpus_slots(self):
        p = AksaraVocabPolicy(target_vocab=10_000)
        slots = p.corpus_slots()
        assert 7_000 <= slots <= 8_000  # 70-80% dari 10k

    def test_kbbi_slots(self):
        p = AksaraVocabPolicy(target_vocab=10_000)
        assert p.corpus_slots() + p.kbbi_slots() == p.target_vocab

    def test_repr(self):
        p = AksaraVocabPolicy()
        r = repr(p)
        assert "AksaraVocabPolicy" in r
        assert "target=" in r


# ─── AksaraVocabValidator — size check ────────────────────────────────────────

class TestSizeCheck:

    def test_size_pass(self):
        p = AksaraVocabPolicy(min_vocab=5_000, max_vocab=15_000, target_vocab=10_000)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(10_000)
        result = v.validate(vocab)
        assert result.check_size[0] == "PASS"

    def test_size_warn_off_target(self):
        p = AksaraVocabPolicy(min_vocab=5_000, max_vocab=15_000, target_vocab=10_000)
        v = AksaraVocabValidator(p)
        # 7000 is in range but far from target 10000
        vocab = make_good_vocab(7_000)
        result = v.validate(vocab)
        assert result.check_size[0] in ("PASS", "WARN")

    def test_size_fail_too_small(self):
        p = AksaraVocabPolicy(min_vocab=5_000, max_vocab=15_000, target_vocab=10_000,
                              corpus_n_sentences=100_000)  # corpus besar → tidak toleran
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(3_000)
        result = v.validate(vocab)
        assert result.check_size[0] == "FAIL"

    def test_size_warn_small_corpus(self):
        """Corpus kecil → ukuran vocab kecil diterima (WARN bukan FAIL)."""
        p = AksaraVocabPolicy(min_vocab=5_000, max_vocab=15_000, target_vocab=10_000,
                              corpus_n_sentences=5_000)  # corpus kecil
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(3_500)
        result = v.validate(vocab)
        assert result.check_size[0] == "WARN"  # bukan FAIL

    def test_size_warn_too_large(self):
        p = AksaraVocabPolicy(min_vocab=5_000, max_vocab=15_000, target_vocab=10_000)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(16_000)
        result = v.validate(vocab)
        assert result.check_size[0] == "WARN"


# ─── AksaraVocabValidator — composition check ─────────────────────────────────

class TestCompositionCheck:

    def test_composition_pass_with_slots(self):
        p = AksaraVocabPolicy(target_vocab=10_000)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(10_000)
        # 75% corpus, 25% kbbi
        result = v.validate(vocab,
                            n_corpus_tokens_from_corpus_slot=7_500,
                            n_kbbi_tokens_from_kbbi_slot=2_495)
        assert result.check_composition[0] == "PASS"

    def test_composition_warn_off_ratio(self):
        p = AksaraVocabPolicy(target_vocab=10_000)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(10_000)
        # 50% corpus, 50% kbbi — di luar range
        result = v.validate(vocab,
                            n_corpus_tokens_from_corpus_slot=5_000,
                            n_kbbi_tokens_from_kbbi_slot=5_000)
        assert result.check_composition[0] == "WARN"

    def test_composition_skip_no_info(self):
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(8_000)
        result = v.validate(vocab)  # tidak ada kbbi_set, n_corpus, n_kbbi
        assert result.check_composition[0] == "SKIP"


# ─── AksaraVocabValidator — coverage check ────────────────────────────────────

class TestCoverageCheck:

    def test_coverage_pass(self):
        p = AksaraVocabPolicy(min_coverage=0.75)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(8_000)
        # Corpus freq: semua kata ada di vocab
        corpus_freq = make_corpus_freq(vocab)
        result = v.validate(vocab, corpus_token_freq=corpus_freq)
        assert result.check_coverage[0] == "PASS"
        assert result.check_coverage[1] >= 0.75

    def test_coverage_fail(self):
        p = AksaraVocabPolicy(min_coverage=0.75)
        v = AksaraVocabValidator(p)
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        # Hanya 4 token di vocab tapi corpus freq punya banyak kata
        corpus_freq = {f"kata_{i}": 100 for i in range(1000)}
        result = v.validate(vocab, corpus_token_freq=corpus_freq)
        assert result.check_coverage[0] == "FAIL"

    def test_coverage_skip_no_freq(self):
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(8_000)
        result = v.validate(vocab)  # tidak ada corpus_freq
        assert result.check_coverage[0] == "SKIP"


# ─── AksaraVocabValidator — OOV check ─────────────────────────────────────────

class TestOOVCheck:

    def test_oov_pass(self):
        """Semua significant tokens ada di vocab → PASS."""
        p = AksaraVocabPolicy(max_oov_rate=0.15)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(8_000)
        # Corpus freq: semua kata dalam vocab punya frekuensi ≥5 (significant)
        corpus_freq = {tok: 10 for tok in vocab if not tok.startswith("<")}
        result = v.validate(vocab, corpus_token_freq=corpus_freq)
        assert result.check_oov[0] == "PASS"
        assert result.check_oov[1] <= 0.15

    def test_oov_fail(self):
        """Banyak significant tokens tidak ada di vocab → FAIL.

        500 token OOV dengan freq=10 (significant), hanya 10 in-vocab.
        OOV rate = 500/510 >> 15%.
        """
        p = AksaraVocabPolicy(max_oov_rate=0.15)
        v = AksaraVocabValidator(p)
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for i in range(10):
            vocab[f"in_vocab_{i}"] = 4 + i
        # 500 significant OOV tokens (freq=10), 10 in-vocab (freq=10)
        corpus_freq = {f"oov_{i}": 10 for i in range(500)}
        corpus_freq.update({f"in_vocab_{i}": 10 for i in range(10)})
        result = v.validate(vocab, corpus_token_freq=corpus_freq)
        assert result.check_oov[0] == "FAIL"
        assert result.check_oov[1] > 0.15

    def test_oov_ignores_hapax(self):
        """Hapax (freq=1) tidak masuk significant → tidak mempengaruhi OOV rate."""
        p = AksaraVocabPolicy(max_oov_rate=0.15)
        v = AksaraVocabValidator(p)
        vocab = make_good_vocab(8_000)
        real_vocab_tokens = [t for t in vocab if not t.startswith("<")]
        # Semua vocab tokens: frekuensi tinggi (100) → significant, in vocab
        corpus_freq = {tok: 100 for tok in real_vocab_tokens}
        # Tambah 50k hapax OOV (freq=1) → di bawah threshold, tidak dihitung
        corpus_freq.update({f"hapax_{i}": 1 for i in range(50_000)})
        result = v.validate(vocab, corpus_token_freq=corpus_freq)
        # Hapax tidak masuk significant → OOV rate = 0% dari significant → PASS
        assert result.check_oov[0] == "PASS"
        assert result.check_oov[1] <= 0.15


# ─── AksaraVocabValidator — domain sanity check ───────────────────────────────

class TestDomainSanityCheck:

    def test_domain_pass(self):
        vocab = make_good_vocab(8_000)  # mengandung semua domain words
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert result.check_domain[0] == "PASS"
        assert result.check_domain[1] >= 0.5

    def test_domain_fail_missing_critical(self):
        """Hapus semua kata dari domain 'pemerintah' → FAIL."""
        vocab = make_good_vocab(8_000)
        # Remove domain words for pemerintah
        for w in ["pemerintah", "hukum", "negara", "kebijakan"]:
            vocab.pop(w, None)
        # Juga hapus dari kata lain di domain itu
        p = AksaraVocabPolicy(domain_seeds={"pemerintah": ["pemerintah", "hukum", "negara", "kebijakan"]})
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        # Score pemerintah = 0/4 = 0% → FAIL
        assert result.check_domain[0] == "FAIL"

    def test_domain_warn_partial(self):
        """Domain 25-50% coverage → WARN."""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        # Hanya 1 dari 4 kata domain ada
        vocab["makan"] = 4
        p = AksaraVocabPolicy(domain_seeds={"makan": ["makan", "nasi", "minum", "makanan"]})
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert result.check_domain[0] in ("WARN", "FAIL")

    def test_domain_details_populated(self):
        vocab = make_good_vocab(8_000)
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert len(result.domain_details) > 0
        assert "makan" in result.domain_details

    def test_domain_seeds_customizable(self):
        """Policy domain seeds bisa dikustomisasi."""
        custom_seeds = {"teknologi": ["komputer", "internet", "data"]}
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                 "komputer": 4, "internet": 5, "data": 6}
        p = AksaraVocabPolicy(domain_seeds=custom_seeds)
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert result.check_domain[0] == "PASS"


# ─── VocabValidationResult ────────────────────────────────────────────────────

class TestVocabValidationResult:

    def test_quality_tier_optimal(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("PASS", 8000, "ok")
        result.check_composition = ("PASS", 0.75, "ok")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        assert result.quality_tier == QualityTier.OPTIMAL
        assert result.all_pass is True
        assert result.passed is True

    def test_quality_tier_valid_with_warn(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("PASS", 8000, "ok")
        result.check_composition = ("WARN", 0.72, "slightly off")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        assert result.quality_tier == QualityTier.VALID
        assert result.all_pass is False
        assert result.passed is True

    def test_quality_tier_degraded_one_fail(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("FAIL", 3000, "too small")
        result.check_composition = ("PASS", 0.75, "ok")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        assert result.quality_tier == QualityTier.DEGRADED
        assert result.passed is False

    def test_quality_tier_experimental_two_fails(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("FAIL", 3000, "too small")
        result.check_composition = ("PASS", 0.75, "ok")
        result.check_coverage    = ("FAIL", 0.40, "low coverage")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        assert result.quality_tier == QualityTier.EXPERIMENTAL
        assert result.passed is False

    def test_assert_valid_passes_on_optimal(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("PASS", 8000, "ok")
        result.check_composition = ("PASS", 0.75, "ok")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        result.assert_valid()  # tidak raise

    def test_assert_valid_does_not_raise_on_degraded(self):
        """DEGRADED tidak raise pada assert_valid() — model tetap bisa jalan."""
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("FAIL", 3000, "too small")
        result.check_composition = ("PASS", 0.75, "ok")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        result.assert_valid()  # DEGRADED tidak raise — hanya EXPERIMENTAL raise

    def test_assert_valid_raises_on_experimental(self):
        """EXPERIMENTAL raise pada assert_valid() — terlalu banyak constraint gagal."""
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("FAIL", 3000, "too small")
        result.check_composition = ("FAIL", 0.50, "off")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        with pytest.raises(ValueError):
            result.assert_valid()

    def test_assert_valid_strict_raises_on_non_optimal(self):
        """strict=True raises pada VALID (ada WARN)."""
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("WARN", 7000, "slightly off")
        result.check_composition = ("PASS", 0.75, "ok")
        result.check_coverage    = ("PASS", 0.80, "ok")
        result.check_oov         = ("PASS", 0.10, "ok")
        result.check_domain      = ("PASS", 0.90, "ok")
        with pytest.raises(ValueError):
            result.assert_valid(strict=True)

    def test_hard_violations_recorded(self):
        """Hard violations dicatat di result.hard_violations."""
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.hard_violations = ["vocab kosong"]
        assert len(result.hard_violations) == 1

    def test_assert_hard_constraints_raises(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.hard_violations = ["special token '<PAD>' tidak ada"]
        with pytest.raises(VocabHardConstraintError):
            result.assert_hard_constraints()

    def test_assert_hard_constraints_passes_when_clean(self):
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.assert_hard_constraints()  # tidak raise

    def test_quality_tier_rank_ordering(self):
        assert QualityTier.rank(QualityTier.OPTIMAL)      > QualityTier.rank(QualityTier.VALID)
        assert QualityTier.rank(QualityTier.VALID)        > QualityTier.rank(QualityTier.DEGRADED)
        assert QualityTier.rank(QualityTier.DEGRADED)     > QualityTier.rank(QualityTier.EXPERIMENTAL)

    def test_print_report_does_not_raise(self, capsys):
        """print_report() tidak pernah raise — bahkan untuk EXPERIMENTAL."""
        p = AksaraVocabPolicy()
        result = VocabValidationResult(policy=p)
        result.check_size        = ("FAIL", 1000, "too small")
        result.check_composition = ("FAIL", 0.50, "bad")
        result.check_coverage    = ("FAIL", 0.30, "low")
        result.check_oov         = ("FAIL", 0.60, "high")
        result.check_domain      = ("FAIL", 0.10, "missing")
        result.hard_violations   = ["no special tokens"]
        result.print_report()  # tidak raise
        captured = capsys.readouterr()
        assert "EXPERIMENTAL" in captured.out
        assert "HARD CONSTRAINT" in captured.out


# ─── Hard constraint detection via validator ──────────────────────────────────

class TestHardConstraintDetection:

    def test_empty_vocab_is_hard_violation(self):
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate({})
        assert any("kosong" in viol for viol in result.hard_violations)

    def test_missing_pad_token_is_hard_violation(self):
        vocab = {"<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "kata": 4}
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert any("<PAD>" in viol for viol in result.hard_violations)

    def test_wrong_pad_id_is_hard_violation(self):
        vocab = {"<PAD>": 99, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "kata": 4}
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert any("<PAD>" in viol for viol in result.hard_violations)

    def test_duplicate_ids_is_hard_violation(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                 "kata_a": 5, "kata_b": 5}  # ID 5 duplikat
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert any("duplikat" in viol for viol in result.hard_violations)

    def test_good_vocab_no_hard_violations(self):
        vocab = make_good_vocab(8_000)
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)
        assert result.hard_violations == []

    def test_hard_violation_does_not_block_soft_checks(self):
        """Hard violations dicatat tapi soft checks tetap jalan."""
        vocab = {"<PAD>": 99, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        result = v.validate(vocab)  # tidak raise
        assert result.hard_violations  # ada hard violation
        # Soft check size masih dijalankan
        assert result.check_size[0] != "SKIP"


# ─── validate_vocab convenience function ──────────────────────────────────────

class TestValidateVocabConvenience:

    def test_basic_call(self):
        vocab = make_good_vocab(8_000)
        result = validate_vocab(
            vocab=vocab,
            n_corpus_sentences=50_000,
            print_report=False,
        )
        assert isinstance(result, VocabValidationResult)
        assert result.quality_tier in (QualityTier.OPTIMAL, QualityTier.VALID,
                                       QualityTier.DEGRADED, QualityTier.EXPERIMENTAL)

    def test_with_freq(self):
        vocab = make_good_vocab(8_000)
        freq = make_corpus_freq(vocab)
        result = validate_vocab(
            vocab=vocab,
            corpus_token_freq=freq,
            n_corpus_sentences=50_000,
            print_report=False,
        )
        assert result.check_coverage[0] in ("PASS", "WARN")
        assert result.check_oov[0] in ("PASS", "WARN")

    def test_experimental_raises_assert_valid(self):
        """EXPERIMENTAL vocab raises assert_valid() karena terlalu banyak constraint gagal."""
        # Vocab sangat kecil + tidak ada domain words = banyak FAIL
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        # Corpus freq dengan banyak token yang OOV
        freq = {f"kata_{i}": 100 for i in range(5000)}
        result = validate_vocab(
            vocab=vocab,
            corpus_token_freq=freq,
            n_corpus_sentences=100_000,
            print_report=False,
        )
        assert result.quality_tier == QualityTier.EXPERIMENTAL
        with pytest.raises(ValueError):
            result.assert_valid()


# ─── DOMAIN_SANITY_SEEDS sanity ───────────────────────────────────────────────

class TestDomainSanitySeeds:

    def test_seeds_not_empty(self):
        assert len(DOMAIN_SANITY_SEEDS) >= 5

    def test_all_seeds_lowercase(self):
        for domain, seeds in DOMAIN_SANITY_SEEDS.items():
            for word in seeds:
                assert word == word.lower(), f"Seed '{word}' in '{domain}' is not lowercase"

    def test_all_seeds_min_length(self):
        for domain, seeds in DOMAIN_SANITY_SEEDS.items():
            assert len(seeds) >= 2, f"Domain '{domain}' has fewer than 2 seeds"
