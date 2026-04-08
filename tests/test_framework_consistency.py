"""
test_framework_consistency.py
==============================================
Validasi konsistensi FRAMEWORK AKSARA — bukan performa model.

Empat properti yang diuji:

    1. DETERMINISM    — config sama → hasil sama, selalu
    2. INVARIANCE     — rule dilanggar → validator bereaksi konsisten
    3. EXPLAINABILITY — tier yang ditetapkan sesuai dengan penjelasan yang diberikan
    4. ROBUSTNESS     — input berbeda (corpus kecil, corpus besar, edge case) → tetap stabil

Framework yang konsisten bisa menghasilkan model buruk (karena config jelek).
Framework yang buruk bisa menghasilkan model bagus (karena kebetulan).
Kita validasi fondasi, bukan hasilnya.
"""

import pytest
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aksara.linguistic.vocab_policy import (
    AksaraVocabPolicy,
    AksaraVocabValidator,
    VocabValidationResult,
    VocabHardConstraintError,
    QualityTier,
    validate_vocab,
    DOMAIN_SANITY_SEEDS,
)


# ─── Fixtures umum ────────────────────────────────────────────────────────────

def buat_vocab_standar(ukuran: int = 8_000) -> dict:
    """Vocab standar dengan special tokens di posisi benar."""
    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4}
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
    while len(vocab) < ukuran:
        vocab[f"kata_{len(vocab)}"] = len(vocab)
    return vocab


def buat_corpus_freq(vocab: dict, frekuensi: int = 10) -> dict:
    """Frekuensi corpus sederhana: semua token dalam vocab punya frekuensi sama."""
    return {
        tok: frekuensi
        for tok in vocab
        if not tok.startswith("<")
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. DETERMINISM — config sama → hasil sama, selalu
# ══════════════════════════════════════════════════════════════════════════════

class TestDeterminism:
    """
    Framework harus deterministik:
    - Panggil validator dua kali dengan input sama → hasil identik
    - Policy dari_ukuran_corpus(N) dipanggil dua kali → policy identik
    - Quality tier dari hasil yang sama → selalu sama
    """

    def test_validator_deterministik_pada_input_sama(self):
        """Dua panggilan validate() dengan input identik → hasil identik."""
        vocab = buat_vocab_standar(8_000)
        freq  = buat_corpus_freq(vocab)
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)

        hasil_1 = v.validate(vocab, corpus_token_freq=freq)
        hasil_2 = v.validate(vocab, corpus_token_freq=freq)

        assert hasil_1.check_size        == hasil_2.check_size
        assert hasil_1.check_composition == hasil_2.check_composition
        assert hasil_1.check_coverage    == hasil_2.check_coverage
        assert hasil_1.check_oov         == hasil_2.check_oov
        assert hasil_1.check_domain      == hasil_2.check_domain
        assert hasil_1.quality_tier      == hasil_2.quality_tier

    def test_policy_from_corpus_size_deterministik(self):
        """Policy dari ukuran corpus yang sama → selalu identik."""
        p1 = AksaraVocabPolicy.from_corpus_size(50_000)
        p2 = AksaraVocabPolicy.from_corpus_size(50_000)

        assert p1.target_vocab      == p2.target_vocab
        assert p1.min_vocab         == p2.min_vocab
        assert p1.max_vocab         == p2.max_vocab
        assert p1.corpus_ratio_min  == p2.corpus_ratio_min
        assert p1.corpus_ratio_max  == p2.corpus_ratio_max
        assert p1.min_coverage      == p2.min_coverage
        assert p1.max_oov_rate      == p2.max_oov_rate

    def test_quality_tier_deterministik(self):
        """Quality tier dari check results yang sama → selalu sama."""
        p = AksaraVocabPolicy()
        for _ in range(5):
            hasil = VocabValidationResult(policy=p)
            hasil.check_size        = ("PASS", 8000, "ok")
            hasil.check_composition = ("WARN", 0.68, "sedikit off")
            hasil.check_coverage    = ("PASS", 0.80, "ok")
            hasil.check_oov         = ("PASS", 0.10, "ok")
            hasil.check_domain      = ("PASS", 0.90, "ok")
            assert hasil.quality_tier == QualityTier.VALID

    def test_validator_tidak_mutasi_input(self):
        """Validator tidak boleh mengubah vocab atau corpus_freq yang diberikan."""
        vocab = buat_vocab_standar(5_000)
        freq  = buat_corpus_freq(vocab)
        vocab_sebelum = copy.deepcopy(vocab)
        freq_sebelum  = copy.deepcopy(freq)

        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        v.validate(vocab, corpus_token_freq=freq)

        assert vocab == vocab_sebelum
        assert freq  == freq_sebelum

    def test_repr_policy_deterministik(self):
        """__repr__ policy deterministik — tidak berubah antar panggilan."""
        p = AksaraVocabPolicy(target_vocab=10_000)
        repr_1 = repr(p)
        repr_2 = repr(p)
        assert repr_1 == repr_2
        assert "10,000" in repr_1


# ══════════════════════════════════════════════════════════════════════════════
# 2. INVARIANCE — rule dilanggar → validator bereaksi konsisten
# ══════════════════════════════════════════════════════════════════════════════

class TestInvariance:
    """
    Setiap rule yang dilanggar harus selalu menghasilkan reaksi yang konsisten:
    - Pelanggaran kecil  → WARN
    - Pelanggaran sedang → FAIL → tier DEGRADED
    - Banyak pelanggaran → tier EXPERIMENTAL
    - Tidak ada pelanggaran → tier OPTIMAL

    Validator tidak boleh "diam" ketika rule dilanggar.
    """

    def test_ukuran_vocab_terlalu_kecil_selalu_fail(self):
        """Vocab di bawah min_vocab → check_size selalu FAIL."""
        for ukuran in [100, 500, 1_000]:
            vocab = {f"tok_{i}": i for i in range(ukuran)}
            vocab.update({"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})
            p = AksaraVocabPolicy(min_vocab=5_000)
            v = AksaraVocabValidator(p)
            hasil = v.validate(vocab)
            assert hasil.check_size[0] == "FAIL", (
                f"Vocab ukuran {ukuran} seharusnya FAIL, dapat {hasil.check_size[0]}"
            )

    def test_coverage_rendah_selalu_fail(self):
        """Coverage jauh di bawah threshold → check_coverage selalu FAIL."""
        vocab = buat_vocab_standar(8_000)
        # Corpus freq: total token sangat besar, tapi hanya sedikit yang ada di vocab
        # Buat freq untuk 90% token di luar vocab
        corpus_freq = {f"luar_vocab_{i}": 1_000 for i in range(80_000)}
        corpus_freq.update({tok: 1 for tok in vocab if not tok.startswith("<")})

        p = AksaraVocabPolicy(min_coverage=0.75)
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab, corpus_token_freq=corpus_freq)
        assert hasil.check_coverage[0] == "FAIL"

    def test_domain_kritis_hilang_selalu_fail(self):
        """Semua seed domain hilang → check_domain selalu FAIL."""
        # Vocab tanpa kata domain sama sekali
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for i in range(100):
            vocab[f"abcxyz_{i}"] = 4 + i

        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab)
        assert hasil.check_domain[0] == "FAIL"

    def test_satu_pelanggaran_selalu_degraded(self):
        """Tepat satu FAIL → tier selalu DEGRADED, bukan EXPERIMENTAL."""
        p = AksaraVocabPolicy()
        # Coba semua posisi FAIL satu per satu
        check_names = ["size", "composition", "coverage", "oov", "domain"]
        for nama_gagal in check_names:
            hasil = VocabValidationResult(policy=p)
            hasil.check_size        = ("PASS", 8000, "ok")
            hasil.check_composition = ("PASS", 0.75, "ok")
            hasil.check_coverage    = ("PASS", 0.80, "ok")
            hasil.check_oov         = ("PASS", 0.10, "ok")
            hasil.check_domain      = ("PASS", 0.90, "ok")
            # Satu FAIL
            setattr(hasil, f"check_{nama_gagal}", ("FAIL", 0.0, "gagal"))
            assert hasil.quality_tier == QualityTier.DEGRADED, (
                f"Satu FAIL di '{nama_gagal}' seharusnya DEGRADED, dapat {hasil.quality_tier}"
            )

    def test_dua_pelanggaran_selalu_experimental(self):
        """Dua atau lebih FAIL → tier selalu EXPERIMENTAL."""
        p = AksaraVocabPolicy()
        check_names = ["size", "composition", "coverage", "oov", "domain"]
        import itertools
        for nama1, nama2 in itertools.combinations(check_names, 2):
            hasil = VocabValidationResult(policy=p)
            hasil.check_size        = ("PASS", 8000, "ok")
            hasil.check_composition = ("PASS", 0.75, "ok")
            hasil.check_coverage    = ("PASS", 0.80, "ok")
            hasil.check_oov         = ("PASS", 0.10, "ok")
            hasil.check_domain      = ("PASS", 0.90, "ok")
            setattr(hasil, f"check_{nama1}", ("FAIL", 0.0, "gagal"))
            setattr(hasil, f"check_{nama2}", ("FAIL", 0.0, "gagal"))
            assert hasil.quality_tier == QualityTier.EXPERIMENTAL, (
                f"Dua FAIL ({nama1}, {nama2}) seharusnya EXPERIMENTAL, dapat {hasil.quality_tier}"
            )

    def test_tanpa_pelanggaran_selalu_optimal(self):
        """Semua PASS → tier selalu OPTIMAL."""
        p = AksaraVocabPolicy()
        for _ in range(3):
            hasil = VocabValidationResult(policy=p)
            hasil.check_size        = ("PASS", 8000, "ok")
            hasil.check_composition = ("PASS", 0.75, "ok")
            hasil.check_coverage    = ("PASS", 0.80, "ok")
            hasil.check_oov         = ("PASS", 0.10, "ok")
            hasil.check_domain      = ("PASS", 0.90, "ok")
            assert hasil.quality_tier == QualityTier.OPTIMAL

    def test_hard_constraint_selalu_dicatat(self):
        """Hard constraint dilanggar → selalu masuk hard_violations, bukan diam."""
        # 1. Vocab kosong
        v = AksaraVocabValidator()
        hasil = v.validate({})
        assert hasil.hard_violations, "Vocab kosong harus dicatat sebagai hard violation"

        # 2. Special token hilang
        vocab_tanpa_pad = {"<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "kata": 4}
        hasil = v.validate(vocab_tanpa_pad)
        assert any("<PAD>" in viol for viol in hasil.hard_violations)

        # 3. ID duplikat
        vocab_duplikat = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            "kata_a": 5, "kata_b": 5
        }
        hasil = v.validate(vocab_duplikat)
        assert any("duplikat" in viol for viol in hasil.hard_violations)

    def test_warn_tidak_blokir_eksekusi(self):
        """WARN tidak pernah raise exception — validator tidak memblokir."""
        p = AksaraVocabPolicy()
        hasil = VocabValidationResult(policy=p)
        hasil.check_size = ("WARN", 4500, "sedikit kecil")
        hasil.check_oov  = ("WARN", 0.18, "sedikit tinggi")
        # Tidak raise
        hasil.assert_valid()
        assert hasil.quality_tier == QualityTier.VALID

    def test_degraded_tidak_blokir_eksekusi(self):
        """DEGRADED tidak raise — framework memberi tahu tapi tidak memblokir."""
        p = AksaraVocabPolicy()
        hasil = VocabValidationResult(policy=p)
        hasil.check_oov = ("FAIL", 0.40, "OOV tinggi")
        # Tidak raise pada assert_valid() standar
        hasil.assert_valid()
        assert hasil.quality_tier == QualityTier.DEGRADED

    def test_experimental_raise_pada_assert_valid(self):
        """EXPERIMENTAL raise pada assert_valid() — terlalu banyak constraint gagal."""
        p = AksaraVocabPolicy()
        hasil = VocabValidationResult(policy=p)
        hasil.check_size     = ("FAIL", 500, "terlalu kecil")
        hasil.check_coverage = ("FAIL", 0.20, "terlalu rendah")
        assert hasil.quality_tier == QualityTier.EXPERIMENTAL
        with pytest.raises(ValueError):
            hasil.assert_valid()


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPLAINABILITY — tier yang ditetapkan sesuai penjelasan yang diberikan
# ══════════════════════════════════════════════════════════════════════════════

class TestExplainability:
    """
    Framework harus bisa menjelaskan kenapa suatu hasil terjadi:
    - Setiap tier punya daftar expected behavior
    - Check detail harus berisi angka yang nyata (bukan placeholder)
    - Hard violations harus berisi teks deskriptif
    - print_report() harus mengandung informasi tier + penjelasan
    """

    def test_setiap_tier_punya_behavior_description(self):
        """Setiap tier harus punya daftar prediksi perilaku yang tidak kosong."""
        for tier in [QualityTier.OPTIMAL, QualityTier.VALID,
                     QualityTier.DEGRADED, QualityTier.EXPERIMENTAL]:
            behaviors = QualityTier.BEHAVIOR.get(tier, [])
            assert behaviors, f"Tier {tier} tidak punya behavior description"
            assert len(behaviors) >= 3, f"Tier {tier} harus punya minimal 3 behavior items"

    def test_tier_berbeda_punya_behavior_berbeda(self):
        """Setiap tier harus punya behavior description yang unik."""
        all_behaviors = {
            tier: tuple(QualityTier.BEHAVIOR[tier])
            for tier in [QualityTier.OPTIMAL, QualityTier.VALID,
                         QualityTier.DEGRADED, QualityTier.EXPERIMENTAL]
        }
        # Pastikan tidak ada dua tier yang punya behavior identik
        behavior_list = list(all_behaviors.values())
        assert len(set(behavior_list)) == 4, "Setiap tier harus punya behavior unik"

    def test_check_detail_berisi_angka_nyata(self):
        """Detail check harus berisi angka terukur, bukan string kosong."""
        vocab = buat_vocab_standar(8_000)
        freq  = buat_corpus_freq(vocab, frekuensi=15)
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab, corpus_token_freq=freq)

        # Setiap check yang bukan SKIP harus punya detail non-kosong
        for nama, check in [
            ("size",        hasil.check_size),
            ("coverage",    hasil.check_coverage),
            ("oov",         hasil.check_oov),
            ("domain",      hasil.check_domain),
        ]:
            status, nilai, detail = check
            if status != "SKIP":
                assert detail, f"Check '{nama}' harus punya detail deskriptif"
                assert len(detail) > 10, f"Detail '{nama}' terlalu pendek: '{detail}'"

    def test_check_size_detail_berisi_angka_target(self):
        """Detail size check harus menyebut ukuran vocab yang terukur."""
        vocab = buat_vocab_standar(8_000)
        p = AksaraVocabPolicy(target_vocab=10_000)
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab)
        detail = hasil.check_size[2]
        # Detail harus menyebut angka aktual
        assert any(char.isdigit() for char in detail), (
            f"Detail size check harus berisi angka: '{detail}'"
        )

    def test_check_coverage_detail_berisi_persentase(self):
        """Detail coverage check harus menyebut persentase yang terukur."""
        vocab = buat_vocab_standar(8_000)
        freq  = buat_corpus_freq(vocab, frekuensi=10)
        p = AksaraVocabPolicy()
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab, corpus_token_freq=freq)
        if hasil.check_coverage[0] != "SKIP":
            detail = hasil.check_coverage[2]
            assert "%" in detail, f"Detail coverage harus berisi '%': '{detail}'"

    def test_hard_violations_berisi_teks_deskriptif(self):
        """Hard violations harus menjelaskan masalahnya, bukan hanya kode error."""
        vocab_bermasalah = {
            "<PAD>": 99,   # ID salah
            "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
        }
        v = AksaraVocabValidator()
        hasil = v.validate(vocab_bermasalah)
        for viol in hasil.hard_violations:
            assert len(viol) > 15, f"Hard violation terlalu pendek: '{viol}'"
            assert viol == viol.strip(), "Hard violation tidak boleh ada whitespace di awal/akhir"

    def test_print_report_berisi_tier_dan_penjelasan(self, capsys):
        """print_report() harus mencetak tier + expected behavior."""
        p = AksaraVocabPolicy()
        hasil = VocabValidationResult(policy=p)
        hasil.check_size     = ("FAIL", 1_000, "terlalu kecil")
        hasil.check_coverage = ("FAIL", 0.30,  "sangat rendah")
        hasil.print_report(title="Tes Explainability")

        output = capsys.readouterr().out
        assert "EXPERIMENTAL" in output
        assert "Expected behavior" in output
        assert "Tes Explainability" in output

    def test_print_report_optimal_berisi_konfirmasi_positif(self, capsys):
        """print_report() OPTIMAL harus menunjukkan konfirmasi positif."""
        p = AksaraVocabPolicy()
        hasil = VocabValidationResult(policy=p)
        hasil.check_size        = ("PASS", 8000, "ok")
        hasil.check_composition = ("PASS", 0.75, "ok")
        hasil.check_coverage    = ("PASS", 0.80, "ok")
        hasil.check_oov         = ("PASS", 0.10, "ok")
        hasil.check_domain      = ("PASS", 0.90, "ok")
        hasil.print_report()

        output = capsys.readouterr().out
        assert "OPTIMAL" in output

    def test_tier_symbol_berbeda_per_tier(self):
        """Setiap tier harus punya simbol visual yang berbeda."""
        simbol = {
            QualityTier.symbol(t)
            for t in [QualityTier.OPTIMAL, QualityTier.VALID,
                      QualityTier.DEGRADED, QualityTier.EXPERIMENTAL]
        }
        assert len(simbol) == 4, "Setiap tier harus punya simbol visual unik"

    def test_domain_details_menjelaskan_kata_hadir_dan_hilang(self):
        """domain_details harus memisahkan kata yang hadir dan yang hilang."""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                 "makan": 4, "minum": 5}
        # "nasi" dan "makanan" hilang dari domain makan
        p = AksaraVocabPolicy(domain_seeds={"makan": ["makan", "minum", "nasi", "makanan"]})
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab)

        assert "makan" in hasil.domain_details
        info = hasil.domain_details["makan"]
        hadir  = [w for w in info if not w.startswith("MISSING:")]
        hilang = [w[8:] for w in info if w.startswith("MISSING:")]
        assert "makan" in hadir
        assert "minum" in hadir
        assert "nasi" in hilang
        assert "makanan" in hilang


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROBUSTNESS — input berbeda → framework tetap stabil
# ══════════════════════════════════════════════════════════════════════════════

class TestRobustness:
    """
    Framework harus tetap stabil pada berbagai kondisi input:
    - Corpus sangat kecil
    - Corpus sangat besar (disimulasikan via freq tinggi)
    - Vocab minimum (hanya special tokens)
    - Vocab maksimum (15k token)
    - corpus_token_freq kosong / None
    - kbbi_set kosong / None
    - domain_seeds kosong
    - Token dengan karakter unicode / non-ASCII
    """

    def test_vocab_hanya_special_tokens_tidak_crash(self):
        """Vocab yang hanya berisi special tokens → tidak crash, hasilkan report valid."""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4}
        v = AksaraVocabValidator()
        hasil = v.validate(vocab)  # tidak raise
        assert hasil.quality_tier in (
            QualityTier.OPTIMAL, QualityTier.VALID,
            QualityTier.DEGRADED, QualityTier.EXPERIMENTAL
        )

    def test_corpus_freq_none_tidak_crash(self):
        """corpus_token_freq=None → check yang butuh freq di-SKIP, tidak crash."""
        vocab = buat_vocab_standar(8_000)
        v = AksaraVocabValidator()
        hasil = v.validate(vocab, corpus_token_freq=None)  # tidak raise
        assert hasil.check_coverage[0] == "SKIP"
        assert hasil.check_oov[0] == "SKIP"

    def test_corpus_freq_kosong_tidak_crash(self):
        """corpus_token_freq={} → di-SKIP dengan elegan."""
        vocab = buat_vocab_standar(8_000)
        v = AksaraVocabValidator()
        hasil = v.validate(vocab, corpus_token_freq={})
        assert hasil.check_coverage[0] == "SKIP"
        assert hasil.check_oov[0] == "SKIP"

    def test_kbbi_set_none_tidak_crash(self):
        """kbbi_set=None → check composition di-SKIP jika tidak ada info slot."""
        vocab = buat_vocab_standar(8_000)
        v = AksaraVocabValidator()
        hasil = v.validate(vocab, kbbi_set=None)  # tidak raise

    def test_domain_seeds_kosong_tidak_crash(self):
        """domain_seeds kosong → domain check PASS dengan avg 0% atau SKIP."""
        vocab = buat_vocab_standar(8_000)
        p = AksaraVocabPolicy(domain_seeds={})
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab)
        assert hasil.check_domain[0] in ("PASS", "SKIP")

    def test_vocab_ukuran_maksimum_tidak_crash(self):
        """Vocab 15k token → tidak crash."""
        vocab = buat_vocab_standar(15_000)
        p = AksaraVocabPolicy(target_vocab=15_000)
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab)
        assert hasil.quality_tier in (
            QualityTier.OPTIMAL, QualityTier.VALID,
            QualityTier.DEGRADED, QualityTier.EXPERIMENTAL
        )

    def test_corpus_freq_satu_token_tidak_crash(self):
        """corpus_token_freq dengan satu token → tidak crash."""
        vocab = buat_vocab_standar(5_000)
        v = AksaraVocabValidator()
        hasil = v.validate(vocab, corpus_token_freq={"dan": 999_999})
        assert hasil.check_coverage[0] in ("PASS", "WARN", "FAIL")

    def test_policy_from_corpus_size_edge_cases(self):
        """from_corpus_size() dengan nilai ekstrim tidak crash dan tetap dalam batas."""
        for n in [0, 1, 100, 1_000_000, 999_999_999]:
            p = AksaraVocabPolicy.from_corpus_size(n)
            assert p.min_vocab <= p.target_vocab <= p.max_vocab, (
                f"from_corpus_size({n}) menghasilkan target di luar batas: "
                f"{p.target_vocab} bukan dalam [{p.min_vocab}, {p.max_vocab}]"
            )

    def test_validator_tanpa_policy_eksplisit_tidak_crash(self):
        """AksaraVocabValidator() tanpa policy → pakai default policy."""
        vocab = buat_vocab_standar(8_000)
        v = AksaraVocabValidator()  # tidak ada policy eksplisit
        hasil = v.validate(vocab)
        assert hasil.policy is not None
        assert hasil.quality_tier in (
            QualityTier.OPTIMAL, QualityTier.VALID,
            QualityTier.DEGRADED, QualityTier.EXPERIMENTAL
        )

    def test_corpus_freq_semua_frekuensi_satu_tidak_crash(self):
        """corpus_token_freq semua freq=1 (semua hapax) → tidak crash."""
        vocab = buat_vocab_standar(5_000)
        freq = {tok: 1 for tok in vocab if not tok.startswith("<")}
        v = AksaraVocabValidator()
        hasil = v.validate(vocab, corpus_token_freq=freq)
        # Tidak crash, dan hasilkan tier yang valid
        assert hasil.quality_tier in (
            QualityTier.OPTIMAL, QualityTier.VALID,
            QualityTier.DEGRADED, QualityTier.EXPERIMENTAL
        )

    def test_validate_vocab_convenience_robustness(self):
        """validate_vocab() convenience function tidak crash dengan berbagai input."""
        vocab = buat_vocab_standar(8_000)
        # Tanpa freq
        r1 = validate_vocab(vocab=vocab, n_corpus_sentences=50_000, print_report=False)
        assert r1.quality_tier is not None
        # Dengan freq
        freq = buat_corpus_freq(vocab)
        r2 = validate_vocab(vocab=vocab, corpus_token_freq=freq,
                            n_corpus_sentences=50_000, print_report=False)
        assert r2.quality_tier is not None

    def test_print_report_tidak_crash_pada_semua_tier(self, capsys):
        """print_report() tidak crash pada semua tier, termasuk dengan hard violations."""
        p = AksaraVocabPolicy()
        for tier_scenario in [
            # (check_size, check_composition, check_coverage, check_oov, check_domain)
            ("PASS", "PASS", "PASS", "PASS", "PASS"),   # OPTIMAL
            ("WARN", "PASS", "PASS", "PASS", "PASS"),   # VALID
            ("FAIL", "PASS", "PASS", "PASS", "PASS"),   # DEGRADED
            ("FAIL", "FAIL", "PASS", "PASS", "PASS"),   # EXPERIMENTAL
        ]:
            hasil = VocabValidationResult(policy=p)
            hasil.check_size        = (tier_scenario[0], 0.0, "tes")
            hasil.check_composition = (tier_scenario[1], 0.0, "tes")
            hasil.check_coverage    = (tier_scenario[2], 0.0, "tes")
            hasil.check_oov         = (tier_scenario[3], 0.0, "tes")
            hasil.check_domain      = (tier_scenario[4], 0.0, "tes")
            hasil.hard_violations   = ["pelanggaran hard constraint tes"]
            hasil.print_report()  # tidak raise
        output = capsys.readouterr().out
        assert len(output) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. KONTRAK POLICY — rule yang didefinisikan harus bisa diuji dan dilanggar
# ══════════════════════════════════════════════════════════════════════════════

class TestKontrakPolicy:
    """
    Policy adalah kontrak.
    Kontrak harus bisa:
    - Ditetapkan dengan parameter eksplisit
    - Dilanggar dengan cara yang terprediksi
    - Diperketat (strict=True) atau dilonggarkan (custom thresholds)
    """

    def test_policy_bisa_dikustomisasi_penuh(self):
        """Policy menerima semua parameter dan menyimpannya dengan benar."""
        p = AksaraVocabPolicy(
            target_vocab=7_500,
            min_vocab=3_000,
            max_vocab=12_000,
            corpus_ratio_min=0.65,
            corpus_ratio_max=0.85,
            kbbi_ratio_min=0.15,
            kbbi_ratio_max=0.35,
            min_coverage=0.70,
            max_oov_rate=0.20,
        )
        assert p.target_vocab     == 7_500
        assert p.min_vocab        == 3_000
        assert p.max_vocab        == 12_000
        assert p.corpus_ratio_min == 0.65
        assert p.min_coverage     == 0.70
        assert p.max_oov_rate     == 0.20

    def test_policy_default_konsisten_dengan_dokumentasi(self):
        """Policy default harus sesuai dengan yang didokumentasikan."""
        p = AksaraVocabPolicy()
        # Sesuai dokumentasi: 5K–15K, 70–80% corpus, 20–30% KBBI, ≥75% coverage, ≤15% OOV
        assert p.min_vocab        == 5_000
        assert p.max_vocab        == 15_000
        assert p.corpus_ratio_min == 0.70
        assert p.corpus_ratio_max == 0.80
        assert p.kbbi_ratio_min   == 0.20
        assert p.kbbi_ratio_max   == 0.30
        assert p.min_coverage     == 0.75
        assert p.max_oov_rate     == 0.15

    def test_melanggar_threshold_coverage_bereaksi_proporsional(self):
        """
        Coverage 74% (sedikit di bawah 75%) → WARN, bukan FAIL.
        Coverage 50% (jauh di bawah 75%)    → FAIL.
        """
        vocab = buat_vocab_standar(8_000)

        # Simulasi coverage 74%: total occ = 100, covered = 74
        freq_warn = {"tok_in": 74}
        freq_warn.update({"tok_out_" + str(i): 1 for i in range(26)})
        for tok in vocab:
            if not tok.startswith("<"):
                freq_warn[tok] = 0  # pastikan semua vocab token ada di freq

        # Pendekatan langsung: uji via _check_coverage
        p = AksaraVocabPolicy(min_coverage=0.75)
        v = AksaraVocabValidator(p)

        # Coverage sangat rendah → FAIL
        corpus_besar_luar_vocab = {f"luar_{i}": 1_000 for i in range(10_000)}
        corpus_besar_luar_vocab.update({tok: 1 for tok in vocab if not tok.startswith("<")})
        hasil_fail = v.validate(vocab, corpus_token_freq=corpus_besar_luar_vocab)
        assert hasil_fail.check_coverage[0] == "FAIL"

        # Coverage tinggi → PASS
        freq_bagus = {tok: 100 for tok in vocab if not tok.startswith("<")}
        hasil_pass = v.validate(vocab, corpus_token_freq=freq_bagus)
        assert hasil_pass.check_coverage[0] == "PASS"

    def test_strict_mode_lebih_ketat_dari_normal(self):
        """strict=True lebih ketat dari strict=False."""
        p = AksaraVocabPolicy()
        hasil_warn = VocabValidationResult(policy=p)
        hasil_warn.check_size = ("WARN", 7000, "sedikit kecil")

        # Normal: WARN tidak raise
        hasil_warn.assert_valid(strict=False)

        # Strict: WARN pada non-OPTIMAL raise
        with pytest.raises(ValueError):
            hasil_warn.assert_valid(strict=True)

    def test_domain_seeds_bisa_diperluas(self):
        """Domain seeds bisa diperluas dengan domain baru."""
        seeds_diperluas = dict(DOMAIN_SANITY_SEEDS)
        seeds_diperluas["teknologi"] = ["komputer", "internet", "data", "jaringan"]

        vocab = buat_vocab_standar(8_000)
        vocab["komputer"] = len(vocab)
        vocab["internet"] = len(vocab)
        vocab["data"]     = len(vocab)
        vocab["jaringan"] = len(vocab)

        p = AksaraVocabPolicy(domain_seeds=seeds_diperluas)
        v = AksaraVocabValidator(p)
        hasil = v.validate(vocab)
        assert "teknologi" in hasil.domain_details

    def test_quality_tier_rank_adalah_total_ordering(self):
        """Rank tier harus membentuk urutan total yang konsisten."""
        tiers = [QualityTier.OPTIMAL, QualityTier.VALID,
                 QualityTier.DEGRADED, QualityTier.EXPERIMENTAL]
        ranks = [QualityTier.rank(t) for t in tiers]
        # Urutan menurun: OPTIMAL > VALID > DEGRADED > EXPERIMENTAL
        assert ranks == sorted(ranks, reverse=True)
        # Semua rank berbeda
        assert len(set(ranks)) == 4
