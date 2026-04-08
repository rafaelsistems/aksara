"""
test_dataset_validator.py — Unit test untuk AksaraDatasetValidator.

Test ini domain-agnostic: tidak ada asumsi konten, hanya validasi struktur.
"""

import json
import tempfile
from pathlib import Path

import pytest

from aksara.data.validator import (
    AksaraDatasetValidator,
    DatasetQualityScore,
    ValidationIssue,
    ValidationReport,
    validate_file,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_jsonl(lines: list, encoding: str = "utf-8") -> Path:
    """Tulis list of objects/strings ke file JSONL temporer."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding=encoding,
        delete=False, newline="\n"
    )
    for line in lines:
        if isinstance(line, str):
            f.write(line + "\n")
        else:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    f.close()
    return Path(f.name)


def v(**kwargs) -> AksaraDatasetValidator:
    """Buat validator dengan default + overrides."""
    defaults = dict(min_tokens=3, max_tokens=40,
                    max_dup_rate=0.20, max_dom_rate=0.50, strict=False)
    defaults.update(kwargs)
    return AksaraDatasetValidator(**defaults)


# ════════════════════════════════════════════════════════════════
# 1. STRUKTUR DASAR
# ════════════════════════════════════════════════════════════════

class TestStructure:

    def test_valid_minimal(self):
        """Kalimat valid minimal: field text ada, panjang cukup."""
        path = make_jsonl([
            {"text": "saya makan nasi di rumah"},
            {"text": "dia membaca buku di perpustakaan"},
            {"text": "mereka bekerja keras setiap hari"},
        ])
        r = v().validate(path)
        assert r.valid_samples == 3
        assert r.passed

    def test_missing_text_field(self):
        """Field text tidak ada → error MISSING_TEXT."""
        path = make_jsonl([{"content": "tanpa field text"}])
        r = v().validate(path)
        codes = [i.code for i in r.errors]
        assert "MISSING_TEXT" in codes
        assert r.valid_samples == 0

    def test_wrong_type_text(self):
        """Field text bukan string → error WRONG_TYPE."""
        path = make_jsonl([{"text": 12345}])
        r = v().validate(path)
        codes = [i.code for i in r.errors]
        assert "WRONG_TYPE" in codes

    def test_empty_text(self):
        """Field text kosong → error EMPTY_TEXT."""
        path = make_jsonl([{"text": "   "}])
        r = v().validate(path)
        codes = [i.code for i in r.errors]
        assert "EMPTY_TEXT" in codes

    def test_invalid_json_line(self):
        """Baris bukan JSON valid → error INVALID_JSON."""
        path = make_jsonl(["{tidak valid json"])
        r = v().validate(path)
        codes = [i.code for i in r.errors]
        assert "INVALID_JSON" in codes

    def test_blank_lines_skipped(self):
        """Baris kosong di-skip tanpa error."""
        path = make_jsonl([
            {"text": "kalimat pertama ini valid sekali"},
            "",
            {"text": "kalimat ketiga ini juga valid"},
        ])
        r = v().validate(path)
        assert r.valid_samples == 2
        assert r.passed

    def test_mixed_valid_invalid(self):
        """Campuran valid dan invalid — hitung masing-masing."""
        path = make_jsonl([
            {"text": "kalimat valid pertama ini bagus"},
            {"no_text": "tidak ada field text"},
            {"text": "kalimat valid ketiga ini juga bagus"},
        ])
        r = v().validate(path)
        assert r.valid_samples == 2
        assert len(r.errors) == 1


# ════════════════════════════════════════════════════════════════
# 2. PANJANG TOKEN
# ════════════════════════════════════════════════════════════════

class TestLength:

    def test_too_short_warning(self):
        """Kalimat terlalu pendek → warning TOO_SHORT (non-strict)."""
        path = make_jsonl([{"text": "ya"}])
        r = v(min_tokens=3).validate(path)
        codes = [i.code for i in r.warnings]
        assert "TOO_SHORT" in codes
        # Non-strict: valid_samples tetap dihitung
        assert r.valid_samples == 1

    def test_too_short_error_in_strict(self):
        """Strict mode: terlalu pendek → error."""
        path = make_jsonl([{"text": "ya tidak"}])
        r = v(min_tokens=3, strict=True).validate(path)
        codes = [i.code for i in r.errors]
        assert "TOO_SHORT" in codes

    def test_too_long_warning(self):
        """Kalimat terlalu panjang → warning TOO_LONG."""
        long_text = " ".join(["kata"] * 50)
        path = make_jsonl([{"text": long_text}])
        r = v(max_tokens=40).validate(path)
        codes = [i.code for i in r.warnings]
        assert "TOO_LONG" in codes

    def test_exactly_min(self):
        """Tepat di batas minimum → tidak ada issue panjang."""
        path = make_jsonl([{"text": "satu dua tiga"}])  # 3 token
        r = v(min_tokens=3).validate(path)
        length_issues = [i for i in r.issues if "SHORT" in i.code or "LONG" in i.code]
        assert len(length_issues) == 0

    def test_exactly_max(self):
        """Tepat di batas maximum → tidak ada issue panjang."""
        text = " ".join(["kata"] * 40)
        path = make_jsonl([{"text": text}])
        r = v(max_tokens=40).validate(path)
        length_issues = [i for i in r.issues if "SHORT" in i.code or "LONG" in i.code]
        assert len(length_issues) == 0


# ════════════════════════════════════════════════════════════════
# 3. NOISE DETECTION
# ════════════════════════════════════════════════════════════════

class TestNoise:

    def test_url_warning(self):
        """Kalimat dengan URL → warning CONTAINS_URL."""
        path = make_jsonl([{"text": "kunjungi website kami di https://example.com sekarang"}])
        r = v().validate(path)
        codes = [i.code for i in r.warnings]
        assert "CONTAINS_URL" in codes

    def test_html_markup_warning(self):
        """Kalimat dengan HTML → warning CONTAINS_MARKUP."""
        path = make_jsonl([{"text": "teks <b>tebal</b> dan <i>miring</i> ini"}])
        r = v().validate(path)
        codes = [i.code for i in r.warnings]
        assert "CONTAINS_MARKUP" in codes

    def test_wiki_markup_warning(self):
        """Kalimat dengan wiki markup → warning CONTAINS_MARKUP."""
        path = make_jsonl([{"text": "ini [[link]] wiki dan {{template}} di sini"}])
        r = v().validate(path)
        codes = [i.code for i in r.warnings]
        assert "CONTAINS_MARKUP" in codes

    def test_null_byte_error(self):
        """Kalimat dengan null byte → error NULL_BYTE."""
        path = make_jsonl([{"text": "kalimat\x00dengan null byte di tengah"}])
        r = v().validate(path)
        codes = [i.code for i in r.errors]
        assert "NULL_BYTE" in codes

    def test_clean_text_no_noise(self):
        """Kalimat bersih → tidak ada noise warning."""
        path = make_jsonl([
            {"text": "saya pergi ke pasar membeli sayuran segar"},
            {"text": "pemerintah menetapkan kebijakan baru untuk rakyat"},
        ])
        r = v().validate(path)
        noise_codes = {"CONTAINS_URL", "CONTAINS_MARKUP", "CONTAINS_TABLE",
                       "NULL_BYTE", "OUT_OF_BMP"}
        found = {i.code for i in r.issues} & noise_codes
        assert len(found) == 0


# ════════════════════════════════════════════════════════════════
# 4. DUPLIKAT & DIVERSITY
# ════════════════════════════════════════════════════════════════

class TestDiversity:

    def test_high_duplicate_rate_warning(self):
        """>20% near-duplicate → warning HIGH_DUPLICATE_RATE."""
        base = "saya makan nasi di rumah setiap hari"
        dupes = [{"text": base}] * 30
        others = [{"text": f"kalimat berbeda nomor {i} ini cukup panjang"} for i in range(10)]
        path = make_jsonl(dupes + others)
        r = v(max_dup_rate=0.20).validate(path)
        codes = [i.code for i in r.issues]
        assert "HIGH_DUPLICATE_RATE" in codes

    def test_low_duplicate_rate_passes(self):
        """Duplikat rendah → tidak ada duplicate warning."""
        sents = [
            {"text": f"ini kalimat ke {i} yang berbeda-beda isinya"}
            for i in range(20)
        ]
        path = make_jsonl(sents)
        r = v(max_dup_rate=0.20).validate(path)
        codes = [i.code for i in r.issues]
        assert "HIGH_DUPLICATE_RATE" not in codes

    def test_low_diversity_warning(self):
        """Satu pola dominasi >50% → warning LOW_DIVERSITY."""
        dominant = [{"text": f"saya makan nasi {i} kali sehari"}
                    for i in range(60)]
        others   = [{"text": f"dia pergi ke {i} tempat berbeda"}
                    for i in range(10)]
        path = make_jsonl(dominant + others)
        r = v(max_dom_rate=0.50).validate(path)
        codes = [i.code for i in r.issues]
        assert "LOW_DIVERSITY" in codes

    def test_diverse_dataset_passes(self):
        """Dataset beragam → tidak ada diversity warning."""
        sents = [
            {"text": "saya makan nasi goreng di warung"},
            {"text": "dia membaca buku di perpustakaan"},
            {"text": "mereka bekerja keras setiap hari"},
            {"text": "pemerintah menetapkan kebijakan baru"},
            {"text": "anak-anak bermain di taman setiap sore"},
            {"text": "ibu memasak sayur bayam untuk makan siang"},
        ]
        path = make_jsonl(sents)
        r = v().validate(path)
        diversity_codes = {"HIGH_DUPLICATE_RATE", "LOW_DIVERSITY"}
        found = {i.code for i in r.issues} & diversity_codes
        assert len(found) == 0


# ════════════════════════════════════════════════════════════════
# 5. VALIDATE_LIST (IN-MEMORY)
# ════════════════════════════════════════════════════════════════

class TestValidateList:

    def test_valid_list(self):
        """validate_list bekerja untuk list of strings."""
        texts = [
            "saya makan nasi di rumah",
            "dia membaca buku di perpustakaan",
            "mereka bekerja keras setiap hari",
        ]
        r = v().validate_list(texts)
        assert r.valid_samples == 3
        assert r.passed

    def test_invalid_in_list(self):
        """validate_list mendeteksi kalimat terlalu pendek."""
        texts = ["oke", "valid kalimat ini cukup panjang", "ya"]
        r = v(min_tokens=3, strict=True).validate_list(texts)
        assert len(r.errors) > 0

    def test_stats_populated(self):
        """Stats valid_rate selalu ada setelah validasi."""
        texts = ["kalimat ini valid dan cukup panjang sekali"] * 5
        r = v().validate_list(texts)
        assert "valid_rate" in r.stats
        assert 0.0 <= r.stats["valid_rate"] <= 1.0


# ════════════════════════════════════════════════════════════════
# 6. DOMAIN AGNOSTICITY
# ════════════════════════════════════════════════════════════════

class TestDomainAgnostic:
    """
    Pastikan validator tidak peduli domain atau konten.
    Kalimat dari domain apapun yang strukturnya benar harus lolos.
    """

    def test_medical_domain_passes(self):
        path = make_jsonl([
            {"text": "pasien didiagnosis mengalami hipertensi stadium dua"},
            {"text": "dokter meresepkan obat antihipertensi dua kali sehari"},
        ])
        r = v().validate(path)
        assert r.valid_samples == 2
        assert r.passed

    def test_legal_domain_passes(self):
        path = make_jsonl([
            {"text": "terdakwa dijatuhi hukuman penjara lima tahun oleh hakim"},
            {"text": "jaksa mengajukan banding atas putusan pengadilan negeri"},
        ])
        r = v().validate(path)
        assert r.valid_samples == 2
        assert r.passed

    def test_informal_slang_passes(self):
        path = make_jsonl([
            {"text": "gue udah makan nih, lo udah belum"},
            {"text": "dia kocak banget sih bikin ngakak terus"},
        ])
        r = v(min_tokens=3).validate(path)
        assert r.valid_samples == 2
        assert r.passed

    def test_technical_domain_passes(self):
        path = make_jsonl([
            {"text": "fungsi rekursif ini memanggil dirinya sendiri hingga base case"},
            {"text": "database terindeks dengan btree untuk query yang lebih cepat"},
        ])
        r = v().validate(path)
        assert r.valid_samples == 2
        assert r.passed

    def test_no_domain_restriction_in_issues(self):
        """Tidak ada issue code yang mengandung 'DOMAIN'."""
        path = make_jsonl([
            {"text": "ini kalimat tentang anime dan manga populer"},
            {"text": "trading kripto memiliki risiko yang sangat tinggi"},
        ])
        r = v().validate(path)
        domain_issues = [i for i in r.issues if "DOMAIN" in i.code]
        assert len(domain_issues) == 0


# ════════════════════════════════════════════════════════════════
# 7. VALIDATE_FILE SHORTHAND
# ════════════════════════════════════════════════════════════════

class TestValidateFile:

    def test_validate_file_returns_report(self):
        path = make_jsonl([
            {"text": "kalimat valid pertama ini bagus"},
            {"text": "kalimat valid kedua ini juga bagus"},
        ])
        r = validate_file(path)
        assert isinstance(r, ValidationReport)
        assert r.valid_samples == 2

    def test_validate_file_fails_on_errors(self):
        path = make_jsonl([{"no_text": "missing text field"}])
        r = validate_file(path)
        assert not r.passed

    def test_report_summary_string(self):
        """summary() harus menghasilkan string yang bisa diprint."""
        path = make_jsonl([{"text": "kalimat valid untuk test summary ini"}])
        r = validate_file(path)
        s = r.summary()
        assert isinstance(s, str)
        assert "valid" in s.lower()


# ════════════════════════════════════════════════════════════════
# 8. EDGE CASES
# ════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_empty_file(self):
        """File kosong → 0 sample, tidak error."""
        path = make_jsonl([])
        r = v().validate(path)
        assert r.total_lines == 0
        assert r.valid_samples == 0
        assert r.passed

    def test_single_valid_line(self):
        """Satu baris valid → passed."""
        path = make_jsonl([{"text": "saya pergi ke pasar"}])
        r = v().validate(path)
        assert r.valid_samples == 1
        assert r.passed

    def test_extra_fields_ignored(self):
        """Field tambahan selain text di-ignore, tidak error."""
        path = make_jsonl([{
            "text": "saya pergi ke pasar membeli sayuran",
            "domain": "daily",
            "source": "synthetic",
            "quality": "high",
            "meta": {"author": "test"},
        }])
        r = v().validate(path)
        assert r.valid_samples == 1
        assert r.passed

    def test_issue_str_representation(self):
        """ValidationIssue.__str__ harus bisa diprint tanpa error."""
        issue = ValidationIssue(
            line_no=42, level="error",
            code="TEST_CODE",
            message="Pesan test",
            text="teks contoh",
        )
        s = str(issue)
        assert "42" in s
        assert "TEST_CODE" in s

    def test_real_dataset_v3(self, tmp_path):
        """Smoke test: validasi corpus_training_v3.jsonl jika ada."""
        corpus = Path("data/corpus_training_v3.jsonl")
        if not corpus.exists():
            pytest.skip("corpus_training_v3.jsonl tidak ditemukan")
        r = validate_file(corpus)
        # Dataset yang sudah dibangun harus >80% valid
        assert r.stats["valid_rate"] > 0.80


# ════════════════════════════════════════════════════════════════
# 9. QUALITY SCORE
# ════════════════════════════════════════════════════════════════

class TestQualityScore:

    def test_quality_score_present(self):
        """Setiap report harus punya quality score setelah validasi."""
        path = make_jsonl([
            {"text": "kalimat valid pertama untuk test ini"},
            {"text": "kalimat valid kedua untuk test ini"},
        ])
        r = validate_file(path)
        assert r.quality is not None
        assert isinstance(r.quality, DatasetQualityScore)

    def test_quality_score_range(self):
        """Semua skor harus dalam rentang 0-100."""
        path = make_jsonl([
            {"text": f"kalimat valid nomor {i} untuk test scoring ini"}
            for i in range(10)
        ])
        r = validate_file(path)
        q = r.quality
        assert 0 <= q.structure  <= 100
        assert 0 <= q.noise      <= 100
        assert 0 <= q.diversity  <= 100
        assert 0 <= q.length     <= 100
        assert 0 <= q.total      <= 100

    def test_excellent_dataset_score(self):
        """Dataset bersih dan beragam harus mendapat grade Excellent atau Good."""
        sents = [
            {"text": f"subjek berbeda nomor {i} melakukan aktivitas sehari-hari ini"}
            for i in range(30)
        ]
        path = make_jsonl(sents)
        r = validate_file(path)
        assert r.quality.total >= 75
        assert r.quality.grade in ("Excellent", "Good")

    def test_poor_dataset_score(self):
        """Dataset dengan banyak error harus mendapat skor rendah."""
        lines = [
            '{"not_text": "missing"}',    # error
            '{"text": ""}',               # error
            'bukan json',                 # error
        ] + [
            json.dumps({"text": f"kalimat {i}"}) for i in range(2)
        ]
        path = make_jsonl(lines)
        r = validate_file(path)
        assert r.quality.structure < 90

    def test_high_duplicate_lowers_diversity_score(self):
        """Dataset dengan banyak duplikat harus punya diversity score rendah."""
        base = "saya makan nasi setiap hari di rumah"
        dupes = [{"text": base}] * 40
        others = [{"text": f"kalimat berbeda ke {i} isinya lain"} for i in range(10)]
        path = make_jsonl(dupes + others)
        r = validate_file(path, max_dup_rate=0.20)
        assert r.quality.diversity < 80

    def test_noisy_dataset_lowers_noise_score(self):
        """Dataset dengan banyak URL/markup harus punya noise score rendah."""
        noisy = [
            {"text": f"kunjungi https://site{i}.com untuk informasi lebih lanjut"}
            for i in range(20)
        ]
        clean = [{"text": "kalimat bersih tanpa noise sama sekali ini"}] * 5
        path = make_jsonl(noisy + clean)
        r = validate_file(path)
        assert r.quality.noise < 80

    def test_grade_thresholds(self):
        """Grade harus sesuai threshold yang didefinisikan."""
        assert DatasetQualityScore(total=95).grade == "Excellent"
        assert DatasetQualityScore(total=80).grade == "Good"
        assert DatasetQualityScore(total=65).grade == "Fair"
        assert DatasetQualityScore(total=40).grade == "Poor"

    def test_ready_for_training(self):
        """ready_for_training harus True untuk dataset berkualitas cukup."""
        assert DatasetQualityScore(total=70, structure=80).ready_for_training is True
        assert DatasetQualityScore(total=50, structure=80).ready_for_training is False
        assert DatasetQualityScore(total=70, structure=60).ready_for_training is False

    def test_quality_score_str(self):
        """__str__ harus menghasilkan output yang bisa diprint tanpa error."""
        q = DatasetQualityScore(structure=90, noise=85, diversity=70, length=95, total=86)
        s = str(q)
        assert "90" in s
        assert "85" in s
        assert "86" in s
        assert "Good" in s

    def test_summary_includes_quality(self):
        """summary() dari report harus include quality score jika ada."""
        path = make_jsonl([
            {"text": "kalimat valid untuk test summary dengan quality score"}
        ])
        r = validate_file(path)
        s = r.summary()
        assert "Structure" in s or "Total" in s

    def test_validate_list_has_quality(self):
        """validate_list() juga harus menghasilkan quality score."""
        texts = [
            "kalimat pertama ini valid dan cukup panjang",
            "kalimat kedua ini juga valid untuk test ini",
            "kalimat ketiga cukup berbeda dari yang sebelumnya",
        ]
        r = v().validate_list(texts)
        assert r.quality is not None
        assert r.quality.total >= 0

    def test_real_dataset_v3_quality(self):
        """corpus_training_v3 harus mendapat score minimal Fair (>=60)."""
        corpus = Path("data/corpus_training_v3.jsonl")
        if not corpus.exists():
            pytest.skip("corpus_training_v3.jsonl tidak ditemukan")
        r = validate_file(corpus)
        assert r.quality is not None
        assert r.quality.total >= 60
        assert r.quality.ready_for_training is True
