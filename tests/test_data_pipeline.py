"""
test_data_pipeline.py — Unit test untuk AksaraDataPipeline.

Test ini domain-agnostic: tidak ada asumsi konten.
"""

import json
import tempfile
from pathlib import Path

import pytest

from aksara.data.pipeline import (
    AksaraDataPipeline,
    PipelineConfig,
    PipelineStats,
    TextCleaner,
    TextDeduplicator,
    TextNormalizer,
    structure_texts,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_jsonl(entries: list) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8",
        delete=False, newline="\n"
    )
    for e in entries:
        if isinstance(e, str):
            f.write(json.dumps({"text": e}, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    f.close()
    return Path(f.name)


def default_pipeline(**kwargs) -> AksaraDataPipeline:
    kwargs.setdefault("min_tokens", 3)
    kwargs.setdefault("max_tokens", 40)
    cfg = PipelineConfig(**kwargs)
    return AksaraDataPipeline(cfg)


# ════════════════════════════════════════════════════════════════
# 1. TextCleaner
# ════════════════════════════════════════════════════════════════

class TestTextCleaner:

    def test_clean_valid(self):
        """Kalimat valid harus lolos clean."""
        cfg     = PipelineConfig(min_tokens=3)
        cleaner = TextCleaner(cfg)
        result  = cleaner.clean("saya makan nasi di rumah")
        assert result == "saya makan nasi di rumah"

    def test_remove_url(self):
        """URL harus dihapus."""
        cfg     = PipelineConfig(remove_urls=True, min_tokens=3)
        cleaner = TextCleaner(cfg)
        result  = cleaner.clean("kunjungi https://example.com untuk info lebih")
        assert result is not None
        assert "https" not in result

    def test_remove_html_markup(self):
        """HTML markup harus dihapus."""
        cfg     = PipelineConfig(remove_markup=True, min_tokens=3)
        cleaner = TextCleaner(cfg)
        result  = cleaner.clean("teks <b>tebal</b> dan biasa ini")
        assert result is not None
        assert "<b>" not in result

    def test_remove_bracket_ref(self):
        """Referensi [1] gaya Wikipedia harus dihapus."""
        cfg     = PipelineConfig(remove_bracket_refs=True, min_tokens=3)
        cleaner = TextCleaner(cfg)
        result  = cleaner.clean("teks ini[1] punya referensi[2] gaya wiki")
        assert result is not None
        assert "[1]" not in result

    def test_too_short_returns_none(self):
        """Kalimat terlalu pendek harus dikembalikan None."""
        cfg     = PipelineConfig(min_tokens=4)
        cleaner = TextCleaner(cfg)
        assert cleaner.clean("ya oke") is None

    def test_too_many_digits_returns_none(self):
        """Baris dominan angka harus dikembalikan None."""
        cfg     = PipelineConfig(max_digit_ratio=0.3, min_tokens=3)
        cleaner = TextCleaner(cfg)
        assert cleaner.clean("1234 5678 9012 3456") is None

    def test_too_many_garbage_returns_none(self):
        """Teks dengan terlalu banyak karakter garbage harus dikembalikan None."""
        cfg     = PipelineConfig(max_garbage_ratio=0.1, min_tokens=3)
        cleaner = TextCleaner(cfg)
        assert cleaner.clean("{{{{{[][][]||||}}}}}") is None

    def test_clean_batch(self):
        """clean_batch harus filter kalimat tidak valid."""
        cfg     = PipelineConfig(min_tokens=3)
        cleaner = TextCleaner(cfg)
        batch   = [
            "kalimat valid ini bagus",
            "ya",               # terlalu pendek
            "kalimat lain yang juga valid",
        ]
        result  = cleaner.clean_batch(batch)
        assert len(result) == 2

    def test_null_byte_removed(self):
        """Null byte harus dihapus dari teks."""
        cfg     = PipelineConfig(min_tokens=3)
        cleaner = TextCleaner(cfg)
        result  = cleaner.clean("teks\x00dengan null ini")
        assert result is not None
        assert "\x00" not in result

    def test_unicode_normalization(self):
        """Karakter unicode harus di-normalize ke NFKC."""
        cfg     = PipelineConfig(min_tokens=3)
        cleaner = TextCleaner(cfg)
        # U+2019 RIGHT SINGLE QUOTATION MARK → harus tetap lolos
        result  = cleaner.clean("kata\u2019kata ini valid sekali")
        assert result is not None


# ════════════════════════════════════════════════════════════════
# 2. TextNormalizer
# ════════════════════════════════════════════════════════════════

class TestTextNormalizer:

    def test_normalize_multiple_spaces(self):
        """Spasi berlebih harus dinormalisasi."""
        cfg   = PipelineConfig(normalize_whitespace=True)
        norm  = TextNormalizer(cfg)
        result = norm.normalize("ini   teks   dengan  spasi  banyak")
        assert "  " not in result

    def test_normalize_multiple_punctuation(self):
        """Tanda baca berulang harus dinormalisasi."""
        cfg   = PipelineConfig()
        norm  = TextNormalizer(cfg)
        result = norm.normalize("ini kalimat!!!! sangat menarik???")
        assert "!!!!" not in result
        assert "???" not in result

    def test_normalize_ellipsis(self):
        """Ellipsis panjang harus dinormalisasi ke 3 titik."""
        cfg   = PipelineConfig()
        norm  = TextNormalizer(cfg)
        result = norm.normalize("ini kalimat......... belum selesai")
        assert "........." not in result

    def test_normalize_tab(self):
        """Tab harus diganti spasi."""
        cfg   = PipelineConfig(normalize_whitespace=True)
        norm  = TextNormalizer(cfg)
        result = norm.normalize("ini\tteks\tdengan\ttab")
        assert "\t" not in result

    def test_normalize_batch(self):
        """normalize_batch harus memproses semua kalimat."""
        cfg   = PipelineConfig()
        norm  = TextNormalizer(cfg)
        batch = [
            "kalimat  dengan  spasi",
            "kalimat normal ini",
        ]
        result = norm.normalize_batch(batch)
        assert len(result) == 2
        assert "  " not in result[0]

    def test_normalize_preserves_meaning(self):
        """Normalisasi tidak boleh menghapus kata-kata penting."""
        cfg   = PipelineConfig()
        norm  = TextNormalizer(cfg)
        text  = "saya pergi ke pasar membeli sayuran segar"
        result = norm.normalize(text)
        assert "saya" in result
        assert "pasar" in result
        assert "sayuran" in result


# ════════════════════════════════════════════════════════════════
# 3. TextDeduplicator
# ════════════════════════════════════════════════════════════════

class TestTextDeduplicator:

    def test_exact_duplicate_removed(self):
        """Kalimat identik harus dihapus."""
        cfg   = PipelineConfig(dedup_prefix_len=60)
        dedup = TextDeduplicator(cfg)
        texts = [
            "saya makan nasi goreng di rumah",
            "saya makan nasi goreng di rumah",  # duplikat
            "dia membaca buku di perpustakaan",
        ]
        result = dedup.deduplicate(texts)
        assert len(result) == 2

    def test_near_duplicate_removed(self):
        """Kalimat near-duplicate (Jaccard tinggi) harus dihapus."""
        cfg   = PipelineConfig(dedup_jaccard_thresh=0.80)
        dedup = TextDeduplicator(cfg)
        # Dua kalimat pertama berbeda satu huruf saja → Jaccard sangat tinggi
        base = "saya makan nasi goreng di rumah setiap hari bersama keluarga"
        near = "saya makan nasi goreng di rumah setiap hari bersama keluargA"
        texts = [
            base,
            near,   # hampir identik, beda satu karakter
            "dia membaca buku di perpustakaan kota besar itu",
        ]
        result = dedup.deduplicate(texts)
        assert len(result) <= 2

    def test_unique_texts_all_pass(self):
        """Kalimat unik semua harus lolos."""
        cfg   = PipelineConfig()
        dedup = TextDeduplicator(cfg)
        texts = [
            "saya makan nasi di rumah",
            "dia membaca buku di sekolah",
            "mereka bekerja di kantor pusat",
            "pemerintah menetapkan kebijakan baru",
        ]
        result = dedup.deduplicate(texts)
        assert len(result) == 4

    def test_reset_clears_state(self):
        """reset() harus menghapus semua state deduplication."""
        cfg   = PipelineConfig()
        dedup = TextDeduplicator(cfg)
        text  = "saya makan nasi di rumah setiap hari"
        assert not dedup.is_duplicate(text)
        assert dedup.is_duplicate(text)     # kedua kali = duplikat
        dedup.reset()
        assert not dedup.is_duplicate(text)  # setelah reset = bukan duplikat


# ════════════════════════════════════════════════════════════════
# 4. structure_texts
# ════════════════════════════════════════════════════════════════

class TestStructureTexts:

    def test_basic_structure(self):
        """Setiap teks harus dibungkus dengan field 'text'."""
        texts  = ["kalimat satu ini", "kalimat dua ini"]
        result = structure_texts(texts, source="test")
        assert len(result) == 2
        assert all("text" in e for e in result)
        assert result[0]["text"] == "kalimat satu ini"

    def test_meta_source(self):
        """Field meta.source harus terisi."""
        result = structure_texts(["kalimat ini valid"], source="wikipedia")
        assert result[0]["meta"]["source"] == "wikipedia"

    def test_extra_meta(self):
        """Field meta tambahan harus dimasukkan."""
        result = structure_texts(
            ["kalimat ini valid"], source="test",
            extra_meta={"quality": "high", "lang": "id"}
        )
        assert result[0]["meta"]["quality"] == "high"
        assert result[0]["meta"]["lang"] == "id"

    def test_empty_input(self):
        """Input kosong → output kosong."""
        assert structure_texts([]) == []


# ════════════════════════════════════════════════════════════════
# 5. AksaraDataPipeline — run()
# ════════════════════════════════════════════════════════════════

class TestPipelineRun:

    def test_run_basic(self):
        """Pipeline dasar harus menghasilkan output bersih."""
        texts = [
            "saya makan nasi goreng di warung dekat rumah",
            "dia membaca buku cerita di perpustakaan sekolah",
            "mereka bekerja keras untuk menyelesaikan proyek",
            "pemerintah menetapkan kebijakan fiskal baru tahun ini",
        ]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        assert res.stats.final_count > 0
        assert res.stats.final_count <= len(texts)
        assert all("text" in e for e in res.entries)

    def test_run_removes_noise(self):
        """Pipeline harus menghapus kalimat dengan noise berat."""
        texts = [
            "kalimat valid ini bagus untuk training",
            "https://example.com kunjungi website kami",   # URL dominan
            "kalimat valid kedua ini juga bagus sekali",
        ]
        p   = default_pipeline(remove_urls=True)
        res = p.run(texts, verbose=False)
        out_texts = [e["text"] for e in res.entries]
        assert not any("https://" in t for t in out_texts)

    def test_run_removes_duplicates(self):
        """Pipeline harus menghapus duplikat."""
        base  = "saya makan nasi goreng di rumah setiap hari"
        texts = [base] * 5 + [
            "kalimat berbeda ini juga ada di sini",
            "kalimat lain yang berbeda juga masuk",
        ]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        out_texts = [e["text"] for e in res.entries]
        assert out_texts.count(base) == 1

    def test_run_stats_populated(self):
        """PipelineStats harus terisi setelah run."""
        texts = ["kalimat valid nomor satu ini", "kalimat valid nomor dua ini"]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        assert res.stats.input_count == 2
        assert res.stats.final_count >= 0
        assert res.stats.quality is not None

    def test_run_writes_output_file(self, tmp_path):
        """output_path harus menghasilkan file JSONL yang valid."""
        texts = [
            "saya pergi ke pasar membeli sayuran segar ini",
            "dia menulis surat panjang untuk temannya di kota",
        ]
        out = tmp_path / "output.jsonl"
        p   = default_pipeline()
        res = p.run(texts, output_path=out, verbose=False)

        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == res.stats.final_count
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj

    def test_run_with_meta(self):
        """extra_meta harus diteruskan ke setiap entry."""
        texts = ["kalimat valid dengan meta untuk test ini"]
        p   = default_pipeline()
        res = p.run(texts, source="test_source",
                    extra_meta={"version": "v1"}, verbose=False)
        if res.entries:
            assert res.entries[0]["meta"]["source"] == "test_source"
            assert res.entries[0]["meta"]["version"] == "v1"

    def test_run_empty_input(self):
        """Input kosong harus menghasilkan output kosong tanpa error."""
        p   = default_pipeline()
        res = p.run([], verbose=False)
        assert res.stats.final_count == 0
        assert res.entries == []

    def test_run_all_noise(self):
        """Input semua noise → output kosong."""
        texts = [
            "ya",       # terlalu pendek
            "ok",       # terlalu pendek
            "1",        # terlalu pendek
        ]
        p   = default_pipeline(min_tokens=4, max_tokens=40)
        res = p.run(texts, verbose=False)
        assert res.stats.final_count == 0


# ════════════════════════════════════════════════════════════════
# 6. AksaraDataPipeline — run_file()
# ════════════════════════════════════════════════════════════════

class TestPipelineRunFile:

    def test_run_file_jsonl(self, tmp_path):
        """run_file() harus bisa baca file JSONL."""
        entries = [
            {"text": "saya makan nasi goreng di warung"},
            {"text": "dia membaca buku di perpustakaan"},
            {"text": "mereka bekerja keras setiap hari"},
        ]
        input_path = tmp_path / "input.jsonl"
        with open(input_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        p   = default_pipeline()
        res = p.run_file(input_path, verbose=False)
        assert res.stats.final_count > 0

    def test_run_file_plain_text(self, tmp_path):
        """run_file() harus bisa baca plain text (1 kalimat/baris)."""
        lines = [
            "saya makan nasi goreng di warung",
            "dia membaca buku di perpustakaan",
            "mereka bekerja keras setiap hari",
        ]
        input_path = tmp_path / "input.txt"
        input_path.write_text("\n".join(lines), encoding="utf-8")

        p   = default_pipeline()
        res = p.run_file(input_path, verbose=False)
        assert res.stats.final_count > 0

    def test_run_file_with_output(self, tmp_path):
        """run_file() dengan output_path harus menulis file JSONL."""
        entries = [
            {"text": "kalimat pertama ini valid dan cukup panjang"},
            {"text": "kalimat kedua ini juga valid dan cukup panjang"},
        ]
        input_path  = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        with open(input_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        p = default_pipeline()
        p.run_file(input_path, output_path=output_path, verbose=False)
        assert output_path.exists()


# ════════════════════════════════════════════════════════════════
# 7. PipelineConfig
# ════════════════════════════════════════════════════════════════

class TestPipelineConfig:

    def test_default_config(self):
        """Config default harus valid dan masuk akal."""
        cfg = PipelineConfig()
        assert cfg.min_tokens > 0
        assert cfg.max_tokens > cfg.min_tokens
        assert 0.0 < cfg.dedup_jaccard_thresh <= 1.0
        assert 0.0 < cfg.max_digit_ratio <= 1.0
        assert 0.0 < cfg.min_alpha_ratio <= 1.0

    def test_custom_config(self):
        """Config custom harus diteruskan ke pipeline."""
        cfg = PipelineConfig(min_tokens=2, max_tokens=20, dedup_jaccard_thresh=0.9)
        p   = AksaraDataPipeline(cfg)
        assert p.config.min_tokens == 2
        assert p.config.max_tokens == 20

    def test_strict_mode(self):
        """Strict mode harus diteruskan ke validator."""
        cfg = PipelineConfig(validator_strict=True)
        p   = AksaraDataPipeline(cfg)
        assert p.validator.strict is True


# ════════════════════════════════════════════════════════════════
# 8. PipelineStats
# ════════════════════════════════════════════════════════════════

class TestPipelineStats:

    def test_rejection_rate_zero(self):
        """Jika semua lolos, rejection rate harus 0."""
        s = PipelineStats(input_count=10, final_count=10)
        assert s.rejection_rate() == 0.0

    def test_rejection_rate_full(self):
        """Jika semua ditolak, rejection rate harus 1."""
        s = PipelineStats(input_count=10, final_count=0)
        assert s.rejection_rate() == 1.0

    def test_rejection_rate_partial(self):
        """Rejection rate parsial harus dihitung dengan benar."""
        s = PipelineStats(input_count=10, final_count=7)
        assert abs(s.rejection_rate() - 0.3) < 0.01

    def test_stats_str(self):
        """__str__ harus bisa diprint tanpa error."""
        s = PipelineStats(input_count=100, after_clean=90,
                          after_normalize=88, after_dedup=80,
                          after_validate=78, final_count=78)
        st = str(s)
        assert "100" in st
        assert "78" in st

    def test_empty_input_no_error(self):
        """Input 0 tidak boleh raise ZeroDivisionError."""
        s = PipelineStats(input_count=0, final_count=0)
        assert s.rejection_rate() == 0.0


# ════════════════════════════════════════════════════════════════
# 9. Domain Agnosticity
# ════════════════════════════════════════════════════════════════

class TestPipelineDomainAgnostic:

    def test_medical_content_passes(self):
        texts = [
            "pasien mengalami gejala demam dan batuk yang berkepanjangan",
            "dokter meresepkan antibiotik setelah melakukan pemeriksaan fisik",
        ]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        assert res.stats.final_count == 2

    def test_legal_content_passes(self):
        texts = [
            "terdakwa dijatuhi vonis penjara atas dakwaan penipuan berat",
            "pengacara mengajukan banding terhadap putusan pengadilan tersebut",
        ]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        assert res.stats.final_count == 2

    def test_technical_content_passes(self):
        texts = [
            "fungsi rekursif memanggil dirinya sendiri hingga base case tercapai",
            "database menggunakan indeks untuk mempercepat proses query pencarian",
        ]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        assert res.stats.final_count == 2

    def test_informal_content_passes(self):
        texts = [
            "gue udah makan siang tadi sama temen-temen di kantin",
            "dia keren banget sih, bisa ngerjain semua tugas sendiri",
        ]
        p   = default_pipeline(min_tokens=3, max_tokens=40)
        res = p.run(texts, verbose=False)
        assert res.stats.final_count == 2

    def test_no_domain_filtering_in_pipeline(self):
        """Pipeline tidak boleh menolak kalimat berdasarkan konten/domain."""
        domains = [
            "pasien menderita hipertensi dan diabetes tipe dua",
            "pemerintah mengesahkan undang-undang perlindungan konsumen",
            "trading crypto memiliki risiko tinggi yang harus dipahami",
            "novel ini menceritakan petualangan di dunia fantasi kuno",
            "kode program ini menggunakan algoritma pencarian biner efisien",
        ]
        p   = default_pipeline()
        res = p.run(domains, verbose=False)
        # Semua harus lolos karena strukturnya valid
        assert res.stats.final_count == len(domains)


# ════════════════════════════════════════════════════════════════
# 10. Integration — pipeline ke validator
# ════════════════════════════════════════════════════════════════

class TestPipelineIntegration:

    def test_pipeline_output_passes_validator(self, tmp_path):
        """Output pipeline harus lolos validator dengan score tinggi."""
        from aksara.data.validator import validate_file

        texts = [
            f"kalimat natural nomor {i} yang beragam dan valid untuk training aksara"
            for i in range(50)
        ]
        out = tmp_path / "gold.jsonl"
        p   = default_pipeline()
        p.run(texts, output_path=out, verbose=False)

        report = validate_file(out)
        assert report.passed
        assert report.quality.total >= 60

    def test_pipeline_removes_what_validator_rejects(self):
        """Kalimat yang ditolak validator tidak boleh masuk output pipeline."""
        texts = [
            "kalimat valid ini cukup panjang untuk training",
            "",                                    # kosong
            "kalimat valid kedua ini juga panjang",
        ]
        p   = default_pipeline()
        res = p.run(texts, verbose=False)
        # Kalimat kosong tidak boleh ada di output
        out_texts = [e["text"] for e in res.entries]
        assert all(len(t.strip()) > 0 for t in out_texts)

    def test_corpus_v3_through_pipeline(self):
        """Smoke test: corpus_training_v3.jsonl harus lolos pipeline."""
        corpus = Path("data/corpus_training_v3.jsonl")
        if not corpus.exists():
            pytest.skip("corpus_training_v3.jsonl tidak ditemukan")

        p   = default_pipeline()
        res = p.run_file(corpus, verbose=False)
        # Pipeline tidak boleh reject >50% kalimat dari dataset yang sudah dibersihkan
        assert res.stats.rejection_rate() < 0.50
        assert res.stats.quality is not None
        assert res.stats.quality.total >= 60
