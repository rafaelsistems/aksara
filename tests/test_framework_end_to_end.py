"""
test_framework_end_to_end.py
============================
Verifikasi end-to-end pipeline AKSARA dari `AksaraFramework.proses()`.

Fokus:
  - instansiasi framework
  - kelengkapan `AksaraState`
  - stabilitas field integrasi CPE/CMC/TDA/KRL
  - kontrak output yang siap dipakai downstream
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aksara.framework import AksaraFramework
from aksara.base.state import AksaraState, ConstraintSatisfaction, PelanggaranConstraint
from aksara.primitives.krl.layer import KRLResult
from aksara.config import AksaraConfig


def _make_framework() -> AksaraFramework:
    """
    Buat framework dengan leksikon KBBI test yang sudah ada di repository.
    """
    kbbi_path = Path("kbbi_core_v2.json")
    if not kbbi_path.exists():
        pytest.skip("kbbi_core_v2.json tidak tersedia di workspace ini")
    return AksaraFramework.dari_kbbi(str(kbbi_path), config=AksaraConfig.default())


@pytest.mark.parametrize(
    "kalimat",
    [
        "Budi membeli beras di pasar.",
        "Beras itu dibeli Budi di pasar.",
        "Makanan tradisional Dompu sangat lezat.",
        "Pemerintah mengeluarkan kebijakan baru itu.",
        "Budi membeli beras. Dia membayar dengan tunai.",
    ],
)
def test_pipeline_end_to_end_produces_aksara_state(kalimat: str):
    fw = _make_framework()
    state = fw.proses(kalimat)

    assert isinstance(state, AksaraState)
    assert state.teks_asli == kalimat.rstrip(".").replace(". ", " ")
    assert state.morfem_states is not None
    assert isinstance(state.energi_total, float)
    assert isinstance(state.energi_per_dimensi, dict)
    assert isinstance(state.pelanggaran, list)
    assert isinstance(state.violation_spans, list)
    assert isinstance(state.fitur_topologis, object)
    assert isinstance(state.anomali_topologis, bool)
    assert state.constraint_satisfaction is None or isinstance(
        state.constraint_satisfaction, ConstraintSatisfaction
    )
    assert state.krl_result is None or isinstance(state.krl_result, KRLResult)


def test_pipeline_end_to_end_populates_key_contracts():
    fw = _make_framework()
    state = fw.proses("Budi membeli beras di pasar.")

    assert isinstance(state, AksaraState)
    assert state.skor_linguistik >= 0.0
    assert state.skor_linguistik <= 1.0

    assert isinstance(state.energi_per_dimensi, dict)
    assert "semantik" in state.energi_per_dimensi or "sintaktis" in state.energi_per_dimensi

    if state.constraint_satisfaction is not None:
        assert isinstance(state.constraint_satisfaction, ConstraintSatisfaction)
        assert 0.0 <= state.constraint_satisfaction.rata_rata <= 1.0

    if state.krl_result is not None:
        assert isinstance(state.krl_result, KRLResult)
        assert state.krl_result.kelengkapan_pemahaman >= 0.0
        assert state.krl_result.kelengkapan_pemahaman <= 1.0


def test_pipeline_end_to_end_state_rendering():
    fw = _make_framework()
    state = fw.proses("Pemerintah mengeluarkan kebijakan baru itu.")

    ringkasan = state.ringkasan()
    jelaskan = state.jelaskan()

    assert isinstance(ringkasan, str)
    assert isinstance(jelaskan, str)
    assert "skor=" in ringkasan
    assert "Kalimat:" in jelaskan
    assert "Constraint" in jelaskan or "Pelanggaran" in jelaskan


def test_pipeline_end_to_end_handles_references():
    fw = _make_framework()
    state_1 = fw.proses("Budi membeli beras di pasar.")
    state_2 = fw.proses("Dia membayar dengan tunai.")

    assert isinstance(state_1, AksaraState)
    assert isinstance(state_2, AksaraState)
    assert state_2.krl_result is None or isinstance(state_2.krl_result, KRLResult)
    if state_2.krl_result is not None:
        assert state_2.krl_result.ikatan_referensi is not None


def test_pipeline_end_to_end_rejects_empty_input_with_clear_message():
    fw = _make_framework()
    state = fw.proses("")

    assert isinstance(state, AksaraState)
    assert state.teks_asli == ""
    assert state.energi_total == 0.0
    assert state.morfem_states == []
    assert state.pelanggaran == []
    assert state.violation_spans == []
    assert state.register == "formal"
    assert state.kelengkapan_struktur == 0.0
    assert state.ringkasan() == "[VALID] skor=0.375 energi=0.000 pelanggaran=0 morfem=0"
    assert "kalimat" in state.jelaskan().lower()
    assert state.krl_result is None or isinstance(state.krl_result, KRLResult)


def test_pipeline_end_to_end_rejects_whitespace_only_input_with_clear_message():
    fw = _make_framework()
    state = fw.proses("   ")

    assert isinstance(state, AksaraState)
    assert state.teks_asli.strip() == ""
    assert state.energi_total == 0.0
    assert state.morfem_states == []
    assert state.pelanggaran == []
    assert state.violation_spans == []
    assert state.register == "formal"
    assert state.kelengkapan_struktur == 0.0
    assert "kalimat" in state.jelaskan().lower()
    assert state.krl_result is None or isinstance(state.krl_result, KRLResult)


def test_pipeline_end_to_end_rejects_punctuation_only_input_with_clear_message():
    fw = _make_framework()
    state = fw.proses("...")

    assert isinstance(state, AksaraState)
    assert state.teks_asli.replace(".", "").strip() == ""
    assert isinstance(state.energi_total, float)
    assert state.pelanggaran == []
    assert state.violation_spans == []
    assert state.register == "formal"
    assert state.kelengkapan_struktur == 0.0
    assert "kalimat" in state.jelaskan().lower()
    assert state.krl_result is None or isinstance(state.krl_result, KRLResult)


def test_pipeline_end_to_end_is_stable_across_repeated_calls():
    fw = _make_framework()
    kalimat = "Budi membeli beras di pasar."

    state_1 = fw.proses(kalimat)
    state_2 = fw.proses(kalimat)

    assert isinstance(state_1, AksaraState)
    assert isinstance(state_2, AksaraState)
    assert state_1.teks_asli == state_2.teks_asli
    assert state_1.ringkasan() == state_2.ringkasan()
    assert state_1.jelaskan() == state_2.jelaskan()
    assert state_1.krl_result is None or isinstance(state_1.krl_result, KRLResult)
    assert state_2.krl_result is None or isinstance(state_2.krl_result, KRLResult)


def test_pipeline_end_to_end_multi_sentence_state_inspection():
    fw = _make_framework()
    kalimat = "Budi membeli beras di pasar. Dia membayar dengan tunai."
    state = fw.proses(kalimat)

    assert isinstance(state, AksaraState)
    assert state.teks_asli == kalimat.rstrip(".").replace(". ", " ")
    assert state.morfem_states is not None
    assert isinstance(state.violation_spans, list)
    assert isinstance(state.pelanggaran, list)
    assert isinstance(state.ringkasan(), str)
    assert isinstance(state.jelaskan(), str)
    if state.krl_result is not None:
        assert isinstance(state.krl_result, KRLResult)
        assert state.krl_result.kelengkapan_pemahaman >= 0.0
        assert state.krl_result.kelengkapan_pemahaman <= 1.0


def test_pipeline_end_to_end_violation_contracts_are_consistent():
    fw = _make_framework()
    state = fw.proses("Makanan tradisional Dompu sangat lezat.")

    assert isinstance(state.pelanggaran, list)
    for p in state.pelanggaran:
        assert isinstance(p, PelanggaranConstraint)
        assert p.dimensi in {"morfologis", "sintaktis", "semantik", "leksikal", "topologis", "animasi", "komposisi"}
        assert isinstance(p.penjelasan, str)
        assert p.penjelasan


def test_pipeline_end_to_end_constraint_satisfaction_shape():
    fw = _make_framework()
    state = fw.proses("Budi membeli beras di pasar.")

    if state.constraint_satisfaction is not None:
        sat = state.constraint_satisfaction
        assert isinstance(sat, ConstraintSatisfaction)
        assert 0.0 <= sat.morfologis <= 1.0
        assert 0.0 <= sat.sintaktis <= 1.0
        assert 0.0 <= sat.semantik <= 1.0
        assert 0.0 <= sat.leksikal <= 1.0
        assert 0.0 <= sat.topologis <= 1.0
        assert 0.0 <= sat.animasi <= 1.0
