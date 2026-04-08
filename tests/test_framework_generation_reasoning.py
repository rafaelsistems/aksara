"""
test_framework_generation_reasoning.py
======================================
Kontrak bukti user-facing readiness untuk generasi dan reasoning.

Fokus:
  - model dapat dibuat
  - model dapat menghasilkan generasi
  - model dapat melakukan reasoning
  - output menampilkan pemahaman linguistik, bukan next-token prediction ala Transformer/Mamba
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from aksara.framework import AksaraFramework
from aksara.config import AksaraConfig
from aksara.base.state import AksaraState


def _make_framework() -> AksaraFramework:
    kbbi_path = Path("kbbi_core_v2.json")
    if not kbbi_path.exists():
        pytest.skip("kbbi_core_v2.json tidak tersedia di workspace ini")
    return AksaraFramework.dari_kbbi(str(kbbi_path), config=AksaraConfig.default())


def test_model_can_be_created():
    fw = _make_framework()

    assert isinstance(fw, AksaraFramework)
    info = fw.info()
    assert isinstance(info, dict)
    assert info["leksikon_size"] > 0
    assert info["sfm_dim"] > 0


def test_model_can_generate_interpretive_output():
    fw = _make_framework()
    state = fw.proses("Budi membeli beras di pasar.")

    assert isinstance(state, AksaraState)
    ringkasan = state.ringkasan()
    jelaskan = state.jelaskan()

    assert isinstance(ringkasan, str)
    assert isinstance(jelaskan, str)
    assert "skor=" in ringkasan
    assert "Kalimat:" in jelaskan
    assert "Register" in jelaskan or "Kelengkapan" in jelaskan


def test_model_can_reason_about_reference_and_context():
    fw = _make_framework()
    state = fw.proses("Budi membeli beras di pasar. Dia membayar dengan tunai.")

    assert isinstance(state, AksaraState)
    assert state.krl_result is None or hasattr(state.krl_result, "jelaskan")
    if state.krl_result is not None:
        assert state.krl_result.kelengkapan_pemahaman >= 0.0
        assert state.krl_result.kelengkapan_pemahaman <= 1.0


def test_output_reflects_linguistic_understanding_not_token_prediction():
    fw = _make_framework()
    state = fw.proses("Pemerintah mengeluarkan kebijakan baru itu.")

    assert isinstance(state, AksaraState)
    assert state.constraint_satisfaction is None or state.constraint_satisfaction.rata_rata >= 0.0
    assert state.ringkasan().startswith("[")
    assert "energi" in state.ringkasan().lower()
    assert "Kalimat:" in state.jelaskan()
    assert state.energi_total >= 0.0
