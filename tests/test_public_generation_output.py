"""
test_public_generation_output.py
================================
Test publik sederhana untuk memastikan jalur generasi/reasoning tetap dapat dijalankan.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aksara.framework import AksaraFramework
from aksara.config import AksaraConfig


def test_public_generation_output_has_text() -> None:
    kbbi_path = Path("kbbi_core_v2.json")
    if not kbbi_path.exists():
        return

    fw = AksaraFramework.dari_kbbi(str(kbbi_path), config=AksaraConfig.default())
    state = fw.proses("Budi membeli beras di pasar.")

    assert state is not None
    assert hasattr(state, "ringkasan")
    assert hasattr(state, "jelaskan")
    assert isinstance(state.ringkasan(), str)
    assert isinstance(state.jelaskan(), str)
