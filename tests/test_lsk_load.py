from pathlib import Path

import pytest

from aksara.linguistic.lsk import KBBIStore, LSKConfig, LapisanSemantikKBBI


KBBI_PATH = Path("kbbi_core_v2.json")


def test_kbbi_file_exists():
    assert KBBI_PATH.exists(), "kbbi_core_v2.json should exist in the project root"


def test_kbbi_store_lookup_returns_tuple_and_pos_list():
    store = KBBIStore(str(KBBI_PATH), max_lemmas=80000)

    word_id, confidence = store.lookup("makan")
    assert isinstance(word_id, int)
    assert isinstance(confidence, float)
    assert word_id >= 0
    assert 0.0 <= confidence <= 1.0

    pos_list = store.get_pos_list("makan")
    assert isinstance(pos_list, list)
    assert all(isinstance(pos, str) for pos in pos_list)
    assert pos_list, "Expected at least one POS tag for a common lemma"


def test_kbbi_store_oov_uses_zero_id_and_low_confidence():
    store = KBBIStore(str(KBBI_PATH), max_lemmas=80000)

    word_id, confidence = store.lookup("asdfghjkl")
    assert word_id == 0
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert confidence <= 0.75


def test_lsk_initialization_and_stats():
    vocab = {"<PAD>": 0, "<UNK>": 1, "makan": 2, "xyz": 3}
    config = LSKConfig(kbbi_path=str(KBBI_PATH), kbbi_vector_dim=16, max_lemmas=80000)
    lsk = LapisanSemantikKBBI(config, vocab)

    stats = lsk.get_stats()
    assert isinstance(stats, dict)
    assert "kbbi_loaded" in stats
    assert "kbbi_entries" in stats
    assert "kbbi_unique_lemmas" in stats
    assert isinstance(lsk._initial_coverage, float)
    assert 0.0 <= lsk._initial_coverage <= 1.0