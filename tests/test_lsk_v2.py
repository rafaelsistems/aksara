from pathlib import Path

import torch

from aksara.linguistic.lsk import KBBIStore, LSKConfig, LapisanSemantikKBBI


KBBI_PATH = Path("kbbi_core_v2.json")


def test_lookup_multi_word_and_case_handling():
    store = KBBIStore(str(KBBI_PATH), max_lemmas=80000)

    for word in ["air mata", "Rumah", "Makan"]:
        word_id, confidence = store.lookup(word)
        assert isinstance(word_id, int)
        assert isinstance(confidence, float)
        assert word_id >= 0
        assert 0.0 <= confidence <= 1.0


def test_lookup_and_pos_list_are_consistent_for_known_lemma():
    store = KBBIStore(str(KBBI_PATH), max_lemmas=80000)

    word_id, confidence = store.lookup("makan")
    pos_list = store.get_pos_list("makan")

    assert word_id > 0
    assert confidence >= 0.75
    assert isinstance(pos_list, list)
    assert pos_list
    assert all(isinstance(pos, str) for pos in pos_list)


def test_lsk_runtime_coverage_tracks_forward_calls():
    vocab = {"<PAD>": 0, "<UNK>": 1, "makan": 2, "besar": 3, "jalan": 4, "xyz": 5}
    config = LSKConfig(kbbi_path=str(KBBI_PATH), kbbi_vector_dim=16, max_lemmas=80000)
    lsk = LapisanSemantikKBBI(config, vocab)

    morpheme_ids = torch.tensor([[2, 3, 4, 5, 0]])
    output_1 = lsk(morpheme_ids)
    output_2 = lsk(morpheme_ids)

    assert output_1.shape == (1, 5, 16)
    assert output_2.shape == (1, 5, 16)
    assert lsk._total_tokens == 10
    assert lsk._kbbi_hits == 6
    assert 0.0 <= lsk.kbbi_coverage <= 1.0
    assert lsk.kbbi_coverage == lsk._kbbi_hits / lsk._total_tokens


def test_lsk_zero_vector_behavior_for_pad_and_oov():
    vocab = {"<PAD>": 0, "<UNK>": 1, "makan": 2, "xyz": 3}
    config = LSKConfig(kbbi_path=str(KBBI_PATH), kbbi_vector_dim=16, max_lemmas=80000)
    lsk = LapisanSemantikKBBI(config, vocab)

    morpheme_ids = torch.tensor([[2, 3, 0]])
    raw = lsk(morpheme_ids, return_raw=True)

    assert raw.shape == (1, 3, 16)
    assert not torch.allclose(raw[0, 0, :], torch.zeros(16), atol=1e-6)
    assert torch.allclose(raw[0, 1, :], torch.zeros(16), atol=1e-6)
    assert torch.allclose(raw[0, 2, :], torch.zeros(16), atol=1e-6)