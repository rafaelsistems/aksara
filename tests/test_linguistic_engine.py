from pathlib import Path

import torch

from aksara.linguistic.lsk import KBBIStore, LSKConfig, LapisanSemantikKBBI


KBBI_PATH = Path("kbbi_core_v2.json")


def test_linguistic_engine_lookup_and_lsk_forward():
    store = KBBIStore(str(KBBI_PATH), max_lemmas=80000)

    known_id, known_conf = store.lookup("makan")
    unknown_id, unknown_conf = store.lookup("qwertyuiop")

    assert isinstance(known_id, int)
    assert isinstance(known_conf, float)
    assert known_id >= 0
    assert 0.0 <= known_conf <= 1.0

    assert isinstance(unknown_id, int)
    assert isinstance(unknown_conf, float)
    assert unknown_id == 0
    assert 0.0 <= unknown_conf <= 1.0

    vocab = {"<PAD>": 0, "<UNK>": 1, "makan": 2, "besar": 3, "jalan": 4, "xyz": 5}
    config = LSKConfig(kbbi_path=str(KBBI_PATH), kbbi_vector_dim=16, max_lemmas=80000)
    lsk = LapisanSemantikKBBI(config, vocab)

    input_ids = torch.tensor([[2, 3, 4, 5, 0]])
    output = lsk(input_ids)

    assert output.shape == (1, 5, 16)
    assert lsk._total_tokens == 5
    assert lsk._kbbi_hits == 3
    assert 0.0 <= lsk.kbbi_coverage <= 1.0
    assert lsk.get_stats()["kbbi_loaded"] in (True, False)