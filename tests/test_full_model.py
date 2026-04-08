"""
Test integrasi AksaraModel v3 — evaluator kebenaran kalimat bahasa Indonesia.
Tidak ada GOS, tidak ada generative. Model mengevaluasi, bukan menghasilkan token.
"""

import pytest
import torch
from aksara.core.model import AksaraModel, AksaraConfig
from aksara.core.bsu import BSUConfig
from aksara.core.meb import MEBConfig
from aksara.core.correctness import CorrectnessConfig
from aksara.linguistic.lps import LPSConfig, build_root_vocab
from aksara.linguistic.lsk import LSKConfig
from aksara.training.loss import CorrectnessLoss
from aksara.data.dataset import AksaraDataset, collate_fn


KALIMAT_BENAR = [
    "saya berjalan di taman setiap pagi",
    "dia membaca buku pelajaran dengan tekun",
    "ibu memasak nasi goreng untuk makan siang",
    "pemerintah menjalankan program pendidikan nasional",
]

KALIMAT_SALAH = [
    "pagi taman di berjalan saya setiap",
    "tekun pelajaran buku membaca dengan dia",
    "goreng nasi memasak siang makan ibu untuk",
    "nasional program pendidikan menjalankan pemerintah",
]

SAMPLE_CORPUS = KALIMAT_BENAR + KALIMAT_SALAH


def make_small_config(kbbi_path: str = "") -> tuple:
    """Buat konfigurasi kecil untuk testing."""
    root_vocab = build_root_vocab(SAMPLE_CORPUS, min_freq=1)

    bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
    meb_cfg = MEBConfig(bsu_config=bsu_cfg, n_layers=2, n_dep_heads=2, kbbi_anchor_dim=16)
    cor_cfg = CorrectnessConfig(bsu_config=bsu_cfg, hidden_dim=64)

    config = AksaraConfig(
        bsu_config=bsu_cfg,
        meb_config=meb_cfg,
        correctness_config=cor_cfg,
        lps_config=LPSConfig(),
        lsk_config=LSKConfig(kbbi_path=kbbi_path),
    )
    return config, root_vocab


class TestAksaraModelForward:

    def setup_method(self):
        self.config, self.vocab = make_small_config()
        self.model = AksaraModel(self.config, self.vocab)
        self.device = torch.device("cpu")

    def test_model_instantiates(self):
        assert self.model is not None
        assert self.model.num_parameters["total"] > 0

    def test_lps_forward(self):
        texts = SAMPLE_CORPUS[:2]
        out = self.model.lps(texts, device=self.device)
        assert "morpheme_ids" in out
        assert "affix_ids" in out
        assert out["morpheme_ids"].shape[0] == 2

    def test_forward_shape(self):
        texts = SAMPLE_CORPUS[:2]
        lps_out = self.model.lps(texts, device=self.device)
        outputs = self.model(lps_out)
        B, L = lps_out["morpheme_ids"].shape
        d = self.config.bsu_config.d_total
        assert outputs["meb_out"].shape == (B, L, d)

    def test_scores_output(self):
        """Forward harus menghasilkan 5 skor: morph, struct, semantic, lexical, total."""
        texts = SAMPLE_CORPUS[:2]
        lps_out = self.model.lps(texts, device=self.device)
        outputs = self.model(lps_out)
        assert "scores" in outputs
        for key in ["morph", "struct", "semantic", "lexical", "total"]:
            assert key in outputs["scores"], f"Skor '{key}' tidak ada"
            s = outputs["scores"][key]
            assert s.shape == (2,), f"Shape skor '{key}' salah: {s.shape}"
            assert ((s >= 0) & (s <= 1)).all(), f"Skor '{key}' di luar [0,1]: {s}"

    def test_forward_with_labels(self):
        """Forward dengan labels harus menghasilkan losses."""
        texts = KALIMAT_BENAR[:2] + KALIMAT_SALAH[:2]
        lps_out = self.model.lps(texts, device=self.device)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        outputs = self.model(lps_out, labels=labels)
        assert "losses" in outputs
        for key in ["l_binary", "l_margin", "l_consist", "total"]:
            assert key in outputs["losses"], f"Loss '{key}' tidak ada"
        total_loss = outputs["losses"]["total"]
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

    def test_forward_no_labels(self):
        """Forward tanpa labels tidak boleh ada 'losses' di output."""
        texts = SAMPLE_CORPUS[:2]
        lps_out = self.model.lps(texts, device=self.device)
        outputs = self.model(lps_out)
        assert "losses" not in outputs
        assert "scores" in outputs
        assert "meb_out" in outputs

    def test_backward_no_nan_grads(self):
        """Backward pass tidak boleh menghasilkan NaN gradients."""
        texts = KALIMAT_BENAR[:2] + KALIMAT_SALAH[:2]
        lps_out = self.model.lps(texts, device=self.device)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        outputs = self.model(lps_out, labels=labels)
        outputs["losses"]["total"].backward()

        nan_params = [
            name for name, p in self.model.named_parameters()
            if p.requires_grad and p.grad is not None and torch.isnan(p.grad).any()
        ]
        assert len(nan_params) == 0, f"NaN gradients in: {nan_params}"

    def test_score_method(self):
        """model.score() harus mengembalikan dict float per kalimat."""
        texts = SAMPLE_CORPUS[:3]
        result = self.model.score(texts)
        assert isinstance(result, dict)
        for key in ["morph", "struct", "semantic", "lexical", "total"]:
            assert key in result
            assert len(result[key]) == 3

    def test_batch_consistency(self):
        """Batch size tidak boleh mempengaruhi output per sampel (eval mode)."""
        self.model.eval()
        text = SAMPLE_CORPUS[0]
        with torch.no_grad():
            lps_single = self.model.lps([text], device=self.device)
            out_single = self.model(lps_single)
            h_single = out_single["meb_out"]

            lps_batch = self.model.lps([text, text], device=self.device)
            out_batch = self.model(lps_batch)
            h_batch = out_batch["meb_out"]

        L = min(h_single.shape[1], h_batch.shape[1])
        assert torch.allclose(h_single[0, :L], h_batch[0, :L], atol=1e-5)


class TestCorrectnessLoss:

    def test_loss_instantiates(self):
        loss = CorrectnessLoss()
        assert loss is not None

    def test_loss_forward(self):
        """Loss harus berjalan dengan benar dan menghasilkan semua komponen."""
        B = 4
        score_total = torch.rand(B)
        scores = {
            "morph":    torch.rand(B),
            "struct":   torch.rand(B),
            "semantic": torch.rand(B),
            "lexical":  torch.rand(B),
            "total":    score_total,
        }
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss_fn = CorrectnessLoss()
        losses = loss_fn(score_total, scores, labels)

        for key in ["l_binary", "l_margin", "l_consist", "total"]:
            assert key in losses
            assert not torch.isnan(losses[key])

    def test_loss_margin_direction(self):
        """Skor benar harus lebih tinggi dari salah setelah beberapa update."""
        loss_fn = CorrectnessLoss(margin=0.3)
        score_total = torch.tensor([0.8, 0.8, 0.2, 0.2])
        scores = {k: score_total.clone() for k in ["morph", "struct", "semantic", "lexical", "total"]}
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        losses = loss_fn(score_total, scores, labels)
        assert losses["l_margin"].item() == 0.0  # margin sudah terpenuhi (0.8-0.2=0.6 > 0.3)


class TestAksaraDatasetIntegration:

    def setup_method(self):
        self.config, self.vocab = make_small_config()
        self.model = AksaraModel(self.config, self.vocab)
        self.dataset = AksaraDataset(SAMPLE_CORPUS, self.vocab, max_length=32)

    def test_dataset_length(self):
        assert len(self.dataset) == len(SAMPLE_CORPUS)

    def test_dataset_item_keys(self):
        item = self.dataset[0]
        assert "morpheme_ids" in item
        assert "affix_ids" in item
        assert "role_ids" in item
        assert "length" in item

    def test_collate_fn(self):
        items = [self.dataset[i] for i in range(3)]
        batch = collate_fn(items)
        assert batch.morpheme_ids.shape[0] == 3
        assert batch.affix_ids.shape == batch.morpheme_ids.shape
        assert batch.role_ids.shape == batch.morpheme_ids.shape

    def test_training_step_from_dataset(self):
        """Training step lengkap dari dataset — forward + backward."""
        from torch.utils.data import DataLoader
        loader = DataLoader(self.dataset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        lps_out = {
            "morpheme_ids": batch.morpheme_ids,
            "affix_ids":    batch.affix_ids,
            "dep_masks":    None,
            "lengths":      batch.lengths,
        }
        # Label bergantian benar/salah untuk test
        B = batch.morpheme_ids.shape[0]
        labels = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(B)])

        outputs = self.model(lps_out, labels=labels)
        assert "losses" in outputs
        loss = outputs["losses"]["total"]
        assert not torch.isnan(loss)
        loss.backward()
