"""
Test MEB - Mesin Evolusi Bahasa
"""

import pytest
import torch
from aksara.core.bsu import BSUConfig
from aksara.core.meb import (
    MesinEvolusiBahasa,
    MEBConfig,
    MorphologyEvolution,
    SyntacticEvolution,
    SemanticGrounding,
    PhiLayer,
)


class TestMorphologyEvolution:

    def setup_method(self):
        self.d_m = 32
        self.affix_vocab = 50
        self.f_morph = MorphologyEvolution(self.d_m, self.affix_vocab)
        self.B, self.L = 2, 8

    def test_output_shape(self):
        h_morph = torch.randn(self.B, self.L, self.d_m)
        affix_ids = torch.randint(0, self.affix_vocab, (self.B, self.L))
        out = self.f_morph(h_morph, affix_ids)
        assert out.shape == (self.B, self.L, self.d_m)

    def test_gradient_flow(self):
        h_morph = torch.randn(self.B, self.L, self.d_m, requires_grad=True)
        affix_ids = torch.randint(0, self.affix_vocab, (self.B, self.L))
        out = self.f_morph(h_morph, affix_ids)
        out.sum().backward()
        assert h_morph.grad is not None


class TestSyntacticEvolution:

    def setup_method(self):
        self.bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        self.d = self.bsu_cfg.d_total
        self.f_syn = SyntacticEvolution(self.d, self.bsu_cfg.d_role, n_heads=4)
        self.B, self.L = 2, 8

    def test_output_shape_no_mask(self):
        h = torch.randn(self.B, self.L, self.d)
        h_role = torch.randn(self.B, self.L, self.bsu_cfg.d_role)
        out = self.f_syn(h, h_role, dep_mask=None)
        assert out.shape == (self.B, self.L, self.d)

    def test_output_shape_with_mask(self):
        h = torch.randn(self.B, self.L, self.d)
        h_role = torch.randn(self.B, self.L, self.bsu_cfg.d_role)
        dep_mask = torch.ones(self.B, self.L, self.L, dtype=torch.bool)
        out = self.f_syn(h, h_role, dep_mask=dep_mask)
        assert out.shape == (self.B, self.L, self.d)

    def test_sparse_mask_not_nan(self):
        """Dengan sparse mask (banyak isolated token), tidak boleh ada NaN."""
        h = torch.randn(self.B, self.L, self.d)
        h_role = torch.randn(self.B, self.L, self.bsu_cfg.d_role)
        # Hanya self-loop
        dep_mask = torch.eye(self.L, dtype=torch.bool).unsqueeze(0).expand(self.B, -1, -1)
        out = self.f_syn(h, h_role, dep_mask=dep_mask)
        assert not torch.isnan(out).any()

    def test_local_window_mask(self):
        mask = SyntacticEvolution._local_window_mask(5, window=1, device=torch.device("cpu"))
        assert mask.shape == (5, 5)
        # Token 0 hanya attend ke 0,1
        assert mask[0, 0].item() is True
        assert mask[0, 1].item() is True
        assert mask[0, 2].item() is False


class TestSemanticGrounding:

    def setup_method(self):
        self.d_sem = 32
        self.kbbi_dim = 16
        self.f_sem = SemanticGrounding(self.d_sem, self.kbbi_dim)
        self.B, self.L = 2, 8

    def test_output_shape(self):
        h_sem = torch.randn(self.B, self.L, self.d_sem)
        anchors = torch.randn(self.B, self.L, self.kbbi_dim)
        out = self.f_sem(h_sem, anchors)
        assert out.shape == (self.B, self.L, self.d_sem)

    def test_zero_anchor_different_output(self):
        """Zero anchor (OOV) harus menghasilkan output berbeda dari non-zero anchor."""
        h_sem = torch.randn(self.B, self.L, self.d_sem)
        zero_anchor = torch.zeros(self.B, self.L, self.kbbi_dim)
        real_anchor = torch.randn(self.B, self.L, self.kbbi_dim)
        out_zero = self.f_sem(h_sem, zero_anchor)
        out_real = self.f_sem(h_sem, real_anchor)
        assert not torch.allclose(out_zero, out_real)


class TestMesinEvolusiBahasa:

    def setup_method(self):
        self.bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        self.cfg = MEBConfig(
            bsu_config=self.bsu_cfg,
            n_layers=2,
            n_dep_heads=2,
            kbbi_anchor_dim=16,
        )
        self.affix_vocab = 50
        self.meb = MesinEvolusiBahasa(self.cfg, self.affix_vocab)
        self.B, self.L = 2, 10
        self.d = self.bsu_cfg.d_total

    def _make_inputs(self):
        bsu_states = torch.randn(self.B, self.L, self.d)
        affix_ids = torch.randint(0, self.affix_vocab, (self.B, self.L))
        kbbi_anchors = torch.randn(self.B, self.L, 16)
        return bsu_states, affix_ids, kbbi_anchors

    def test_output_shape(self):
        bsu, aff, kbbi = self._make_inputs()
        h_final, _ = self.meb(bsu, aff, kbbi)
        assert h_final.shape == (self.B, self.L, self.d)

    def test_return_all_layers(self):
        bsu, aff, kbbi = self._make_inputs()
        h_final, layer_states = self.meb(bsu, aff, kbbi, return_all_layers=True)
        assert len(layer_states) == self.cfg.n_layers
        for ls in layer_states:
            assert ls.shape == (self.B, self.L, self.d)

    def test_output_different_from_input(self):
        """MEB harus mengubah state, bukan identity function."""
        bsu, aff, kbbi = self._make_inputs()
        h_final, _ = self.meb(bsu, aff, kbbi)
        assert not torch.allclose(h_final, bsu)

    def test_gradient_flow(self):
        bsu, aff, kbbi = self._make_inputs()
        bsu.requires_grad_(True)
        h_final, _ = self.meb(bsu, aff, kbbi)
        h_final.sum().backward()
        assert bsu.grad is not None
        assert not torch.isnan(bsu.grad).any()

    def test_with_dep_mask(self):
        bsu, aff, kbbi = self._make_inputs()
        dep_mask = torch.ones(self.B, self.L, self.L, dtype=torch.bool)
        h_final, _ = self.meb(bsu, aff, kbbi, dep_mask=dep_mask)
        assert h_final.shape == (self.B, self.L, self.d)
        assert not torch.isnan(h_final).any()

    def test_n_layers_count(self):
        assert len(self.meb.layers) == self.cfg.n_layers
