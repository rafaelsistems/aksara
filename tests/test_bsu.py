"""
Test BSU - Bahasa State Unit
"""

import pytest
import torch
from aksara.core.bsu import BahasaStateUnit, BSUConfig


class TestBSUConfig:

    def test_d_total(self):
        cfg = BSUConfig(d_morpheme=64, d_semantic=64, d_role=32, d_context=64)
        assert cfg.d_total == 224

    def test_slot_offsets(self):
        cfg = BSUConfig(d_morpheme=64, d_semantic=64, d_role=32, d_context=64)
        offsets = cfg.slot_offsets
        assert offsets["morpheme"] == 0
        assert offsets["semantic"] == 64
        assert offsets["role"] == 128
        assert offsets["context"] == 160

    def test_slot_sizes(self):
        cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        sizes = cfg.slot_sizes
        assert sizes["morpheme"] == 32
        assert sizes["semantic"] == 32
        assert sizes["role"] == 16
        assert sizes["context"] == 32


class TestBahasaStateUnit:

    def setup_method(self):
        self.config = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        self.bsu = BahasaStateUnit(
            config=self.config,
            vocab_size=1000,
            affix_vocab_size=50,
            role_vocab_size=8,
            kbbi_input_dim=16,
        )
        self.B, self.L = 2, 10

    def _make_inputs(self, with_roles=False):
        morpheme_ids = torch.randint(0, 1000, (self.B, self.L))
        affix_ids = torch.randint(0, 50, (self.B, self.L))
        kbbi_vectors = torch.randn(self.B, self.L, 16)
        role_ids = torch.randint(0, 8, (self.B, self.L)) if with_roles else None
        return morpheme_ids, affix_ids, kbbi_vectors, role_ids

    def test_output_shape(self):
        m, a, k, _ = self._make_inputs()
        bsu_states, slots = self.bsu(m, a, k)
        assert bsu_states.shape == (self.B, self.L, self.config.d_total)

    def test_slots_keys(self):
        m, a, k, _ = self._make_inputs()
        _, slots = self.bsu(m, a, k)
        assert set(slots.keys()) == {"morpheme", "semantic", "role", "context"}

    def test_slot_dimensions(self):
        m, a, k, r = self._make_inputs(with_roles=True)
        _, slots = self.bsu(m, a, k, r)
        assert slots["morpheme"].shape == (self.B, self.L, self.config.d_morpheme)
        assert slots["semantic"].shape == (self.B, self.L, self.config.d_semantic)
        assert slots["role"].shape == (self.B, self.L, self.config.d_role)
        assert slots["context"].shape == (self.B, self.L, self.config.d_context)

    def test_no_role_ids_zero_role_slot(self):
        m, a, k, _ = self._make_inputs(with_roles=False)
        bsu_states, slots = self.bsu(m, a, k, role_ids=None)
        # Role slot harus nol jika tidak ada role_ids
        assert slots["role"].abs().sum().item() == 0.0

    def test_get_slot_extraction(self):
        m, a, k, _ = self._make_inputs()
        bsu_states, _ = self.bsu(m, a, k)
        morph_extracted = self.bsu.get_slot(bsu_states, "morpheme")
        assert morph_extracted.shape[-1] == self.config.d_morpheme

    def test_gradients_flow(self):
        m, a, k, _ = self._make_inputs()
        bsu_states, _ = self.bsu(m, a, k)
        loss = bsu_states.sum()
        loss.backward()
        for name, param in self.bsu.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_different_kbbi_vectors(self):
        """BSU output harus berbeda untuk KBBI vector yang berbeda."""
        m = torch.randint(0, 1000, (self.B, self.L))
        a = torch.randint(0, 50, (self.B, self.L))
        k1 = torch.randn(self.B, self.L, 16)
        k2 = torch.randn(self.B, self.L, 16)
        out1, _ = self.bsu(m, a, k1)
        out2, _ = self.bsu(m, a, k2)
        assert not torch.allclose(out1, out2)
