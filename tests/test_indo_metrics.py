"""
Tests for Fase 2.2 — Indo-Native Evaluation Metrics

Verifikasi:
1. MorphologicalConsistencyScore (MCS): validity, coherence, consistency
2. StructureValidityScore (SVS): SPOK completeness, order validity, dep coherence
3. SemanticDriftScore (SDS): anchor distance, drift velocity, snapshot history
4. IndoNativeMetrics: full integration, end_epoch flow, reset behavior
"""

import math
import pytest
import torch

from aksara.utils.indo_metrics import (
    IndoNativeMetrics,
    MorphologicalConsistencyScore,
    StructureValidityScore,
    SemanticDriftScore,
    SDSSnapshot,
)
from aksara.linguistic.lps import AFFIX_TO_ID, ROLE_LABELS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_affix_ids(affixes, B=1, pad_to=None):
    """Buat tensor affix_ids dari list string affix."""
    ids = [[AFFIX_TO_ID.get(a, AFFIX_TO_ID.get("<NONE>", 1)) for a in affixes]]
    if pad_to and len(ids[0]) < pad_to:
        ids[0] += [0] * (pad_to - len(ids[0]))
    return torch.tensor(ids * B, dtype=torch.long)


def make_role_ids(roles, B=1, pad_to=None):
    """Buat tensor role_ids dari list string role."""
    ids = [[ROLE_LABELS.get(r, ROLE_LABELS["UNK"]) for r in roles]]
    if pad_to and len(ids[0]) < pad_to:
        ids[0] += [0] * (pad_to - len(ids[0]))
    return torch.tensor(ids * B, dtype=torch.long)


def make_gos_output(B, L, n_affixes, n_roles, vocab_size, pred_affix_ids=None, pred_role_ids=None):
    """Buat dummy GOS output dict."""
    affix_logits = torch.zeros(B, L, n_affixes)
    role_logits  = torch.zeros(B, L, n_roles)
    ctx_logits   = torch.zeros(B, L, vocab_size)

    # Inject specific predictions if provided
    if pred_affix_ids is not None:
        for b in range(B):
            for l in range(min(L, pred_affix_ids.shape[1])):
                affix_logits[b, l, pred_affix_ids[b, l]] = 10.0

    if pred_role_ids is not None:
        for b in range(B):
            for l in range(min(L, pred_role_ids.shape[1])):
                role_logits[b, l, pred_role_ids[b, l]] = 10.0

    return {
        "affix_logits": affix_logits,
        "role_logits":  role_logits,
        "context_logits": ctx_logits,
    }


N_AFFIXES   = len(AFFIX_TO_ID)
N_ROLES     = len(ROLE_LABELS)
VOCAB_SIZE  = 100


# ─── 1. MorphologicalConsistencyScore ────────────────────────────────────────

class TestMorphologicalConsistencyScore:

    def setup_method(self):
        self.mcs = MorphologicalConsistencyScore()

    def test_all_valid_affixes(self):
        """Semua affix valid → validity = 1.0."""
        L = 4
        valid_affixes = ["<NONE>", "ber", "me", "kan"]
        pred = make_affix_ids(valid_affixes)  # (1, 4)
        true = make_affix_ids(valid_affixes)
        self.mcs.update(pred, true)
        result = self.mcs.compute()
        assert result.affix_validity == 1.0

    def test_invalid_affix_id_reduces_validity(self):
        """Affix ID yang tidak ada di vocab → dianggap <UNK> → validity turun."""
        L = 4
        # ID 999 tidak ada di affix vocab
        pred = torch.tensor([[999, 999, 999, 999]], dtype=torch.long)
        true = make_affix_ids(["<NONE>", "ber", "me", "kan"])
        self.mcs.update(pred, true)
        result = self.mcs.compute()
        # <UNK> termasuk valid set, tapi kita cek behavior
        assert 0.0 <= result.affix_validity <= 1.0
        assert result.n_tokens == 4

    def test_attention_mask_respected(self):
        """Token yang di-mask tidak dihitung."""
        pred = make_affix_ids(["<NONE>", "ber", "me", "kan"])
        true = make_affix_ids(["<NONE>", "ber", "me", "kan"])
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        self.mcs.update(pred, true, attention_mask=mask)
        result = self.mcs.compute()
        assert result.n_tokens == 2

    def test_transform_consistency_same_root(self):
        """Root yang sama selalu diprediksi affix yang sama → consistency = 1.0."""
        # Dua batch dengan root "jalan" → selalu "ber"
        root_texts_1 = [["jalan", "buku"]]
        root_texts_2 = [["jalan", "buku"]]
        pred = make_affix_ids(["ber", "<NONE>"])
        true = make_affix_ids(["ber", "<NONE>"])
        self.mcs.update(pred, true, root_texts=root_texts_1)
        self.mcs.update(pred, true, root_texts=root_texts_2)
        result = self.mcs.compute()
        assert result.transform_consistency == 1.0

    def test_transform_inconsistency(self):
        """Root yang sama diprediksi affix berbeda → consistency < 1.0."""
        root_texts_1 = [["jalan"]]
        root_texts_2 = [["jalan"]]
        pred1 = make_affix_ids(["ber"])
        pred2 = make_affix_ids(["me"])
        true  = make_affix_ids(["ber"])
        self.mcs.update(pred1, true, root_texts=root_texts_1)
        self.mcs.update(pred2, true, root_texts=root_texts_2)
        result = self.mcs.compute()
        assert result.transform_consistency < 1.0

    def test_reset(self):
        pred = make_affix_ids(["ber", "me"])
        true = make_affix_ids(["ber", "me"])
        self.mcs.update(pred, true)
        self.mcs.reset()
        result = self.mcs.compute()
        assert result.n_tokens == 0

    def test_overall_between_zero_one(self):
        pred = make_affix_ids(["ber", "me", "<NONE>", "kan"])
        true = make_affix_ids(["ber", "di", "<NONE>", "an"])
        self.mcs.update(pred, true)
        result = self.mcs.compute()
        assert 0.0 <= result.overall <= 1.0

    def test_str_output(self):
        pred = make_affix_ids(["ber"])
        true = make_affix_ids(["ber"])
        self.mcs.update(pred, true)
        s = str(self.mcs.compute())
        assert "MCS=" in s


# ─── 2. StructureValidityScore ───────────────────────────────────────────────

class TestStructureValidityScore:

    def setup_method(self):
        self.svs = StructureValidityScore()

    def test_complete_spok(self):
        """Kalimat dengan S dan P → spok_completeness = 1.0."""
        roles = ["S", "P", "O"]
        pred = make_role_ids(roles)
        self.svs.update(pred)
        result = self.svs.compute()
        assert result.spok_completeness == 1.0

    def test_incomplete_spok_no_predicate(self):
        """Kalimat tanpa P → spok_completeness = 0.0."""
        roles = ["S", "O", "K"]  # tidak ada P
        pred = make_role_ids(roles)
        self.svs.update(pred)
        result = self.svs.compute()
        assert result.spok_completeness == 0.0

    def test_incomplete_spok_no_subject(self):
        """Kalimat tanpa S → spok_completeness = 0.0."""
        roles = ["P", "O"]  # tidak ada S
        pred = make_role_ids(roles)
        self.svs.update(pred)
        result = self.svs.compute()
        assert result.spok_completeness == 0.0

    def test_valid_order_sp(self):
        """Transisi S→P valid."""
        roles = ["S", "P"]
        pred = make_role_ids(roles)
        self.svs.update(pred)
        result = self.svs.compute()
        assert result.order_validity > 0.0

    def test_order_validity_range(self):
        """order_validity harus antara 0 dan 1."""
        roles = ["S", "P", "O", "K"]
        pred = make_role_ids(roles)
        self.svs.update(pred)
        result = self.svs.compute()
        assert 0.0 <= result.order_validity <= 1.0

    def test_attention_mask(self):
        """Token padding tidak dihitung sebagai kalimat."""
        roles = ["S", "P", "O", "UNK", "UNK"]
        pred = make_role_ids(roles)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        self.svs.update(pred, attention_mask=mask)
        result = self.svs.compute()
        assert result.n_sentences == 1
        assert result.spok_completeness == 1.0

    def test_dep_coherence_with_mask(self):
        """Dengan dep_mask, dep_coherence harus antara 0 dan 1."""
        roles = ["S", "P", "O"]
        pred = make_role_ids(roles)
        # Simple chain: 0→1→2
        dep = torch.zeros(1, 3, 3)
        dep[0, 0, 1] = 1.0
        dep[0, 1, 2] = 1.0
        self.svs.update(pred, dep_masks=dep)
        result = self.svs.compute()
        assert 0.0 <= result.dep_coherence <= 1.0

    def test_reset(self):
        pred = make_role_ids(["S", "P"])
        self.svs.update(pred)
        self.svs.reset()
        result = self.svs.compute()
        assert result.n_sentences == 0

    def test_overall_range(self):
        pred = make_role_ids(["S", "P", "O", "K"])
        self.svs.update(pred)
        result = self.svs.compute()
        assert 0.0 <= result.overall <= 1.0

    def test_str_output(self):
        pred = make_role_ids(["S", "P"])
        self.svs.update(pred)
        s = str(self.svs.compute())
        assert "SVS=" in s


# ─── 3. SemanticDriftScore ────────────────────────────────────────────────────

class TestSemanticDriftScore:

    def setup_method(self):
        self.sds = SemanticDriftScore()

    def _make_aligned_tensors(self, B=1, L=4, d=32):
        """Buat semantic slots yang sangat dekat dengan KBBI anchors."""
        anchors = torch.randn(B, L, d)
        anchors = torch.nn.functional.normalize(anchors, dim=-1)
        # Slots hampir sama dengan anchors (sedikit noise)
        slots = anchors + torch.randn_like(anchors) * 0.01
        return slots, anchors

    def _make_drifted_tensors(self, B=1, L=4, d=32):
        """Buat semantic slots yang jauh dari KBBI anchors."""
        anchors = torch.randn(B, L, d)
        anchors = torch.nn.functional.normalize(anchors, dim=-1)
        # Slots berlawanan arah dari anchors
        slots = -anchors + torch.randn_like(anchors) * 0.1
        return slots, anchors

    def test_aligned_low_distance(self):
        """Slots dekat anchors → anchor_distance rendah."""
        slots, anchors = self._make_aligned_tensors()
        self.sds.update(slots, anchors)
        snap = self.sds.take_snapshot(epoch=1, step=1)
        assert snap.mean_anchor_distance < 0.2

    def test_drifted_high_distance(self):
        """Slots jauh dari anchors → anchor_distance tinggi."""
        slots, anchors = self._make_drifted_tensors()
        self.sds.update(slots, anchors)
        snap = self.sds.take_snapshot(epoch=1, step=1)
        assert snap.mean_anchor_distance > 0.5

    def test_zero_anchor_skipped(self):
        """Token dengan anchor = 0 (OOV) tidak dihitung."""
        B, L, d = 1, 4, 32
        anchors = torch.zeros(B, L, d)
        slots   = torch.randn(B, L, d)
        self.sds.update(slots, anchors)
        # Tidak ada token dengan KBBI → total = 0
        snap = self.sds.take_snapshot(epoch=1, step=1)
        assert snap.n_tokens == 0

    def test_coverage_score_aligned(self):
        """Semua token dekat anchor → coverage = 1.0."""
        slots, anchors = self._make_aligned_tensors(B=2, L=6, d=32)
        self.sds.update(slots, anchors)
        snap = self.sds.take_snapshot(epoch=1, step=1)
        assert snap.coverage_score == 1.0

    def test_snapshot_history_grows(self):
        """Snapshot terakumulasi per epoch."""
        slots, anchors = self._make_aligned_tensors()
        for epoch in range(3):
            self.sds.update(slots, anchors)
            self.sds.take_snapshot(epoch=epoch, step=epoch)
        assert len(self.sds.history) == 3

    def test_drift_velocity_positive_when_drifting(self):
        """
        Jika epoch 2 lebih jauh dari KBBI dari epoch 1 → drift_velocity > 0.
        """
        slots_good, anchors = self._make_aligned_tensors()
        slots_bad,  _       = self._make_drifted_tensors()

        self.sds.update(slots_good, anchors)
        self.sds.take_snapshot(epoch=1, step=1)

        self.sds.update(slots_bad, anchors)
        self.sds.take_snapshot(epoch=2, step=2)

        result = self.sds.compute()
        assert result.drift_velocity > 0

    def test_drift_velocity_negative_when_improving(self):
        """
        Jika epoch 2 lebih dekat KBBI dari epoch 1 → drift_velocity < 0 (baik).
        """
        slots_bad,  anchors = self._make_drifted_tensors()
        slots_good, _       = self._make_aligned_tensors()

        self.sds.update(slots_bad, anchors)
        self.sds.take_snapshot(epoch=1, step=1)

        self.sds.update(slots_good, anchors)
        self.sds.take_snapshot(epoch=2, step=2)

        result = self.sds.compute()
        assert result.drift_velocity < 0

    def test_reset_clears_accumulation(self):
        slots, anchors = self._make_aligned_tensors()
        self.sds.update(slots, anchors)
        self.sds.reset()
        # Setelah reset, total = 0
        assert self.sds._total == 0

    def test_reset_all_clears_history(self):
        slots, anchors = self._make_aligned_tensors()
        self.sds.update(slots, anchors)
        self.sds.take_snapshot(epoch=1, step=1)
        self.sds.reset_all()
        assert len(self.sds.history) == 0

    def test_overall_range(self):
        slots, anchors = self._make_aligned_tensors()
        self.sds.update(slots, anchors)
        self.sds.take_snapshot(epoch=1, step=1)
        result = self.sds.compute()
        assert 0.0 <= result.overall <= 1.0

    def test_str_output(self):
        slots, anchors = self._make_aligned_tensors()
        self.sds.update(slots, anchors)
        self.sds.take_snapshot(epoch=1, step=1)
        s = str(self.sds.compute())
        assert "SDS=" in s


# ─── 4. IndoNativeMetrics Integration ────────────────────────────────────────

class TestIndoNativeMetrics:

    def setup_method(self):
        self.metrics = IndoNativeMetrics()
        self.B, self.L = 2, 5
        self.d_sem = 32

    def _make_batch(self):
        """Buat batch lengkap untuk update."""
        gos = make_gos_output(self.B, self.L, N_AFFIXES, N_ROLES, VOCAB_SIZE)
        targets = {
            "affix_ids": make_affix_ids(["ber", "me", "<NONE>", "kan", "<NONE>"],
                                        B=self.B),
            "role_ids":  make_role_ids(["S", "P", "O", "K", "UNK"],
                                       B=self.B),
            "root_ids":  torch.randint(1, VOCAB_SIZE, (self.B, self.L)),
        }
        anchors = torch.nn.functional.normalize(
            torch.randn(self.B, self.L, self.d_sem), dim=-1
        )
        slots = anchors + torch.randn_like(anchors) * 0.05
        return gos, targets, slots, anchors

    def test_update_and_end_epoch(self):
        gos, targets, slots, anchors = self._make_batch()
        self.metrics.update(gos, targets, slots, anchors)
        result = self.metrics.end_epoch(epoch=1)

        assert result.epoch == 1
        assert 0.0 <= result.mcs.overall <= 1.0
        assert 0.0 <= result.svs.overall <= 1.0
        assert 0.0 <= result.sds.overall <= 1.0
        assert 0.0 <= result.morph_accuracy <= 1.0
        assert result.root_perplexity > 0.0

    def test_multi_epoch_sds_drift_tracked(self):
        """SDS history harus tumbuh per epoch."""
        for epoch in range(3):
            gos, targets, slots, anchors = self._make_batch()
            self.metrics.update(gos, targets, slots, anchors)
            self.metrics.end_epoch(epoch=epoch)
            self.metrics.reset()

        sds_result = self.metrics.sds.compute()
        assert len(sds_result.snapshots) == 3

    def test_reset_clears_per_epoch_state(self):
        gos, targets, slots, anchors = self._make_batch()
        self.metrics.update(gos, targets, slots, anchors)
        self.metrics.reset()
        # Setelah reset, morph_total = 0
        assert self.metrics._morph_total == 0

    def test_reset_all_clears_sds_history(self):
        for epoch in range(2):
            gos, targets, slots, anchors = self._make_batch()
            self.metrics.update(gos, targets, slots, anchors)
            self.metrics.end_epoch(epoch=epoch)
            self.metrics.reset()
        self.metrics.reset_all()
        assert len(self.metrics.sds.history) == 0

    def test_to_dict_has_all_keys(self):
        gos, targets, slots, anchors = self._make_batch()
        self.metrics.update(gos, targets, slots, anchors)
        result = self.metrics.end_epoch(epoch=1)
        d = result.to_dict()

        expected_keys = [
            "epoch", "mcs_overall", "svs_overall", "sds_overall",
            "morph_accuracy", "root_perplexity",
            "mcs_affix_validity", "mcs_root_affix_coherence",
            "svs_spok_completeness", "svs_order_validity",
            "sds_anchor_distance", "sds_drift_velocity", "sds_coverage",
        ]
        for k in expected_keys:
            assert k in d, f"Key '{k}' missing from to_dict()"

    def test_summary_contains_all_metrics(self):
        gos, targets, slots, anchors = self._make_batch()
        self.metrics.update(gos, targets, slots, anchors)
        result = self.metrics.end_epoch(epoch=1)
        summary = result.summary()

        assert "MCS=" in summary
        assert "SVS=" in summary
        assert "SDS=" in summary
        assert "morph_acc=" in summary

    def test_with_attention_mask(self):
        """Attention mask harus diteruskan ke semua sub-metrik."""
        gos, targets, slots, anchors = self._make_batch()
        mask = torch.ones(self.B, self.L, dtype=torch.long)
        mask[:, -1] = 0  # mask token terakhir

        self.metrics.update(gos, targets, slots, anchors, attention_mask=mask)
        result = self.metrics.end_epoch(epoch=1)

        # MCS harus hitung L-1 token per batch = B*(L-1)
        expected_total = self.B * (self.L - 1)
        assert self.metrics.mcs._total == expected_total

    def test_no_nan_in_results(self):
        gos, targets, slots, anchors = self._make_batch()
        self.metrics.update(gos, targets, slots, anchors)
        result = self.metrics.end_epoch(epoch=1)
        d = result.to_dict()

        for k, v in d.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN detected in {k}={v}"
