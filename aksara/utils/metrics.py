"""
AksaraMetrics - Metrik evaluasi khusus untuk AKSARA.
Mengukur aspek linguistik Indonesia, bukan hanya perplexity.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from aksara.linguistic.lps import MorfologiAnalyzer, AFFIX_TO_ID


class AksaraMetrics:
    """
    Metrik evaluasi untuk model AKSARA.

    Berbeda dari metrik LLM biasa — ini mengukur:
    1. Morfologi accuracy: seberapa sering affix prediction benar
    2. Struktur accuracy: seberapa sering S-P-O-K prediction benar
    3. KBBI alignment: seberapa dekat semantic slots ke KBBI anchors
    4. Perplexity di level root word (bukan subword token)
    """

    def __init__(self):
        self.analyzer = MorfologiAnalyzer()
        self._reset()

    def _reset(self):
        self._morph_correct = 0
        self._morph_total = 0
        self._struct_correct = 0
        self._struct_total = 0
        self._kbbi_sim_sum = 0.0
        self._kbbi_count = 0
        self._ctx_nll_sum = 0.0
        self._ctx_steps = 0

    def update(
        self,
        gos_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        semantic_slots: torch.Tensor,
        kbbi_anchors: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Update metrik dari satu batch.

        Args:
            gos_output     : output dari GOS (logits)
            targets        : ground truth labels
            semantic_slots : (B, L, d_sem)
            kbbi_anchors   : (B, L, d_sem)
            attention_mask : (B, L) — 1 untuk real token
        """
        with torch.no_grad():
            # --- Morfologi Accuracy ---
            pred_affix = gos_output["affix_logits"].argmax(dim=-1)  # (B, L)
            true_affix = targets["affix_ids"]                        # (B, L)

            if attention_mask is not None:
                valid = attention_mask.bool()
                self._morph_correct += (pred_affix[valid] == true_affix[valid]).sum().item()
                self._morph_total += valid.sum().item()
            else:
                self._morph_correct += (pred_affix == true_affix).sum().item()
                self._morph_total += pred_affix.numel()

            # --- Struktur Accuracy (role S/P/O/K) ---
            pred_role = gos_output["role_logits"].argmax(dim=-1)   # (B, L)
            true_role = targets["role_ids"]                         # (B, L)
            # Hanya hitung untuk token yang punya role label (bukan UNK=0)
            role_valid = (true_role > 0)
            if attention_mask is not None:
                role_valid = role_valid & attention_mask.bool()

            if role_valid.any():
                self._struct_correct += (pred_role[role_valid] == true_role[role_valid]).sum().item()
                self._struct_total += role_valid.sum().item()

            # --- KBBI Alignment (cosine similarity) ---
            if semantic_slots.shape == kbbi_anchors.shape:
                cos_sim = F.cosine_similarity(semantic_slots, kbbi_anchors, dim=-1)  # (B, L)
                # Hanya untuk token dengan KBBI entry (anchor != 0)
                has_kbbi = (kbbi_anchors.abs().sum(dim=-1) > 0)
                if attention_mask is not None:
                    has_kbbi = has_kbbi & attention_mask.bool()
                if has_kbbi.any():
                    self._kbbi_sim_sum += cos_sim[has_kbbi].sum().item()
                    self._kbbi_count += has_kbbi.sum().item()

            # --- Context NLL (perplexity di level root word) ---
            ctx_logits = gos_output["context_logits"]    # (B, L, V)
            true_root = targets["root_ids"]              # (B, L)

            if ctx_logits.shape[1] > 1:
                logits_shift = ctx_logits[:, :-1, :].contiguous()
                targets_shift = true_root[:, 1:].contiguous()

                nll = F.cross_entropy(
                    logits_shift.view(-1, logits_shift.size(-1)),
                    targets_shift.view(-1),
                    ignore_index=0,
                    reduction="mean",
                )
                self._ctx_nll_sum += nll.item()
                self._ctx_steps += 1

    def compute(self) -> Dict[str, float]:
        """
        Hitung semua metrik dari akumulasi update.

        Returns:
            dict metrik:
                morph_accuracy  : akurasi prediksi affix (0..1)
                struct_accuracy : akurasi prediksi S-P-O-K (0..1)
                kbbi_alignment  : rata-rata cosine similarity ke KBBI
                root_perplexity : perplexity di level root word
        """
        results = {}

        # Morfologi accuracy
        if self._morph_total > 0:
            results["morph_accuracy"] = self._morph_correct / self._morph_total
        else:
            results["morph_accuracy"] = 0.0

        # Struktur accuracy
        if self._struct_total > 0:
            results["struct_accuracy"] = self._struct_correct / self._struct_total
        else:
            results["struct_accuracy"] = None  # Tidak ada label role

        # KBBI alignment
        if self._kbbi_count > 0:
            results["kbbi_alignment"] = self._kbbi_sim_sum / self._kbbi_count
        else:
            results["kbbi_alignment"] = 0.0

        # Root perplexity
        if self._ctx_steps > 0:
            import math
            avg_nll = self._ctx_nll_sum / self._ctx_steps
            results["root_perplexity"] = math.exp(min(avg_nll, 20))  # cap untuk stabilitas
        else:
            results["root_perplexity"] = float("inf")

        return results

    def reset(self):
        """Reset semua akumulasi."""
        self._reset()

    @staticmethod
    def evaluate_morfologi(
        texts: List[str],
        predicted_affixes: List[List[str]],
        id_to_affix: Dict[int, str],
    ) -> Dict[str, float]:
        """
        Evaluasi morfologi secara standalone (tanpa model).
        Bandingkan prediksi affix dengan rule-based analyzer.
        """
        analyzer = MorfologiAnalyzer()
        correct = 0
        total = 0

        for text, pred_aff_seq in zip(texts, predicted_affixes):
            words = text.split()
            for word, pred_aff in zip(words, pred_aff_seq):
                _, true_aff = analyzer.best(word)
                if pred_aff == true_aff:
                    correct += 1
                total += 1

        return {"morph_accuracy": correct / max(total, 1)}

    @staticmethod
    def word_level_accuracy(
        pred_ids: torch.Tensor,
        true_ids: torch.Tensor,
        ignore_index: int = 0,
    ) -> float:
        """Akurasi prediksi root word (token-level)."""
        mask = true_ids != ignore_index
        if not mask.any():
            return 0.0
        correct = (pred_ids[mask] == true_ids[mask]).sum().item()
        return correct / mask.sum().item()
