"""
CorrectnessLoss — Loss untuk training AKSARA sebagai evaluator kebenaran kalimat.

Tidak ada cross-entropy prediksi token. Tidak ada autoregressive loss.

Komponen:
  L_binary   : BCE loss — skor total harus tinggi untuk kalimat benar, rendah untuk salah
  L_margin   : Margin loss — skor kalimat benar harus lebih tinggi dari yang salah (min margin=0.3)
  L_consist  : Konsistensi antar skor — morph/struct/semantic tidak boleh terlalu divergen
  L_calibrate: Skor tidak boleh collapse ke 0.5 (dull point) — dorong distribusi bimodal

Total = λ_b * L_binary + λ_m * L_margin + λ_c * L_consist + λ_cal * L_calibrate
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectnessLoss(nn.Module):
    """
    Loss untuk training AKSARA sebagai evaluator kebenaran kalimat.

    Tidak ada prediksi token. Tidak ada autoregressive loss.

    Input per batch selalu berupa pasangan (benar, salah) yang di-interleave:
      index genap  → kalimat benar  (label = 1.0)
      index ganjil → kalimat salah  (label = 0.0)

    Komponen:
      L_binary   : BCE(skor_total, label) — sinyal utama
      L_margin   : skor_benar harus > skor_salah + margin (default 0.3)
      L_consist  : variance antar 4 sub-skor tidak boleh terlalu besar
    """

    def __init__(
        self,
        margin: float = 0.3,
        lambda_binary: float = 1.0,
        lambda_margin: float = 0.3,
        lambda_consist: float = 0.1,
    ):
        super().__init__()
        self.margin        = margin
        self.lambda_binary = lambda_binary
        self.lambda_margin = lambda_margin
        self.lambda_consist= lambda_consist
        self.bce = nn.BCELoss()

    def forward(
        self,
        score_total: torch.Tensor,
        scores: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            score_total : (B,) — skor gabungan dari CorrectnessHead
            scores      : dict {morph, struct, semantic, lexical, total} — (B,) tiap key
            labels      : (B,) float — 1.0 benar, 0.0 salah

        Returns:
            dict losses dengan semua komponen + total
        """
        device = score_total.device
        losses: Dict[str, torch.Tensor] = {}

        # ── L_binary: BCE — skor total harus sesuai label ─────────────────
        l_binary = self.bce(score_total, labels)
        losses["l_binary"] = torch.nan_to_num(l_binary, nan=0.0)

        # ── L_margin: skor kalimat benar harus lebih tinggi dari yang salah ─
        # Pisahkan skor berdasarkan label
        pos_mask = labels > 0.5   # (B,) — indeks kalimat benar
        neg_mask = ~pos_mask      # (B,) — indeks kalimat salah

        if pos_mask.any() and neg_mask.any():
            s_pos = score_total[pos_mask].mean()   # rata-rata skor benar
            s_neg = score_total[neg_mask].mean()   # rata-rata skor salah
            gap = s_pos - s_neg
            # Soft margin: hanya penalize jika gap NEGATIF atau sangat kecil
            # Saat model random (gap=0), loss = 0 — tidak inflate
            l_margin = F.relu(-gap)
        else:
            l_margin = torch.tensor(0.0, device=device)
        losses["l_margin"] = torch.nan_to_num(l_margin, nan=0.0)

        # ── L_consist: konsistensi antar 4 sub-skor ───────────────────────
        sub_scores = torch.stack([
            scores["morph"], scores["struct"],
            scores["semantic"], scores["lexical"],
        ], dim=1)  # (B, 4)
        # Variance antar sub-skor per kalimat — harus kecil (skor konsisten)
        l_consist = sub_scores.var(dim=1).mean()
        losses["l_consist"] = torch.nan_to_num(l_consist, nan=0.0)

        # ── Total ─────────────────────────────────────────────────────────
        total = (
            self.lambda_binary  * losses["l_binary"]
            + self.lambda_margin  * losses["l_margin"]
            + self.lambda_consist * losses["l_consist"]
        )
        losses["total"] = torch.nan_to_num(total, nan=0.0)

        return losses


class AksaraLoss(CorrectnessLoss):
    """Alias backward-compat. Gunakan CorrectnessLoss."""
    def __init__(self, *args, **kwargs):
        super().__init__()
