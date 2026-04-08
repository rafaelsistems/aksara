"""
PD - Pengendali Dinamik
Bukan scheduler biasa. Ini mengatur lambda weights dan learning rate
secara adaptif berdasarkan signal linguistik dari training.

Mengatur:
  - Bobot lambda untuk setiap komponen loss
  - Learning rate per parameter group
  - Fokus pembelajaran berdasarkan kelemahan model saat ini

Berdasarkan:
  - Error morfologi (l_morph tinggi → tingkatkan λ_morph)
  - Inkonsistensi struktur (l_struct tinggi → tingkatkan λ_struct)
  - Entropy output (entropy tinggi → turunkan lr)
  - Divergence dari KBBI (l_sem tinggi → tingkatkan λ_sem)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn


@dataclass
class PDConfig:
    # Range lambda weights
    lambda_morph_range: Tuple[float, float] = (0.3, 2.0)
    lambda_struct_range: Tuple[float, float] = (0.3, 1.5)
    lambda_sem_range: Tuple[float, float] = (0.2, 1.2)
    lambda_ctx_range: Tuple[float, float] = (0.5, 2.0)
    lambda_ar_range: Tuple[float, float] = (0.5, 3.0)
    # AR range lebih lebar: saat model belum bisa generate, AR loss harus dominan.
    # Setelah convergence, bisa turun untuk beri ruang komponen lain.

    # Nilai awal lambda
    lambda_morph_init: float = 1.0
    lambda_struct_init: float = 0.8
    lambda_sem_init: float = 0.6
    lambda_ctx_init: float = 1.0
    lambda_ar_init: float = 1.5
    # AR lebih tinggi dari default karena phi_seq perlu sinyal kuat di awal training.

    # Adaptasi rate
    adaptation_rate: float = 0.05   # seberapa cepat lambda berubah
    ema_alpha: float = 0.9           # exponential moving average untuk smoothing

    # Threshold untuk adaptasi
    morph_error_threshold: float = 0.3    # jika l_morph > ini, boost lambda
    struct_error_threshold: float = 0.4
    sem_error_threshold: float = 0.5
    ctx_error_threshold: float = 0.3
    ar_error_threshold: float = 0.5
    # AR threshold sengaja lebih tinggi karena AR loss awal memang besar
    # (model belum tahu cara generate). Baru diboost jika tidak turun.

    # Entropy regularization
    entropy_penalty: float = 0.01    # penalti entropy output terlalu tinggi

    # History window untuk trend detection
    history_window: int = 50


class LossHistory:
    """Track riwayat loss untuk deteksi trend."""

    def __init__(self, window: int = 50):
        self.window = window
        self._history: Dict[str, List[float]] = {
            "l_morph": [], "l_struct": [], "l_sem": [], "l_ctx": [],
            "l_ar": [], "total": []
        }
        self._ema: Dict[str, float] = {}

    def update(self, losses: Dict[str, torch.Tensor], alpha: float = 0.9):
        """Update history dengan losses terbaru."""
        for key in self._history:
            if key in losses:
                val = losses[key].item() if torch.is_tensor(losses[key]) else float(losses[key])
                self._history[key].append(val)

                # Trim ke window
                if len(self._history[key]) > self.window:
                    self._history[key].pop(0)

                # Update EMA
                if key not in self._ema:
                    self._ema[key] = val
                else:
                    self._ema[key] = alpha * self._ema[key] + (1 - alpha) * val

    def get_ema(self, key: str) -> float:
        return self._ema.get(key, 1.0)

    def get_trend(self, key: str) -> float:
        """
        Hitung trend loss (positif = memburuk, negatif = membaik).
        Menggunakan slope linear sederhana.
        """
        history = self._history.get(key, [])
        if len(history) < 5:
            return 0.0
        n = len(history)
        x_mean = (n - 1) / 2.0
        y_mean = sum(history) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(history))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if abs(denominator) < 1e-8:
            return 0.0
        return numerator / denominator

    def is_stagnating(self, key: str, threshold: float = 1e-4) -> bool:
        """Deteksi apakah loss tidak bergerak."""
        return abs(self.get_trend(key)) < threshold

    @property
    def recent_losses(self) -> Dict[str, float]:
        """Rata-rata 10 step terakhir."""
        result = {}
        for key, hist in self._history.items():
            if hist:
                result[key] = sum(hist[-10:]) / len(hist[-10:])
        return result


class PengendaliDinamik:
    """
    PD: Pengendali Dinamik

    Mengadaptasi lambda weights secara otomatis berdasarkan
    signal linguistik dari training progress.

    Bukan nn.Module karena tidak perlu gradien — ini controller eksternal.
    """

    def __init__(self, config: PDConfig):
        self.config = config
        self.history = LossHistory(config.history_window)
        self.step = 0

        # State lambda saat ini
        self.lambdas = {
            "morph":  config.lambda_morph_init,
            "struct": config.lambda_struct_init,
            "sem":    config.lambda_sem_init,
            "ctx":    config.lambda_ctx_init,
            "ar":     config.lambda_ar_init,
        }

        # Ranges untuk clipping
        self._ranges = {
            "morph":  config.lambda_morph_range,
            "struct": config.lambda_struct_range,
            "sem":    config.lambda_sem_range,
            "ctx":    config.lambda_ctx_range,
            "ar":     config.lambda_ar_range,
        }

        # Thresholds
        self._thresholds = {
            "morph":  config.morph_error_threshold,
            "struct": config.struct_error_threshold,
            "sem":    config.sem_error_threshold,
            "ctx":    config.ctx_error_threshold,
            "ar":     config.ar_error_threshold,
        }

        # Mapping loss key → lambda key
        self._loss_to_lambda = {
            "l_morph":  "morph",
            "l_struct": "struct",
            "l_sem":    "sem",
            "l_ctx":    "ctx",
            "l_ar":     "ar",
        }

    def step_update(
        self,
        losses: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None,
        output_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Update lambda weights berdasarkan losses saat ini.
        Dipanggil setelah setiap backward step.

        Args:
            losses        : dict output dari AksaraLoss.forward()
            optimizer     : jika diberikan, PD juga adjust learning rate
            output_logits : (B, L, V) untuk entropy measurement

        Returns:
            current lambdas yang akan dipakai di step berikutnya
        """
        self.step += 1
        self.history.update(losses, alpha=self.config.ema_alpha)

        # Adaptasi setiap N steps (bukan setiap step untuk stabilitas)
        if self.step % 10 == 0:
            self._adapt_lambdas()

        # Adaptasi learning rate berdasarkan entropy (jika ada)
        if optimizer is not None and output_logits is not None:
            self._adapt_lr(optimizer, output_logits)

        return dict(self.lambdas)

    def _adapt_lambdas(self):
        """
        Adaptasi lambda: jika loss komponen X di atas threshold → boost λ_X.
        Jika loss sudah rendah → turunkan λ_X untuk beri ruang komponen lain.
        """
        recent = self.history.recent_losses

        for loss_key, lam_key in self._loss_to_lambda.items():
            current_loss = recent.get(loss_key, 0.0)
            threshold = self._thresholds[lam_key]
            rate = self.config.adaptation_rate
            lo, hi = self._ranges[lam_key]

            if current_loss > threshold * 1.5:
                # Loss tinggi banget → boost agresif
                self.lambdas[lam_key] *= (1.0 + rate * 2)
            elif current_loss > threshold:
                # Loss di atas threshold → boost sedang
                self.lambdas[lam_key] *= (1.0 + rate)
            elif current_loss < threshold * 0.5:
                # Loss sudah sangat rendah → kurangi fokus
                self.lambdas[lam_key] *= (1.0 - rate * 0.5)

            # Clamp ke range
            self.lambdas[lam_key] = max(lo, min(hi, self.lambdas[lam_key]))

        # Normalisasi agar total lambda tidak meledak
        # Batas dinaikkan ke 8.0 karena sekarang ada 5 komponen (morph+struct+sem+ctx+ar)
        total_lam = sum(self.lambdas.values())
        if total_lam > 8.0:
            factor = 8.0 / total_lam
            for k in self.lambdas:
                self.lambdas[k] *= factor

    def _adapt_lr(
        self,
        optimizer: torch.optim.Optimizer,
        output_logits: torch.Tensor,
    ):
        """
        Adaptasi learning rate berdasarkan entropy output.
        Entropy tinggi = model tidak yakin = kurangi lr sedikit.
        """
        with torch.no_grad():
            probs = torch.softmax(output_logits.view(-1, output_logits.size(-1)), dim=-1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()

        # Normalize entropy (0..1 approximate)
        max_entropy = math.log(output_logits.size(-1))
        normalized_entropy = entropy / max(max_entropy, 1.0)

        # Jika entropy sangat tinggi → scale down lr sedikit
        if normalized_entropy > 0.9:
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.999  # perlahan saja

    def get_lambdas(self) -> Dict[str, float]:
        """Ambil lambda weights saat ini."""
        return dict(self.lambdas)

    def get_diagnostics(self) -> Dict:
        """Informasi diagnostik untuk logging."""
        tracked = ["l_morph", "l_struct", "l_sem", "l_ctx", "l_ar", "total"]
        return {
            "step": self.step,
            "lambdas": self.get_lambdas(),
            "loss_ema": {k: self.history.get_ema(k) for k in tracked},
            "trends": {k: self.history.get_trend(k) for k in tracked[:-1]},
            "stagnating": {k: self.history.is_stagnating(k) for k in tracked[:-1]},
        }

    def reset(self):
        """Reset history dan lambdas ke nilai awal."""
        cfg = self.config
        self.lambdas = {
            "morph":  cfg.lambda_morph_init,
            "struct": cfg.lambda_struct_init,
            "sem":    cfg.lambda_sem_init,
            "ctx":    cfg.lambda_ctx_init,
            "ar":     cfg.lambda_ar_init,
        }
        self.history = LossHistory(cfg.history_window)
        self.step = 0
