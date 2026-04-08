"""
AksaraTrainer - Training loop untuk AKSARA.
Mengintegrasikan PD (Pengendali Dinamik) untuk adaptasi lambda weights.
"""

import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aksara.core.model import AksaraModel
from aksara.training.pd import PengendaliDinamik, PDConfig
from aksara.data.dataset import AksaraBatch, collate_fn


@dataclass
class TrainerConfig:
    output_dir: str = "aksara_output"
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_every_n_steps: int = 500
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_pd: bool = True                  # gunakan Pengendali Dinamik
    pd_config: PDConfig = field(default_factory=PDConfig)
    fp16: bool = False                   # mixed precision (jika GPU mendukung)
    seed: int = 42


class AksaraTrainer:
    """
    Training loop untuk AksaraModel.

    Mengintegrasikan:
    - Multi-component loss (AksaraLoss)
    - Pengendali Dinamik (PD) untuk adaptive lambda weights
    - Warmup + cosine LR schedule
    - Gradient clipping
    - Checkpoint saving

    Penggunaan:
        trainer = AksaraTrainer(model, train_dataset, eval_dataset, config)
        trainer.train()
    """

    def __init__(
        self,
        model: AksaraModel,
        train_dataset,
        eval_dataset=None,
        config: TrainerConfig = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainerConfig()
        self.callbacks = callbacks or []

        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Pengendali Dinamik
        self.pd = PengendaliDinamik(config.pd_config) if config.use_pd else None

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 and self.device.type == "cuda" else None

        self.global_step = 0
        self.best_eval_loss = float("inf")
        self._train_losses: List[Dict] = []

        os.makedirs(config.output_dir, exist_ok=True)

        torch.manual_seed(config.seed)

    def _get_lr(self, step: int) -> float:
        """Warmup + cosine decay schedule."""
        warmup = self.config.warmup_steps
        if step < warmup:
            return self.config.learning_rate * (step + 1) / warmup
        total_steps = self.config.num_epochs * len(self.train_dataset) // self.config.batch_size
        progress = (step - warmup) / max(total_steps - warmup, 1)
        import math
        return self.config.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _build_dep_masks(self, batch: AksaraBatch) -> torch.Tensor:
        """
        Bangun dependency masks dari batch menggunakan LPS.

        Dependency mask adalah pembeda fundamental AKSARA dari Transformer:
        f_syn beroperasi pada dependency graph O(n·deg), bukan full attention O(n²).
        Tanpa dep_masks, f_syn fallback ke local window — kehilangan identitas.

        Returns:
            dep_masks : (B, L, L) bool tensor
        """
        B = batch.morpheme_ids.size(0)
        L = batch.morpheme_ids.size(1)
        dep_masks = torch.zeros(B, L, L, dtype=torch.bool, device=self.device)

        for i in range(B):
            actual_len = batch.lengths[i].item()
            # Bangun mask untuk token aktual (bukan padding)
            dummy_tokens = ["_"] * actual_len  # build_dep_mask hanya pakai panjang list
            mask_i = self.model.lps.build_dep_mask(dummy_tokens, L)
            # Zero-out koneksi ke/dari padding positions
            mask_i[actual_len:, :] = False
            mask_i[:, actual_len:] = False
            dep_masks[i] = mask_i.to(self.device)

        return dep_masks

    def _training_step(self, batch: AksaraBatch) -> Dict:
        """Satu langkah training."""
        self.model.train()
        batch = batch.to(self.device)

        # Bangun dependency masks — ini yang membuat f_syn beroperasi
        # pada dependency graph, bukan sekadar local attention
        dep_masks = self._build_dep_masks(batch)

        # Siapkan lps_output dari batch
        lps_output = {
            "morpheme_ids": batch.morpheme_ids,
            "affix_ids":    batch.affix_ids,
            "dep_masks":    dep_masks,
            "lengths":      batch.lengths,
            "max_len":      batch.morpheme_ids.size(1),
        }

        targets = batch.as_targets()

        # Ambil lambda dari PD
        lambdas = self.pd.get_lambdas() if self.pd else None

        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(lps_output, targets=targets, lambdas=lambdas)
            loss = outputs["losses"]["total"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(lps_output, targets=targets, lambdas=lambdas)
            loss = outputs["losses"]["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        self.optimizer.zero_grad()

        # Update PD
        if self.pd:
            gos_logits = outputs["gos_out"].get("context_logits")
            self.pd.step_update(
                outputs["losses"],
                optimizer=self.optimizer,
                output_logits=gos_logits,
            )

        return {k: v.item() if torch.is_tensor(v) else v for k, v in outputs["losses"].items()}

    @torch.no_grad()
    def _eval_step(self, batch: AksaraBatch) -> Dict:
        """Satu langkah evaluasi."""
        self.model.eval()
        batch = batch.to(self.device)

        dep_masks = self._build_dep_masks(batch)

        lps_output = {
            "morpheme_ids": batch.morpheme_ids,
            "affix_ids":    batch.affix_ids,
            "dep_masks":    dep_masks,
            "lengths":      batch.lengths,
            "max_len":      batch.morpheme_ids.size(1),
        }

        targets = batch.as_targets()
        outputs = self.model(lps_output, targets=targets)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in outputs["losses"].items()}

    def train(self):
        """Main training loop."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )

        print(f"[AKSARA] Mulai training — {self.model.num_parameters['trainable']:,} parameter")
        print(f"[AKSARA] Device: {self.device} | Epochs: {self.config.num_epochs}")
        print(f"[AKSARA] KBBI coverage: {self.model.lsk.kbbi_coverage:.1%}")

        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            t0 = time.time()

            for batch in train_loader:
                # Update LR
                lr = self._get_lr(self.global_step)
                self._set_lr(lr)

                step_losses = self._training_step(batch)
                epoch_losses.append(step_losses)
                self._train_losses.append(step_losses)
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    self._log_step(step_losses, epoch, lr)

                # Evaluasi
                if eval_loader and self.global_step % self.config.eval_every_n_steps == 0:
                    eval_loss = self._evaluate(eval_loader)
                    print(f"[Eval step {self.global_step}] total={eval_loss['total']:.4f}")

                    if eval_loss["total"] < self.best_eval_loss:
                        self.best_eval_loss = eval_loss["total"]
                        self._save_checkpoint("best")

                # Checkpoint reguler
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                # Callbacks
                for cb in self.callbacks:
                    cb(self, step_losses, self.global_step)

            # Summary per epoch
            avg = self._average_losses(epoch_losses)
            elapsed = time.time() - t0
            print(f"\n[Epoch {epoch+1}/{self.config.num_epochs}] "
                  f"total={avg['total']:.4f} | morph={avg.get('l_morph', 0):.4f} | "
                  f"struct={avg.get('l_struct', 0):.4f} | sem={avg.get('l_sem', 0):.4f} | "
                  f"ctx={avg.get('l_ctx', 0):.4f} | {elapsed:.1f}s")

            if self.pd:
                diag = self.pd.get_diagnostics()
                print(f"  PD λ: morph={diag['lambdas']['morph']:.3f} "
                      f"struct={diag['lambdas']['struct']:.3f} "
                      f"sem={diag['lambdas']['sem']:.3f} "
                      f"ctx={diag['lambdas']['ctx']:.3f}")

        self._save_checkpoint("final")
        print(f"\n[AKSARA] Training selesai. Best eval loss: {self.best_eval_loss:.4f}")
        return self._train_losses

    def _evaluate(self, loader: DataLoader) -> Dict:
        all_losses = []
        for batch in loader:
            losses = self._eval_step(batch)
            all_losses.append(losses)
        return self._average_losses(all_losses)

    @staticmethod
    def _average_losses(losses_list: List[Dict]) -> Dict:
        if not losses_list:
            return {}
        keys = [k for k in losses_list[0] if k != "lambdas"]
        return {k: sum(d.get(k, 0) for d in losses_list) / len(losses_list) for k in keys}

    def _log_step(self, losses: Dict, epoch: int, lr: float):
        pd_info = ""
        if self.pd:
            lam = self.pd.get_lambdas()
            pd_info = f" | λ_m={lam['morph']:.2f} λ_s={lam['struct']:.2f} λ_k={lam['sem']:.2f} λ_c={lam['ctx']:.2f}"
        print(f"[Step {self.global_step} | E{epoch+1}] "
              f"total={losses.get('total', 0):.4f} "
              f"morph={losses.get('l_morph', 0):.4f} "
              f"struct={losses.get('l_struct', 0):.4f} "
              f"sem={losses.get('l_sem', 0):.4f} "
              f"ctx={losses.get('l_ctx', 0):.4f} "
              f"lr={lr:.2e}{pd_info}")

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.config.output_dir, f"checkpoint_{tag}")
        self.model.save(path)
        if self.pd:
            diag = self.pd.get_diagnostics()
            with open(os.path.join(path, "pd_state.json"), "w") as f:
                json.dump(diag, f, indent=2)
