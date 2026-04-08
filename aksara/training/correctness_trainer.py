"""
correctness_trainer.py — Training loop untuk AKSARA v3 (Evaluator Kebenaran Kalimat).

Tidak ada next-token prediction. Tidak ada autoregressive loop.
Model dilatih untuk membedakan kalimat BENAR vs SALAH secara linguistik Indonesia.

Pipeline training:
  1. Load corpus paired (benar/salah)
  2. Forward pass → CorrectnessHead → 4 skor
  3. CorrectnessLoss (BCE + margin + consistency + calibration)
  4. Backward + clip_grad + step

Fungsi utama:
  train_correctness(model, corpus_paired, vocab, ...) → loss_history
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aksara.data.dataset import AksaraDataset, collate_fn


# ── Dataset untuk corpus paired ───────────────────────────────────────────────

class PairedCorpusDataset(Dataset):
    """
    Dataset untuk corpus pasangan (benar, salah).

    Format input: list of dict {"text": str, "label": int}
    Label: 1 = benar, 0 = salah

    Setiap item dikembalikan sebagai (text, label).
    Collation menggunakan PairedBatchCollator terpisah.
    """

    def __init__(self, records: List[Dict], min_len: int = 2):
        self.records = [
            r for r in records
            if len(r["text"].split()) >= min_len
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        return self.records[idx]


def paired_collate_fn(
    batch: List[Dict],
    root_vocab: Dict[str, int],
    device: torch.device,
    max_length: int = 32,
) -> Tuple[Dict, torch.Tensor]:
    """
    Collate function untuk PairedCorpusDataset.

    Returns:
        lps_input : dict siap untuk model.forward()
        labels    : (B,) float tensor — 1.0 benar, 0.0 salah
    """
    texts  = [r["text"] for r in batch]
    labels = torch.tensor(
        [float(r["label"]) for r in batch],
        dtype=torch.float32, device=device,
    )

    # Gunakan AksaraDataset untuk tokenisasi (LPS + morfem decomposition)
    ds = AksaraDataset(texts, root_vocab, max_length=max_length, min_length=1)
    items = [ds[i] for i in range(len(ds))]
    ak_batch = collate_fn(items)
    ak_batch = ak_batch.to(device)

    L = ak_batch.morpheme_ids.shape[1]
    dep_masks = torch.zeros(
        ak_batch.morpheme_ids.shape[0], L, L,
        dtype=torch.bool, device=device,
    )

    lps_input = {
        "morpheme_ids":   ak_batch.morpheme_ids,
        "affix_ids":      ak_batch.affix_ids,
        "dep_masks":      dep_masks,
        "attention_mask": ak_batch.attention_mask,
        "lengths":        ak_batch.lengths,
    }

    return lps_input, labels


# ── Loader corpus paired ──────────────────────────────────────────────────────

def load_paired_corpus(path: str, n: int = 0) -> List[Dict]:
    """
    Load corpus paired dari JSONL.
    Format: {"text": str, "label": int, "source": str}
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" in obj and "label" in obj:
                    records.append(obj)
            except json.JSONDecodeError:
                continue
            if n and len(records) >= n:
                break
    return records


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_correctness(
    model,
    paired_records: List[Dict],
    root_vocab: Dict[str, int],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    verbose: bool = True,
    label: str = "",
) -> List[float]:
    """
    Training loop evaluator kebenaran kalimat.

    Tidak ada next-token prediction. Tidak ada autoregressive loop.
    Sinyal training murni dari pasangan (kalimat_benar, kalimat_salah).

    Args:
        model         : AksaraModel v3
        paired_records: list of {"text": str, "label": int}
        root_vocab    : mapping token → id
        device        : torch device
        epochs        : jumlah epoch
        batch_size    : ukuran batch (disarankan genap — balance benar/salah)
        lr            : learning rate awal
        verbose       : cetak progress
        label         : label untuk logging

    Returns:
        loss_history : list float, satu nilai per epoch
    """
    dataset = PairedCorpusDataset(paired_records)

    def _collate(batch):
        return paired_collate_fn(batch, root_vocab, device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        drop_last=False,
    )

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=lr * 0.05
    )

    loss_history: List[float] = []
    best_loss      = float("inf")
    patience_count = 0
    early_stop     = 12

    n_pos = sum(1 for r in paired_records if r["label"] == 1)
    n_neg = sum(1 for r in paired_records if r["label"] == 0)

    if verbose:
        tag = f"[{label}] " if label else ""
        print(f"  {tag}Dataset: {len(paired_records):,} records "
              f"(+{n_pos:,} / -{n_neg:,})")
        print(f"  {tag}Epochs : {epochs}  batch={batch_size}  lr={lr:.1e}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for lps_input, labels in loader:
            out  = model(lps_input, labels=labels)
            loss = out["losses"]["total"]

            if loss.isnan() or loss.isinf():
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        loss_history.append(avg_loss)
        sched.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss      = avg_loss
            patience_count = 0
        else:
            patience_count += 1

        if verbose and epoch % 5 == 0:
            cur_lr = opt.param_groups[0]["lr"]
            tag    = f"[{label}] " if label else ""
            print(f"  {tag}[Epoch {epoch:>3}/{epochs}]  "
                  f"loss={avg_loss:.4f}  lr={cur_lr:.2e}  "
                  f"patience={patience_count}")

        if patience_count >= early_stop:
            if verbose:
                tag = f"[{label}] " if label else ""
                print(f"  {tag}[early stop] epoch {epoch} — "
                      f"loss tidak improve {early_stop} epoch")
            break

    return loss_history


# ── Evaluasi Cepat ────────────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(
    model,
    paired_records: List[Dict],
    root_vocab: Dict[str, int],
    device: torch.device,
    n_samples: int = 200,
) -> Dict[str, float]:
    """
    Evaluasi cepat model setelah training.

    Metrics:
      accuracy    : proporsi prediksi benar (threshold 0.5)
      avg_pos     : rata-rata skor kalimat benar
      avg_neg     : rata-rata skor kalimat salah
      separation  : avg_pos - avg_neg (semakin besar semakin baik)

    Returns:
        dict metrics
    """
    model.eval()
    sample = random.sample(paired_records, min(n_samples, len(paired_records)))

    pos_scores: List[float] = []
    neg_scores: List[float] = []
    correct = 0

    # Proses satu per satu untuk simplisitas (eval tidak butuh speed)
    for rec in sample:
        result = model.score([rec["text"]])
        s = result["total"][0]

        if rec["label"] == 1:
            pos_scores.append(s)
            if s >= 0.5:
                correct += 1
        else:
            neg_scores.append(s)
            if s < 0.5:
                correct += 1

    accuracy   = correct / len(sample)
    avg_pos    = sum(pos_scores) / max(len(pos_scores), 1)
    avg_neg    = sum(neg_scores) / max(len(neg_scores), 1)
    separation = avg_pos - avg_neg

    return {
        "accuracy":   accuracy,
        "avg_pos":    avg_pos,
        "avg_neg":    avg_neg,
        "separation": separation,
        "n_eval":     len(sample),
    }
