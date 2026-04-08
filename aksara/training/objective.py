"""
objective.py — Objective Layer untuk AKSARA.

Komponen:
  embedding_relation_loss  : cek apakah verb dan objek "nyambung" secara semantik
                             berbasis semantic_slots dari BSU — BUKAN embedding baru
  cooccurrence_loss        : cek apakah kombinasi token masuk akal secara statistik
                             berbasis PMI matrix yang diprecompute dari corpus
  build_cooccurrence_matrix: precompute PMI(a, b) dari corpus, domain-agnostic
  negative_sample          : buat kalimat negatif via token shuffle

Prinsip implementasi:
  - Tidak ada hardcode domain atau aturan semantik manual
  - Semua berbasis representasi model sendiri (BSU semantic_slots)
  - Co-occurrence berbasis data corpus, bukan library NLP eksternal
  - Tidak ada layer baru — hanya operasi di atas output BSU yang sudah ada
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ── Co-occurrence Matrix ──────────────────────────────────────────────────────

def build_cooccurrence_matrix(
    corpus: List[str],
    root_vocab: Dict[str, int],
    window: int = 4,
    min_count: int = 2,
) -> Dict[Tuple[int, int], float]:
    """
    Precompute PMI co-occurrence matrix dari corpus.

    PMI(a, b) = log [ P(a,b) / (P(a) * P(b)) ]

    Semakin tinggi PMI, semakin sering a dan b muncul bersama
    dibanding ekspektasi random. Dipakai sebagai sinyal "masuk akal".

    Args:
        corpus     : list kalimat teks mentah
        root_vocab : mapping token → id
        window     : ukuran jendela co-occurrence (default 4)
        min_count  : pasangan dengan co-count < min_count dibuang

    Returns:
        dict {(id_a, id_b): pmi_score} — hanya PMI positif yang disimpan
    """
    token_count: Dict[int, int] = defaultdict(int)
    pair_count: Dict[Tuple[int, int], int] = defaultdict(int)
    total_tokens = 0

    for text in corpus:
        tokens = text.lower().split()
        ids = [root_vocab[t] for t in tokens if t in root_vocab]
        for i, a in enumerate(ids):
            token_count[a] += 1
            total_tokens += 1
            # Window co-occurrence (tidak arahkan — simetrik)
            for j in range(i + 1, min(i + 1 + window, len(ids))):
                b = ids[j]
                if a != b:
                    pair_count[(min(a, b), max(a, b))] += 1

    if total_tokens == 0:
        return {}

    # Hitung PMI — hanya simpan yang positif dan >= min_count
    co_matrix: Dict[Tuple[int, int], float] = {}
    log_total = math.log(total_tokens + 1e-9)

    for (a, b), co_c in pair_count.items():
        if co_c < min_count:
            continue
        p_ab = co_c / total_tokens
        p_a  = token_count[a] / total_tokens
        p_b  = token_count[b] / total_tokens
        pmi  = math.log(p_ab / (p_a * p_b + 1e-9) + 1e-9)
        if pmi > 0:
            co_matrix[(a, b)] = pmi

    return co_matrix


# ── Embedding Relation Loss ───────────────────────────────────────────────────

def embedding_relation_loss(
    semantic_slots: torch.Tensor,
    morpheme_ids: torch.Tensor,
    verb_token_ids: Set[int],
    min_sim_target: float = 0.3,
) -> torch.Tensor:
    """
    Cek apakah kata-kata dalam kalimat "nyambung secara makna".

    Menggunakan semantic_slots dari BSU output — BUKAN embedding layer baru.
    BSU semantic slot (d_semantic dim) sudah meng-encode makna per token.

    Strategi:
      1. Temukan posisi verb dalam kalimat (via morpheme_ids vs verb_token_ids)
      2. Temukan posisi non-verb setelah verb (kandidat objek/pelengkap)
      3. Hitung cosine similarity antara semantic slot verb dan objek
      4. Loss = max(0, min_sim_target - avg_sim) — dorong similarity minimum

    Ini domain-agnostic: tidak ada hardcode "makan → nasi".
    Model sendiri yang harus belajar bahwa verb dan objeknya semantically dekat.

    Args:
        semantic_slots  : (B, L, d_sem) — semantic slots dari BSU
        morpheme_ids    : (B, L) — token ids per posisi
        verb_token_ids  : set of int — ids token yang dianggap verb (dari root_vocab)
        min_sim_target  : target cosine similarity minimum (default 0.3)

    Returns:
        scalar tensor — 0.0 jika tidak ada verb atau tidak ada objek kandidat
    """
    device = semantic_slots.device
    B, L, d = semantic_slots.shape

    if L < 2:
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    n_valid = 0

    for b in range(B):
        ids = morpheme_ids[b]  # (L,)
        sem = semantic_slots[b]  # (L, d_sem)

        # Temukan posisi verb pertama
        verb_pos = -1
        for pos in range(L):
            if ids[pos].item() in verb_token_ids:
                verb_pos = pos
                break

        if verb_pos == -1 or verb_pos >= L - 1:
            continue

        # Semantic slot verb
        v_sem = sem[verb_pos]  # (d_sem,)

        # Semua posisi setelah verb — kandidat objek/pelengkap
        after_verb = sem[verb_pos + 1:]  # (L - verb_pos - 1, d_sem)
        if after_verb.size(0) == 0:
            continue

        # Cosine similarity verb vs tiap token setelahnya
        v_norm  = F.normalize(v_sem.unsqueeze(0), dim=-1)   # (1, d_sem)
        o_norm  = F.normalize(after_verb, dim=-1)            # (k, d_sem)
        sims    = (v_norm * o_norm).sum(dim=-1)              # (k,)

        avg_sim = sims.mean()

        # Dorong similarity minimal min_sim_target
        deficit = (min_sim_target - avg_sim).clamp(min=0.0)
        total_loss = total_loss + deficit
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / n_valid


# ── Co-occurrence Loss ────────────────────────────────────────────────────────

def cooccurrence_loss(
    morpheme_ids: torch.Tensor,
    co_matrix: Dict[Tuple[int, int], float],
    window: int = 3,
) -> torch.Tensor:
    """
    Cek apakah kombinasi token masuk akal berdasarkan PMI co-occurrence.

    Loss = -mean(PMI(a, b)) untuk semua pasangan dalam window.
    Semakin tinggi PMI pasangan token, semakin kecil loss.
    Jika token jarang muncul bersama (PMI rendah atau tidak ada), loss naik.

    Args:
        morpheme_ids : (B, L) — token ids per posisi
        co_matrix    : {(id_a, id_b): pmi} dari build_cooccurrence_matrix()
        window       : ukuran window lookup (default 3)

    Returns:
        scalar tensor — 0.0 jika tidak ada pasangan yang dikenal
    """
    device = morpheme_ids.device
    B, L = morpheme_ids.shape

    total_pmi = 0.0
    n_pairs = 0

    for b in range(B):
        ids = morpheme_ids[b].tolist()
        for i in range(L):
            a = ids[i]
            if a == 0:  # padding
                continue
            for j in range(i + 1, min(i + 1 + window, L)):
                bb = ids[j]
                if bb == 0:
                    continue
                key = (min(a, bb), max(a, bb))
                if key in co_matrix:
                    total_pmi += co_matrix[key]
                    n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    # Negatif mean PMI → semakin tinggi PMI, semakin rendah loss
    avg_pmi = total_pmi / n_pairs
    return torch.tensor(-avg_pmi, dtype=torch.float32, device=device)


# ── Negative Sampling ─────────────────────────────────────────────────────────

def make_negative_batch(
    morpheme_ids: torch.Tensor,
    n_neg: int = 1,
) -> torch.Tensor:
    """
    Buat kalimat negatif via token shuffle dalam batch.

    Prinsip: urutan token yang benar mengandung struktur makna.
    Versi yang di-shuffle merusak struktur — model harus belajar perbedaannya.

    Ini bukan hardcode domain — semua kalimat di-corrupt dengan cara sama.

    Args:
        morpheme_ids : (B, L) — token ids dari batch asli
        n_neg        : jumlah versi negatif per kalimat (default 1)

    Returns:
        (B * n_neg, L) — batch negatif, setiap baris token diacak
    """
    B, L = morpheme_ids.shape
    device = morpheme_ids.device
    negs = []

    for _ in range(n_neg):
        neg = morpheme_ids.clone()
        for b in range(B):
            # Cari posisi non-padding
            non_pad = (neg[b] != 0).nonzero(as_tuple=True)[0]
            if len(non_pad) > 2:
                # Shuffle hanya posisi non-padding
                perm = non_pad[torch.randperm(len(non_pad), device=device)]
                neg[b, non_pad] = neg[b, perm]
        negs.append(neg)

    return torch.cat(negs, dim=0)  # (B * n_neg, L)


# ── CompositeLoss ─────────────────────────────────────────────────────────────

class CompositeLoss:
    """
    Wrapper yang menggabungkan AksaraLoss dengan objective tambahan.

    Tidak ada layer baru. Semua input diambil dari output forward() yang sudah ada:
      - semantic_slots  : dari bsu_original (sudah ada di model.forward output)
      - morpheme_ids    : dari input batch
      - co_matrix       : diprecompute sekali dari corpus sebelum training

    Args:
        lambda_rel  : bobot embedding_relation_loss (default 0.3)
        lambda_co   : bobot cooccurrence_loss (default 0.1)
        verb_ids    : set int — token ids yang dianggap verb
        co_matrix   : PMI matrix dari build_cooccurrence_matrix()
    """

    def __init__(
        self,
        lambda_rel: float = 0.3,
        lambda_co: float = 0.1,
        verb_ids: Optional[Set[int]] = None,
        co_matrix: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        self.lambda_rel = lambda_rel
        self.lambda_co  = lambda_co
        self.verb_ids   = verb_ids or set()
        self.co_matrix  = co_matrix or {}

    def __call__(
        self,
        base_loss: torch.Tensor,
        model_output: Dict,
        morpheme_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Hitung composite loss = base_loss + λ_rel * L_rel + λ_co * L_co.

        Args:
            base_loss    : loss dari AksaraLoss (sudah include CE, contrast, dll)
            model_output : output dari model.forward() — perlu semantic_slots
            morpheme_ids : (B, L) token ids dari batch

        Returns:
            (total_loss, breakdown_dict) — breakdown untuk logging
        """
        breakdown: Dict[str, float] = {"base": base_loss.item()}
        total = base_loss

        # ── Embedding Relation Loss ──────────────────────────────────────────
        semantic_slots = model_output.get("semantic_slots")  # (B, L, d_sem)
        if semantic_slots is not None and self.verb_ids:
            l_rel = embedding_relation_loss(
                semantic_slots, morpheme_ids, self.verb_ids,
            )
            total = total + self.lambda_rel * l_rel
            breakdown["rel"] = l_rel.item()
        else:
            breakdown["rel"] = 0.0

        # ── Co-occurrence Loss ───────────────────────────────────────────────
        if self.co_matrix:
            l_co = cooccurrence_loss(morpheme_ids, self.co_matrix)
            total = total + self.lambda_co * l_co
            breakdown["co"] = l_co.item()
        else:
            breakdown["co"] = 0.0

        breakdown["total"] = total.item()
        return total, breakdown
