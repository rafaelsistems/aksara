"""
metrics.py — Evaluation Layer domain-agnostic untuk AKSARA.

Metric wajib:
  1. verb_hit_rate   : apakah model menangkap intent dasar (verb muncul)
  2. structure_score : SVO completeness berbasis posisi token
  3. relation_score  : avg cosine similarity verb-object dari semantic_slots BSU
  4. coherence_score : berbasis co-occurrence PMI dari corpus
  5. diversity_score : cek model tidak stuck / collapse

Output format:
  {
    "verb_hit":    0.65,
    "structure":   0.72,
    "relation":    0.48,
    "coherence":   0.61,
    "diversity":   0.80,
  }

Prinsip:
  - Tidak ada hardcode domain atau semantic rule manual
  - Semua metric berbasis output model + representasi internal BSU
  - Co-occurrence dari corpus, bukan library NLP eksternal
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ── Verb Hit Rate ─────────────────────────────────────────────────────────────

def verb_hit_rate(
    generated_ids: List[List[int]],
    verb_token_ids: Set[int],
) -> float:
    """
    Proporsi output yang mengandung minimal satu verb dari verb_token_ids.

    Ini mengukur apakah model menangkap intent dasar — bukan konten spesifik.
    Domain-agnostic: verb_token_ids dibangun dari root_vocab, bukan hardcode.

    Args:
        generated_ids  : list of list of int — token ids tiap output
        verb_token_ids : set of int — token ids yang dianggap verb

    Returns:
        float [0, 1]
    """
    if not generated_ids:
        return 0.0
    hits = sum(
        1 for ids in generated_ids
        if any(i in verb_token_ids for i in ids)
    )
    return hits / len(generated_ids)


# ── Structure Score ───────────────────────────────────────────────────────────

def structure_score(
    generated_ids: List[List[int]],
    verb_token_ids: Set[int],
    subject_token_ids: Optional[Set[int]] = None,
    object_token_ids: Optional[Set[int]] = None,
) -> float:
    """
    SVO completeness score — berapa banyak elemen struktur dasar yang muncul.

    Scoring per output:
      - Ada verb                 → +1/3
      - Ada kandidat subjek      → +1/3  (jika subject_token_ids diberikan)
      - Ada token setelah verb   → +1/3  (proxy objek jika object_token_ids tidak ada)

    Jika subject_token_ids dan object_token_ids tidak diberikan,
    proxy digunakan: token pertama = subjek, token setelah verb = objek.

    Args:
        generated_ids     : list of list of int — output token ids
        verb_token_ids    : set of int — verb ids
        subject_token_ids : opsional — set of int — token ids pronoun/nama subjek
        object_token_ids  : opsional — set of int — token ids kandidat objek

    Returns:
        float [0, 1] — rata-rata SVO score per output
    """
    if not generated_ids:
        return 0.0

    scores = []
    for ids in generated_ids:
        if not ids:
            scores.append(0.0)
            continue

        score = 0.0
        components = 0

        # Deteksi verb
        verb_pos = -1
        for i, t in enumerate(ids):
            if t in verb_token_ids:
                verb_pos = i
                break

        # Komponen 1: ada verb
        components += 1
        if verb_pos != -1:
            score += 1.0

        # Komponen 2: ada subjek (posisi sebelum verb, atau token pertama)
        components += 1
        if subject_token_ids:
            has_subject = any(t in subject_token_ids for t in ids[:max(verb_pos, 1)])
        else:
            # Proxy: token pertama dianggap posisi subjek
            has_subject = verb_pos > 0
        if has_subject:
            score += 1.0

        # Komponen 3: ada objek (token setelah verb)
        components += 1
        if verb_pos != -1 and verb_pos < len(ids) - 1:
            after_verb = ids[verb_pos + 1:]
            if object_token_ids:
                has_object = any(t in object_token_ids for t in after_verb)
            else:
                # Proxy: ada token apapun setelah verb
                has_object = len(after_verb) > 0
            if has_object:
                score += 1.0

        scores.append(score / components)

    return sum(scores) / len(scores)


# ── Relation Score ────────────────────────────────────────────────────────────

@torch.no_grad()
def relation_score(
    semantic_slots: torch.Tensor,
    morpheme_ids: torch.Tensor,
    verb_token_ids: Set[int],
) -> float:
    """
    Rata-rata cosine similarity antara semantic slot verb dan token setelahnya.

    Menggunakan BSU semantic_slots — representasi semantik internal model.
    Tidak ada embedding baru, tidak ada hardcode domain.

    Semakin tinggi score, semakin "nyambung" verb dan objek dalam ruang semantik.

    Args:
        semantic_slots  : (B, L, d_sem) — dari model_output["semantic_slots"]
        morpheme_ids    : (B, L) — token ids
        verb_token_ids  : set of int — verb ids

    Returns:
        float [−1, 1] — rata-rata cosine similarity, 0.0 jika tidak ada verb
    """
    B, L, _ = semantic_slots.shape
    all_sims = []

    for b in range(B):
        ids = morpheme_ids[b].tolist()
        sem = semantic_slots[b]  # (L, d_sem)

        verb_pos = -1
        for i, t in enumerate(ids):
            if t in verb_token_ids:
                verb_pos = i
                break

        if verb_pos == -1 or verb_pos >= L - 1:
            continue

        v_sem    = sem[verb_pos]
        after    = sem[verb_pos + 1:]
        if after.size(0) == 0:
            continue

        v_norm   = F.normalize(v_sem.unsqueeze(0), dim=-1)
        o_norm   = F.normalize(after, dim=-1)
        sims     = (v_norm * o_norm).sum(dim=-1)
        all_sims.append(sims.mean().item())

    return sum(all_sims) / len(all_sims) if all_sims else 0.0


# ── Coherence Score ───────────────────────────────────────────────────────────

def coherence_score(
    generated_ids: List[List[int]],
    co_matrix: Dict[Tuple[int, int], float],
    window: int = 3,
) -> float:
    """
    Skor koherensi berbasis PMI co-occurrence dari corpus.

    Rata-rata PMI pasangan token yang muncul dalam window.
    Semakin tinggi, semakin sering kombinasi token ini muncul di corpus asli.

    Args:
        generated_ids : list of list of int — output token ids
        co_matrix     : {(id_a, id_b): pmi} dari build_cooccurrence_matrix()
        window        : ukuran window lookup

    Returns:
        float — rata-rata PMI per output, 0.0 jika tidak ada pasangan dikenal
    """
    if not generated_ids or not co_matrix:
        return 0.0

    output_scores = []
    for ids in generated_ids:
        pmis = []
        for i in range(len(ids)):
            a = ids[i]
            if a == 0:
                continue
            for j in range(i + 1, min(i + 1 + window, len(ids))):
                b = ids[j]
                if b == 0:
                    continue
                key = (min(a, b), max(a, b))
                if key in co_matrix:
                    pmis.append(co_matrix[key])
        output_scores.append(sum(pmis) / len(pmis) if pmis else 0.0)

    return sum(output_scores) / len(output_scores)


# ── Diversity Score ───────────────────────────────────────────────────────────

def diversity_score(generated_ids: List[List[int]]) -> float:
    """
    Ukur keberagaman output — cegah model stuck / collapse.

    Berbasis type-token ratio (TTR) yang dinormalisasi:
      - TTR = unique_tokens / total_tokens per output
      - Score = rata-rata TTR semua output

    Score rendah (< 0.3) = model sering mengulang token yang sama.
    Score tinggi (> 0.7) = output beragam.

    Args:
        generated_ids : list of list of int — output token ids

    Returns:
        float [0, 1]
    """
    if not generated_ids:
        return 0.0

    ttrs = []
    for ids in generated_ids:
        clean = [i for i in ids if i != 0]
        if not clean:
            ttrs.append(0.0)
        else:
            ttrs.append(len(set(clean)) / len(clean))

    return sum(ttrs) / len(ttrs)


# ── AksaraMetrics ─────────────────────────────────────────────────────────────

class AksaraMetrics:
    """
    Evaluasi lengkap output model AKSARA.

    Semua metric domain-agnostic — tidak ada aturan semantik manual.
    verb_token_ids dan co_matrix dibangun dari root_vocab + corpus.

    Cara pakai:
        metrics = AksaraMetrics(verb_token_ids, co_matrix)
        result  = metrics.evaluate(model_output, morpheme_ids, generated_ids)
    """

    def __init__(
        self,
        verb_token_ids: Set[int],
        co_matrix: Optional[Dict[Tuple[int, int], float]] = None,
        subject_token_ids: Optional[Set[int]] = None,
    ):
        self.verb_token_ids    = verb_token_ids
        self.co_matrix         = co_matrix or {}
        self.subject_token_ids = subject_token_ids

    def evaluate(
        self,
        model_output: Dict,
        morpheme_ids: torch.Tensor,
        generated_ids: List[List[int]],
    ) -> Dict[str, float]:
        """
        Hitung semua metric dari satu batch evaluasi.

        Args:
            model_output  : output dari model.forward() — perlu semantic_slots
            morpheme_ids  : (B, L) token ids dari batch input
            generated_ids : list of list of int — decoded output per sampel

        Returns:
            dict dengan semua metric
        """
        result: Dict[str, float] = {}

        # 1. Verb Hit Rate
        result["verb_hit"] = verb_hit_rate(generated_ids, self.verb_token_ids)

        # 2. Structure Score
        result["structure"] = structure_score(
            generated_ids, self.verb_token_ids,
            subject_token_ids=self.subject_token_ids,
        )

        # 3. Relation Score (berbasis BSU semantic slots)
        sem_slots = model_output.get("semantic_slots")
        if sem_slots is not None and sem_slots.requires_grad is False or sem_slots is not None:
            result["relation"] = relation_score(
                sem_slots, morpheme_ids, self.verb_token_ids,
            )
        else:
            result["relation"] = 0.0

        # 4. Coherence Score (berbasis PMI)
        result["coherence"] = coherence_score(
            generated_ids, self.co_matrix,
        )

        # 5. Diversity Score
        result["diversity"] = diversity_score(generated_ids)

        return result


def evaluate(
    model,
    corpus: List[str],
    root_vocab: Dict[str, int],
    device: torch.device,
    verb_token_ids: Set[int],
    co_matrix: Optional[Dict[Tuple[int, int], float]] = None,
    n_samples: int = 32,
) -> Dict[str, float]:
    """
    Convenience function — evaluasi model langsung dari corpus.

    Ambil n_samples kalimat dari corpus sebagai prompt,
    generate output, hitung semua metric.

    Args:
        model          : AksaraModel
        corpus         : list kalimat untuk dijadikan prompt
        root_vocab     : mapping token → id
        device         : torch device
        verb_token_ids : set of int — verb ids
        co_matrix      : PMI matrix (opsional)
        n_samples      : jumlah sampel yang dievaluasi

    Returns:
        dict metric lengkap
    """
    import random as _random

    metrics_calc = AksaraMetrics(verb_token_ids, co_matrix)

    # Sample prompt dari corpus
    prompts = _random.sample(corpus, min(n_samples, len(corpus)))

    model.eval()
    with torch.no_grad():
        gen_out = model.generate(prompts, max_length=10, temperature=0.7, min_length=3)

    generated_texts = gen_out.get("generated_texts", [])

    # Decode ids untuk metric
    inv_vocab = {v: k for k, v in root_vocab.items()}
    generated_ids = []
    for text in generated_texts:
        ids = [root_vocab.get(t, 0) for t in text.lower().split() if t]
        generated_ids.append(ids)

    # Encode prompt untuk mendapatkan semantic_slots
    from aksara.data.dataset import AksaraDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = AksaraDataset(prompts, root_vocab, max_length=32, min_length=1)
    loader  = DataLoader(dataset, batch_size=min(n_samples, 16),
                         shuffle=False, collate_fn=collate_fn)

    all_sem_slots  = []
    all_morph_ids  = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            L = batch.morpheme_ids.shape[1]
            dep = torch.zeros(batch.morpheme_ids.shape[0], L, L,
                              dtype=torch.bool, device=device)
            lps_dict = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids":    batch.affix_ids,
                "dep_masks":    dep,
                "lengths":      batch.lengths,
            }
            out = model(lps_dict)
            if "semantic_slots" in out:
                all_sem_slots.append(out["semantic_slots"])
                all_morph_ids.append(batch.morpheme_ids)
            break  # satu batch sudah cukup untuk estimasi

    if all_sem_slots:
        sem_slots   = all_sem_slots[0]
        morph_ids   = all_morph_ids[0]
        model_output = {"semantic_slots": sem_slots}
        gen_ids_batch = generated_ids[:sem_slots.size(0)]
    else:
        model_output = {}
        morph_ids    = torch.zeros(1, 1, dtype=torch.long, device=device)
        gen_ids_batch = generated_ids

    return metrics_calc.evaluate(model_output, morph_ids, gen_ids_batch)
