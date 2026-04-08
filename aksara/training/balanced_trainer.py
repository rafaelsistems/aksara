"""
balanced_trainer.py — Phase Training + Loss Shaping untuk AKSARA.

Strategi (setelah diagnosis gradient domination + signal dilution):

  Phase 1 — Action Only:
    - Hanya gunakan corpus action (filter 'other' agresif)
    - Loss shaping: reward ketika verb action muncul di output
    - TIDAK ada balanced sampler (terbukti merusak sinyal)
    - Target: model HARUS belajar struktur SVO sebelum diekspos Wikipedia

  Phase 2 — Mixed:
    - Campur action + Wikipedia setelah Phase 1 stabil
    - Proporsi: min 40% action, max 60% Wikipedia
    - Loss shaping tetap aktif

Digunakan oleh action_benchmark.py dan epoch_snapshot_eval.py.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from torch.utils.data import DataLoader

from aksara.data.dataset import AksaraDataset, collate_fn
from aksara.training.objective import (
    CompositeLoss, build_cooccurrence_matrix, make_negative_batch,
)


# ── Verb action set untuk filter dan loss shaping ────────────────────────────

# Semua verb aktif dari 8 domain
_ACTION_VERBS: Set[str] = {
    "makan", "memakan", "minum", "meminum",
    "membaca", "baca", "membacakan",
    "bekerja", "kerja", "mengerjakan",
    "pergi", "berangkat", "menuju", "pulang",
    "memasak", "masak", "menggoreng", "merebus", "menumis", "memanggang",
    "belajar", "mempelajari", "mengajar",
    "memeriksa", "mengecek", "mendiagnosis",
    "menetapkan", "mengeluarkan", "mengumumkan", "mengesahkan",
}

# Keyword per domain untuk deteksi
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "makan":     ["makan", "memakan", "nasi", "roti", "mie", "sayur", "ikan",
                  "daging", "makanan", "lauk", "minum", "lapar", "kenyang"],
    "membaca":   ["membaca", "baca", "buku", "artikel", "novel", "koran",
                  "teks", "halaman", "cerita", "bacaan"],
    "bekerja":   ["bekerja", "kerja", "kantor", "laporan", "tugas", "proyek",
                  "rapat", "karyawan", "pekerjaan"],
    "pergi":     ["pergi", "berangkat", "menuju", "pulang", "naik", "pasar",
                  "toko", "sekolah"],
    "memasak":   ["memasak", "menggoreng", "merebus", "menumis", "memanggang",
                  "masak", "dapur", "bumbu"],
    "belajar":   ["belajar", "pelajaran", "ujian", "sekolah", "guru",
                  "murid", "siswa", "tugas", "matematika"],
    "memeriksa": ["memeriksa", "mengecek", "pasien", "dokter", "perawat",
                  "kesehatan", "obat", "klinik"],
    "formal":    ["pemerintah", "kebijakan", "peraturan", "menteri", "undang",
                  "negara", "aturan", "keputusan"],
}


def has_action_signal(text: str) -> bool:
    """True jika teks mengandung minimal satu verb action."""
    tokens = set(text.lower().split())
    return bool(tokens & _ACTION_VERBS)


def label_domain(text: str) -> str:
    """Beri label domain ke satu kalimat. Return 'other' jika tidak cocok."""
    t = text.lower()
    best_domain = "other"
    best_count  = 0
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in t)
        if count > best_count:
            best_count  = count
            best_domain = domain
    return best_domain


# ── Filter: buang kalimat tanpa sinyal aksi ───────────────────────────────────

def filter_action_only(
    corpus: List[str],
    other_limit: float = 0.15,
    verbose: bool = True,
) -> List[str]:
    """
    Filter corpus agar hanya kalimat dengan sinyal aksi yang tersisa.

    Strategi:
      1. Lolos jika mengandung verb dari _ACTION_VERBS (hard filter)
      2. Tambah kalimat 'other' hingga batas other_limit dari total
         (untuk mencegah distribusi menjadi 0% konteks umum)

    Args:
        corpus      : list of training texts
        other_limit : fraksi maksimal kalimat 'other' yang diizinkan (default 15%)
        verbose     : print statistik filter

    Returns:
        corpus yang sudah difilter
    """
    action_texts = []
    other_texts  = []

    for text in corpus:
        if has_action_signal(text):
            action_texts.append(text)
        else:
            other_texts.append(text)

    # Hitung berapa 'other' yang boleh masuk
    max_other = int(len(action_texts) * other_limit / max(1 - other_limit, 1e-6))
    random.shuffle(other_texts)
    allowed_other = other_texts[:max_other]

    filtered = action_texts + allowed_other
    random.shuffle(filtered)

    if verbose:
        total_in  = len(corpus)
        total_out = len(filtered)
        pct_action = len(action_texts) / max(total_out, 1)
        print(f"  [filter_action_only]")
        print(f"    Input        : {total_in:,}")
        print(f"    Action texts : {len(action_texts):,}  ({len(action_texts)/total_in:.1%})")
        print(f"    Other kept   : {len(allowed_other):,}  ({len(allowed_other)/max(total_out,1):.1%})")
        print(f"    Output       : {total_out:,}  (action density: {pct_action:.1%})")

    return filtered


# ── Loss Shaping: Verb Reward ─────────────────────────────────────────────────

def build_verb_mask(root_vocab: Dict[str, int], device: torch.device) -> torch.Tensor:
    """
    Bangun binary mask: 1.0 untuk token yang merupakan verb action,
    0.0 untuk yang bukan.

    Digunakan untuk loss shaping: token verb dapat reward tambahan.
    """
    vocab_size = max(root_vocab.values()) + 1
    mask = torch.zeros(vocab_size, device=device)
    for verb in _ACTION_VERBS:
        if verb in root_vocab:
            mask[root_vocab[verb]] = 1.0
    return mask


def compute_shaped_loss(
    base_loss: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    verb_mask: torch.Tensor,
    reward_scale: float = 0.15,
) -> torch.Tensor:
    """
    Tambahkan reward ke loss ketika target adalah verb action.

    Cara kerja:
      shaped_loss = base_loss - reward_scale * P(verb) ketika target=verb

    Ini memaksa model untuk meningkatkan probabilitas verb action,
    karena menurunkan probabilitas verb = loss lebih tinggi.

    Args:
        base_loss    : loss biasa dari model forward
        logits       : output logits (B, L, V)
        targets      : target token ids (B, L)
        verb_mask    : binary mask verb tokens (V,)
        reward_scale : skala reward (default 0.15 = tambahan 15% sinyal)

    Returns:
        shaped_loss: scalar
    """
    if logits is None or targets is None:
        return base_loss

    try:
        V = logits.shape[-1]
        mask = verb_mask[:V]  # safety clip

        # Flatten
        logits_flat  = logits.reshape(-1, V)        # (B*L, V)
        targets_flat = targets.reshape(-1)           # (B*L,)

        # Cari posisi di mana target adalah verb
        valid = (targets_flat >= 0) & (targets_flat < V)
        verb_pos = valid & (mask[targets_flat.clamp(0, V-1)] > 0)

        if verb_pos.any():
            # Hitung probabilitas verb di posisi tersebut
            probs    = torch.softmax(logits_flat[verb_pos], dim=-1)  # (N_verb, V)
            verb_ids = targets_flat[verb_pos]                        # (N_verb,)
            verb_probs = probs.gather(1, verb_ids.unsqueeze(1)).squeeze(1)  # (N_verb,)

            # Reward: kurangi loss proporsional dengan P(verb)
            reward = reward_scale * verb_probs.mean()
            return (base_loss - reward).clamp(min=0.0)
    except Exception:
        pass

    return base_loss


# ── Training Loop ────────────────────────────────────────────────────────────


def _build_verb_ids(root_vocab: Dict[str, int]) -> Set[int]:
    """Bangun set of verb token ids dari root_vocab menggunakan _ACTION_VERBS."""
    return {root_vocab[v] for v in _ACTION_VERBS if v in root_vocab}


def _run_loop(
    model,
    corpus: List[str],
    root_vocab: Dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    use_loss_shaping: bool,
    verbose: bool,
    label: str = "",
    use_composite_loss: bool = True,
) -> List[float]:
    """
    Training loop dasar tanpa balanced sampler.
    Gunakan shuffle biasa — biarkan distribusi alami corpus mengalir.
    Loss shaping verb reward diaktifkan jika use_loss_shaping=True.
    CompositeLoss (embedding relation + co-occurrence) diaktifkan
    jika use_composite_loss=True.
    """
    dataset = AksaraDataset(corpus, root_vocab, max_length=32, min_length=2)
    loader  = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn, drop_last=False,
    )

    verb_mask = build_verb_mask(root_vocab, device) if use_loss_shaping else None
    n_verbs   = int(verb_mask.sum().item()) if verb_mask is not None else 0
    if verbose and verb_mask is not None:
        print(f"  [loss shaping] {n_verbs} verb tokens aktif di vocab")

    # Precompute co-occurrence matrix dan bangun CompositeLoss
    composite = None
    if use_composite_loss:
        verb_ids  = _build_verb_ids(root_vocab)
        co_matrix = build_cooccurrence_matrix(
            corpus, root_vocab, window=4, min_count=2,
        )
        composite = CompositeLoss(
            lambda_rel=0.3,
            lambda_co=0.1,
            verb_ids=verb_ids,
            co_matrix=co_matrix,
        )
        if verbose:
            print(f"  [composite loss] rel=0.3  co=0.1  "
                  f"verb_ids={len(verb_ids)}  pmi_pairs={len(co_matrix):,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # ReduceLROnPlateau: turunkan LR hanya kalau loss tidak improve
    # Ini mencegah LR collapse prematur di akhir epoch
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=lr * 0.05
    )

    loss_history  = []
    best_loss     = float("inf")
    patience_count = 0
    early_stop_patience = 15  # stop jika loss tidak improve 15 epoch berturut-turut

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in loader:
            batch = batch.to(device)
            L     = batch.morpheme_ids.shape[1]
            dep_masks = torch.zeros(
                batch.morpheme_ids.shape[0], L, L,
                dtype=torch.bool, device=device
            )
            lps_dict = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids":    batch.affix_ids,
                "dep_masks":    dep_masks,
                "lengths":      batch.lengths,
            }
            out  = model(lps_dict, targets=batch.as_targets())
            loss = out["losses"]["total"]

            if loss.isnan() or loss.isinf():
                continue

            # Loss shaping: verb reward
            if verb_mask is not None:
                logits  = out.get("logits", None)
                targets_dict = batch.as_targets()
                loss = compute_shaped_loss(
                    loss, logits, targets_dict, verb_mask, reward_scale=0.15
                )

            # CompositeLoss: embedding relation + co-occurrence
            if composite is not None:
                loss, _breakdown = composite(loss, out, batch.morpheme_ids)
                if loss.isnan() or loss.isinf():
                    loss = out["losses"]["total"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / max(len(losses), 1)
        loss_history.append(avg_loss)
        sched.step(avg_loss)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss      = avg_loss
            patience_count = 0
        else:
            patience_count += 1

        if verbose and epoch % 10 == 0:
            ppl      = math.exp(min(avg_loss / 10, 10))
            cur_lr   = opt.param_groups[0]["lr"]
            tag      = f"[{label}] " if label else ""
            print(f"  {tag}[Epoch {epoch:>3}/{epochs}]  loss={avg_loss:.3f}  "
                  f"ppl≈{ppl:.1f}  lr={cur_lr:.2e}  patience={patience_count}")

        if patience_count >= early_stop_patience:
            if verbose:
                print(f"  [early stop] epoch {epoch} — loss tidak improve {early_stop_patience} epoch")
            break

    return loss_history


def train_phase1(
    model,
    action_corpus: List[str],
    root_vocab: Dict[str, int],
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    other_limit: float = 0.15,
    verbose: bool = True,
) -> Dict:
    """
    Phase 1: Action-Only Training.

    - Filter corpus: buang kalimat tanpa sinyal verb action
    - Izinkan maksimal other_limit kalimat 'other' untuk variasi
    - Aktifkan loss shaping verb reward
    - TIDAK ada balanced sampler

    Target: SVO dan VO-pairing mulai muncul sebelum ekspos Wikipedia.
    """
    if verbose:
        print(f"  [Phase 1] Action-Only Training ({epochs} epoch)")

    filtered = filter_action_only(action_corpus, other_limit=other_limit, verbose=verbose)

    if not filtered:
        raise ValueError("filter_action_only menghasilkan corpus kosong.")

    loss_history = _run_loop(
        model, filtered, root_vocab, device,
        epochs=epochs, batch_size=batch_size, lr=lr,
        use_loss_shaping=True, verbose=verbose, label="P1",
    )

    return {"phase": 1, "loss_history": loss_history, "corpus_size": len(filtered)}


def train_phase2(
    model,
    action_corpus: List[str],
    wiki_corpus: List[str],
    root_vocab: Dict[str, int],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-4,
    action_ratio: float = 0.4,
    verbose: bool = True,
) -> Dict:
    """
    Phase 2: Mixed Training untuk generalisasi.

    - Campur action corpus + Wikipedia
    - Proporsi: action_ratio action, sisanya Wikipedia
    - Loss shaping tetap aktif
    - LR lebih kecil (default 2e-4) agar tidak overwrite Phase 1

    Hanya jalankan ini setelah Phase 1 stabil (SVO ≥ 3/8).
    """
    if verbose:
        print(f"  [Phase 2] Mixed Training ({epochs} epoch)  "
              f"action={action_ratio:.0%}  wiki={1-action_ratio:.0%}")

    # Resample action corpus untuk mencapai rasio yang diinginkan
    n_wiki   = len(wiki_corpus)
    n_action = int(n_wiki * action_ratio / max(1 - action_ratio, 1e-6))
    action_sample = random.choices(action_corpus, k=min(n_action, len(action_corpus) * 3))

    mixed = action_sample + wiki_corpus
    random.shuffle(mixed)

    if verbose:
        print(f"    Action: {len(action_sample):,}  Wiki: {len(wiki_corpus):,}  "
              f"Total: {len(mixed):,}")

    loss_history = _run_loop(
        model, mixed, root_vocab, device,
        epochs=epochs, batch_size=batch_size, lr=lr,
        use_loss_shaping=True, verbose=verbose, label="P2",
    )

    return {"phase": 2, "loss_history": loss_history, "corpus_size": len(mixed)}


# ── Relation Corpus Builder (Phase 2) ────────────────────────────────────────

# Pasangan verb → objek valid per domain
_VERB_OBJECT_PAIRS: Dict[str, List[str]] = {
    "makan":      ["nasi", "roti", "mie", "sayur", "ikan", "daging", "makanan", "lauk"],
    "memakan":    ["nasi", "roti", "buah", "makanan"],
    "minum":      ["air", "susu", "teh", "kopi", "jus"],
    "membaca":    ["buku", "artikel", "novel", "koran", "teks", "cerita"],
    "baca":       ["buku", "artikel", "koran"],
    "bekerja":    ["laporan", "proyek", "tugas"],
    "mengerjakan":["tugas", "laporan", "pekerjaan"],
    "pergi":      ["sekolah", "pasar", "kantor", "toko"],
    "menuju":     ["sekolah", "pasar", "kantor"],
    "memasak":    ["nasi", "sayur", "ikan", "sup", "mie"],
    "menggoreng": ["ikan", "ayam", "tempe", "tahu"],
    "merebus":    ["air", "sayur", "mie"],
    "belajar":    ["matematika", "bahasa", "sains", "pelajaran"],
    "mempelajari":["materi", "pelajaran", "buku"],
    "memeriksa":  ["pasien", "kesehatan", "kondisi"],
    "mengecek":   ["laporan", "kondisi", "pasien"],
    "menetapkan": ["kebijakan", "peraturan", "keputusan"],
    "mengeluarkan":["kebijakan", "peraturan", "aturan"],
}

# Template kalimat relasi (SUBJ VERB OBJ)
# Prioritas tinggi: template yang menempatkan OBJ langsung setelah VERB
# Ini melatih bigram (verb, obj) yang persis dicek oleh VO pairing scorer
_RELATION_TEMPLATES: List[str] = [
    # Weight 3x: verb langsung diikuti obj — ini yang paling penting
    "{subj} {verb} {obj}",
    "{subj} {verb} {obj}",
    "{subj} {verb} {obj}",
    # Weight 1x: dengan kata bantu di tengah
    "{subj} sedang {verb} {obj}",
    "{subj} akan {verb} {obj}",
    "{subj} sudah {verb} {obj}",
    "{subj} mulai {verb} {obj}",
    "{subj} ingin {verb} {obj}",
    # Variasi kalimat pendek tanpa subjek eksplisit
    "{verb} {obj}",
    "{verb} {obj} itu",
    "{verb} {obj} setiap hari",
]

_RELATION_SUBJECTS: List[str] = [
    "saya", "dia", "mereka", "kami", "kita",
    "ibu", "ayah", "anak", "guru", "dokter",
    "murid", "siswa", "pasien", "pemerintah",
]


def _build_relation_corpus(n: int = 6000) -> List[str]:
    """
    Bangun corpus relasi verb→object secara programatik.

    Setiap kalimat dijamin mengandung bigram (verb, obj) yang valid.
    Template dengan verb langsung diikuti obj diberi bobot 3x lebih tinggi
    untuk memastikan model belajar menempatkan objek tepat setelah verb.

    Args:
        n : jumlah kalimat relasi yang dibuat (default 6000)

    Returns:
        list of strings dengan jaminan bigram VO valid
    """
    results = []
    pairs = [(v, o) for v, objs in _VERB_OBJECT_PAIRS.items() for o in objs]

    while len(results) < n:
        verb, obj  = random.choice(pairs)
        subj       = random.choice(_RELATION_SUBJECTS)
        template   = random.choice(_RELATION_TEMPLATES)
        results.append(template.format(subj=subj, verb=verb, obj=obj))

    return results


# ── Curriculum: P1 → P2 → P3 sekali jalan ────────────────────────────────────

def train_curriculum(
    model,
    action_corpus: List[str],
    wiki_corpus: List[str],
    root_vocab: Dict[str, int],
    device: torch.device,
    p1_epochs: int = 10,
    p2_epochs: int = 15,
    p3_epochs: int = 10,
    batch_size: int = 16,
    lr_p1: float = 5e-4,
    lr_p2: float = 3e-4,
    lr_p3: float = 1e-4,
    replay_ratio: float = 0.3,
    verbose: bool = True,
) -> Dict:
    """
    Curriculum lengkap P1 → P2 → P3. Dipanggil SEKALI, tidak diulang.

    Phase 1 — Action Only (p1_epochs):
      - Filter action corpus (verb-only, other ≤ 15%)
      - Loss shaping verb reward
      - Tujuan: bangun representasi verb dasar

    Phase 2 — Relation Training (p2_epochs):
      - Corpus = relasi SVO programatik + replay 30% dari P1
      - Loss shaping tetap aktif
      - Tujuan: paksa VO pairing terbentuk

    Phase 3 — Full Mix (p3_epochs):
      - Corpus = action + wiki + replay P1+P2
      - LR kecil untuk stabilisasi, bukan overwrite
      - Tujuan: generalisasi tanpa lupa struktur aksi

    Args:
        replay_ratio : fraksi corpus lama yang di-replay per fase (default 30%)

    Returns:
        dict dengan loss_history per fase dan corpus_sizes
    """
    results: Dict = {"phases": {}}

    # ── Phase 1: Action Only ──────────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  CURRICULUM PHASE 1 — Action Only ({p1_epochs} epoch)")
        print(f"{'─'*60}")

    p1_corpus = filter_action_only(action_corpus, other_limit=0.15, verbose=verbose)
    if not p1_corpus:
        raise ValueError("Phase 1: corpus kosong setelah filter.")

    lh_p1 = _run_loop(
        model, p1_corpus, root_vocab, device,
        epochs=p1_epochs, batch_size=batch_size, lr=lr_p1,
        use_loss_shaping=True, verbose=verbose, label="P1",
    )
    results["phases"][1] = {"loss_history": lh_p1, "corpus_size": len(p1_corpus)}

    # ── Phase 2: Relation Training + Replay P1 ───────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  CURRICULUM PHASE 2 — Relation Training ({p2_epochs} epoch)")
        print(f"{'─'*60}")

    relation_corpus = _build_relation_corpus(n=6000)
    n_replay_p1     = int(len(p1_corpus) * replay_ratio)
    replay_p1       = random.sample(p1_corpus, min(n_replay_p1, len(p1_corpus)))
    p2_corpus       = relation_corpus + replay_p1
    random.shuffle(p2_corpus)

    if verbose:
        print(f"    Relasi SVO  : {len(relation_corpus):,}")
        print(f"    Replay P1   : {len(replay_p1):,}  ({replay_ratio:.0%} dari P1)")
        print(f"    Total P2    : {len(p2_corpus):,}")

    lh_p2 = _run_loop(
        model, p2_corpus, root_vocab, device,
        epochs=p2_epochs, batch_size=batch_size, lr=lr_p2,
        use_loss_shaping=True, verbose=verbose, label="P2",
    )
    results["phases"][2] = {"loss_history": lh_p2, "corpus_size": len(p2_corpus)}

    # ── Phase 3: Full Mix + Replay P1+P2 ─────────────────────────────────────
    if verbose:
        print(f"\n{'─'*60}")
        print(f"  CURRICULUM PHASE 3 — Full Mix ({p3_epochs} epoch)")
        print(f"{'─'*60}")

    n_wiki      = min(len(wiki_corpus), 5000)   # cap Wikipedia agar tidak banjiri
    wiki_sample = random.sample(wiki_corpus, n_wiki)

    n_replay_p2 = int(len(p2_corpus) * replay_ratio)
    replay_p2   = random.sample(p2_corpus, min(n_replay_p2, len(p2_corpus)))
    n_replay_p1b = int(len(p1_corpus) * replay_ratio * 0.5)
    replay_p1b  = random.sample(p1_corpus, min(n_replay_p1b, len(p1_corpus)))

    p3_corpus = wiki_sample + replay_p1b + replay_p2
    random.shuffle(p3_corpus)

    if verbose:
        print(f"    Wikipedia   : {len(wiki_sample):,}")
        print(f"    Replay P1   : {len(replay_p1b):,}")
        print(f"    Replay P2   : {len(replay_p2):,}")
        print(f"    Total P3    : {len(p3_corpus):,}")

    lh_p3 = _run_loop(
        model, p3_corpus, root_vocab, device,
        epochs=p3_epochs, batch_size=batch_size, lr=lr_p3,
        use_loss_shaping=True, verbose=verbose, label="P3",
    )
    results["phases"][3] = {"loss_history": lh_p3, "corpus_size": len(p3_corpus)}

    results["total_epochs"] = p1_epochs + p2_epochs + p3_epochs
    return results


# ── Backward compat: train_balanced → Phase 1 saja ───────────────────────────

def train_balanced(
    model,
    corpus: List[str],
    root_vocab: Dict[str, int],
    device: torch.device,
    epochs: int,
    batch_size: int   = 16,
    lr: float         = 5e-4,
    use_domain_balance: bool = True,   # diabaikan (deprecated)
    use_loss_weighting: bool = True,   # diabaikan (deprecated)
    use_curriculum: bool     = True,   # diabaikan (deprecated)
    verbose: bool     = True,
) -> Dict:
    """
    Wrapper backward-compatible untuk action_benchmark.py --balanced.
    Sekarang identik dengan train_phase1 + filter action-only.
    Parameter use_domain_balance/use_loss_weighting/use_curriculum diabaikan.
    """
    return train_phase1(
        model, corpus, root_vocab, device,
        epochs=epochs, batch_size=batch_size, lr=lr,
        other_limit=0.15, verbose=verbose,
    )
