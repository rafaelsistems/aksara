"""
AKSARA — Stress Test Linguistik (3-Layer)

Tujuan: Hancurkan model dengan data noisy untuk membuktikan
ketahanan representasi linguistik AKSARA.

3 Layer Stress:
  Layer 1 — Surface Noise: typo, huruf hilang, spasi kacau
  Layer 2 — Morphological Corruption: affix salah, double affix, affix acak
  Layer 3 — Structural Chaos: tanpa SPOK, urutan acak, missing words

Metrik:
  - MCS, SVS, SDS per layer
  - Recovery Score: seberapa baik model "memperbaiki" struktur
  - Degradation Curve: noise ↑ → performa ↓ (seberapa cepat?)

Cara jalankan:
    python examples/stress_test.py --kbbi kbbi_true_clean_production.json
    python examples/stress_test.py  # tanpa KBBI
"""

import argparse
import sys
import random
import time
import math
import copy
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from aksara.core.model import AksaraModel, AksaraConfig
from aksara.core.bsu import BSUConfig
from aksara.core.meb import MEBConfig
from aksara.linguistic.lps import LPSConfig, build_root_vocab
from aksara.linguistic.lsk import LSKConfig
from aksara.core.gos import GOSConfig
from aksara.training.loss import LossConfig
from aksara.data.dataset import AksaraDataset, collate_fn
from aksara.utils.indo_metrics import IndoNativeMetrics

# ─── Clean Corpus ────────────────────────────────────────────────────────────

CLEAN_CORPUS = [
    "pemerintah indonesia menetapkan kebijakan baru tentang pendidikan nasional",
    "presiden menandatangani peraturan tentang pengembangan ekonomi digital",
    "kementerian kesehatan mengumumkan program vaksinasi untuk seluruh masyarakat",
    "bank indonesia mempertahankan suku bunga acuan pada level yang stabil",
    "saya berjalan di taman setiap pagi untuk menjaga kesehatan tubuh",
    "dia membaca buku pelajaran dengan tekun dan penuh semangat belajar",
    "anak-anak bermain dengan gembira di halaman sekolah yang luas",
    "petani menanam padi di sawah yang subur setiap musim hujan tiba",
    "guru mengajarkan matematika kepada murid-murid yang duduk dengan tertib",
    "pembangunan berkelanjutan harus mempertimbangkan dampak lingkungan hidup",
    "pengembangan sumber daya manusia merupakan investasi jangka panjang",
    "pemberantasan korupsi memerlukan keterlibatan seluruh lapisan masyarakat",
    "mahkamah agung memutuskan perkara sengketa tanah di provinsi jawa tengah",
    "mereka berlari kencang menuju garis finis dengan penuh semangat",
    "kebijakan baru itu mendapat dukungan luas dari berbagai kalangan",
]


# ─── Layer 1: Surface Noise ─────────────────────────────────────────────────

def apply_typo(word: str, prob: float = 0.3) -> str:
    """Simulasi typo: swap huruf, huruf ganda, huruf hilang."""
    if len(word) < 3 or random.random() > prob:
        return word
    op = random.choice(["swap", "double", "drop", "insert"])
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)
    if op == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif op == "double":
        chars.insert(idx, chars[idx])
    elif op == "drop" and len(chars) > 2:
        chars.pop(idx)
    elif op == "insert":
        chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz"))
    return "".join(chars)


def surface_noise(text: str, intensity: float = 0.3) -> str:
    """Layer 1: typo, huruf hilang, spasi kacau."""
    words = text.split()
    noisy = []
    for w in words:
        w = apply_typo(w, prob=intensity)
        # Spasi kacau
        if random.random() < intensity * 0.3:
            w = w + " " + w[:2]  # fragment
        noisy.append(w)
    # Kadang hilangkan spasi
    result = " ".join(noisy)
    if random.random() < intensity * 0.2:
        idx = random.randint(0, max(0, len(result) - 5))
        result = result[:idx] + result[idx:].replace(" ", "", 1)
    return result


# ─── Layer 2: Morphological Corruption ──────────────────────────────────────

FAKE_PREFIXES = ["meng", "mem", "ber", "ter", "di", "ke", "se", "per"]
FAKE_SUFFIXES = ["kan", "an", "i", "nya"]

def morph_corrupt(text: str, intensity: float = 0.3) -> str:
    """Layer 2: affix salah, double affix, affix acak."""
    words = text.split()
    corrupted = []
    for w in words:
        if len(w) < 4 or random.random() > intensity:
            corrupted.append(w)
            continue
        op = random.choice(["wrong_prefix", "double_affix", "strip_affix", "wrong_suffix"])
        if op == "wrong_prefix":
            prefix = random.choice(FAKE_PREFIXES)
            corrupted.append(prefix + w)
        elif op == "double_affix":
            prefix = random.choice(FAKE_PREFIXES)
            suffix = random.choice(FAKE_SUFFIXES)
            corrupted.append(prefix + w + suffix)
        elif op == "strip_affix":
            # Strip prefix jika ada
            for p in FAKE_PREFIXES:
                if w.startswith(p) and len(w) > len(p) + 2:
                    w = w[len(p):]
                    break
            corrupted.append(w)
        elif op == "wrong_suffix":
            suffix = random.choice(FAKE_SUFFIXES)
            corrupted.append(w + suffix)
    return " ".join(corrupted)


# ─── Layer 3: Structural Chaos ──────────────────────────────────────────────

def structural_chaos(text: str, intensity: float = 0.3) -> str:
    """Layer 3: tanpa SPOK jelas, urutan acak, missing words."""
    words = text.split()
    if not words:
        return text

    ops = []
    if random.random() < intensity:
        ops.append("shuffle")
    if random.random() < intensity:
        ops.append("drop")
    if random.random() < intensity:
        ops.append("repeat")
    if not ops:
        ops.append(random.choice(["shuffle", "drop", "repeat"]))

    for op in ops:
        if op == "shuffle":
            # Acak sebagian urutan kata
            n_shuffle = max(2, int(len(words) * intensity))
            indices = random.sample(range(len(words)), min(n_shuffle, len(words)))
            shuffled_words = [words[i] for i in indices]
            random.shuffle(shuffled_words)
            for i, idx in enumerate(indices):
                words[idx] = shuffled_words[i]
        elif op == "drop":
            # Hilangkan beberapa kata
            n_drop = max(1, int(len(words) * intensity * 0.5))
            for _ in range(n_drop):
                if len(words) > 2:
                    words.pop(random.randint(0, len(words) - 1))
        elif op == "repeat":
            # Ulangi kata secara acak
            if words:
                idx = random.randint(0, len(words) - 1)
                words.insert(idx, words[idx])

    return " ".join(words)


# ─── Combined Noise at Different Intensities ────────────────────────────────

def apply_noise_layer(corpus: List[str], layer: str, intensity: float) -> List[str]:
    """Terapkan satu layer noise pada corpus."""
    fn = {
        "surface": surface_noise,
        "morphological": morph_corrupt,
        "structural": structural_chaos,
    }[layer]
    return [fn(text, intensity) for text in corpus]


def apply_all_layers(corpus: List[str], intensity: float) -> List[str]:
    """Terapkan semua layer noise secara berurutan."""
    result = corpus
    for layer in ["surface", "morphological", "structural"]:
        result = apply_noise_layer(result, layer, intensity)
    return result


# ─── Model & Training Helpers ───────────────────────────────────────────────

def make_model(corpus, kbbi_path="", device="cpu"):
    root_vocab = build_root_vocab(corpus, min_freq=1)
    bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
    meb_cfg = MEBConfig(bsu_config=bsu_cfg, n_layers=3, n_dep_heads=2, kbbi_anchor_dim=16)
    config = AksaraConfig(
        vocab_size=len(root_vocab),
        bsu_config=bsu_cfg,
        meb_config=meb_cfg,
        lps_config=LPSConfig(),
        lsk_config=LSKConfig(kbbi_path=kbbi_path, kbbi_vector_dim=16),
        gos_config=GOSConfig(bsu_config=bsu_cfg, vocab_size=len(root_vocab)),
        loss_config=LossConfig(),
        kbbi_path=kbbi_path,
        max_seq_len=48,
        dropout=0.05,
    )
    model = AksaraModel(config, root_vocab).to(device)
    return model, root_vocab


def build_dep_masks(model, batch, device):
    B = batch.morpheme_ids.size(0)
    L = batch.morpheme_ids.size(1)
    dep_masks = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    for i in range(B):
        actual_len = batch.lengths[i].item()
        dummy_tokens = ["_"] * actual_len
        mask_i = model.lps.build_dep_mask(dummy_tokens, L)
        mask_i[actual_len:, :] = False
        mask_i[:, actual_len:] = False
        dep_masks[i] = mask_i.to(device)
    return dep_masks


def train_and_evaluate(model, root_vocab, train_corpus, eval_corpus, device,
                       epochs=5, lr=5e-4, label=""):
    """Train model dan evaluasi pada eval_corpus. Return metrik."""
    train_ds = AksaraDataset(train_corpus, root_vocab, max_length=48, min_length=1)
    eval_ds = AksaraDataset(eval_corpus, root_vocab, max_length=48, min_length=1)

    if len(train_ds) == 0 or len(eval_ds) == 0:
        return {"loss": float("nan"), "mcs": 0, "svs": 0, "sds": 0,
                "morph_acc": 0, "root_ppl": float("inf")}

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn, drop_last=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            dep_masks_t = build_dep_masks(model, batch, device)
            lps_dict = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids": batch.affix_ids,
                "dep_masks": dep_masks_t,
                "lengths": batch.lengths,
            }
            targets = batch.as_targets()
            outputs = model(lps_dict, targets=targets)
            loss = outputs["losses"]["total"]
            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    # Evaluate
    model.eval()
    metrics = IndoNativeMetrics()
    eval_losses = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            dep_masks_e = build_dep_masks(model, batch, device)
            lps_dict = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids": batch.affix_ids,
                "dep_masks": dep_masks_e,
                "lengths": batch.lengths,
            }
            targets = batch.as_targets()
            outputs = model(lps_dict, targets=targets)

            if "losses" in outputs:
                eval_losses.append(outputs["losses"]["total"].item())

            aux = outputs["aux"]
            metrics.update(
                gos_output=outputs["gos_out"],
                targets=targets,
                semantic_slots=aux["semantic_slots"],
                kbbi_anchors=aux["kbbi_anchors_proj"],
                attention_mask=batch.attention_mask,
            )

    result = metrics.end_epoch(epoch=0)
    avg_loss = sum(eval_losses) / max(len(eval_losses), 1)

    return {
        "loss": avg_loss,
        "mcs": result.mcs.overall,
        "svs": result.svs.overall,
        "sds": result.sds.overall,
        "morph_acc": result.morph_accuracy,
        "root_ppl": result.root_perplexity,
    }


# ─── Main Stress Test ───────────────────────────────────────────────────────

def run_stress_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    print(f"\n{'='*70}")
    print("  AKSARA — Stress Test Linguistik (3-Layer)")
    print(f"{'='*70}")
    print(f"  Device : {device}")
    print(f"  KBBI   : {args.kbbi if args.kbbi else 'tidak aktif'}")
    print(f"  Epochs : {args.epochs}")
    print()

    # ── 1. Train model pada clean data ───────────────────────────────────────
    print("[1/3] Training model pada clean data...")
    all_corpus = CLEAN_CORPUS * 3  # augment via repetition
    model, root_vocab = make_model(all_corpus, args.kbbi or "", device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Parameters: {n_params:,}")

    # Train pada clean data
    model_state_clean = None
    train_and_evaluate(model, root_vocab, all_corpus, CLEAN_CORPUS, device,
                       epochs=args.epochs, label="clean_train")
    model_state_clean = copy.deepcopy(model.state_dict())

    # ── 2. Evaluate pada setiap layer noise × intensitas ─────────────────────
    print("\n[2/3] Evaluating degradation across noise layers & intensities...\n")

    layers = ["surface", "morphological", "structural", "all"]
    intensities = [0.1, 0.2, 0.3, 0.5, 0.7]

    results = {}

    # Baseline: clean → clean
    model.load_state_dict(model_state_clean)
    baseline = train_and_evaluate(model, root_vocab, [], CLEAN_CORPUS, device,
                                  epochs=0, label="baseline")
    # Evaluate clean (no training, just eval)
    model.eval()
    metrics_clean = IndoNativeMetrics()
    eval_ds = AksaraDataset(CLEAN_CORPUS, root_vocab, max_length=48, min_length=1)
    eval_loader = DataLoader(eval_ds, batch_size=4, collate_fn=collate_fn, drop_last=False)
    clean_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            dep_masks_e = build_dep_masks(model, batch, device)
            lps_dict = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids": batch.affix_ids,
                "dep_masks": dep_masks_e,
                "lengths": batch.lengths,
            }
            targets = batch.as_targets()
            outputs = model(lps_dict, targets=targets)
            if "losses" in outputs:
                clean_losses.append(outputs["losses"]["total"].item())
            aux = outputs["aux"]
            metrics_clean.update(
                gos_output=outputs["gos_out"],
                targets=targets,
                semantic_slots=aux["semantic_slots"],
                kbbi_anchors=aux["kbbi_anchors_proj"],
                attention_mask=batch.attention_mask,
            )
    clean_result = metrics_clean.end_epoch(epoch=0)
    baseline = {
        "loss": sum(clean_losses) / max(len(clean_losses), 1),
        "mcs": clean_result.mcs.overall,
        "svs": clean_result.svs.overall,
        "sds": clean_result.sds.overall,
        "morph_acc": clean_result.morph_accuracy,
        "root_ppl": clean_result.root_perplexity,
    }
    results[("clean", 0.0)] = baseline

    # Header
    print(f"{'Layer':<15} {'Intens':>6} | {'Loss':>8} | {'MCS':>6} | {'SVS':>6} | "
          f"{'SDS':>6} | {'morph%':>7} | {'ppl':>7} | {'Δloss':>7}")
    print("-" * 85)
    print(f"{'clean':<15} {'0.0':>6} | {baseline['loss']:>8.4f} | {baseline['mcs']:>6.3f} | "
          f"{baseline['svs']:>6.3f} | {baseline['sds']:>6.3f} | {baseline['morph_acc']:>7.3f} | "
          f"{baseline['root_ppl']:>7.1f} | {'---':>7}")

    for layer in layers:
        for intensity in intensities:
            model.load_state_dict(model_state_clean)

            if layer == "all":
                noisy_corpus = apply_all_layers(CLEAN_CORPUS, intensity)
            else:
                noisy_corpus = apply_noise_layer(CLEAN_CORPUS, layer, intensity)

            # Evaluate model (trained on clean) on noisy data
            noisy_ds = AksaraDataset(noisy_corpus, root_vocab, max_length=48, min_length=1)
            if len(noisy_ds) == 0:
                continue
            noisy_loader = DataLoader(noisy_ds, batch_size=4, collate_fn=collate_fn,
                                      drop_last=False)

            model.eval()
            metrics_noisy = IndoNativeMetrics()
            noisy_losses = []
            with torch.no_grad():
                for batch in noisy_loader:
                    batch = batch.to(device)
                    dep_masks_e = build_dep_masks(model, batch, device)
                    lps_dict = {
                        "morpheme_ids": batch.morpheme_ids,
                        "affix_ids": batch.affix_ids,
                        "dep_masks": dep_masks_e,
                        "lengths": batch.lengths,
                    }
                    targets = batch.as_targets()
                    outputs = model(lps_dict, targets=targets)
                    if "losses" in outputs:
                        noisy_losses.append(outputs["losses"]["total"].item())
                    aux = outputs["aux"]
                    metrics_noisy.update(
                        gos_output=outputs["gos_out"],
                        targets=targets,
                        semantic_slots=aux["semantic_slots"],
                        kbbi_anchors=aux["kbbi_anchors_proj"],
                        attention_mask=batch.attention_mask,
                    )

            noisy_result = metrics_noisy.end_epoch(epoch=0)
            r = {
                "loss": sum(noisy_losses) / max(len(noisy_losses), 1),
                "mcs": noisy_result.mcs.overall,
                "svs": noisy_result.svs.overall,
                "sds": noisy_result.sds.overall,
                "morph_acc": noisy_result.morph_accuracy,
                "root_ppl": noisy_result.root_perplexity,
            }
            results[(layer, intensity)] = r

            delta_loss = r["loss"] - baseline["loss"]
            print(f"{layer:<15} {intensity:>6.1f} | {r['loss']:>8.4f} | {r['mcs']:>6.3f} | "
                  f"{r['svs']:>6.3f} | {r['sds']:>6.3f} | {r['morph_acc']:>7.3f} | "
                  f"{r['root_ppl']:>7.1f} | {delta_loss:>+7.4f}")

    # ── 3. Degradation Curve & Recovery Score ────────────────────────────────
    print(f"\n[3/3] Degradation Analysis...\n")

    for layer in layers:
        layer_results = [(i, results.get((layer, i))) for i in intensities
                         if (layer, i) in results]
        if not layer_results:
            continue

        # Degradation rate: slope of loss increase per intensity unit
        if len(layer_results) >= 2:
            x = [lr[0] for lr in layer_results]
            y = [lr[1]["loss"] - baseline["loss"] for lr in layer_results]
            # Simple linear regression slope
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi ** 2 for xi in x)
            denom = n * sum_x2 - sum_x ** 2
            slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0

            # Recovery score: 1 - (max_degradation / baseline_loss)
            max_deg = max(y) if y else 0
            recovery = max(0, 1.0 - max_deg / max(baseline["loss"], 1e-6))

            # MCS stability: min MCS across intensities
            mcs_min = min(lr[1]["mcs"] for lr in layer_results)
            mcs_stability = mcs_min / max(baseline["mcs"], 1e-6)

            print(f"  [{layer.upper()}]")
            print(f"    Degradation slope : {slope:+.4f} (loss increase per 0.1 intensity)")
            print(f"    Recovery score    : {recovery:.3f} (1.0 = perfect, 0.0 = total collapse)")
            print(f"    MCS stability     : {mcs_stability:.3f} (min_MCS / baseline_MCS)")
            print(f"    Worst loss delta  : {max_deg:+.4f}")
            print()

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")

    # Check: apakah model degradasi pelan (bagus) atau tajam (buruk)?
    all_results = [(i, results.get(("all", i))) for i in intensities
                   if ("all", i) in results]
    if all_results:
        worst = max(all_results, key=lambda x: x[1]["loss"])
        best_mcs = min(r[1]["mcs"] for r in all_results)

        if worst[1]["loss"] < baseline["loss"] * 2.0:
            print("  ✅ Model degradasi PELAN di bawah noise gabungan — ketahanan baik")
        else:
            print("  ⚠️  Model degradasi TAJAM di bawah noise gabungan — perlu perbaikan")

        if best_mcs > 0.5:
            print("  ✅ MCS tetap di atas 0.5 bahkan di noise tertinggi — morfologi robust")
        else:
            print("  ⚠️  MCS turun di bawah 0.5 — morfologi tidak cukup robust")
    else:
        print("  ⚠️  Tidak cukup data untuk verdict")

    print()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKSARA Stress Test Linguistik")
    parser.add_argument("--kbbi", type=str, default="", help="Path ke KBBI JSON")
    parser.add_argument("--epochs", type=int, default=8, help="Epoch training pada clean data")
    args = parser.parse_args()
    run_stress_test(args)
