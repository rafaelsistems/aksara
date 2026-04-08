"""
AKSARA — Ablation Study

Tujuan: Buktikan setiap komponen AKSARA berkontribusi nyata,
bukan sekadar dekorasi arsitektural.

7 Konfigurasi Ablation:
  1. FULL          — semua komponen aktif (baseline)
  2. NO_MORPH      — disable f_morph (evolusi morfologi)
  3. NO_SYN        — disable f_syn (evolusi sintaktik)
  4. NO_SEM        — disable f_sem (grounding semantik KBBI)
  5. NO_MORPH_SYN  — disable f_morph + f_syn (interaksi)
  6. NO_MORPH_SEM  — disable f_morph + f_sem (interaksi)
  7. NO_SYN_SEM    — disable f_syn + f_sem (interaksi)

Setiap konfigurasi dijalankan 3 seeds untuk stabilitas statistik.

Metrik per konfigurasi:
  - Final loss (total, l_morph, l_struct, l_sem, l_ctx)
  - Convergence speed (epoch di mana loss < threshold)
  - Loss variance across seeds
  - Gradient norms (per komponen)
  - MCS, SVS, SDS

Cara jalankan:
    python examples/ablation_study.py --kbbi kbbi_true_clean_production.json
    python examples/ablation_study.py  # tanpa KBBI
"""

import argparse
import sys
import time
import copy
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

# ─── Corpus ──────────────────────────────────────────────────────────────────

CORPUS = [
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
] * 3  # augment via repetition


# ─── Ablation Configurations ────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "FULL":          [],                          # semua aktif
    "NO_MORPH":      ["f_morph"],                 # tanpa evolusi morfologi
    "NO_SYN":        ["f_syn"],                   # tanpa evolusi sintaktik
    "NO_SEM":        ["f_sem"],                   # tanpa grounding semantik
    "NO_MORPH_SYN":  ["f_morph", "f_syn"],        # tanpa morph + syn
    "NO_MORPH_SEM":  ["f_morph", "f_sem"],        # tanpa morph + sem
    "NO_SYN_SEM":    ["f_syn", "f_sem"],          # tanpa syn + sem
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def compute_grad_norms(model) -> Dict[str, float]:
    """Compute gradient norms per komponen."""
    norms = {}
    component_map = {
        "bsu": model.bsu,
        "meb": model.meb,
        "gos": model.gos,
        "lsk": model.lsk,
    }
    for name, module in component_map.items():
        total_norm = 0.0
        count = 0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
                count += 1
        norms[name] = math.sqrt(total_norm) if count > 0 else 0.0
    return norms


def run_single_config(
    config_name: str,
    disabled_components: List[str],
    seed: int,
    corpus: List[str],
    kbbi_path: str,
    device: torch.device,
    epochs: int = 10,
    lr: float = 5e-4,
) -> Dict:
    """Jalankan satu konfigurasi ablation dengan seed tertentu."""
    torch.manual_seed(seed)

    model, root_vocab = make_model(corpus, kbbi_path, device)

    # Apply ablation
    for comp in disabled_components:
        model.disable(comp)

    ablation_info = model.get_ablation_config()
    n_trainable = ablation_info["trainable_params"]

    dataset = AksaraDataset(corpus, root_vocab, max_length=48, min_length=1)
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    # Training loop
    epoch_losses = []
    epoch_metrics = []
    grad_norms_history = []
    convergence_epoch = -1
    convergence_threshold = None  # set after first epoch

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        batch_grad_norms = []

        for batch in loader:
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

                # Capture grad norms before step
                gnorms = compute_grad_norms(model)
                batch_grad_norms.append(gnorms)

                optimizer.step()
                batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / max(len(batch_losses), 1)
        epoch_losses.append(avg_loss)

        # Set convergence threshold after first epoch
        if epoch == 0:
            convergence_threshold = avg_loss * 0.5  # 50% of initial loss

        if convergence_epoch < 0 and avg_loss < convergence_threshold:
            convergence_epoch = epoch

        # Average grad norms for this epoch
        if batch_grad_norms:
            avg_gnorms = {}
            for key in batch_grad_norms[0]:
                avg_gnorms[key] = sum(g[key] for g in batch_grad_norms) / len(batch_grad_norms)
            grad_norms_history.append(avg_gnorms)

        # Eval metrics
        model.eval()
        metrics = IndoNativeMetrics()
        eval_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, drop_last=False)
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
                aux = outputs["aux"]
                metrics.update(
                    gos_output=outputs["gos_out"],
                    targets=targets,
                    semantic_slots=aux["semantic_slots"],
                    kbbi_anchors=aux["kbbi_anchors_proj"],
                    attention_mask=batch.attention_mask,
                )
        result = metrics.end_epoch(epoch=epoch)
        epoch_metrics.append({
            "mcs": result.mcs.overall,
            "svs": result.svs.overall,
            "sds": result.sds.overall,
            "morph_acc": result.morph_accuracy,
            "root_ppl": result.root_perplexity,
        })

    # Final eval with loss components
    model.eval()
    final_losses = {}
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
            for k, v in outputs["losses"].items():
                if k not in final_losses:
                    final_losses[k] = []
                final_losses[k].append(v.item() if torch.is_tensor(v) else v)
            break  # satu batch cukup untuk loss components

    avg_final_losses = {k: sum(v) / len(v) for k, v in final_losses.items()}

    # Reset ablation
    model.reset_ablation()

    return {
        "config": config_name,
        "seed": seed,
        "disabled": disabled_components,
        "n_trainable": n_trainable,
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else float("nan"),
        "final_loss_components": avg_final_losses,
        "convergence_epoch": convergence_epoch,
        "convergence_threshold": convergence_threshold,
        "grad_norms": grad_norms_history[-1] if grad_norms_history else {},
        "final_metrics": epoch_metrics[-1] if epoch_metrics else {},
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_ablation_study(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [42, 123, 777]

    print(f"\n{'='*80}")
    print("  AKSARA — Ablation Study")
    print(f"{'='*80}")
    print(f"  Device : {device}")
    print(f"  KBBI   : {args.kbbi if args.kbbi else 'tidak aktif'}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Seeds  : {seeds}")
    print()

    all_results = {}

    for config_name, disabled in ABLATION_CONFIGS.items():
        print(f"\n{'─'*60}")
        print(f"  Config: {config_name}")
        print(f"  Disabled: {disabled if disabled else '(none — full model)'}")
        print(f"{'─'*60}")

        config_results = []
        for seed in seeds:
            t0 = time.time()
            result = run_single_config(
                config_name=config_name,
                disabled_components=disabled,
                seed=seed,
                corpus=CORPUS,
                kbbi_path=args.kbbi or "",
                device=device,
                epochs=args.epochs,
            )
            elapsed = time.time() - t0
            config_results.append(result)

            fm = result["final_metrics"]
            print(f"    seed={seed:>3d} | loss={result['final_loss']:.4f} | "
                  f"conv@{result['convergence_epoch']:>2d} | "
                  f"MCS={fm.get('mcs', 0):.3f} SVS={fm.get('svs', 0):.3f} "
                  f"SDS={fm.get('sds', 0):.3f} | {elapsed:.1f}s")

        all_results[config_name] = config_results

    # ── Summary Table ────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  ABLATION SUMMARY TABLE")
    print(f"{'='*80}\n")

    header = (f"{'Config':<16} | {'Loss':>8} {'±σ':>6} | {'Conv':>4} | "
              f"{'MCS':>6} | {'SVS':>6} | {'SDS':>6} | {'morph%':>7} | "
              f"{'ppl':>7} | {'params':>8}")
    print(header)
    print("-" * len(header))

    full_loss = None

    for config_name in ABLATION_CONFIGS:
        results = all_results[config_name]
        losses = [r["final_loss"] for r in results]
        avg_loss = sum(losses) / len(losses)
        std_loss = (sum((l - avg_loss) ** 2 for l in losses) / len(losses)) ** 0.5

        conv_epochs = [r["convergence_epoch"] for r in results]
        avg_conv = sum(conv_epochs) / len(conv_epochs)

        mcs_vals = [r["final_metrics"].get("mcs", 0) for r in results]
        svs_vals = [r["final_metrics"].get("svs", 0) for r in results]
        sds_vals = [r["final_metrics"].get("sds", 0) for r in results]
        morph_vals = [r["final_metrics"].get("morph_acc", 0) for r in results]
        ppl_vals = [r["final_metrics"].get("root_ppl", float("inf")) for r in results]

        avg_mcs = sum(mcs_vals) / len(mcs_vals)
        avg_svs = sum(svs_vals) / len(svs_vals)
        avg_sds = sum(sds_vals) / len(sds_vals)
        avg_morph = sum(morph_vals) / len(morph_vals)
        avg_ppl = sum(ppl_vals) / len(ppl_vals)

        n_params = results[0]["n_trainable"]

        if config_name == "FULL":
            full_loss = avg_loss

        print(f"{config_name:<16} | {avg_loss:>8.4f} {std_loss:>5.4f} | {avg_conv:>4.1f} | "
              f"{avg_mcs:>6.3f} | {avg_svs:>6.3f} | {avg_sds:>6.3f} | {avg_morph:>7.3f} | "
              f"{avg_ppl:>7.1f} | {n_params:>8,}")

    # ── Gradient Norms ───────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  GRADIENT NORMS (final epoch, seed=42)")
    print(f"{'='*80}\n")

    gnorm_header = f"{'Config':<16} | {'BSU':>8} | {'MEB':>8} | {'GOS':>8} | {'LSK':>8}"
    print(gnorm_header)
    print("-" * len(gnorm_header))

    for config_name in ABLATION_CONFIGS:
        results = all_results[config_name]
        # Use first seed result
        gnorms = results[0].get("grad_norms", {})
        print(f"{config_name:<16} | {gnorms.get('bsu', 0):>8.4f} | "
              f"{gnorms.get('meb', 0):>8.4f} | {gnorms.get('gos', 0):>8.4f} | "
              f"{gnorms.get('lsk', 0):>8.4f}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  ABLATION VERDICT")
    print(f"{'='*80}\n")

    if full_loss is not None:
        for config_name, disabled in ABLATION_CONFIGS.items():
            if config_name == "FULL":
                continue
            results = all_results[config_name]
            avg_loss = sum(r["final_loss"] for r in results) / len(results)
            delta = avg_loss - full_loss
            pct = (delta / full_loss) * 100 if full_loss > 0 else 0

            avg_mcs = sum(r["final_metrics"].get("mcs", 0) for r in results) / len(results)
            full_mcs = sum(r["final_metrics"].get("mcs", 0)
                          for r in all_results["FULL"]) / len(all_results["FULL"])
            mcs_delta = avg_mcs - full_mcs

            if delta > 0.01:
                verdict = "✅ PENTING"
            elif delta > 0.001:
                verdict = "⚠️  MINOR"
            else:
                verdict = "❌ TIDAK SIGNIFIKAN"

            disabled_str = " + ".join(disabled)
            print(f"  {config_name:<16} (tanpa {disabled_str})")
            print(f"    Loss Δ: {delta:+.4f} ({pct:+.1f}%) | MCS Δ: {mcs_delta:+.3f} → {verdict}")
            print()

    print()
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKSARA Ablation Study")
    parser.add_argument("--kbbi", type=str, default="", help="Path ke KBBI JSON")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch per konfigurasi")
    args = parser.parse_args()
    run_ablation_study(args)
