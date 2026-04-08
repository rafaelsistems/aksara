"""
AKSARA Full Training — 2000 Data, 20 Epochs
Autoregressive BSU-level generation training.

Ini BUKAN test minimal. Ini training penuh dengan:
- 2000 kalimat dari KBBI corpus
- 20 epochs
- Teacher forcing + L_fluency
- Output inspection setiap 5 epoch
- Autoregressive generation test di akhir

Usage:
    py examples/train_full_autoregressive.py
"""

import sys
import os
import time
import json
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# AKSARA imports
from aksara.core.bsu import BSUConfig
from aksara.core.meb import MEBConfig
from aksara.core.gos import GOSConfig
from aksara.core.model import AksaraModel, AksaraConfig
from aksara.linguistic.lps import (
    LPSConfig, MorfologiAnalyzer, AFFIX_TO_ID, ROLE_LABELS, build_root_vocab
)
from aksara.linguistic.lsk import LSKConfig
from aksara.data.dataset import AksaraDataset, collate_fn
from aksara.data.corpus_builder import KBBICorpusBuilder
from aksara.training.loss import AksaraLoss
from aksara.training.pd import PengendaliDinamik, PDConfig


def main():
    print("=" * 70)
    print("  AKSARA FULL TRAINING — 2000 Data, 20 Epochs")
    print("  Autoregressive BSU-Level Generation")
    print("=" * 70)

    KBBI_PATH = "kbbi_true_clean_production.json"
    TARGET_DATA = 2000
    NUM_EPOCHS = 20
    BATCH_SIZE = 16
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    OUTPUT_DIR = "aksara_output/full_training"

    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ─── Step 1: Build Corpus ───
    print("\n[1/6] Building corpus from KBBI...")
    t0 = time.time()

    builder = KBBICorpusBuilder(KBBI_PATH, seed=SEED)
    corpus = builder.build_corpus(target_size=TARGET_DATA, min_words=4, max_words=30)
    stats = builder.get_stats(corpus)

    print(f"  Corpus: {len(corpus)} sentences")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Avg words/sentence: {stats['avg_words_per_sentence']:.1f}")
    print(f"  KBBI coverage: {stats['kbbi_vocab_coverage']:.1%}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ─── Step 2: Build Vocab ───
    print("\n[2/6] Building vocabulary...")
    t0 = time.time()

    root_vocab = build_root_vocab(corpus, min_freq=2)
    vocab_size = len(root_vocab)
    print(f"  Vocab size: {vocab_size}")

    # Known words for morphology
    known_words = set()
    if os.path.exists(KBBI_PATH):
        with open(KBBI_PATH, "r", encoding="utf-8") as f:
            kbbi_data = json.load(f)
        for entry in kbbi_data.get("entries", []):
            lemma = entry.get("lemma", "").lower().strip()
            if lemma:
                known_words.add(lemma)
    print(f"  Known words (KBBI): {len(known_words)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ─── Step 3: Build Dataset ───
    print("\n[3/6] Building dataset...")
    t0 = time.time()

    # Split: 90% train, 10% eval
    split_idx = int(len(corpus) * 0.9)
    train_texts = corpus[:split_idx]
    eval_texts = corpus[split_idx:]

    train_dataset = AksaraDataset(train_texts, root_vocab, max_length=64)
    eval_dataset = AksaraDataset(eval_texts, root_vocab, max_length=64)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    print(f"  Train: {len(train_dataset)} sentences")
    print(f"  Eval: {len(eval_dataset)} sentences")
    print(f"  Batches/epoch: {len(train_loader)}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ─── Step 4: Build Model ───
    print("\n[4/6] Building AKSARA model...")
    t0 = time.time()

    bsu_config = BSUConfig(d_morpheme=64, d_semantic=64, d_role=32, d_context=64)
    meb_config = MEBConfig(bsu_config=bsu_config, n_layers=4, n_dep_heads=4)
    gos_config = GOSConfig(
        bsu_config=bsu_config,
        vocab_size=vocab_size,
        affix_vocab_size=len(AFFIX_TO_ID),
        role_vocab_size=len(ROLE_LABELS),
        phi_seq_layers=2,
        alpha_context=0.7,
        teacher_forcing=True,
    )
    lps_config = LPSConfig()
    lsk_config = LSKConfig(kbbi_path=KBBI_PATH)

    aksara_config = AksaraConfig(
        bsu_config=bsu_config,
        meb_config=meb_config,
        gos_config=gos_config,
        lps_config=lps_config,
        lsk_config=lsk_config,
        lambda_root=2.0,
        lambda_fluency=0.1,
    )

    model = AksaraModel(aksara_config, root_vocab, known_words=known_words)
    model = model.to(DEVICE)

    params = model.num_parameters
    print(f"  Parameters: {params['trainable']:,} trainable / {params['total']:,} total")
    print(f"  KBBI coverage: {model.lsk.kbbi_coverage:.1%}")
    print(f"  Device: {DEVICE}")
    print(f"  d_total (BSU): {bsu_config.d_total}")
    print(f"  MEB layers: {meb_config.n_layers}")
    print(f"  Phi_seq layers: {gos_config.phi_seq_layers}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ─── Step 5: Training Loop ───
    print("\n[5/6] Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print("-" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    pd = PengendaliDinamik(PDConfig())

    # Warmup + cosine schedule
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return LR * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    global_step = 0
    best_eval_loss = float("inf")
    epoch_history = []

    training_start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = []
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(torch.device(DEVICE))

            # LR schedule
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Build dependency masks
            B = batch.morpheme_ids.size(0)
            L = batch.morpheme_ids.size(1)
            dep_masks = torch.zeros(B, L, L, dtype=torch.bool, device=DEVICE)
            for i in range(B):
                actual_len = batch.lengths[i].item()
                dummy_tokens = ["_"] * actual_len
                mask_i = model.lps.build_dep_mask(dummy_tokens, L)
                mask_i[actual_len:, :] = False
                mask_i[:, actual_len:] = False
                dep_masks[i] = mask_i.to(DEVICE)

            lps_output = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids": batch.affix_ids,
                "dep_masks": dep_masks,
                "lengths": batch.lengths,
                "max_len": L,
            }

            targets = batch.as_targets()
            lambdas = pd.get_lambdas()

            # Forward
            outputs = model(lps_output, targets=targets, lambdas=lambdas)
            loss = outputs["losses"]["total"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # PD update
            gos_logits = outputs["gos_out"].get("context_logits")
            pd.step_update(outputs["losses"], optimizer=optimizer, output_logits=gos_logits)

            step_losses = {k: v.item() if torch.is_tensor(v) else v
                          for k, v in outputs["losses"].items()}
            epoch_losses.append(step_losses)
            global_step += 1

        # ─── Epoch Summary ───
        avg_losses = {}
        for key in epoch_losses[0]:
            vals = [d[key] for d in epoch_losses if key in d]
            avg_losses[key] = sum(vals) / len(vals) if vals else 0.0

        epoch_time = time.time() - epoch_start
        lam = pd.get_lambdas()

        print(f"[Epoch {epoch+1:2d}/{NUM_EPOCHS}] "
              f"total={avg_losses['total']:.4f} "
              f"root={avg_losses.get('l_root', 0):.4f} "
              f"morph={avg_losses.get('l_morph', 0):.4f} "
              f"struct={avg_losses.get('l_struct', 0):.4f} "
              f"sem={avg_losses.get('l_sem', 0):.4f} "
              f"ctx={avg_losses.get('l_ctx', 0):.4f} "
              f"fluency={avg_losses.get('l_fluency', 0):.4f} "
              f"lr={lr:.2e} "
              f"| {epoch_time:.1f}s")
        print(f"  PD lambda: morph={lam['morph']:.3f} struct={lam['struct']:.3f} "
              f"sem={lam['sem']:.3f} ctx={lam['ctx']:.3f}")

        epoch_history.append(avg_losses)

        # ─── Eval every 5 epochs ───
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            model.eval()
            eval_losses = []
            with torch.no_grad():
                for batch in eval_loader:
                    batch = batch.to(torch.device(DEVICE))
                    B = batch.morpheme_ids.size(0)
                    L = batch.morpheme_ids.size(1)
                    dep_masks = torch.zeros(B, L, L, dtype=torch.bool, device=DEVICE)
                    for i in range(B):
                        actual_len = batch.lengths[i].item()
                        mask_i = model.lps.build_dep_mask(["_"] * actual_len, L)
                        mask_i[actual_len:, :] = False
                        mask_i[:, actual_len:] = False
                        dep_masks[i] = mask_i.to(DEVICE)

                    lps_output = {
                        "morpheme_ids": batch.morpheme_ids,
                        "affix_ids": batch.affix_ids,
                        "dep_masks": dep_masks,
                        "lengths": batch.lengths,
                        "max_len": L,
                    }
                    targets = batch.as_targets()
                    outputs = model(lps_output, targets=targets)
                    eval_losses.append({
                        k: v.item() if torch.is_tensor(v) else v
                        for k, v in outputs["losses"].items()
                    })

            avg_eval = {}
            for key in eval_losses[0]:
                vals = [d[key] for d in eval_losses if key in d]
                avg_eval[key] = sum(vals) / len(vals) if vals else 0.0

            print(f"  >>> EVAL: total={avg_eval['total']:.4f} "
                  f"root={avg_eval.get('l_root', 0):.4f} "
                  f"fluency={avg_eval.get('l_fluency', 0):.4f}")

            if avg_eval["total"] < best_eval_loss:
                best_eval_loss = avg_eval["total"]
                model.save(os.path.join(OUTPUT_DIR, "checkpoint_best"))
                print(f"  >>> NEW BEST! Saved checkpoint.")

            # ─── Output Inspection ───
            print(f"\n  --- Output Inspection (Epoch {epoch+1}) ---")
            inspect_model(model, train_texts[:5], root_vocab, DEVICE)
            print()

    # ─── Final Save ───
    model.save(os.path.join(OUTPUT_DIR, "checkpoint_final"))
    total_time = time.time() - training_start

    print("=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Loss reduction: {epoch_history[0]['total']:.4f} -> {epoch_history[-1]['total']:.4f}")
    if epoch_history[0]['total'] > 0:
        reduction_pct = (1 - epoch_history[-1]['total'] / epoch_history[0]['total']) * 100
        print(f"  Reduction: {reduction_pct:.1f}%")
    print("=" * 70)

    # ─── Step 6: Final Generation Test ───
    print("\n[6/6] Autoregressive Generation Test...")
    test_generation(model, train_texts[:10], root_vocab, DEVICE)

    # Save results
    results = {
        "epochs": NUM_EPOCHS,
        "data_size": len(corpus),
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "vocab_size": vocab_size,
        "parameters": params,
        "best_eval_loss": best_eval_loss,
        "final_train_loss": epoch_history[-1]["total"],
        "initial_train_loss": epoch_history[0]["total"],
        "total_time_seconds": total_time,
        "epoch_history": epoch_history,
    }
    with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/training_results.json")


def inspect_model(model, texts, root_vocab, device):
    """Inspect model predictions on sample texts."""
    model.eval()
    root_vocab_inv = {v: k for k, v in root_vocab.items()}

    with torch.no_grad():
        for i, text in enumerate(texts[:3]):
            words = text.split()[:10]
            short_text = " ".join(words)

            # Encode
            lps_out = model.lps([short_text], device=torch.device(device))
            morpheme_ids = lps_out["morpheme_ids"]
            affix_ids = lps_out["affix_ids"]

            kbbi_vectors = model.lsk(morpheme_ids, return_raw=True)
            bsu_states, _ = model.bsu(
                morpheme_ids=morpheme_ids,
                affix_ids=affix_ids,
                kbbi_vectors=kbbi_vectors,
            )
            bsu_original = bsu_states.clone()

            kbbi_anchors = model.lsk.get_anchors(morpheme_ids)
            dep_masks = lps_out.get("dep_masks")
            meb_out, _ = model.meb(
                bsu_states=bsu_states,
                affix_ids=affix_ids,
                kbbi_anchors=kbbi_anchors,
                dep_mask=dep_masks,
            )

            gos_out = model.gos(h_evolved=meb_out, h_original=bsu_original)

            # Root accuracy
            pred_root_ids = gos_out["root_logits"].argmax(dim=-1)  # (1, L)
            input_ids = morpheme_ids[0]
            pred_ids = pred_root_ids[0]
            L = min(len(input_ids), len(pred_ids))

            correct = 0
            total = 0
            for t in range(L):
                inp_id = input_ids[t].item()
                if inp_id == 0:  # padding
                    break
                pred_id = pred_ids[t].item()
                total += 1
                if inp_id == pred_id:
                    correct += 1

            acc = correct / total if total > 0 else 0
            input_words = [root_vocab_inv.get(input_ids[t].item(), "?") for t in range(min(8, L))]
            pred_words = [root_vocab_inv.get(pred_ids[t].item(), "?") for t in range(min(8, L))]

            print(f"  [{i+1}] Input:  {' '.join(input_words)}")
            print(f"       Pred:   {' '.join(pred_words)}")
            print(f"       Root acc: {acc:.1%} ({correct}/{total})")


def test_generation(model, texts, root_vocab, device):
    """Test autoregressive generation."""
    model.eval()
    print("-" * 50)

    prompts = texts[:5]
    for i, text in enumerate(prompts):
        words = text.split()[:6]
        prompt = " ".join(words)

        try:
            gen_out = model.generate(
                texts=[prompt],
                max_length=15,
                temperature=0.8,
            )
            generated = gen_out["generated_texts"][0]
            gen_len = gen_out["generated_root_ids"].size(1)

            print(f"  [{i+1}] Prompt:    {prompt}")
            print(f"       Generated: {generated}")
            print(f"       Length:    {gen_len} BSUs")

            # Fluency check: are there repeated words?
            gen_words = generated.split()
            if gen_words:
                unique_ratio = len(set(gen_words)) / len(gen_words)
                print(f"       Unique ratio: {unique_ratio:.1%}")
            print()

        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            print()

    print("-" * 50)
    print("  Generation test complete.")


if __name__ == "__main__":
    main()
