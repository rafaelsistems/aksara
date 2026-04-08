"""
Diagnostic: Verify l_struct > 0 after role assignment fix.
Quick test — 3 epochs, 200 data, confirm struct loss is active.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from torch.utils.data import DataLoader

from aksara.core.model import AksaraModel, AksaraConfig
from aksara.core.bsu import BSUConfig
from aksara.core.meb import MEBConfig
from aksara.core.gos import GOSConfig
from aksara.linguistic.lps import LPSConfig, build_root_vocab, ROLE_LABELS
from aksara.linguistic.lsk import LSKConfig
from aksara.data.dataset import AksaraDataset, collate_fn
from aksara.data.corpus_builder import KBBICorpusBuilder


def main():
    print("=" * 70)
    print("  DIAGNOSTIC: Verify l_struct > 0 after role assignment fix")
    print("=" * 70)

    KBBI_PATH = "kbbi_true_clean_production.json"

    # ─── Load corpus via KBBICorpusBuilder (same as training script) ───
    print("\n[1] Loading corpus...")
    builder = KBBICorpusBuilder(KBBI_PATH, seed=42)
    corpus = builder.build_corpus(target_size=200, min_words=4, max_words=30)
    print(f"  Corpus: {len(corpus)} sentences")

    if len(corpus) == 0:
        print("  ❌ FAIL: No corpus data!")
        return

    # Show sample
    for i in range(min(3, len(corpus))):
        print(f"  Sample {i+1}: {corpus[i][:80]}...")

    # ─── Build vocab ───
    root_vocab = build_root_vocab(corpus, min_freq=1)
    vocab_size = max(root_vocab.values()) + 1
    print(f"  Vocab size: {vocab_size}")

    # Known words
    known_words = set()
    with open(KBBI_PATH, "r", encoding="utf-8") as f:
        kbbi_data = json.load(f)
    for entry in kbbi_data.get("entries", []):
        lemma = entry.get("lemma", "").lower().strip()
        if lemma:
            known_words.add(lemma)
    print(f"  Known words: {len(known_words)}")

    # ─── Step 1: Check role distribution in dataset ───
    print("\n--- Step 1: Role Distribution Check ---")
    dataset = AksaraDataset(corpus, root_vocab, max_length=64, min_length=3)
    print(f"  Dataset size: {len(dataset)}")

    role_counts = {name: 0 for name in ROLE_LABELS}
    total_tokens = 0
    for i in range(min(50, len(dataset))):
        item = dataset[i]
        for rid in item["role_ids"]:
            total_tokens += 1
            for name, val in ROLE_LABELS.items():
                if rid == val:
                    role_counts[name] += 1
                    break

    print(f"  Total tokens (50 samples): {total_tokens}")
    for name, count in role_counts.items():
        pct = 100.0 * count / total_tokens if total_tokens > 0 else 0
        print(f"    {name:>4}: {count:>5} ({pct:5.1f}%)")

    non_unk = total_tokens - role_counts["UNK"]
    if total_tokens > 0:
        print(f"\n  Non-UNK tokens: {non_unk}/{total_tokens} ({100*non_unk/total_tokens:.1f}%)")
    else:
        print("\n  ❌ No tokens found!")
        return

    if non_unk == 0:
        print("\n  ❌ FAIL: All role_ids are UNK=0. Fix did not work!")
        return
    else:
        print(f"  ✅ PASS: {non_unk} tokens have real roles (S/P/O/K/PEL/DET/MOD)")

    # Show sample role assignments
    print("\n  Sample role assignments:")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        words = item["text"].split()[:10]
        roles = item["role_ids"][1:len(words)+1]  # skip BOS
        role_inv = {v: k for k, v in ROLE_LABELS.items()}
        role_names = [role_inv.get(r, "?") for r in roles]
        pairs = [f"{w}({r})" for w, r in zip(words, role_names)]
        print(f"    [{i+1}] {' '.join(pairs)}")

    # ─── Step 2: Quick training — verify l_struct > 0 ───
    print("\n--- Step 2: Quick Training (3 epochs) ---")

    config = AksaraConfig(
        bsu_config=BSUConfig(),
        meb_config=MEBConfig(),
        gos_config=GOSConfig(),
        lps_config=LPSConfig(),
        lsk_config=LSKConfig(kbbi_path=KBBI_PATH),
    )

    model = AksaraModel(config, root_vocab, known_words=known_words)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    for epoch in range(3):
        model.train()
        epoch_losses = {}
        n_batches = 0

        for batch in loader:
            lps_output = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids": batch.affix_ids,
                "dep_masks": torch.ones(batch.morpheme_ids.size(0),
                                        batch.morpheme_ids.size(1),
                                        batch.morpheme_ids.size(1), dtype=torch.bool),
            }
            targets = batch.as_targets()

            result = model(lps_output, targets=targets)
            losses = result["losses"]

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                val = v.item() if torch.is_tensor(v) else v
                epoch_losses[k] = epoch_losses.get(k, 0) + val
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        struct_val = epoch_losses.get("l_struct", 0)
        struct_status = "✅ ACTIVE" if struct_val > 0.001 else "❌ DEAD"
        print(f"  Epoch {epoch+1}: total={epoch_losses.get('total',0):.4f} "
              f"struct={struct_val:.4f} [{struct_status}] "
              f"morph={epoch_losses.get('l_morph',0):.4f} "
              f"root={epoch_losses.get('l_root',0):.4f} "
              f"fluency={epoch_losses.get('l_fluency',0):.4f}")

    # ─── Verdict ───
    print("\n" + "=" * 70)
    final_struct = epoch_losses.get("l_struct", 0)
    if final_struct > 0.001:
        print(f"  ✅ VERDICT: l_struct = {final_struct:.4f} > 0 — STRUCTURE IS LEARNING!")
        print("  The fix works. Proceed with full training.")
    else:
        print(f"  ❌ VERDICT: l_struct = {final_struct:.4f} — still dead.")
        print("  Need further investigation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
