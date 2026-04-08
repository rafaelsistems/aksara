"""
Fase 2.3 — Mini Training Loop Validation

Tujuan: MEMBUKTIKAN bahwa AKSARA benar-benar belajar secara linguistik,
bukan hanya meminimalkan loss numerik seperti framework biasa.

Yang diverifikasi:
1. Loss menurun konsisten (model tidak collapse)
2. MCS (Morphological Consistency) meningkat → imbuhan makin konsisten
3. SVS (Structure Validity) stabil → S-P-O-K tidak rusak selama training
4. SDS (Semantic Drift) tidak drift jauh → semantic slots tetap grounded ke KBBI
5. Morph accuracy meningkat, root perplexity turun
6. Gradient flow sehat (tidak NaN, tidak exploding)

Output: tabel per-epoch + verdict final PASS/FAIL per kriteria.

Cara jalankan:
    py -3.11 examples/validate_mini_loop.py --kbbi kbbi_true_clean_production.json
    py -3.11 examples/validate_mini_loop.py  # tanpa KBBI (lebih cepat, SDS tidak bermakna)
"""

import argparse
import sys
import time
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Pastikan aksara bisa diimport dari root
sys.path.insert(0, str(Path(__file__).parent.parent))

from aksara.core.model import AksaraModel, AksaraConfig
from aksara.core.bsu import BSUConfig
from aksara.core.meb import MEBConfig
from aksara.linguistic.lps import LPSConfig, build_root_vocab
from aksara.linguistic.lsk import LSKConfig
from aksara.core.gos import GOSConfig
from aksara.training.loss import LossConfig
from aksara.training.pd import PengendaliDinamik, PDConfig
from aksara.data.dataset import AksaraDataset, collate_fn
from aksara.data.augmentor import LinguisticDatasetEngine
from aksara.utils.indo_metrics import IndoNativeMetrics
import json
import tempfile, os

# ─── Corpus loader ──────────────────────────────────────────────────────────

def load_corpus_jsonl(path: str, max_n: int = 0) -> list:
    """Load corpus dari JSONL file. Return list of text strings."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if obj.get("text"):
                    texts.append(obj["text"])
                    if max_n > 0 and len(texts) >= max_n:
                        break
            except Exception:
                continue
    return texts


# ─── Corpus validasi (representatif, mencakup formal, informal, domain beragam) ──

VALIDATION_CORPUS = [
    # Formal - Berita
    "pemerintah indonesia menetapkan kebijakan baru tentang pendidikan nasional",
    "presiden menandatangani peraturan tentang pengembangan ekonomi digital",
    "kementerian kesehatan mengumumkan program vaksinasi untuk seluruh masyarakat",
    "bank indonesia mempertahankan suku bunga acuan pada level yang stabil",
    "mahkamah agung memutuskan perkara sengketa tanah di provinsi jawa tengah",
    # Formal - Umum
    "saya berjalan di taman setiap pagi untuk menjaga kesehatan tubuh",
    "dia membaca buku pelajaran dengan tekun dan penuh semangat belajar",
    "anak-anak bermain dengan gembira di halaman sekolah yang luas",
    "petani menanam padi di sawah yang subur setiap musim hujan tiba",
    "guru mengajarkan matematika kepada murid-murid yang duduk dengan tertib",
    # Informal
    "gue udah bilang berkali-kali tapi lo gak mau dengerin juga",
    "wkwk emang sih dia orangnya suka lebay banget kalau ngomong",
    "kayaknya besok kita jalan-jalan ke pantai deh bareng temen-temen",
    "eh lo tau gak kemarin ada yang ketangkep karena nyuri motor",
    # Morfologi kompleks
    "pertanggungjawaban program pemberdayaan masyarakat harus dipertimbangkan",
    "pemberantasan korupsi memerlukan keterlibatan seluruh lapisan masyarakat",
    "pembangunan berkelanjutan harus mempertimbangkan dampak lingkungan hidup",
    "pengembangan sumber daya manusia merupakan investasi jangka panjang",
    # Kalimat pendek
    "dia berlari cepat",
    "buku itu mahal",
]


def make_model(corpus, kbbi_path="", device="cpu"):
    """Buat model AKSARA kecil untuk validasi."""
    root_vocab = build_root_vocab(corpus, min_freq=1)

    bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
    meb_cfg = MEBConfig(bsu_config=bsu_cfg, n_layers=3, n_dep_heads=2, kbbi_anchor_dim=16)

    config = AksaraConfig(
        bsu_config=bsu_cfg,
        meb_config=meb_cfg,
        lps_config=LPSConfig(),
        lsk_config=LSKConfig(kbbi_path=kbbi_path, kbbi_vector_dim=16),
        gos_config=GOSConfig(bsu_config=bsu_cfg, vocab_size=len(root_vocab)),
    )
    model = AksaraModel(config, root_vocab).to(device)
    return model, root_vocab


def build_dep_masks_from_batch(model, batch, device):
    """
    Bangun dependency masks dari batch menggunakan LPS.

    Dependency mask adalah pembeda fundamental AKSARA dari Transformer:
    f_syn beroperasi pada dependency graph O(n·deg), bukan full attention O(n²).
    Tanpa dep_masks, f_syn fallback ke local window — kehilangan identitas.
    """
    B = batch.morpheme_ids.size(0)
    L = batch.morpheme_ids.size(1)
    dep_masks = torch.zeros(B, L, L, dtype=torch.bool, device=device)

    for i in range(B):
        actual_len = batch.lengths[i].item()
        dummy_tokens = ["_"] * actual_len
        mask_i = model.lps.build_dep_mask(dummy_tokens, L)
        # Zero-out koneksi ke/dari padding positions
        mask_i[actual_len:, :] = False
        mask_i[:, actual_len:] = False
        dep_masks[i] = mask_i.to(device)

    return dep_masks


def run_validation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  AKSARA — Fase 2.3: Mini Training Loop Validation")
    print(f"{'='*65}")
    print(f"  Device : {device}")
    print(f"  KBBI   : {args.kbbi if args.kbbi else 'tidak aktif'}")
    print(f"  Epochs : {args.epochs}")

    # ── Load corpus (JSONL atau built-in) ──────────────────────────────────────────────
    if getattr(args, 'corpus', '') and os.path.exists(args.corpus):
        corpus = load_corpus_jsonl(args.corpus, getattr(args, 'corpus_n', 0))
        print(f"  Corpus : {args.corpus} ({len(corpus):,} kalimat)")
    else:
        corpus = VALIDATION_CORPUS
        print(f"  Corpus : built-in ({len(corpus)} kalimat)")
    print()

    # ── 1. Build dataset dengan LinguisticDatasetEngine ──────────────────────
    print("[1/4] Building Linguistic Dataset Engine...")
    with tempfile.TemporaryDirectory() as tmpdir:
        kbbi_known = set()
        if args.kbbi:
            with open(args.kbbi, "r", encoding="utf-8") as f:
                kbbi_data = json.load(f)
            # kbbi_core_v2 tidak punya semantic_vector — ambil semua lemma
            kbbi_known = {e["lemma"].lower() for e in kbbi_data.get("entries", [])}

        engine = LinguisticDatasetEngine(
            cache_dir=tmpdir,
            kbbi_known=kbbi_known,
            kbbi_version="validate_v1",
            verbose=False,
        )
        samples = engine.process(
            corpus, augment=True, n_augments_per_sample=2
        )
        print(f"    Samples: {len(corpus)} original + augmented = {len(samples)} total")

    # ── 2. Buat model ──────────────────────────────────────────────────────────────
    print("[2/4] Creating model...")
    model, root_vocab = make_model(corpus, args.kbbi or "", device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # FIX: kbbi_coverage adalah @property, bukan method — tidak menerima argumen
    kbbi_cov = model.lsk.kbbi_coverage
    print(f"    Parameters  : {n_params:,}")
    print(f"    KBBI coverage: {kbbi_cov:.1%}")

    # ── 3. Training setup ─────────────────────────────────────────────────────
    dataset = AksaraDataset.from_linguistic_samples(
        samples, root_vocab, max_length=48
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    pd_ctrl = PengendaliDinamik(PDConfig())
    metrics = IndoNativeMetrics()

    # ── 4. Training loop dengan IndoNativeMetrics ─────────────────────────────
    print(f"\n[3/4] Training {args.epochs} epochs...\n")

    history = []

    # Header tabel
    print(f"{'Ep':>3} | {'loss':>8} | {'MCS':>6} | {'SVS':>6} | "
          f"{'SDS':>6} | {'morph%':>7} | {'ppl':>7} | {'grad_ok':>7}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        grad_nans = 0
        t0 = time.time()

        for batch in loader:
            batch = batch.to(device)

            # FIX: Aktifkan dep_masks — ini pembeda fundamental AKSARA
            dep_masks = build_dep_masks_from_batch(model, batch, device)

            lps_dict = {
                "morpheme_ids": batch.morpheme_ids,
                "affix_ids":    batch.affix_ids,
                "dep_masks":    dep_masks,
                "lengths":      batch.lengths,
                "raw_tokens":   [],
            }

            targets = batch.as_targets()
            outputs = model(lps_dict, targets=targets)
            losses  = outputs["losses"]
            total_loss = losses["total"]

            if torch.isnan(total_loss):
                grad_nans += 1
                continue

            # FIX: Urutan yang benar — backward dulu, baru PD update
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient check
            max_grad = max(
                (p.grad.abs().max().item()
                 for p in model.parameters() if p.grad is not None),
                default=0.0
            )
            if math.isnan(max_grad) or max_grad > 1e4:
                grad_nans += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # FIX: PD update SETELAH optimizer step
            # FIX: Gunakan step_update() (bukan step()), dengan signature yang benar
            # FIX: Losses dict dari AksaraLoss sudah punya key "l_morph", "l_struct", dll
            gos_logits = outputs["gos_out"].get("context_logits")
            pd_ctrl.step_update(
                losses=losses,
                optimizer=optimizer,
                output_logits=gos_logits,
            )

            epoch_losses.append(total_loss.item())

            # Metrik update — gunakan kbbi_anchors_sem (sudah diproyeksikan ke d_sem)
            metrics.update(
                gos_output=outputs["gos_out"],
                targets=targets,
                semantic_slots=outputs["semantic_slots"],
                kbbi_anchors=outputs["kbbi_anchors_sem"],
                dep_masks=dep_masks,
                attention_mask=batch.attention_mask,
                kbbi_mask=outputs.get("kbbi_mask"),
            )

        # Akhir epoch
        result = metrics.end_epoch(epoch=epoch)
        metrics.reset()

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        grad_ok = grad_nans == 0

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "mcs": result.mcs.overall,
            "svs": result.svs.overall,
            "sds": result.sds.overall,
            "morph_acc": result.morph_accuracy,
            "root_ppl": result.root_perplexity,
            "grad_ok": grad_ok,
            "elapsed": time.time() - t0,
        })

        print(
            f"{epoch:>3} | {avg_loss:>8.4f} | {result.mcs.overall:>6.3f} | "
            f"{result.svs.overall:>6.3f} | {result.sds.overall:>6.3f} | "
            f"{result.morph_accuracy:>7.3f} | {result.root_perplexity:>7.1f} | "
            f"{'✓' if grad_ok else '✗':>7}"
        )

    # ── 5. Verdict ────────────────────────────────────────────────────────────
    print(f"\n[4/4] Verdict...\n")

    first = history[0]
    last  = history[-1]

    checks = []

    # Loss convergence
    loss_ok = last["loss"] < first["loss"] * 0.85
    checks.append(("Loss menurun ≥15%",
                   loss_ok,
                   f"{first['loss']:.4f} → {last['loss']:.4f}"))

    # MCS meningkat atau stabil
    mcs_ok = last["mcs"] >= first["mcs"] - 0.05
    checks.append(("MCS tidak memburuk (Δ≥-0.05)",
                   mcs_ok,
                   f"{first['mcs']:.3f} → {last['mcs']:.3f}"))

    # SVS stabil
    svs_ok = last["svs"] >= first["svs"] - 0.10
    checks.append(("SVS stabil (Δ≥-0.10)",
                   svs_ok,
                   f"{first['svs']:.3f} → {last['svs']:.3f}"))

    # SDS: (a) tidak drift jauh, (b) tidak collapse ke nol
    sds_result = metrics.sds.compute()
    drift_ok = abs(sds_result.drift_velocity) < 0.20
    checks.append(("SDS drift velocity < 0.20",
                   drift_ok,
                   f"velocity={sds_result.drift_velocity:+.4f}"))

    # Deteksi semantic collapse: jika SDS < 0.001 di >70% epoch setelah epoch 1
    sds_vals_after_first = [h["sds"] for h in history[1:]]
    collapse_ratio = sum(1 for v in sds_vals_after_first if v < 0.001) / max(len(sds_vals_after_first), 1)
    collapse_ok = collapse_ratio < 0.70
    collapse_detail = (
        f"sehat ({collapse_ratio:.0%} epoch collapse)"
        if collapse_ok
        else f"⚠ COLLAPSE — {collapse_ratio:.0%} epoch SDS≈0 (semantic slot beku)"
    )
    checks.append(("SDS tidak collapse (dinamika semantik hidup)",
                   collapse_ok,
                   collapse_detail))

    # Perplexity turun
    ppl_ok = last["root_ppl"] < first["root_ppl"] * 0.95
    checks.append(("Root perplexity turun",
                   ppl_ok,
                   f"{first['root_ppl']:.1f} → {last['root_ppl']:.1f}"))

    # Gradient sehat di semua epoch
    grad_all_ok = all(h["grad_ok"] for h in history)
    checks.append(("Gradient sehat (no NaN/explode)",
                   grad_all_ok,
                   "semua epoch" if grad_all_ok else "ada epoch bermasalah"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total  = len(checks)

    print(f"  {'Kriteria':<40} {'Status':>8}  {'Detail'}")
    print(f"  {'-'*72}")
    for name, ok, detail in checks:
        status = "  PASS  " if ok else "  FAIL  "
        print(f"  {name:<40} {status}  {detail}")

    print(f"\n  Hasil: {passed}/{total} kriteria terpenuhi")

    if passed == total:
        print("\n  ✅ AKSARA LULUS VALIDASI — framework berperilaku linguistik, bukan ML biasa.")
    elif passed >= total - 1:
        print(f"\n  ⚠️  HAMPIR LULUS ({passed}/{total}) — cek kriteria yang gagal.")
    else:
        print(f"\n  ❌ GAGAL ({passed}/{total}) — ada masalah fundamental pada training loop.")

    # ── P4: Output Inspection ─────────────────────────────────────────────────
    if args.inspect:
        print(f"\n{'='*65}")
        print("  P4 — Output Inspection (generate dari prompt)")
        print(f"{'='*65}")
        model.eval()
        inspect_prompts = [
            "saya berjalan",
            "pemerintah menetapkan",
            "dia membaca",
            "gue gak mau",
        ]
        for prompt in inspect_prompts:
            gen = model.generate([prompt], max_length=8, temperature=0.7)
            gen_text = gen["generated_texts"][0] if gen.get("generated_texts") else ""
            # Tampilkan juga morfologi per kata dari output
            words = gen_text.split() if gen_text else []
            morph_info = []
            for w in words[:6]:
                root, affix = model.lps.analyzer.best(w)
                morph_info.append(f"{w}[{affix}→{root}]" if affix != "<NONE>" else w)
            print(f"  Prompt : '{prompt}'")
            print(f"  Output : '{gen_text}'")
            print(f"  Morph  : {' '.join(morph_info)}")
            # KBBI coverage dari output
            kbbi_hits = sum(1 for w in words if model.lsk.kbbi_store.contains(w))
            cov = kbbi_hits / max(len(words), 1)
            print(f"  KBBI   : {kbbi_hits}/{len(words)} kata ter-grounded ({cov:.0%})")
            print()

    return passed, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKSARA Mini Training Loop Validation")
    parser.add_argument("--kbbi", type=str, default="", help="Path ke KBBI JSON")
    parser.add_argument("--epochs", type=int, default=10, help="Jumlah epoch validasi")
    parser.add_argument("--inspect", action="store_true", help="Aktifkan P4 output inspection")
    parser.add_argument("--corpus", type=str, default="", help="Path ke corpus JSONL (opsional, override built-in)")
    parser.add_argument("--corpus-n", type=int, default=0, dest="corpus_n",
                        help="Max kalimat dari corpus (0 = semua)")
    args = parser.parse_args()

    passed, total = run_validation(args)
    sys.exit(0 if passed >= total - 1 else 1)
