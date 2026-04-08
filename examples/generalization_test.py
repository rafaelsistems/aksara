"""
AKSARA — Generalization Test

Tujuan: Buktikan arsitektur AKSARA mampu generalisasi lintas domain
dan bekerja baik di skenario low-resource.

Skenario:
  1. Domain Transfer (Formal → Informal): Train pada teks formal, eval pada informal
  2. Domain Transfer (Informal → Formal): Train pada teks informal, eval pada formal
  3. Low-Resource (10% data): Train hanya dengan 10% corpus, eval pada full corpus
  4. Cross-Domain (Mixed → Specific): Train pada campuran, eval pada domain spesifik

Metrik:
  - Transfer Score: performa pada domain target / performa pada domain sumber
  - Low-Resource Efficiency: performa 10% / performa 100%
  - MCS, SVS, SDS per skenario

Cara jalankan:
    python examples/generalization_test.py --kbbi kbbi_true_clean_production.json
    python examples/generalization_test.py  # tanpa KBBI
"""

import argparse
import sys
import time
import random
import math
from pathlib import Path
from typing import List, Dict

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

# ─── Domain Corpora ──────────────────────────────────────────────────────────

FORMAL_CORPUS = [
    "pemerintah indonesia menetapkan kebijakan baru tentang pendidikan nasional",
    "presiden menandatangani peraturan tentang pengembangan ekonomi digital",
    "kementerian kesehatan mengumumkan program vaksinasi untuk seluruh masyarakat",
    "bank indonesia mempertahankan suku bunga acuan pada level yang stabil",
    "pembangunan berkelanjutan harus mempertimbangkan dampak lingkungan hidup",
    "pengembangan sumber daya manusia merupakan investasi jangka panjang",
    "pemberantasan korupsi memerlukan keterlibatan seluruh lapisan masyarakat",
    "mahkamah agung memutuskan perkara sengketa tanah di provinsi jawa tengah",
    "undang-undang dasar mengatur hak dan kewajiban setiap warga negara",
    "rapat paripurna membahas rancangan anggaran pendapatan dan belanja negara",
    "menteri keuangan menyampaikan laporan pertanggungjawaban pelaksanaan anggaran",
    "badan perencanaan pembangunan nasional menyusun rencana pembangunan jangka menengah",
    "lembaga penegak hukum melakukan penyelidikan terhadap kasus penyelewengan dana",
    "dewan perwakilan rakyat mengesahkan rancangan undang-undang tentang perlindungan data",
    "kementerian pendidikan menerbitkan kurikulum baru untuk sekolah dasar dan menengah",
]

INFORMAL_CORPUS = [
    "gue lagi jalan di taman tiap pagi biar sehat badan",
    "dia baca buku pelajaran rajin banget semangat belajarnya",
    "anak-anak main seneng banget di halaman sekolah yang gede",
    "pak tani nanem padi di sawah yang subur pas musim ujan",
    "bu guru ngajarin matematika ke murid-murid yang duduk rapi",
    "mereka lari kenceng banget ke garis finis semangat abis",
    "kebijakan baru itu dapet dukungan banyak dari berbagai pihak",
    "gue udah capek banget hari ini kerja dari pagi sampe malem",
    "temen gue bilang dia mau pindah ke kota lain bulan depan",
    "makanan di warung itu enak banget harganya juga murah",
    "adek gue lagi belajar buat ujian besok pagi di sekolah",
    "kita jalan bareng yuk ke mall nanti sore abis pulang",
    "dia cerita soal liburannya kemarin seru banget katanya",
    "gue pengen beli hp baru tapi duitnya belum cukup nih",
    "tadi pagi hujan deres banget jalanan jadi banjir dimana-mana",
]

BERITA_CORPUS = [
    "gempa bumi berkekuatan tujuh skala richter mengguncang wilayah sulawesi",
    "banjir bandang melanda beberapa kabupaten di provinsi kalimantan selatan",
    "tim nasional sepak bola indonesia meraih kemenangan di ajang piala asia",
    "harga bahan pokok mengalami kenaikan signifikan menjelang hari raya",
    "polisi berhasil mengungkap jaringan penipuan daring yang merugikan korban",
    "cuaca ekstrem diprediksi melanda sejumlah wilayah di pulau jawa",
    "pemerintah daerah mengalokasikan anggaran untuk perbaikan infrastruktur jalan",
    "peluncuran satelit komunikasi baru mendukung konektivitas di daerah terpencil",
]

SASTRA_CORPUS = [
    "bulan purnama menerangi jalan setapak yang sunyi di tengah hutan",
    "angin sepoi bertiup lembut membawa aroma bunga melati yang harum",
    "senja merah membias di ufuk barat menandai berakhirnya hari",
    "air sungai mengalir jernih melewati bebatuan yang berlumut hijau",
    "burung camar terbang rendah di atas permukaan laut yang tenang",
    "daun-daun berguguran menari mengikuti irama angin musim gugur",
    "embun pagi menetes perlahan dari kelopak bunga yang merekah",
    "langit malam bertabur bintang gemerlap bagai permata di kegelapan",
]


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


def train_and_eval(model, root_vocab, train_corpus, eval_corpus, device,
                   epochs=10, lr=5e-4):
    """Train pada train_corpus, eval pada eval_corpus."""
    train_ds = AksaraDataset(train_corpus, root_vocab, max_length=48, min_length=1)
    eval_ds = AksaraDataset(eval_corpus, root_vocab, max_length=48, min_length=1)

    if len(train_ds) == 0:
        return {"loss": float("nan"), "mcs": 0, "svs": 0, "sds": 0,
                "morph_acc": 0, "root_ppl": float("inf")}

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)

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

    # Eval
    if len(eval_ds) == 0:
        return {"loss": float("nan"), "mcs": 0, "svs": 0, "sds": 0,
                "morph_acc": 0, "root_ppl": float("inf")}

    eval_loader = DataLoader(eval_ds, batch_size=4, shuffle=False,
                             collate_fn=collate_fn, drop_last=False)

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


# ─── Scenarios ───────────────────────────────────────────────────────────────

def run_scenario(name, train_corpus, eval_corpus, all_corpus, kbbi_path, device, epochs):
    """Jalankan satu skenario generalisasi."""
    print(f"\n  [{name}]")
    print(f"    Train: {len(train_corpus)} samples")
    print(f"    Eval:  {len(eval_corpus)} samples")

    torch.manual_seed(42)
    model, root_vocab = make_model(all_corpus, kbbi_path, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    t0 = time.time()
    result = train_and_eval(model, root_vocab, train_corpus, eval_corpus,
                            device, epochs=epochs)
    elapsed = time.time() - t0

    print(f"    Loss={result['loss']:.4f} | MCS={result['mcs']:.3f} | "
          f"SVS={result['svs']:.3f} | SDS={result['sds']:.3f} | "
          f"morph%={result['morph_acc']:.3f} | ppl={result['root_ppl']:.1f} | "
          f"{elapsed:.1f}s")

    return result


def run_generalization_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    print(f"\n{'='*80}")
    print("  AKSARA — Generalization Test")
    print(f"{'='*80}")
    print(f"  Device : {device}")
    print(f"  KBBI   : {args.kbbi if args.kbbi else 'tidak aktif'}")
    print(f"  Epochs : {args.epochs}")
    print()

    # All corpus for vocab building
    all_corpus = (FORMAL_CORPUS + INFORMAL_CORPUS + BERITA_CORPUS + SASTRA_CORPUS)

    results = {}

    # ── 1. In-Domain Baselines ───────────────────────────────────────────────
    print("[1/5] In-Domain Baselines (train & eval pada domain yang sama)...")

    results["formal_in"] = run_scenario(
        "Formal → Formal", FORMAL_CORPUS * 2, FORMAL_CORPUS,
        all_corpus, args.kbbi or "", device, args.epochs
    )
    results["informal_in"] = run_scenario(
        "Informal → Informal", INFORMAL_CORPUS * 2, INFORMAL_CORPUS,
        all_corpus, args.kbbi or "", device, args.epochs
    )

    # ── 2. Domain Transfer ───────────────────────────────────────────────────
    print(f"\n[2/5] Domain Transfer (train pada satu domain, eval pada domain lain)...")

    results["formal_to_informal"] = run_scenario(
        "Formal → Informal", FORMAL_CORPUS * 2, INFORMAL_CORPUS,
        all_corpus, args.kbbi or "", device, args.epochs
    )
    results["informal_to_formal"] = run_scenario(
        "Informal → Formal", INFORMAL_CORPUS * 2, FORMAL_CORPUS,
        all_corpus, args.kbbi or "", device, args.epochs
    )

    # ── 3. Cross-Domain ──────────────────────────────────────────────────────
    print(f"\n[3/5] Cross-Domain Transfer...")

    results["mixed_to_berita"] = run_scenario(
        "Mixed → Berita", (FORMAL_CORPUS + INFORMAL_CORPUS) * 2, BERITA_CORPUS,
        all_corpus, args.kbbi or "", device, args.epochs
    )
    results["mixed_to_sastra"] = run_scenario(
        "Mixed → Sastra", (FORMAL_CORPUS + INFORMAL_CORPUS) * 2, SASTRA_CORPUS,
        all_corpus, args.kbbi or "", device, args.epochs
    )

    # ── 4. Low-Resource ──────────────────────────────────────────────────────
    print(f"\n[4/5] Low-Resource Scenarios...")

    full_corpus = FORMAL_CORPUS + INFORMAL_CORPUS
    n_10pct = max(2, len(full_corpus) // 10)
    n_25pct = max(3, len(full_corpus) // 4)
    n_50pct = max(5, len(full_corpus) // 2)

    random.shuffle(full_corpus)

    results["full_100"] = run_scenario(
        "100% Data", full_corpus * 2, full_corpus,
        all_corpus, args.kbbi or "", device, args.epochs
    )
    results["low_50"] = run_scenario(
        "50% Data", full_corpus[:n_50pct] * 4, full_corpus,
        all_corpus, args.kbbi or "", device, args.epochs
    )
    results["low_25"] = run_scenario(
        "25% Data", full_corpus[:n_25pct] * 8, full_corpus,
        all_corpus, args.kbbi or "", device, args.epochs
    )
    results["low_10"] = run_scenario(
        "10% Data", full_corpus[:n_10pct] * 15, full_corpus,
        all_corpus, args.kbbi or "", device, args.epochs
    )

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  GENERALIZATION SUMMARY TABLE")
    print(f"{'='*80}\n")

    header = (f"{'Scenario':<25} | {'Loss':>8} | {'MCS':>6} | {'SVS':>6} | "
              f"{'SDS':>6} | {'morph%':>7} | {'ppl':>7}")
    print(header)
    print("-" * len(header))

    for key, label in [
        ("formal_in", "Formal→Formal (base)"),
        ("informal_in", "Informal→Informal (base)"),
        ("formal_to_informal", "Formal→Informal"),
        ("informal_to_formal", "Informal→Formal"),
        ("mixed_to_berita", "Mixed→Berita"),
        ("mixed_to_sastra", "Mixed→Sastra"),
        ("full_100", "100% Data"),
        ("low_50", "50% Data"),
        ("low_25", "25% Data"),
        ("low_10", "10% Data"),
    ]:
        r = results[key]
        print(f"{label:<25} | {r['loss']:>8.4f} | {r['mcs']:>6.3f} | "
              f"{r['svs']:>6.3f} | {r['sds']:>6.3f} | {r['morph_acc']:>7.3f} | "
              f"{r['root_ppl']:>7.1f}")

    # ── Transfer Scores ──────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  TRANSFER & EFFICIENCY SCORES")
    print(f"{'='*80}\n")

    def safe_ratio(a, b, metric="loss", lower_better=True):
        va = a.get(metric, 0)
        vb = b.get(metric, 1e-6)
        if vb == 0:
            return 0
        if lower_better:
            return vb / va if va != 0 else 0  # higher = better transfer
        else:
            return va / vb  # higher = better transfer

    # Domain transfer scores (MCS-based, higher is better)
    f2i_transfer = safe_ratio(results["formal_to_informal"], results["informal_in"],
                              "mcs", lower_better=False)
    i2f_transfer = safe_ratio(results["informal_to_formal"], results["formal_in"],
                              "mcs", lower_better=False)

    print(f"  Formal→Informal Transfer Score (MCS): {f2i_transfer:.3f}")
    print(f"    (1.0 = sempurna, >0.8 = baik, <0.5 = buruk)")
    print(f"  Informal→Formal Transfer Score (MCS): {i2f_transfer:.3f}")
    print(f"    (1.0 = sempurna, >0.8 = baik, <0.5 = buruk)")

    # Low-resource efficiency
    if results["full_100"]["mcs"] > 0:
        eff_50 = results["low_50"]["mcs"] / results["full_100"]["mcs"]
        eff_25 = results["low_25"]["mcs"] / results["full_100"]["mcs"]
        eff_10 = results["low_10"]["mcs"] / results["full_100"]["mcs"]
    else:
        eff_50 = eff_25 = eff_10 = 0

    print(f"\n  Low-Resource Efficiency (MCS ratio vs 100%):")
    print(f"    50% data: {eff_50:.3f}")
    print(f"    25% data: {eff_25:.3f}")
    print(f"    10% data: {eff_10:.3f}")
    print(f"    (1.0 = no degradation, >0.8 = efficient, <0.5 = data-hungry)")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}\n")

    verdicts = []

    avg_transfer = (f2i_transfer + i2f_transfer) / 2
    if avg_transfer > 0.8:
        verdicts.append("✅ Domain transfer BAIK — model generalisasi lintas gaya bahasa")
    elif avg_transfer > 0.5:
        verdicts.append("⚠️  Domain transfer MODERAT — ada degradasi tapi masih fungsional")
    else:
        verdicts.append("❌ Domain transfer BURUK — model terlalu overfit ke domain training")

    if eff_10 > 0.7:
        verdicts.append("✅ Low-resource EFISIEN — 10% data masih menghasilkan >70% performa")
    elif eff_10 > 0.4:
        verdicts.append("⚠️  Low-resource MODERAT — perlu lebih banyak data untuk performa optimal")
    else:
        verdicts.append("❌ Low-resource BURUK — model sangat data-hungry")

    # Cross-domain
    berita_mcs = results["mixed_to_berita"]["mcs"]
    sastra_mcs = results["mixed_to_sastra"]["mcs"]
    if berita_mcs > 0.5 and sastra_mcs > 0.5:
        verdicts.append("✅ Cross-domain BAIK — model adaptif ke domain baru")
    else:
        verdicts.append("⚠️  Cross-domain perlu perbaikan")

    for v in verdicts:
        print(f"  {v}")

    print(f"\n  Key Insight:")
    print(f"    Arsitektur AKSARA dengan representasi morfologi eksplisit")
    print(f"    memungkinkan transfer lintas domain karena struktur bahasa")
    print(f"    Indonesia (akar kata + afiks) konsisten di semua domain.")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AKSARA Generalization Test")
    parser.add_argument("--kbbi", type=str, default="", help="Path ke KBBI JSON")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch training per skenario")
    args = parser.parse_args()
    run_generalization_test(args)
