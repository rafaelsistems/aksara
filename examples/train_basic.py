"""
Contoh penggunaan AKSARA untuk melatih model bahasa Indonesia dari scratch.

Jalankan:
    python examples/train_basic.py

Atau dengan KBBI:
    python examples/train_basic.py --kbbi path/to/kbbi_true_clean_production.json
"""

import argparse
import torch
from torch.utils.data import DataLoader

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aksara.core.model import AksaraModel, AksaraConfig
from aksara.core.bsu import BSUConfig
from aksara.core.meb import MEBConfig
from aksara.linguistic.lps import LPSConfig, build_root_vocab
from aksara.linguistic.lsk import LSKConfig
from aksara.core.gos import GOSConfig
from aksara.training.loss import LossConfig
from aksara.training.pd import PDConfig
from aksara.data.dataset import AksaraDataset, collate_fn
from aksara.data.tokenizer import AksaraTokenizer
from aksara.utils.trainer import AksaraTrainer, TrainerConfig
from aksara.utils.metrics import AksaraMetrics


# ─── Contoh corpus Indonesia kecil ────────────────────────────────────────────
SAMPLE_CORPUS = [
    "saya berjalan di taman setiap pagi hari",
    "dia membaca buku pelajaran dengan tekun dan rajin",
    "rumah besar itu terletak di tepi pantai yang indah",
    "makanan enak selalu membuat kita bahagia dan kenyang",
    "pemerintah menjalankan program pendidikan nasional dengan baik",
    "mereka berlari kencang menuju garis finis dengan semangat",
    "kebijakan baru itu mendapat dukungan luas dari masyarakat",
    "anak-anak bermain dengan gembira di halaman sekolah",
    "pengembangan teknologi informasi sangat penting bagi kemajuan bangsa",
    "keadilan sosial merupakan fondasi negara yang demokratis",
    "pertanian merupakan sektor penting dalam perekonomian Indonesia",
    "kebudayaan Indonesia sangat beragam dan kaya",
    "pendidikan karakter perlu ditanamkan sejak dini",
    "pembangunan infrastruktur terus dilakukan di seluruh wilayah",
    "pelestarian lingkungan hidup menjadi tanggung jawab bersama",
    "inovasi dan kreativitas mendorong kemajuan peradaban manusia",
    "persatuan dan kesatuan bangsa harus selalu dijaga",
    "kesehatan masyarakat perlu mendapat perhatian serius dari pemerintah",
    "penggunaan bahasa Indonesia yang baik dan benar wajib diterapkan",
    "kerjasama antarnegara diperlukan untuk mengatasi masalah global",
]


def main():
    parser = argparse.ArgumentParser(description="AKSARA Training Example")
    parser.add_argument("--kbbi", type=str, default="", help="Path ke KBBI JSON")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="aksara_output")
    parser.add_argument("--corpus", type=str, default="", help="Path ke file corpus (opsional)")
    args = parser.parse_args()

    print("=" * 60)
    print("  AKSARA - Framework Model Bahasa Indonesia Native")
    print("  by Emylton Leunufna")
    print("=" * 60)

    # ── 1. Load/buat corpus ──────────────────────────────────────────
    if args.corpus and os.path.exists(args.corpus):
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
        print(f"[+] Corpus loaded: {len(corpus)} kalimat dari {args.corpus}")
    else:
        corpus = SAMPLE_CORPUS
        print(f"[+] Menggunakan sample corpus: {len(corpus)} kalimat")

    # ── 2. Bangun vocabulary ─────────────────────────────────────────
    print("[+] Membangun root vocabulary...")
    root_vocab = build_root_vocab(corpus, min_freq=1)
    print(f"    Vocab size: {len(root_vocab)} root words")

    # ── 3. Bangun tokenizer ──────────────────────────────────────────
    tokenizer = AksaraTokenizer(root_vocab)
    print(f"    Affix vocab size: {tokenizer.affix_vocab_size}")

    # ── 4. Konfigurasi model ─────────────────────────────────────────
    bsu_cfg = BSUConfig(
        d_morpheme=64,
        d_semantic=64,
        d_role=32,
        d_context=64,
        dropout=0.1,
    )
    meb_cfg = MEBConfig(
        bsu_config=bsu_cfg,
        n_layers=4,
        n_dep_heads=4,
        kbbi_anchor_dim=16,
        ffn_expansion=4,
        dropout=0.1,
    )

    config = AksaraConfig(
        vocab_size=len(root_vocab),
        bsu_config=bsu_cfg,
        meb_config=meb_cfg,
        lps_config=LPSConfig(use_soft_segmentation=True),
        lsk_config=LSKConfig(kbbi_path=args.kbbi, freeze_kbbi=True),
        gos_config=GOSConfig(bsu_config=bsu_cfg, vocab_size=len(root_vocab)),
        loss_config=LossConfig(
            lambda_morph=1.0,
            lambda_struct=0.8,
            lambda_sem=0.6,
            lambda_ctx=1.0,
        ),
        kbbi_path=args.kbbi,
        max_seq_len=128,
    )

    # ── 5. Buat model ────────────────────────────────────────────────
    model = AksaraModel(config, root_vocab)
    params = model.num_parameters
    print(f"\n[+] Model AKSARA berhasil dibuat:")
    print(f"    Total parameter    : {params['total']:,}")
    print(f"    Parameter trainable: {params['trainable']:,}")

    if args.kbbi:
        print(f"    KBBI coverage      : {model.lsk.kbbi_coverage:.1%}")
    else:
        print(f"    KBBI: tidak dimuat (jalankan dengan --kbbi untuk aktifkan)")

    # ── 6. Dataset ───────────────────────────────────────────────────
    split = int(len(corpus) * 0.8)
    train_corpus = corpus[:split]
    eval_corpus = corpus[split:] if len(corpus[split:]) > 0 else corpus[:2]

    train_dataset = AksaraDataset(train_corpus, root_vocab, max_length=64)
    eval_dataset = AksaraDataset(eval_corpus, root_vocab, max_length=64)

    print(f"\n[+] Dataset:")
    print(f"    Train: {len(train_dataset)} sampel")
    print(f"    Eval : {len(eval_dataset)} sampel")

    # ── 7. Training ──────────────────────────────────────────────────
    trainer_cfg = TrainerConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=20,
        log_every_n_steps=10,
        eval_every_n_steps=50,
        save_every_n_steps=200,
        use_pd=True,
        pd_config=PDConfig(),
    )

    trainer = AksaraTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=trainer_cfg,
    )

    print(f"\n[+] Memulai training dengan Pengendali Dinamik (PD)...")
    print(f"    Loss: L_morph + L_struct + L_sem + L_ctx\n")

    trainer.train()

    # ── 8. Demo inference ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Demo: Representasi BSU")
    print("=" * 60)
    demo_texts = [
        "berjalan di taman",
        "membaca buku pelajaran",
    ]
    for text in demo_texts:
        print(f"\nInput  : '{text}'")
        # Gunakan analyzer dari model (sudah di-init dengan KBBI known_words)
        words = text.split()
        for word in words:
            root, affix = model.lps.analyzer.best(word)
            print(f"  '{word}' → root='{root}' affix='{affix}'")
        # BSU representasi
        model_device = next(model.parameters()).device
        h = model.get_bsu_representation([text], device=model_device)
        print(f"  BSU shape: {h.shape} (batch=1, seq={h.shape[1]}, d={h.shape[2]})")

    print("\n[AKSARA] Selesai.")


if __name__ == "__main__":
    main()
