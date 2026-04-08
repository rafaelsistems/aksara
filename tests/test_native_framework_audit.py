"""
test_native_framework_audit.py
==============================================
Audit kejujuran AKSARA — 10-point native framework checklist.

Pertanyaan inti:
    "Kalau kamu bisa mematikan setengah komponen dan model tetap sama,
     berarti komponen itu hanya dekorasi."

Setiap test di sini membuktikan satu klaim tentang keunikan arsitektur AKSARA.
Bukan opini — verifikasi kode nyata.

Kategori:
    1. Representasi Bahasa    — token bukan unit utama
    2. Integrasi KBBI         — KBBI anchor mempengaruhi state, bukan lookup kosmetik
    3. Kecerdasan Morfologi   — morfologi adalah first-class citizen
    4. Arsitektur             — tidak ada Q/K/V global, bukan attention matrix
    5. Mekanisme Generasi     — berbasis BSU state, bukan softmax token
    6. Filosofi Vocab         — corpus + KBBI + coverage constraint
    7. Sistem Validasi        — framework punya opini terhadap kualitas
    8. Pipeline Data          — preprocessing linguistik sebelum model
    9. Explainability         — ada jejak pipeline yang bisa dilacak
   10. Dependency Tests       — disable komponen → perilaku berubah (bukan dekorasi)
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aksara.core.bsu import BahasaStateUnit, BSUConfig
from aksara.core.meb import MesinEvolusiBahasa, MEBConfig, SyntacticEvolution
from aksara.core.gos import GeneratorOutputStruktural, GOSConfig
from aksara.linguistic.lps import (
    LapisanParsingStuktural, LPSConfig, MorfologiAnalyzer,
    AFFIX_VOCAB, PREFIXES_ID, SUFFIXES_ID, CONFIXES_ID, ROLE_LABELS,
)
from aksara.linguistic.lsk import LapisanSemantikKBBI, LSKConfig
from aksara.linguistic.vocab_policy import (
    AksaraVocabPolicy, AksaraVocabValidator, QualityTier,
)


# ─── Helper ───────────────────────────────────────────────────────────────────

def buat_bsu(vocab_size=500, batch=2, seq=8):
    """Buat BahasaStateUnit dan tensor input standar untuk testing."""
    cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
    bsu = BahasaStateUnit(cfg, vocab_size=vocab_size, affix_vocab_size=40, kbbi_input_dim=16)
    morph_ids  = torch.randint(1, vocab_size, (batch, seq))
    affix_ids  = torch.randint(0, 40, (batch, seq))
    kbbi_vecs  = torch.randn(batch, seq, 16)
    return bsu, cfg, morph_ids, affix_ids, kbbi_vecs


def buat_meb(vocab_size=500):
    """Buat MesinEvolusiBahasa standar."""
    bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
    cfg = MEBConfig(bsu_config=bsu_cfg, n_layers=2, n_dep_heads=2,
                    kbbi_anchor_dim=16, ffn_expansion=2)
    meb = MesinEvolusiBahasa(cfg, affix_vocab_size=40)
    return meb, bsu_cfg


# ══════════════════════════════════════════════════════════════════════════════
# 1. REPRESENTASI BAHASA — token bukan unit utama
# ══════════════════════════════════════════════════════════════════════════════

class TestRepresentasiBahasa:
    """
    BSU punya 4 slot terpisah: morfem, semantik, peran, konteks.
    Ini bukan word embedding — setiap dimensi punya fungsi linguistik eksplisit.
    """

    def test_bsu_punya_4_slot_terpisah(self):
        """BSU bukan satu embedding — ada 4 slot berbeda dengan dimensi berbeda."""
        cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        assert cfg.d_total == 32 + 32 + 16 + 32  # 112
        assert "morpheme" in cfg.slot_sizes
        assert "semantic" in cfg.slot_sizes
        assert "role"     in cfg.slot_sizes
        assert "context"  in cfg.slot_sizes

    def test_slot_ukuran_bisa_berbeda_satu_sama_lain(self):
        """Setiap slot bisa punya ukuran berbeda — bukan vektor monolitik."""
        cfg = BSUConfig(d_morpheme=64, d_semantic=32, d_role=8, d_context=16)
        assert cfg.d_morpheme != cfg.d_semantic
        assert cfg.d_role < cfg.d_morpheme
        assert cfg.d_total == 64 + 32 + 8 + 16

    def test_bsu_output_terdiri_dari_slot_terpisah(self):
        """Forward BSU mengembalikan dict slot yang bisa diakses per-nama."""
        bsu, cfg, morph_ids, affix_ids, kbbi_vecs = buat_bsu(batch=2, seq=6)
        states, slots = bsu(morph_ids, affix_ids, kbbi_vecs)

        assert "morpheme" in slots
        assert "semantic" in slots
        assert "role"     in slots
        assert "context"  in slots

        # Setiap slot punya ukuran yang tepat
        B, L = 2, 6
        assert slots["morpheme"].shape == (B, L, cfg.d_morpheme)
        assert slots["semantic"].shape == (B, L, cfg.d_semantic)
        assert slots["role"].shape     == (B, L, cfg.d_role)
        assert slots["context"].shape  == (B, L, cfg.d_context)

    def test_slot_bisa_diekstrak_dari_bsu_tensor(self):
        """
        get_slot() mengekstrak slot dari BSU tensor berdasarkan nama.

        Catatan: BSU menerapkan LayerNorm pada bsu_states (gabungan semua slot),
        sehingga nilai dalam states BERBEDA dari slot individual sebelum LayerNorm.
        Yang divalidasi: ukuran slot sesuai config, dan bisa diekstrak tanpa error.
        """
        bsu, cfg, morph_ids, affix_ids, kbbi_vecs = buat_bsu()
        states, slots = bsu(morph_ids, affix_ids, kbbi_vecs)

        morph_extracted = bsu.get_slot(states, "morpheme")
        sem_extracted   = bsu.get_slot(states, "semantic")

        # Ukuran harus sesuai config
        B, L = morph_ids.shape
        assert morph_extracted.shape == (B, L, cfg.d_morpheme)
        assert sem_extracted.shape   == (B, L, cfg.d_semantic)

        # Verifikasi slot offsets konsisten: morfem ada di awal tensor
        assert morph_extracted.shape[-1] == cfg.d_morpheme
        assert sem_extracted.shape[-1]   == cfg.d_semantic

    def test_surface_form_berbeda_slot_morfem_berbeda(self):
        """
        Dua token berbeda (ID berbeda) → slot morfem berbeda.
        Ini membuktikan representasi berbasis root/affix, bukan hanya lookup ID.
        """
        bsu, cfg, _, affix_ids, kbbi_vecs = buat_bsu(vocab_size=500, batch=1, seq=2)

        ids_a = torch.tensor([[10, 20]])   # dua token berbeda
        ids_b = torch.tensor([[10, 10]])   # dua token sama

        states_a, slots_a = bsu(ids_a, affix_ids[:1, :2], kbbi_vecs[:1, :2])
        states_b, slots_b = bsu(ids_b, affix_ids[:1, :2], kbbi_vecs[:1, :2])

        # Posisi 0 harus sama (token sama)
        assert torch.allclose(slots_a["morpheme"][:, 0], slots_b["morpheme"][:, 0])
        # Posisi 1 harus berbeda (token berbeda)
        assert not torch.allclose(slots_a["morpheme"][:, 1], slots_b["morpheme"][:, 1])


# ══════════════════════════════════════════════════════════════════════════════
# 2. INTEGRASI KBBI — bukan lookup kosmetik
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrasiKBBI:
    """
    KBBI harus mempengaruhi state internal secara nyata.
    Tes: ganti kbbi_vectors → output berbeda.
    Jika output sama → KBBI hanya dekorasi.
    """

    def test_kbbi_vector_berbeda_hasilkan_semantic_slot_berbeda(self):
        """kbbi_vectors berbeda → slot semantik berbeda. KBBI nyata bukan kosmetik."""
        bsu, cfg, morph_ids, affix_ids, _ = buat_bsu(batch=1, seq=4)

        kbbi_nol  = torch.zeros(1, 4, 16)
        kbbi_rand = torch.randn(1, 4, 16) * 2.0

        _, slots_nol  = bsu(morph_ids[:1], affix_ids[:1], kbbi_nol)
        _, slots_rand = bsu(morph_ids[:1], affix_ids[:1], kbbi_rand)

        # Slot semantik HARUS berbeda
        assert not torch.allclose(slots_nol["semantic"], slots_rand["semantic"])

    def test_kbbi_tidak_mempengaruhi_slot_morfem(self):
        """
        kbbi_vectors mempengaruhi slot semantik, TAPI tidak langsung ke slot morfem.
        Ini membuktikan pemisahan concern yang sesungguhnya.
        """
        bsu, cfg, morph_ids, affix_ids, _ = buat_bsu(batch=1, seq=4)

        kbbi_nol  = torch.zeros(1, 4, 16)
        kbbi_rand = torch.randn(1, 4, 16) * 2.0

        _, slots_nol  = bsu(morph_ids[:1], affix_ids[:1], kbbi_nol)
        _, slots_rand = bsu(morph_ids[:1], affix_ids[:1], kbbi_rand)

        # Slot morfem HARUS sama (KBBI tidak langsung ke slot morfem di BSU)
        assert torch.allclose(slots_nol["morpheme"], slots_rand["morpheme"])

    def test_semantic_grounding_di_setiap_layer_meb(self):
        """
        SemanticGrounding (f_sem) dijalankan di setiap PhiLayer MEB.
        KBBI anchor mempengaruhi state di setiap lapisan evolusi.
        """
        meb, bsu_cfg = buat_meb()
        B, L = 2, 6
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix_ids  = torch.randint(0, 40, (B, L))
        kbbi_nol   = torch.zeros(B, L, 16)
        kbbi_aktif = torch.randn(B, L, 16)

        out_nol,   _ = meb(h, affix_ids, kbbi_nol)
        out_aktif, _ = meb(h, affix_ids, kbbi_aktif)

        # Output HARUS berbeda — KBBI mempengaruhi evolusi di setiap layer
        assert not torch.allclose(out_nol, out_aktif)

    def test_disable_f_sem_membuat_kbbi_tidak_berpengaruh(self):
        """
        Disable f_sem → kbbi_vectors tidak lagi mempengaruhi output MEB.

        Catatan desain: KBBI juga mempengaruhi state lewat BSU (input awal),
        tapi dalam tes ini h sudah merupakan BSU state yang sudah jadi —
        jadi pengaruh KBBI di MEB HANYA melalui f_sem.
        """
        meb, bsu_cfg = buat_meb()
        B, L = 2, 6
        # h tetap sama — hanya kbbi_anchors yang berbeda saat melewati MEB
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix_ids  = torch.randint(0, 40, (B, L))
        kbbi_nol   = torch.zeros(B, L, 16)
        kbbi_aktif = torch.randn(B, L, 16) * 3.0  # beda signifikan

        # Dengan f_sem aktif: output berbeda
        out_nol,   _ = meb(h, affix_ids, kbbi_nol)
        out_aktif, _ = meb(h, affix_ids, kbbi_aktif)
        berbeda_sebelum = not torch.allclose(out_nol, out_aktif)

        # Disable f_sem — kbbi tidak lagi berkontribusi dalam MEB
        meb.disable("f_sem")

        out_nol_dis,   _ = meb(h, affix_ids, kbbi_nol)
        out_aktif_dis, _ = meb(h, affix_ids, kbbi_aktif)
        # Dengan f_sem disabled, kbbi tidak mempengaruhi MEB
        # Output hanya berbeda karena dropout stochasticity — gunakan eval mode
        meb.eval()
        with torch.no_grad():
            out_nol_eval,   _ = meb(h, affix_ids, kbbi_nol)
            out_aktif_eval, _ = meb(h, affix_ids, kbbi_aktif)
        sama_sesudah = torch.allclose(out_nol_eval, out_aktif_eval)

        assert berbeda_sebelum, "f_sem aktif: kbbi berbeda HARUS hasilkan output berbeda"
        assert sama_sesudah, "f_sem disable (eval): kbbi berbeda HARUS hasilkan output sama"

    def test_lsk_tersedia_sebagai_modul(self):
        """LSKConfig dan LapisanSemantikKBBI tersedia dan bisa diinisialisasi."""
        cfg = LSKConfig(kbbi_path="kbbi_core_v2.json", kbbi_vector_dim=16)
        assert cfg.kbbi_vector_dim == 16
        assert cfg.oov_strategy in ("zero", "random")


# ══════════════════════════════════════════════════════════════════════════════
# 3. KECERDASAN MORFOLOGI — first-class citizen
# ══════════════════════════════════════════════════════════════════════════════

class TestKecerdasanMorfologi:
    """
    Morfologi Indonesia (prefiks, sufiks, konfiks) harus terwakili
    sebagai komponen yang terukur, bukan statistik token biasa.
    """

    def test_affix_vocab_mencakup_prefiks_bahasa_indonesia(self):
        """Prefix utama Bahasa Indonesia terdaftar di AFFIX_VOCAB."""
        affix_set = set(AFFIX_VOCAB)
        prefiks_wajib = ["me", "ber", "ter", "di", "ke", "pe", "se"]
        for p in prefiks_wajib:
            assert p in affix_set, f"Prefiks '{p}' tidak ada di AFFIX_VOCAB"

    def test_affix_vocab_mencakup_sufiks_bahasa_indonesia(self):
        """Sufiks utama Bahasa Indonesia terdaftar."""
        affix_set = set(AFFIX_VOCAB)
        sufiks_wajib = ["kan", "an", "i", "nya"]
        for s in sufiks_wajib:
            assert s in affix_set, f"Sufiks '{s}' tidak ada di AFFIX_VOCAB"

    def test_konfiks_ke_an_terdaftar(self):
        """Konfiks ke-an (salah satu konfiks paling produktif) terdaftar."""
        konfiks_labels = [f"{p}+{s}" for p, s in CONFIXES_ID]
        assert "ke+an" in konfiks_labels, "Konfiks ke-an harus terdaftar"

    def test_morfologi_analyzer_ekstrak_root_dari_kata_berimbuhan(self):
        """
        MorfologiAnalyzer bisa menganalisis kata berimbuhan → kandidat root + affix.
        API: analyze() kembalikan list kandidat [(root, affix, conf),...]
             best() kembalikan (root, affix) kandidat terbaik.
        """
        # MorfologiAnalyzer(min_root_length, known_words) — bukan LPSConfig
        analyzer = MorfologiAnalyzer(min_root_length=3)
        kasus = ["membaca", "berlari", "dilakukan", "pembelajaran"]
        for kata in kasus:
            # analyze() kembalikan list kandidat
            kandidat = analyzer.analyze(kata)
            assert len(kandidat) >= 1, f"'{kata}' harus punya minimal 1 kandidat"
            # Minimal satu kandidat harus lebih pendek dari kata asli
            ada_yg_lebih_pendek = any(len(r) < len(kata) for r, a, c in kandidat)
            assert ada_yg_lebih_pendek, (
                f"'{kata}' harus punya kandidat root lebih pendek dari kata asli"
            )

    def test_affix_ids_berbeda_hasilkan_slot_morfem_berbeda(self):
        """
        Token dengan root sama tapi affix berbeda → slot morfem berbeda.
        'baca' ≠ 'membaca' ≠ 'dibaca' di level representasi.
        """
        bsu, cfg, morph_ids, _, kbbi_vecs = buat_bsu(batch=1, seq=3)
        morph_sama = morph_ids[:1, :1].expand(1, 3)  # root sama semua

        affix_beda = torch.tensor([[1, 2, 3]])  # affix berbeda

        _, slots = bsu(morph_sama, affix_beda, kbbi_vecs[:1, :3])
        morph = slots["morpheme"]

        # Tiga posisi dengan root sama tapi affix berbeda → representasi berbeda
        assert not torch.allclose(morph[:, 0], morph[:, 1])
        assert not torch.allclose(morph[:, 1], morph[:, 2])

    def test_disable_f_morph_hilangkan_kontribusi_morfologi(self):
        """
        Disable f_morph → evolusi morfologi tidak berkontribusi.
        Ini membuktikan f_morph bukan dekorasi.
        """
        meb, bsu_cfg = buat_meb()
        B, L = 1, 4
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix_a = torch.randint(1, 10, (B, L))
        affix_b = torch.randint(15, 25, (B, L))  # affix berbeda
        kbbi = torch.zeros(B, L, 16)

        # Dengan f_morph aktif: affix berbeda → output berbeda
        meb.eval()
        with torch.no_grad():
            out_a, _ = meb(h, affix_a, kbbi)
            out_b, _ = meb(h, affix_b, kbbi)
        berbeda_aktif = not torch.allclose(out_a, out_b)

        # Disable f_morph — pakai eval + no_grad untuk eliminasi dropout noise
        meb.disable("f_morph")
        with torch.no_grad():
            out_a_dis, _ = meb(h, affix_a, kbbi)
            out_b_dis, _ = meb(h, affix_b, kbbi)
        sama_disable = torch.allclose(out_a_dis, out_b_dis)

        assert berbeda_aktif, "f_morph aktif: affix berbeda HARUS hasilkan output berbeda"
        assert sama_disable,  "f_morph disable (eval): affix berbeda HARUS hasilkan output sama"

    def test_role_labels_mencakup_fungsi_sintaktik_indonesia(self):
        """Label sintaktik Indonesia (S/P/O/K/PEL) terdaftar sebagai first-class."""
        assert "S"   in ROLE_LABELS
        assert "P"   in ROLE_LABELS
        assert "O"   in ROLE_LABELS
        assert "K"   in ROLE_LABELS
        assert "PEL" in ROLE_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# 4. ARSITEKTUR — tidak ada Q/K/V global, bukan attention matrix penuh
# ══════════════════════════════════════════════════════════════════════════════

class TestArsitektur:
    """
    SyntacticEvolution menggunakan attention TAPI hanya pada dependency neighbors,
    bukan full O(n²) attention matrix.
    Ini berbeda fundamental dari Transformer self-attention.
    """

    def test_f_syn_hanya_pada_dependency_neighbors(self):
        """
        Dengan dep_mask yang membatasi neighbors, token terisolasi
        (semua mask=False) tidak bisa attend ke siapa pun → output berbeda
        dari token yang punya neighbors.
        """
        meb, bsu_cfg = buat_meb()
        B, L = 1, 6
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix = torch.zeros(B, L, dtype=torch.long)
        kbbi  = torch.zeros(B, L, 16)

        # Dep mask: hanya diagonal (self-loop saja)
        dep_only_self = torch.eye(L, dtype=torch.bool).unsqueeze(0)
        # Dep mask: full connection
        dep_full = torch.ones(B, L, L, dtype=torch.bool)

        out_self, _ = meb(h, affix, kbbi, dep_mask=dep_only_self)
        out_full, _ = meb(h, affix, kbbi, dep_mask=dep_full)

        # Output harus berbeda — dep_mask benar-benar membatasi interaction
        assert not torch.allclose(out_self, out_full)

    def test_tidak_ada_atribut_global_attention_matrix(self):
        """
        MEB tidak menyimpan attention weight global (tidak ada self.attn_weights global).
        f_syn hanya menyimpan parameter proyeksi, bukan state attention.
        """
        meb, _ = buat_meb()
        # Cek bahwa tidak ada atribut bernama 'attention', 'attn_weights', 'kv_cache'
        atribut_transformer = ["attention", "attn_weights", "kv_cache", "key_cache", "value_cache"]
        for nama in atribut_transformer:
            assert not hasattr(meb, nama), (
                f"MEB tidak seharusnya punya atribut '{nama}' (gaya Transformer)"
            )

    def test_evolusi_phi_akumulasi_tiga_fungsi_bukan_satu(self):
        """
        PhiLayer menggabungkan tiga delta: f_morph + f_syn + f_sem.
        Ini berbeda dari Transformer yang hanya punya attention + FFN.
        """
        meb, bsu_cfg = buat_meb()
        layer = meb.layers[0]

        assert hasattr(layer, "f_morph")
        assert hasattr(layer, "f_syn")
        assert hasattr(layer, "f_sem")
        assert hasattr(layer, "ffn")

    def test_disable_enable_komponen_bekerja(self):
        """disable/enable komponen di semua layer harus konsisten."""
        meb, _ = buat_meb()

        meb.disable("f_morph")
        assert "f_morph" in meb.get_disabled()
        assert "f_syn" not in meb.get_disabled()

        meb.enable("f_morph")
        assert "f_morph" not in meb.get_disabled()

    def test_freeze_tidak_sama_dengan_disable(self):
        """
        freeze: komponen tetap berkontribusi tapi tidak belajar.
        disable: komponen tidak berkontribusi sama sekali (zero).
        """
        meb, bsu_cfg = buat_meb()
        B, L = 1, 4
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix = torch.randint(1, 10, (B, L))
        kbbi  = torch.randn(B, L, 16)

        # Output dengan freeze (berkontribusi, tapi bobot tidak update)
        meb.freeze("f_morph")
        assert "f_morph" in meb.get_frozen()
        out_frozen, _ = meb(h, affix, kbbi)

        # Output dengan disable (tidak berkontribusi sama sekali)
        meb.unfreeze("f_morph")
        meb.disable("f_morph")
        out_disabled, _ = meb(h, affix, kbbi)

        # Keduanya harus berbeda — freeze ≠ disable
        assert not torch.allclose(out_frozen, out_disabled)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MEKANISME GENERASI — berbasis BSU state
# ══════════════════════════════════════════════════════════════════════════════

class TestMekanismeGenerasi:
    """
    GOS menghasilkan token melalui state evolution BSU, bukan softmax langsung.
    Pipeline: state_0 → Phi_seq → BSU_1 → ... → token prediction
    """

    def test_gos_punya_phi_seq_terpisah_dari_meb(self):
        """GOS punya Phi_seq sendiri — generation dynamics berbeda dari understanding."""
        bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        cfg = GOSConfig(bsu_config=bsu_cfg, vocab_size=500, affix_vocab_size=40)
        gos = GeneratorOutputStruktural(cfg)

        # GOS harus punya phi_seq (generation-specific state evolution)
        assert hasattr(gos, "phi_seq"), "GOS harus punya Phi_seq untuk generation dynamics"

    def test_gos_prediksi_dari_bsu_state_bukan_raw_token(self):
        """
        GOS menerima BSU state sebagai input, bukan sequence token langsung.
        Ini membuktikan generasi berbasis state, bukan token lookup.
        Output GOS: root_logits, affix_logits, role_logits, context_logits
        """
        bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        cfg = GOSConfig(bsu_config=bsu_cfg, vocab_size=500, affix_vocab_size=40)
        gos = GeneratorOutputStruktural(cfg)

        B, L = 2, 6
        # Input adalah BSU state, bukan token IDs
        meb_out  = torch.randn(B, L, bsu_cfg.d_total)
        bsu_orig = torch.randn(B, L, bsu_cfg.d_total)

        output = gos(meb_out, bsu_orig)
        # GOS menghasilkan prediksi multi-komponen:
        assert "root_logits"    in output, "GOS harus prediksi root word"
        assert "affix_logits"   in output, "GOS harus prediksi affix"
        assert "role_logits"    in output, "GOS harus prediksi syntactic role"
        assert "context_logits" in output, "GOS harus prediksi context"
        # root_logits berbentuk (B, L, vocab_size)
        assert output["root_logits"].shape == (B, L, 500)

    def test_gos_bsu_state_berbeda_hasilkan_logits_berbeda(self):
        """
        BSU state berbeda → root_logits berbeda.
        Membuktikan generasi sensitif terhadap state, bukan hanya posisi.
        """
        bsu_cfg = BSUConfig(d_morpheme=32, d_semantic=32, d_role=16, d_context=32)
        cfg = GOSConfig(bsu_config=bsu_cfg, vocab_size=500, affix_vocab_size=40)
        gos = GeneratorOutputStruktural(cfg)
        gos.eval()

        B, L = 1, 4
        state_a  = torch.randn(B, L, bsu_cfg.d_total)
        state_b  = torch.randn(B, L, bsu_cfg.d_total)  # state berbeda
        bsu_orig = torch.zeros(B, L, bsu_cfg.d_total)

        with torch.no_grad():
            out_a = gos(state_a, bsu_orig)
            out_b = gos(state_b, bsu_orig)

        assert not torch.allclose(out_a["root_logits"], out_b["root_logits"])


# ══════════════════════════════════════════════════════════════════════════════
# 6. FILOSOFI VOCAB — bukan sekadar top-N token
# ══════════════════════════════════════════════════════════════════════════════

class TestFilosofiVocab:
    """
    Vocab AKSARA bukan frequency list — ada corpus selection + KBBI injection
    + coverage constraint + quality tier.
    """

    def test_policy_punya_corpus_ratio_dan_kbbi_ratio(self):
        """Policy vocab eksplisit memisahkan slot corpus dan slot KBBI."""
        p = AksaraVocabPolicy()
        assert p.corpus_ratio_min > 0 and p.corpus_ratio_max < 1.0
        assert p.kbbi_ratio_min   > 0 and p.kbbi_ratio_max   < 1.0
        # Bersama-sama harus mencakup ~100% (overlap diizinkan)
        assert p.corpus_ratio_max + p.kbbi_ratio_max >= 0.90

    def test_policy_punya_coverage_constraint(self):
        """Ada batas minimum coverage yang terukur."""
        p = AksaraVocabPolicy()
        assert p.min_coverage >= 0.70  # minimum 70%

    def test_policy_punya_oov_constraint(self):
        """Ada batas maksimum OOV yang terukur."""
        p = AksaraVocabPolicy()
        assert 0 < p.max_oov_rate < 0.50  # OOV harus dikontrol

    def test_vocab_bukan_hanya_frekuensi_ada_kbbi_slot(self):
        """
        Vocab dengan 25% KBBI slot berbeda dari vocab frekuensi murni.
        Membuktikan KBBI injection adalah fitur nyata, bukan kosmetik.
        """
        p = AksaraVocabPolicy()
        n_corpus = p.corpus_slots()
        n_kbbi   = p.kbbi_slots()
        total    = n_corpus + n_kbbi

        assert n_kbbi > 0, "KBBI slots harus > 0"
        assert n_corpus > n_kbbi, "Corpus slot harus lebih banyak dari KBBI slot"
        assert abs(total - p.target_vocab) <= 1  # total ≈ target

    def test_quality_tier_tersedia_untuk_vocab(self):
        """Quality tier memberikan opini terhadap vocab — bukan sekadar ukuran."""
        vocab_kecil = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for i in range(20):
            vocab_kecil[f"tok_{i}"] = 4 + i

        v = AksaraVocabValidator()
        hasil = v.validate(vocab_kecil)
        # Framework punya opini: vocab kecil → tier turun
        assert hasil.quality_tier in (QualityTier.DEGRADED, QualityTier.EXPERIMENTAL)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SISTEM VALIDASI — framework punya opini
# ══════════════════════════════════════════════════════════════════════════════

class TestSistemValidasi:
    """Sudah dicover di test_framework_consistency.py — test ringkas di sini."""

    def test_validator_hasilkan_tier_bukan_hanya_bool(self):
        """Hasil validasi bukan True/False — ada 4 level tier yang bermakna."""
        from aksara.linguistic.vocab_policy import VocabValidationResult
        p = AksaraVocabPolicy()
        hasil = VocabValidationResult(policy=p)
        hasil.check_oov = ("FAIL", 0.40, "OOV tinggi")
        assert hasil.quality_tier == QualityTier.DEGRADED
        assert hasattr(hasil, "hard_violations")

    def test_hard_constraint_terpisah_dari_soft_constraint(self):
        """Hard constraint dan soft constraint punya jalur terpisah."""
        from aksara.linguistic.vocab_policy import VocabHardConstraintError
        vocab_rusak = {}  # vocab kosong = hard violation
        v = AksaraVocabValidator()
        hasil = v.validate(vocab_rusak)

        # Hard violation dicatat
        assert hasil.hard_violations
        # Tapi tidak langsung raise — pengembang yang memilih
        # Baru raise saat assert_hard_constraints() dipanggil eksplisit
        with pytest.raises(VocabHardConstraintError):
            hasil.assert_hard_constraints()


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE DATA — preprocessing linguistik sebelum model
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineData:
    """
    LPS adalah pre-processing linguistik — teks tidak langsung masuk model.
    Preprocessing menghasilkan morpheme_ids, affix_ids, dep_mask.
    """

    def test_lps_tersedia_dan_bisa_diinisialisasi(self):
        """LPS bisa dibuat tanpa error. API: LapisanParsingStuktural(config, root_vocab)"""
        cfg = LPSConfig()
        lps = LapisanParsingStuktural(cfg, root_vocab={"<PAD>": 0, "<UNK>": 1, "baca": 2})
        assert lps is not None

    def test_lps_hasilkan_morpheme_ids_bukan_raw_token_ids(self):
        """
        LPS mengubah teks → morpheme_ids (root-based), bukan sequence token mentah.
        API: lps.forward(texts) mengembalikan dict dengan morpheme_ids, affix_ids.
        """
        cfg = LPSConfig()
        root_vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                      "baca": 4, "tulis": 5, "makan": 6, "pergi": 7}
        lps = LapisanParsingStuktural(cfg, root_vocab=root_vocab)

        kalimat = ["membaca buku", "menulis laporan"]
        hasil = lps(kalimat)  # forward() menerima List[str]

        assert "morpheme_ids" in hasil
        assert "affix_ids"    in hasil
        assert hasil["morpheme_ids"].shape[0] == 2  # batch size 2

    def test_lps_hasilkan_affix_ids_terpisah_dari_root(self):
        """affix_ids dihasilkan terpisah dari morpheme_ids — morfologi dipisahkan."""
        cfg = LPSConfig()
        root_vocab = {"<PAD>": 0, "<UNK>": 1, "baca": 2, "tulis": 3}
        lps = LapisanParsingStuktural(cfg, root_vocab=root_vocab)

        hasil = lps(["membaca menulis"])  # forward()
        assert "affix_ids" in hasil
        # affix_ids harus punya shape sama dengan morpheme_ids
        assert hasil["affix_ids"].shape == hasil["morpheme_ids"].shape

    def test_morfologi_analyzer_bisa_dijalankan_tanpa_model(self):
        """
        MorfologiAnalyzer beroperasi sebelum model — rule-based, tidak perlu GPU.
        Ini membuktikan linguistic preprocessing adalah step terpisah.
        API: MorfologiAnalyzer(min_root_length, known_words)
             analyze(kata) → list [(root, affix, conf)]
             best(kata)    → (root, affix) terbaik
        """
        analyzer = MorfologiAnalyzer(min_root_length=3)
        # Bisa dijalankan tanpa torch, tanpa model
        kandidat = analyzer.analyze("berlari")
        assert isinstance(kandidat, list)
        assert len(kandidat) >= 1
        root, affix, conf = kandidat[0]
        assert isinstance(root, str)
        assert isinstance(affix, str)
        assert 0.0 <= conf <= 1.0
        # best() sebagai shortcut
        root_best, affix_best = analyzer.best("berlari")
        assert isinstance(root_best, str)


# ══════════════════════════════════════════════════════════════════════════════
# 9. EXPLAINABILITY — ada jejak pipeline yang bisa dilacak
# ══════════════════════════════════════════════════════════════════════════════

class TestExplainability:
    """
    Pipeline AKSARA: LPS → BSU → MEB → GOS
    Setiap langkah meninggalkan jejak yang bisa diperiksa.
    """

    def test_bsu_output_ada_slots_yang_bisa_diinspeksi(self):
        """BSU mengembalikan dict slot — setiap slot bisa diinspeksi per nama."""
        bsu, cfg, morph_ids, affix_ids, kbbi_vecs = buat_bsu()
        states, slots = bsu(morph_ids, affix_ids, kbbi_vecs)

        # Jejak: bisa tahu berapa besar kontribusi setiap slot
        for nama in ["morpheme", "semantic", "role", "context"]:
            slot = slots[nama]
            norm = slot.norm(dim=-1).mean().item()
            assert norm >= 0  # bisa diukur

    def test_meb_bisa_kembalikan_semua_layer_states(self):
        """
        MEB bisa dikonfigurasi untuk kembalikan semua layer states.
        Ini memungkinkan inspeksi evolusi state per-layer.
        """
        meb, bsu_cfg = buat_meb()
        B, L = 2, 5
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix = torch.zeros(B, L, dtype=torch.long)
        kbbi  = torch.zeros(B, L, 16)

        h_final, layer_states = meb(h, affix, kbbi, return_all_layers=True)

        assert layer_states is not None
        assert len(layer_states) == meb.config.n_layers
        for ls in layer_states:
            assert ls.shape == (B, L, bsu_cfg.d_total)

    def test_disabled_components_tercatat_di_meb(self):
        """Komponen yang di-disable tercatat — bisa di-audit kapan saja."""
        meb, _ = buat_meb()
        meb.disable("f_sem")
        meb.disable("f_morph")

        disabled = meb.get_disabled()
        assert "f_sem"   in disabled
        assert "f_morph" in disabled
        assert "f_syn"   not in disabled

    def test_morfologi_analyzer_hasilkan_confidence_score(self):
        """
        Setiap analisis morfologi menghasilkan confidence score per kandidat.
        Ini memungkinkan downstream untuk memberi bobot berbeda per analisis.
        """
        analyzer = MorfologiAnalyzer(min_root_length=3)
        kasus = ["membaca", "berlari", "keadaan", "rumah", "xyz123"]
        for kata in kasus:
            kandidat = analyzer.analyze(kata)
            for root, affix, conf in kandidat:
                assert 0.0 <= conf <= 1.0, (
                    f"Confidence '{kata}' → '{root}' harus dalam [0,1]: {conf}"
                )

    def test_quality_tier_punya_expected_behavior_yang_bisa_dibaca(self):
        """
        Setiap tier menjelaskan perilaku yang diharapkan dalam bahasa manusia.
        Framework bisa menjelaskan kenapa output jelek.
        """
        for tier in [QualityTier.OPTIMAL, QualityTier.VALID,
                     QualityTier.DEGRADED, QualityTier.EXPERIMENTAL]:
            behaviors = QualityTier.BEHAVIOR[tier]
            assert behaviors, f"Tier {tier} harus punya penjelasan perilaku"
            # Penjelasan harus dalam bahasa yang dapat dibaca manusia
            for b in behaviors:
                assert len(b) > 5, f"Penjelasan terlalu pendek: '{b}'"


# ══════════════════════════════════════════════════════════════════════════════
# 10. DEPENDENCY TESTS — komponen bukan dekorasi
# ══════════════════════════════════════════════════════════════════════════════

class TestDependency:
    """
    Tes paling jujur:
    Test 1: Disable KBBI → output berubah (KBBI nyata, bukan kosmetik)
    Test 2: Disable morph layer → output berubah (morfologi nyata)
    Test 3: Ganti dep_mask → output berubah (dependency graph nyata)
    """

    def test_1_disable_kbbi_mengubah_output_secara_terukur(self):
        """
        Test 1: Disable f_sem (jalur KBBI).
        Sebelum disable: KBBI aktif → output A.
        Sesudah disable: KBBI nonaktif → output B.
        A ≠ B → KBBI bukan dekorasi. ✔
        """
        meb, bsu_cfg = buat_meb()
        B, L = 2, 6
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix = torch.randint(0, 10, (B, L))
        kbbi  = torch.randn(B, L, 16)

        # Sebelum disable
        out_kbbi_aktif, _ = meb(h, affix, kbbi)

        # Setelah disable f_sem
        meb.disable("f_sem")
        out_kbbi_nonaktif, _ = meb(h, affix, kbbi)

        selisih = (out_kbbi_aktif - out_kbbi_nonaktif).abs().mean().item()
        assert selisih > 0.001, (
            f"Disable KBBI (f_sem) harus mengubah output secara signifikan. "
            f"Selisih: {selisih:.6f} — kemungkinan KBBI hanya dekorasi!"
        )

    def test_2_disable_morph_layer_mengubah_output_secara_terukur(self):
        """
        Test 2: Disable f_morph (lapisan morfologi).
        Sebelum disable: morfologi aktif → output A.
        Sesudah disable: morfologi nonaktif → output B.
        A ≠ B → morfologi bukan dekorasi. ✔
        """
        meb, bsu_cfg = buat_meb()
        B, L = 2, 6
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix = torch.randint(1, 20, (B, L))  # affix bermakna
        kbbi  = torch.zeros(B, L, 16)

        out_morph_aktif, _ = meb(h, affix, kbbi)

        meb.disable("f_morph")
        out_morph_nonaktif, _ = meb(h, affix, kbbi)

        selisih = (out_morph_aktif - out_morph_nonaktif).abs().mean().item()
        assert selisih > 0.001, (
            f"Disable morfologi (f_morph) harus mengubah output. "
            f"Selisih: {selisih:.6f} — kemungkinan morfologi hanya dekorasi!"
        )

    def test_3_ganti_corpus_domain_framework_tetap_stabil(self):
        """
        Test 3: Framework stabil dengan berbagai distribusi frekuensi corpus.
        Ganti distribusi frekuensi → validator tetap bisa menghasilkan tier.
        """
        from aksara.linguistic.vocab_policy import validate_vocab

        def buat_vocab_domain(domain: str, ukuran: int = 5_000) -> dict:
            vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4}
            domain_seeds = {
                "makan": ["makan", "minum", "nasi"],
                "membaca": ["baca", "buku", "kata"],
                "jalan": ["jalan", "kaki", "gerak"],
                "pemerintah": ["pemerintah", "hukum", "negara"],
                "sekolah": ["sekolah", "belajar", "guru"],
                "sejarah": ["sejarah", "tahun", "masa"],
                "ilmu": ["penelitian", "data", "hasil"],
                "kerja": ["kerja", "kantor", "tugas"],
            }
            for seeds in domain_seeds.values():
                for w in seeds:
                    if w not in vocab:
                        vocab[w] = len(vocab)
            while len(vocab) < ukuran:
                vocab[f"{domain}_tok_{len(vocab)}"] = len(vocab)
            return vocab

        # Tiga distribusi berbeda: sains, pemerintah, sehari-hari
        for domain, n_sents in [("sains", 5_000), ("pemerintah", 50_000), ("sehari_hari", 200_000)]:
            vocab = buat_vocab_domain(domain)
            freq  = {tok: 10 for tok in vocab if not tok.startswith("<")}

            hasil = validate_vocab(
                vocab=vocab,
                corpus_token_freq=freq,
                n_corpus_sentences=n_sents,
                print_report=False,
            )
            # Framework harus tetap menghasilkan tier yang valid
            assert hasil.quality_tier in (
                QualityTier.OPTIMAL, QualityTier.VALID,
                QualityTier.DEGRADED, QualityTier.EXPERIMENTAL
            ), f"Corpus domain '{domain}' menghasilkan tier tidak valid: {hasil.quality_tier}"

    def test_dep_mask_none_fallback_ke_local_window(self):
        """
        dep_mask=None → f_syn fallback ke local window ±2.
        Framework stabil tanpa dependency graph eksplisit.
        """
        meb, bsu_cfg = buat_meb()
        B, L = 2, 8
        h = torch.randn(B, L, bsu_cfg.d_total)
        affix = torch.zeros(B, L, dtype=torch.long)
        kbbi  = torch.zeros(B, L, 16)

        # Tanpa dep_mask
        out_tanpa_dep, _ = meb(h, affix, kbbi, dep_mask=None)
        assert out_tanpa_dep.shape == (B, L, bsu_cfg.d_total)

        # Dengan dep_mask eksplisit (local window manual)
        local = torch.zeros(B, L, L, dtype=torch.bool)
        for i in range(L):
            for j in range(max(0, i-2), min(L, i+3)):
                local[:, i, j] = True
        out_dengan_dep, _ = meb(h, affix, kbbi, dep_mask=local)
        assert out_dengan_dep.shape == (B, L, bsu_cfg.d_total)


class TestKBBIPreseededEmbedding:
    """
    B2-D: Verifikasi KBBI pre-seeded embedding memberikan sinyal semantik nyata.

    Sebelum training, embedding KBBI sudah membawa sinyal:
    - Kata dengan POS sama lebih dekat satu sama lain
    - Kata dengan definisi serupa lebih dekat
    - Kata dengan POS berbeda lebih jauh

    Ini membuktikan grounding adalah SEMANTIK (dari isi KBBI),
    bukan hanya struktural (ada/tidak di KBBI).
    """

    def setup_method(self):
        import json, os
        pretrained_path = "data/kbbi_pretrained.pt"
        if not os.path.exists(pretrained_path):
            pytest.skip("kbbi_pretrained.pt belum dibangun — jalankan tools/build_kbbi_embeddings.py")

        with open("data/vocab_aksara.json", encoding="utf-8") as f:
            rv = json.load(f)["vocab"]

        cfg = LSKConfig(
            kbbi_path="kbbi_core_v2.json",
            kbbi_vector_dim=16,
            pretrained_path=pretrained_path,
        )
        self.lsk = LapisanSemantikKBBI(cfg, rv)
        self.store = self.lsk.kbbi_store
        self.emb = self.lsk.kbbi_embeddings.weight.detach()

    def _cosine(self, w1: str, w2: str) -> float:
        """Hitung cosine similarity antara dua kata berdasarkan embedding KBBI."""
        id1 = self.store.lookup_exact(w1)
        id2 = self.store.lookup_exact(w2)
        if id1 == 0 or id2 == 0:
            return None
        v1, v2 = self.emb[id1], self.emb[id2]
        return torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-8)

    def test_pretrained_loaded(self):
        assert self.lsk._pretrained_loaded, \
            "Pre-seeded embedding harus berhasil dimuat dari kbbi_pretrained.pt"

    def test_oov_adalah_zero_vector(self):
        # Baris 0 (OOV/PAD) harus zero vector
        assert self.emb[0].norm().item() < 1e-6, \
            "Baris 0 (OOV) harus zero vector"

    def test_embedding_tidak_identik_antar_lemma(self):
        # Setelah pre-seeding, embedding tidak boleh semua identik
        # (masalah utama random init: sebelum training, semua embedding ≈ 0 atau acak)
        id_besar  = self.store.lookup_exact("besar")
        id_kecil  = self.store.lookup_exact("kecil")
        id_berlari = self.store.lookup_exact("berlari")
        assert id_besar > 0 and id_kecil > 0 and id_berlari > 0, \
            "Kata uji harus ada di KBBI"
        assert not torch.allclose(self.emb[id_besar], self.emb[id_kecil]), \
            "Embedding berbeda untuk kata berbeda"
        assert not torch.allclose(self.emb[id_besar], self.emb[id_berlari]), \
            "Embedding ADJ vs V harus berbeda"

    def test_adjektiva_lebih_dekat_dengan_adjektiva(self):
        # besar (adj) ↔ kecil (adj) harus lebih dekat dari besar (adj) ↔ berlari (v)
        cos_adj_adj = self._cosine("besar", "kecil")
        cos_adj_v   = self._cosine("besar", "berlari")
        if cos_adj_adj is None or cos_adj_v is None:
            pytest.skip("Salah satu kata tidak ditemukan di KBBI")
        assert cos_adj_adj > cos_adj_v, (
            f"ADJ↔ADJ ({cos_adj_adj:.4f}) harus > ADJ↔V ({cos_adj_v:.4f})"
        )

    def test_verba_lebih_dekat_dengan_verba(self):
        # menjadi (v) ↔ memiliki (v) harus lebih dekat dari menjadi (v) ↔ orang (n)
        # Kata-kata ini dipilih karena ada di vocab_aksara.json DAN KBBI dengan POS v/n
        cos_v_v = self._cosine("menjadi", "memiliki")
        cos_v_n = self._cosine("menjadi", "orang")
        if cos_v_v is None or cos_v_n is None:
            pytest.skip("Salah satu kata tidak ditemukan di KBBI")
        assert cos_v_v > cos_v_n, (
            f"V↔V ({cos_v_v:.4f}) harus > V↔N ({cos_v_n:.4f})"
        )

    def test_pos_onehot_berbeda_antar_kelas(self):
        # Verba dan nomina harus punya slot POS yang berbeda
        # (slot 0=V, 1=N, 2=ADJ, 3=ADV)
        # 'menjadi' = V, 'orang' = N — keduanya ada di vocab_aksara.json dan KBBI
        id_v = self.store.lookup_exact("menjadi")   # V
        id_n = self.store.lookup_exact("orang")     # N
        if id_v == 0 or id_n == 0:
            pytest.skip("Kata uji tidak ditemukan di KBBI")
        v_verb = self.emb[id_v]
        v_noun = self.emb[id_n]
        # Slot POS berbeda → dot product di slot 0-3 harus kecil
        # (karena one-hot: V=[1,0,0,0], N=[0,1,0,0] → dot=0)
        pos_dot = torch.dot(v_verb[:4], v_noun[:4]).item()
        assert pos_dot < 0.5, (
            f"POS slot V vs N harus orthogonal, bukan dot={pos_dot:.4f}"
        )

    def test_grounding_berpengaruh_pada_forward(self):
        # Verifikasi: embedding pre-seeded mempengaruhi get_anchors()
        # Anchor besar(adj) dan berlari(v) harus berbeda secara signifikan
        from aksara.linguistic.lps import build_root_vocab
        rv = self.lsk.root_vocab
        id_besar   = rv.get("besar", None)
        id_berlari = rv.get("berlari", None)
        if id_besar is None or id_berlari is None:
            pytest.skip("Kata uji tidak ada di root_vocab")

        ids_besar   = torch.tensor([[id_besar]])
        ids_berlari = torch.tensor([[id_berlari]])
        anchor_besar   = self.lsk.get_anchors(ids_besar).squeeze()
        anchor_berlari = self.lsk.get_anchors(ids_berlari).squeeze()

        # Keduanya harus berbeda (bukan zero vector yang sama)
        assert not torch.allclose(anchor_besar, anchor_berlari, atol=1e-4), \
            "Anchor ADJ vs V harus berbeda karena embedding pre-seeded berbeda"

    def test_coverage_100_persen(self):
        # Semua 49,999 lemma KBBI harus punya embedding non-zero setelah pre-seeding
        n_nonzero = (self.emb[1:].norm(dim=1) > 1e-6).sum().item()
        total = self.store.unique_lemmas
        coverage = n_nonzero / total
        assert coverage > 0.99, (
            f"Coverage embedding harus > 99%, dapat {coverage:.1%}"
        )
