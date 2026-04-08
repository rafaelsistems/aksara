"""
Test LPS - Lapisan Parsing Struktural
"""

import pytest
from aksara.linguistic.lps import (
    MorfologiAnalyzer,
    LapisanParsingStuktural,
    LPSConfig,
    build_root_vocab,
    AFFIX_VOCAB,
)


class TestMorfologiAnalyzer:

    def setup_method(self):
        self.analyzer = MorfologiAnalyzer(min_root_length=3)

    def test_prefix_ber(self):
        root, affix = self.analyzer.best("berjalan")
        assert affix == "ber"
        assert root == "jalan"

    def test_prefix_me(self):
        root, affix = self.analyzer.best("membaca")
        assert "mem" in affix or affix == "me"

    def test_prefix_di(self):
        root, affix = self.analyzer.best("dimakan")
        assert affix == "di" or "di" in affix

    def test_suffix_an(self):
        root, affix = self.analyzer.best("makanan")
        assert affix == "an"
        assert root == "makan"

    def test_suffix_kan(self):
        root, affix = self.analyzer.best("jalankan")
        assert affix == "kan"
        assert root == "jalan"

    def test_confix_ke_an(self):
        candidates = self.analyzer.analyze("keadilan")
        affix_labels = [c[1] for c in candidates]
        assert "ke+an" in affix_labels or any("ke" in a for a in affix_labels)

    def test_no_affix(self):
        root, affix = self.analyzer.best("rumah")
        assert root == "rumah"
        assert affix == "<NONE>"

    def test_root_minimum_length(self):
        # Root pendek tidak boleh dihasilkan
        candidates = self.analyzer.analyze("berak")
        for root, affix, conf in candidates:
            if affix != "<NONE>":
                assert len(root) >= self.analyzer.min_root

    def test_candidates_confidence(self):
        candidates = self.analyzer.analyze("berjalan")
        assert len(candidates) >= 1
        for root, affix, conf in candidates:
            assert 0.0 <= conf <= 1.0


class TestLapisanParsingStruktural:

    def setup_method(self):
        # Bangun vocab kecil untuk testing
        corpus = [
            "berjalan di taman", "membaca buku pelajaran",
            "rumah besar di kota", "makanan enak sekali",
        ]
        self.vocab = build_root_vocab(corpus, min_freq=1)
        self.lps = LapisanParsingStuktural(LPSConfig(), self.vocab)

    def test_tokenize_basic(self):
        tokens = self.lps.tokenize("berjalan di taman")
        assert "berjalan" in tokens
        assert "di" in tokens
        assert "taman" in tokens

    def test_tokenize_punctuation(self):
        tokens = self.lps.tokenize("Dia berlari, cepat sekali.")
        assert "," in tokens or len(tokens) >= 4

    def test_encode_sequence_length(self):
        tokens = ["berjalan", "di", "taman"]
        enc = self.lps.encode_sequence(tokens, max_len=5)
        assert len(enc["morpheme_ids"]) == 5
        assert len(enc["affix_ids"]) == 5

    def test_encode_padding(self):
        tokens = ["berjalan"]
        enc = self.lps.encode_sequence(tokens, max_len=4)
        assert enc["morpheme_ids"][1:] == [0, 0, 0]

    def test_forward_batch(self):
        import torch
        texts = ["berjalan di taman", "membaca buku"]
        out = self.lps(texts, device=torch.device("cpu"))

        assert "morpheme_ids" in out
        assert "affix_ids" in out
        assert "dep_masks" in out
        assert out["morpheme_ids"].shape[0] == 2
        assert out["dep_masks"].shape[1] == out["dep_masks"].shape[2]

    def test_dep_mask_symmetric(self):
        import torch
        tokens = ["saya", "makan", "nasi"]
        mask = self.lps.build_dep_mask(tokens, L=3)
        # Self-loop harus ada
        for i in range(3):
            assert mask[i, i].item() is True
        # Mask harus simetris (undirected)
        assert (mask == mask.T).all()

    def test_build_root_vocab(self):
        corpus = ["berjalan cepat", "berlari kencang", "berjalan santai"]
        vocab = build_root_vocab(corpus, min_freq=1)
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert vocab["<PAD>"] == 0


class TestDepParserBI:
    """
    B1-D: Verifikasi dependency parser Bahasa Indonesia.

    Parser harus menghasilkan dependency tree linguistik, bukan sliding window.
    Setiap test memverifikasi satu kaidah tata bahasa Indonesia.
    """

    def setup_method(self):
        from aksara.linguistic.lsk import KBBIStore
        import json, os
        # Gunakan KBBIStore jika tersedia untuk POS tagging akurat
        kbbi_path = "kbbi_core_v2.json"
        self.store = KBBIStore(kbbi_path) if os.path.exists(kbbi_path) else None
        rv = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        # Tambah kata uji ke vocab
        for w in ["anak", "membaca", "buku", "dia", "sedang", "menulis",
                  "surat", "mereka", "pergi", "ke", "sekolah", "yang",
                  "dibaca", "para", "siswa", "besar", "itu", "ibu",
                  "memasak", "dan", "sangat", "menarik"]:
            if w not in rv:
                rv[w] = len(rv)
        self.lps = LapisanParsingStuktural(
            LPSConfig(), rv, kbbi_store=self.store
        )

    # ── POS Tagging ──────────────────────────────────────────────────────────

    def test_pos_verba_prefiks_me(self):
        # me- + root verba → V
        assert self.lps._pos_tag("membaca") == "V"
        assert self.lps._pos_tag("menulis") == "V"
        assert self.lps._pos_tag("memasak") == "V"

    def test_pos_verba_prefiks_ber(self):
        assert self.lps._pos_tag("berjalan") == "V"
        assert self.lps._pos_tag("berlari") == "V"

    def test_pos_verba_prefiks_di(self):
        assert self.lps._pos_tag("dibaca") == "V"
        assert self.lps._pos_tag("dimakan") == "V"

    def test_pos_nomina_kbbi(self):
        # Kata nomina yang ada di KBBI harus dapat POS N
        if self.store is None:
            pytest.skip("KBBI tidak tersedia")
        assert self.lps._pos_tag("buku") == "N"
        assert self.lps._pos_tag("anak") == "N"

    def test_pos_nomina_tidak_terpengaruh_prefix_se(self):
        # 'sekolah' punya prefix se- tapi KBBI mengatakan N
        if self.store is None:
            pytest.skip("KBBI tidak tersedia")
        assert self.lps._pos_tag("sekolah") == "N"

    def test_pos_function_words_tetap(self):
        # Function words harus selalu dapat POS yang sama tanpa KBBI
        assert self.lps._pos_tag("dan") == "CONJ"
        assert self.lps._pos_tag("yang") == "REL"
        assert self.lps._pos_tag("di") == "P"
        assert self.lps._pos_tag("dia") == "PRON"
        assert self.lps._pos_tag("sedang") == "ADV"
        assert self.lps._pos_tag("itu") == "DET"
        assert self.lps._pos_tag("para") == "DET"

    def test_pos_numeralia(self):
        assert self.lps._pos_tag("123") == "NUM"
        assert self.lps._pos_tag("2025") == "NUM"
        assert self.lps._pos_tag("satu") == "NUM"
        assert self.lps._pos_tag("dua") == "NUM"

    # ── Head-Finding ─────────────────────────────────────────────────────────

    def test_head_svo_dasar(self):
        # "anak membaca buku" — verba adalah root, N attach ke V
        tokens = ["anak", "membaca", "buku"]
        pos    = [self.lps._pos_tag(t) for t in tokens]
        heads  = self.lps._find_heads(tokens, pos)
        v_idx  = tokens.index("membaca")
        assert heads[v_idx] == -1,                    "membaca harus ROOT"
        assert heads[tokens.index("anak")] == v_idx,  "anak harus attach ke membaca"
        assert heads[tokens.index("buku")] == v_idx,  "buku harus attach ke membaca"

    def test_head_adv_attach_ke_verba(self):
        # "dia sedang menulis surat" — ADV attach ke V
        tokens = ["dia", "sedang", "menulis", "surat"]
        pos    = [self.lps._pos_tag(t) for t in tokens]
        heads  = self.lps._find_heads(tokens, pos)
        v_idx  = tokens.index("menulis")
        assert heads[tokens.index("sedang")] == v_idx, "sedang(ADV) harus attach ke menulis"

    def test_head_preposisi_ke_attach_verba(self):
        # "mereka pergi ke sekolah" — P attach ke V, N setelah P attach ke P
        tokens = ["mereka", "pergi", "ke", "sekolah"]
        pos    = [self.lps._pos_tag(t) for t in tokens]
        heads  = self.lps._find_heads(tokens, pos)
        v_idx  = tokens.index("pergi")
        p_idx  = tokens.index("ke")
        assert heads[p_idx] == v_idx,                   "ke(P) harus attach ke pergi"
        assert heads[tokens.index("sekolah")] == p_idx, "sekolah harus attach ke ke(P)"

    def test_head_relativizer_yang(self):
        # "buku yang dibaca anak" — 'yang' attach ke N sebelumnya (buku)
        tokens = ["buku", "yang", "dibaca", "anak"]
        pos    = [self.lps._pos_tag(t) for t in tokens]
        heads  = self.lps._find_heads(tokens, pos)
        assert heads[tokens.index("yang")] == tokens.index("buku"), \
            "yang(REL) harus attach ke buku"

    def test_head_det_post_nominal(self):
        # "buku itu" — 'itu' (post-nominal DET) attach ke N di kiri
        tokens = ["buku", "itu"]
        pos    = [self.lps._pos_tag(t) for t in tokens]
        heads  = self.lps._find_heads(tokens, pos)
        assert heads[tokens.index("itu")] == tokens.index("buku"), \
            "itu(DET post-nominal) harus attach ke buku"

    def test_head_det_pre_nominal(self):
        # "para siswa" — 'para' (pre-nominal DET) attach ke N di kanan
        tokens = ["para", "siswa"]
        pos    = [self.lps._pos_tag(t) for t in tokens]
        heads  = self.lps._find_heads(tokens, pos)
        assert heads[tokens.index("para")] == tokens.index("siswa"), \
            "para(DET pre-nominal) harus attach ke siswa"

    # ── Dependency Mask ───────────────────────────────────────────────────────

    def test_dep_mask_bukan_full_matrix(self):
        # Dep mask dari parser linguistik BUKAN full matrix (O(n^2))
        # Untuk kalimat panjang, jumlah True jauh lebih sedikit dari n^2
        import torch
        tokens = ["anak", "sedang", "membaca", "buku", "yang", "sangat", "menarik"]
        n = len(tokens)
        mask = self.lps.build_dep_mask(tokens, n)
        n_edges = mask.sum().item()
        # Full attention: n^2 = 49. Dependency tree: jauh lebih sedikit.
        # Dengan sibling edges, masih << n^2
        assert n_edges < n * n, \
            f"dep_mask bukan full matrix: {n_edges} edges < {n*n} (full)"

    def test_dep_mask_simetris(self):
        import torch
        tokens = ["anak", "membaca", "buku"]
        mask = self.lps.build_dep_mask(tokens, 3)
        assert (mask == mask.T).all(), "dep_mask harus simetris (undirected graph)"

    def test_dep_mask_self_loop(self):
        import torch
        tokens = ["anak", "membaca", "buku"]
        mask = self.lps.build_dep_mask(tokens, 3)
        for i in range(3):
            assert mask[i, i].item(), f"token {i} harus punya self-loop"

    def test_dep_mask_svo_edge_linguistik(self):
        # Verifikasi: edge S→V dan O→V ada di mask
        import torch
        tokens = ["anak", "membaca", "buku"]
        mask = self.lps.build_dep_mask(tokens, 3)
        # anak(0)↔membaca(1) dan buku(2)↔membaca(1) harus ada
        assert mask[0, 1].item(), "edge anak↔membaca harus ada"
        assert mask[1, 0].item(), "edge membaca↔anak harus ada (simetris)"
        assert mask[2, 1].item(), "edge buku↔membaca harus ada"
        assert mask[1, 2].item(), "edge membaca↔buku harus ada (simetris)"

    def test_dep_mask_berbeda_dari_window(self):
        # Dep mask linguistik HARUS berbeda dari sliding window lama
        # untuk kalimat yang punya struktur dependency nyata
        import torch
        tokens = ["buku", "yang", "dibaca", "anak"]
        n = len(tokens)
        mask_linguistik = self.lps.build_dep_mask(tokens, n)

        # Buat window mask manual (bootstrap lama)
        window = self.lps.config.dep_window
        mask_window = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            mask_window[i, i] = True
            for j in range(max(0, i - window), min(n, i + window + 1)):
                mask_window[i, j] = True
                mask_window[j, i] = True

        # Harus ada perbedaan (kalimat ≤ 2*window+1 token mungkin sama,
        # tapi untuk kalimat panjang harus berbeda)
        # Untuk L=4 dengan window=4: window mask bisa sama, jadi cek kalimat panjang
        tokens_panjang = ["anak", "sedang", "membaca", "buku", "yang", "menarik",
                          "di", "perpustakaan", "sekolah"]
        n2 = len(tokens_panjang)
        mask_l = self.lps.build_dep_mask(tokens_panjang, n2)
        mask_w = torch.zeros(n2, n2, dtype=torch.bool)
        for i in range(n2):
            mask_w[i, i] = True
            for j in range(max(0, i - window), min(n2, i + window + 1)):
                mask_w[i, j] = True
                mask_w[j, i] = True
        assert not (mask_l == mask_w).all(), \
            "dep_mask linguistik harus berbeda dari sliding window untuk kalimat panjang"
