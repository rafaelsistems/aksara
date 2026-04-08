"""
LPS - Lapisan Parsing Struktural
Mengubah teks menjadi unit linguistik Indonesia yang terstruktur.

DESAIN KRITIS: LPS menggunakan pendekatan SOFT/PROBABILISTIC —
bukan hard rule-based parsing. Ini memungkinkan model belajar
dari kesalahan dan memperbaiki representasinya sendiri.

Pipeline:
  teks → tokenisasi morfem → distribusi affix → peran sintaktik awal
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Daftar affix Bahasa Indonesia (first-class citizen)
# ─────────────────────────────────────────────

PREFIXES_ID = [
    "me", "meng", "mem", "men", "meny",
    "ber", "be",
    "ter",
    "di",
    "ke",
    "pe", "peng", "pem", "pen", "peny", "per",
    "se",
    "mempel", "memper", "diper",
]

SUFFIXES_ID = [
    "kan", "an", "i",
    "nya", "ku", "mu",
    "lah", "kah", "tah", "pun",
]

CONFIXES_ID = [
    ("ke", "an"), ("per", "an"), ("pe", "an"),
    ("ber", "an"), ("se", "nya"),
]

# Pemetaan POS dari KBBI ke kategori internal LPS
# KBBI menggunakan: v, n, adj, adv, p, num, pron, cak, ark, ref, bt, ki, ...
KBBI_POS_MAP: Dict[str, str] = {
    "v":    "V",    # verba (kata kerja)
    "vi":   "V",
    "vt":   "V",
    "n":    "N",    # nomina (kata benda)
    "adj":  "ADJ",  # adjektiva (kata sifat)
    "adv":  "ADV",  # adverbia (kata keterangan)
    "p":    "P",    # partikel / preposisi
    "pron": "PRON", # pronomina (kata ganti)
    "num":  "NUM",  # numeralia (kata bilangan)
    "cak":  "N",    # cakap — umumnya nomina informal
    "ki":   "ADJ",  # kiasan — umumnya adjektiva
    "ark":  "N",    # arkais — umumnya nomina
    "ref":  "N",    # referensi silang — fallback ke N
    "bt":   "N",    # botani — umumnya nomina
    "konjung": "CONJ",
    "konj":    "CONJ",
}

# Prefix Bahasa Indonesia → kecenderungan POS hasilnya
# me-/ber-/ter-/di- → hampir selalu verba
# pe-/ke- → nomina (pelaku/abstrak)
# se- → adverbia/adjektiva
PREFIX_POS_HINT: Dict[str, str] = {
    "me":  "V", "meng": "V", "mem": "V", "men": "V", "meny": "V",
    "ber": "V", "be":   "V",
    "ter": "V",
    "di":  "V",
    "pe":  "N", "peng": "N", "pem": "N", "pen": "N", "peny": "N",
    "per": "N",
    "ke":  "N",
    "se":  "ADV",
    "mempel": "V", "memper": "V", "diper": "V",
}

# Kata fungsi (function words) dengan POS tetap — tidak perlu KBBI
FUNCTION_WORDS: Dict[str, str] = {
    # Pronomina
    "saya": "PRON", "aku": "PRON", "kamu": "PRON", "kau": "PRON",
    "dia": "PRON", "ia": "PRON", "kami": "PRON", "kita": "PRON",
    "mereka": "PRON", "anda": "PRON",
    # Preposisi / Konjungsi
    "di": "P", "ke": "P", "dari": "P", "dengan": "P", "untuk": "P",
    "pada": "P", "oleh": "P", "tentang": "P", "dalam": "P", "atas": "P",
    "bagi": "P", "antara": "P", "sekitar": "P", "melalui": "P",
    "dan": "CONJ", "atau": "CONJ", "tetapi": "CONJ", "namun": "CONJ",
    "karena": "CONJ", "sehingga": "CONJ", "agar": "CONJ", "jika": "CONJ",
    "bahwa": "CONJ", "meskipun": "CONJ", "walaupun": "CONJ",
    "yang": "REL",  # relativizer — kelas khusus BI
    # Adverbia umum
    "sudah": "ADV", "sudah": "ADV", "telah": "ADV", "akan": "ADV",
    "sedang": "ADV", "belum": "ADV", "tidak": "ADV", "bukan": "ADV",
    "juga": "ADV", "hanya": "ADV", "sangat": "ADV", "lebih": "ADV",
    "paling": "ADV", "sudah": "ADV", "masih": "ADV", "selalu": "ADV",
    "lagi": "ADV", "pun": "ADV",
    # Determiner
    "ini": "DET", "itu": "DET", "tersebut": "DET",
    "para": "DET", "semua": "DET", "setiap": "DET", "beberapa": "DET",
    "suatu": "DET", "sebuah": "DET", "seorang": "DET", "segala": "DET",
    # Numeralia
    "satu": "NUM", "dua": "NUM", "tiga": "NUM", "empat": "NUM",
    "lima": "NUM", "enam": "NUM", "tujuh": "NUM", "delapan": "NUM",
    "sembilan": "NUM", "sepuluh": "NUM", "seratus": "NUM", "seribu": "NUM",
    "pertama": "NUM", "kedua": "NUM", "ketiga": "NUM",
}

# Label peran sintaktik
ROLE_LABELS = {
    "UNK": 0,
    "S":   1,   # Subjek
    "P":   2,   # Predikat
    "O":   3,   # Objek
    "K":   4,   # Keterangan
    "PEL": 5,   # Pelengkap
    "DET": 6,   # Determiner
    "MOD": 7,   # Modifier
}

# Label affix (untuk vocab affix)
AFFIX_VOCAB = ["<PAD>", "<NONE>"] + PREFIXES_ID + SUFFIXES_ID + [
    f"{p}+{s}" for p, s in CONFIXES_ID
]
AFFIX_TO_ID = {a: i for i, a in enumerate(AFFIX_VOCAB)}


@dataclass
class LPSConfig:
    max_word_length: int = 32
    min_root_length: int = 3        # panjang minimum root setelah stripping affix
    use_soft_segmentation: bool = True  # probabilistik vs. hard rule
    soft_temp: float = 1.0          # temperature untuk distribusi affix
    build_dep_graph: bool = True    # apakah LPS membangun dependency mask
    dep_window: int = 4             # fallback window jika tidak ada rule-based dep


class MorfologiAnalyzer:
    """
    Analisis morfologi berbasis aturan untuk Bahasa Indonesia.
    Menghasilkan (root_candidate, affix_label, confidence).

    Ini BUKAN NLP library eksternal — murni aturan morfologi Indonesia
    yang bisa di-differentiate melalui soft weighting.
    """

    def __init__(self, min_root_length: int = 3, known_words: set = None):
        self.min_root = min_root_length
        self.known_words = known_words or set()  # KBBI lemma set sebagai validator
        self._build_rules()

    def _build_rules(self):
        """Bangun aturan stripping affix berdasarkan morfologi Indonesia."""
        self.prefix_rules = sorted(PREFIXES_ID, key=len, reverse=True)
        self.suffix_rules = sorted(SUFFIXES_ID, key=len, reverse=True)

    def analyze(self, word: str) -> List[Tuple[str, str, float]]:
        """
        Analisis satu kata. Kembalikan daftar kandidat:
        [(root, affix_label, confidence), ...]

        confidence tinggi = lebih yakin ini segmentasi yang benar.
        """
        word = word.lower().strip()
        candidates = []

        # Kandidat 0: tidak ada affix (root = word itu sendiri)
        # Boost confidence jika kata ini ada di KBBI sebagai lemma — artinya kata utuh
        # TAPI: jika ada prefix yang menghasilkan root yang JUGA ada di KBBI,
        # maka kata ini adalah bentuk turunan dan prefix harus menang.
        # Contoh: 'membaca' ada di KBBI, tapi 'baca' juga ada → mem+baca menang.
        #         'taman' ada di KBBI, strip 'an' → 'tam' tidak ada di KBBI → <NONE> menang.
        has_valid_prefix_root = False
        for prefix in self.prefix_rules:
            if word.startswith(prefix):
                root_cand = word[len(prefix):]
                if len(root_cand) >= self.min_root and root_cand in self.known_words:
                    has_valid_prefix_root = True
                    break
        if word in self.known_words and not has_valid_prefix_root:
            none_conf = 0.97
        else:
            none_conf = 0.5
        candidates.append((word, "<NONE>", none_conf))

        # Strip prefix — lebih tinggi confidence base dari suffix
        for prefix in self.prefix_rules:
            if word.startswith(prefix):
                root_cand = word[len(prefix):]
                if len(root_cand) >= self.min_root:
                    # Prefix lebih reliable daripada suffix: base 0.6
                    conf = 0.6 + 0.05 * len(prefix)
                    candidates.append((root_cand, prefix, min(conf, 0.95)))

        # Strip suffix — base lebih rendah dari prefix
        for suffix in self.suffix_rules:
            if word.endswith(suffix):
                root_cand = word[:-len(suffix)]
                if len(root_cand) >= self.min_root:
                    conf = 0.5 + 0.05 * len(suffix)
                    candidates.append((root_cand, suffix, min(conf, 0.85)))

        # Strip confix — hanya menang jika root confix LEBIH PENDEK dari prefix-only root
        # DAN tidak ada kandidat prefix tunggal yang sudah menghasilkan root yang valid
        for prefix, suffix in CONFIXES_ID:
            if word.startswith(prefix) and word.endswith(suffix):
                root_cand = word[len(prefix):-len(suffix)]
                prefix_only_root = word[len(prefix):]
                # Confix hanya valid jika root lebih pendek (suffix benar-benar dipotong)
                # dan root confix bukan sama dengan prefix-only root (berarti suffix tidak ada)
                if (len(root_cand) >= self.min_root
                        and len(root_cand) < len(prefix_only_root)
                        and root_cand != prefix_only_root):
                    affix_label = f"{prefix}+{suffix}"
                    # Confidence confix = prefix_conf - 0.1 agar prefix selalu menang jika ambigu
                    prefix_conf = 0.6 + 0.05 * len(prefix)
                    conf = min(prefix_conf - 0.1, 0.80)
                    candidates.append((root_cand, affix_label, conf))

        return candidates

    def best(self, word: str) -> Tuple[str, str]:
        """Ambil kandidat terbaik (greedy, untuk preprocessing)."""
        candidates = self.analyze(word)
        best = max(candidates, key=lambda x: x[2])
        return best[0], best[1]


class LapisanParsingStuktural(nn.Module):
    """
    LPS: Lapisan Parsing Struktural

    Fungsi utama:
    1. Tokenisasi berbasis kata (bukan subword BPE)
    2. Analisis morfologi → distribusi affix (soft)
    3. Inisialisasi dependency mask (rule-based dengan fallback window)
    4. Normalisasi teks Indonesia (EYD, ragam formal/informal)

    Output yang diberikan ke BahasaStateUnit:
    - morpheme_ids  : (B, L) — root word indices
    - affix_ids     : (B, L) — dominant affix indices
    - dep_mask      : (B, L, L) bool — dependency adjacency
    - raw_tokens    : List[List[str]] — untuk debugging
    """

    def __init__(self, config: LPSConfig, root_vocab: Dict[str, int],
                 known_words: set = None,
                 kbbi_store=None):
        super().__init__()
        self.config = config
        self.root_vocab = root_vocab
        self.affix_vocab = AFFIX_TO_ID
        self.role_labels = ROLE_LABELS
        self.analyzer = MorfologiAnalyzer(config.min_root_length, known_words)
        # KBBIStore opsional — jika tersedia, POS tagging jauh lebih akurat
        self._kbbi_store = kbbi_store

        # NOTE: Soft segmentation (use_soft_segmentation) akan diimplementasi
        # di versi mendatang. Saat ini LPS menggunakan rule-based morphology
        # dengan confidence scoring — yang sudah merupakan pendekatan probabilistik
        # karena setiap kandidat affix memiliki confidence score yang bisa
        # digunakan untuk weighted decision.
        #
        # Implementasi soft segmentation yang benar memerlukan:
        # - Character-level encoder untuk kata
        # - Distribusi probabilistik atas affix candidates
        # - Differentiable morpheme boundary detection
        # Ini akan menjadi bagian dari evolusi arsitektur AKSARA.

    def tokenize(self, text: str) -> List[str]:
        """Tokenisasi teks Indonesia berbasis kata."""
        # Normalisasi dasar
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        # Pisahkan tanda baca dari kata
        text = re.sub(r'([.,!?;:"])', r' \1 ', text)
        tokens = [t for t in text.split() if t]
        return tokens

    def encode_sequence(
        self,
        tokens: List[str],
        max_len: Optional[int] = None,
    ) -> Dict:
        """
        Encode satu sequence token → morpheme_ids, affix_ids.

        Returns dict dengan keys:
            morpheme_ids, affix_ids, affix_confidences, raw_tokens
        """
        UNK = self.root_vocab.get("<UNK>", 1)
        PAD_ROOT = self.root_vocab.get("<PAD>", 0)

        morpheme_ids = []
        affix_ids = []
        affix_confs = []

        for tok in tokens:
            root, affix = self.analyzer.best(tok)
            root_id = self.root_vocab.get(root, self.root_vocab.get(tok, UNK))
            affix_id = self.affix_vocab.get(affix, self.affix_vocab.get("<NONE>", 1))

            candidates = self.analyzer.analyze(tok)
            best_conf = max(c[2] for c in candidates) if candidates else 0.5

            morpheme_ids.append(root_id)
            affix_ids.append(affix_id)
            affix_confs.append(best_conf)

        if max_len is not None:
            # Pad atau truncate
            L = len(morpheme_ids)
            if L < max_len:
                pad_n = max_len - L
                morpheme_ids += [PAD_ROOT] * pad_n
                affix_ids += [0] * pad_n
                affix_confs += [0.0] * pad_n
            else:
                morpheme_ids = morpheme_ids[:max_len]
                affix_ids = affix_ids[:max_len]
                affix_confs = affix_confs[:max_len]

        return {
            "morpheme_ids": morpheme_ids,
            "affix_ids": affix_ids,
            "affix_confidences": affix_confs,
            "raw_tokens": tokens,
        }

    def _pos_tag(self, token: str) -> str:
        """
        Tentukan POS tag untuk satu token Bahasa Indonesia.

        Prioritas lookup (dari paling reliable ke fallback):
          1. Function words tetap (pronomina, preposisi, konjungsi, det, num)
          2. Token numerik → NUM
          3. KBBI exact match → POS dari KBBI
          4. Prefix morphology → kecenderungan POS dari PREFIX_POS_HINT
          5. Suffix heuristik (kata berakhiran -an → N, -i/-kan → V)
          6. Fallback → N (nomina paling umum dalam BI)

        Returns: POS string dari {V, N, ADJ, ADV, P, PRON, NUM, CONJ, REL, DET}
        """
        word = token.lower().strip()

        # 1. Function words — fixed mapping, tidak perlu lookup lebih lanjut
        if word in FUNCTION_WORDS:
            return FUNCTION_WORDS[word]

        # 2. Angka / numeralia
        if word.isdigit() or (word.replace('.', '').replace(',', '').isdigit()):
            return "NUM"

        # 3. KBBI exact match
        # PRIORITAS: KBBI menang atas prefix rule.
        # Contoh: 'sekolah' punya prefix 'se-' tapi KBBI tahu itu N, bukan ADV.
        # PENGECUALIAN: jika POS KBBI adalah 'ref' (referensi silang), berarti KBBI
        # tidak punya definisi mandiri untuk kata ini — lanjut ke prefix rule.
        if self._kbbi_store is not None:
            raw_pos = self._kbbi_store.get_pos(word)
            if raw_pos and raw_pos != "ref":
                mapped = KBBI_POS_MAP.get(raw_pos, None)
                if mapped:
                    return mapped

        # 4. Prefix morphology → kecenderungan POS
        # Hanya dipakai jika KBBI tidak punya entry definitif (atau word OOV KBBI).
        for prefix in PREFIXES_ID:
            if word.startswith(prefix):
                root_cand = word[len(prefix):]
                if len(root_cand) >= self.config.min_root_length:
                    if self._kbbi_store is not None:
                        root_pos = self._kbbi_store.get_pos(root_cand)
                        # Root ada di KBBI dan root bukan 'ref' → prefix rule valid
                        if root_pos and root_pos != "ref":
                            hint = PREFIX_POS_HINT.get(prefix)
                            if hint:
                                return hint
                    else:
                        hint = PREFIX_POS_HINT.get(prefix)
                        if hint:
                            return hint

        # 5. Suffix heuristik
        if word.endswith("an") and len(word) > 4:
            return "N"   # ke-an, per-an, atau kata benda berakhiran -an
        if word.endswith("kan") and len(word) > 5:
            return "V"   # kata kerja transitif
        if word.endswith("i") and len(word) > 4:
            return "V"   # kata kerja + objek (menemani, mendatangi)
        if word.endswith("nya") and len(word) > 5:
            return "N"   # pronomina enklitik → tetap N/PRON

        # 6. Fallback: nomina (paling umum dalam teks BI)
        return "N"

    def _pos_tag_sequence(self, tokens: List[str]) -> List[str]:
        """
        POS tagging untuk seluruh sequence dengan multi-POS disambiguation.

        Lebih akurat dari memanggil _pos_tag() satu per satu karena:
        - Kata ambigu (mis. 'bisa': N=kemampuan, V=dapat, N=racun)
          mendapat POS terbaik berdasarkan POS tetangga di kalimat.
        - Menggunakan KBBIStore.get_pos_context() untuk disambiguation.

        Dua tahap:
          1. Pass pertama: tag semua token dengan _pos_tag() (tanpa konteks)
          2. Pass kedua: untuk token yang POS-nya ambigu (ada lebih dari 1 POS
             di KBBI), lakukan re-tagging dengan konteks dari pass pertama.

        Args:
            tokens : List[str] token yang sudah ditokenisasi

        Returns:
            pos_tags : List[str] POS untuk setiap token, dengan konteks.
        """
        n = len(tokens)

        # Pass 1: POS tanpa konteks (baseline)
        pos_tags = [self._pos_tag(tok) for tok in tokens]

        # Pass 2: disambiguation untuk token yang KBBI-nya multi-POS
        if self._kbbi_store is None:
            return pos_tags

        for i, tok in enumerate(tokens):
            word = tok.lower().strip()

            # Skip function words — POS-nya fixed, tidak perlu disambiguation
            if word in FUNCTION_WORDS:
                continue

            # Cek apakah kata ini punya lebih dari 1 POS di KBBI
            pos_list = self._kbbi_store.get_pos_list(word, top_n=5)
            if len(pos_list) <= 1:
                # Tidak ambigu, tidak perlu disambiguation
                continue

            # Ada ambiguitas POS di KBBI — gunakan konteks tetangga
            # Kumpulkan POS tetangga kiri dan kanan (dari pass 1)
            left_ctx  = pos_tags[max(0, i - 2) : i]
            right_ctx = pos_tags[i + 1 : min(n, i + 3)]
            neighbor_pos = left_ctx + right_ctx

            # Minta KBBIStore untuk disambiguasi berbasis konteks
            best_raw_pos = self._kbbi_store.get_pos_context(
                word, neighbor_pos=neighbor_pos
            )
            if best_raw_pos and best_raw_pos != "ref":
                mapped = KBBI_POS_MAP.get(best_raw_pos, None)
                if mapped:
                    pos_tags[i] = mapped

        return pos_tags

    def _find_heads(self, tokens: List[str], pos_tags: List[str]) -> List[int]:
        """
        Tentukan head (governor) untuk setiap token dalam kalimat.

        Algoritma berbasis kaidah Bahasa Indonesia:
        - Kalimat BI dominan SVO / SPO / SPOK
        - Verba utama adalah root kalimat (head = -1 untuk root)
        - Verba intransitif: subjek (N/PRON sebelum V) → attach ke V
        - Verba transitif: subjek + objek (N setelah V) → attach ke V
        - Adjektiva / Adverbia → attach ke N/V terdekat (head-right untuk BI)
        - Preposisi → attach ke V terdekat; NP setelah preposisi → attach ke P
        - Determiner / Numeralia → attach ke N terdekat
        - Konjungsi → attach ke token setelahnya (conjoin-right)
        - Relativizer 'yang' → attach ke N sebelumnya (modifikasi N)
        - Fallback: attach ke token sebelumnya (projective tree)

        Returns:
            heads: List[int], heads[i] = indeks head token i,
                   -1 = root kalimat
        """
        n = len(tokens)
        if n == 0:
            return []
        if n == 1:
            return [-1]

        heads = [-1] * n

        # Langkah 1: Temukan verba utama (root kalimat)
        # Prioritaskan V pertama yang bukan modalitas/aspektual
        MODAL_VERBS = {"adalah", "merupakan", "ialah", "yakni", "yaitu"}
        root_idx = -1
        # Cari V yang bukan auxiliary/modal
        for i, (tok, pos) in enumerate(zip(tokens, pos_tags)):
            if pos == "V" and tok.lower() not in MODAL_VERBS:
                root_idx = i
                break
        # Fallback: V apa pun (termasuk kopula)
        if root_idx == -1:
            for i, pos in enumerate(pos_tags):
                if pos == "V":
                    root_idx = i
                    break

        # Fallback: kalimat nominal (tanpa verba) — N/PRON pertama jadi root
        # Contoh: "para siswa" (DET + N), "buku pelajaran" (N + N)
        if root_idx == -1:
            for i, pos in enumerate(pos_tags):
                if pos in ("N", "PRON"):
                    root_idx = i
                    break

        # Fallback terakhir: token pertama
        if root_idx == -1:
            root_idx = 0

        heads[root_idx] = -1  # root kalimat

        # Langkah 2: Assign head untuk setiap token
        for i, (tok, pos) in enumerate(zip(tokens, pos_tags)):
            if i == root_idx:
                continue
            word = tok.lower()

            if pos == "N" or pos == "PRON":
                # N/PRON sebelum root → subjek → attach ke root
                # N/PRON setelah root → objek → attach ke root
                # TAPI: jika ada P di antara N dan root → attach ke P
                left_p = -1
                for j in range(min(i, root_idx) + 1, max(i, root_idx)):
                    if pos_tags[j] == "P":
                        left_p = j
                        break
                heads[i] = left_p if left_p != -1 else root_idx

            elif pos == "V" and i != root_idx:
                # Verba non-root (klausa bawahan) → attach ke root
                heads[i] = root_idx

            elif pos == "ADJ":
                # Adjektiva → attach ke N terdekat (kiri dulu, lalu kanan)
                # BI: adjektiva biasanya SETELAH nomina (head-right modifier)
                found = False
                # Cari N di kiri (head)
                for j in range(i - 1, -1, -1):
                    if pos_tags[j] == "N" or pos_tags[j] == "PRON":
                        heads[i] = j
                        found = True
                        break
                if not found:
                    # Fallback: N di kanan
                    for j in range(i + 1, n):
                        if pos_tags[j] == "N":
                            heads[i] = j
                            found = True
                            break
                if not found:
                    heads[i] = root_idx

            elif pos == "ADV":
                # Adverbia → attach ke V terdekat, atau root
                found = False
                for dist in range(1, n):
                    for sign in (-1, 1):
                        j = i + sign * dist
                        if 0 <= j < n and pos_tags[j] == "V":
                            heads[i] = j
                            found = True
                            break
                    if found:
                        break
                if not found:
                    heads[i] = root_idx

            elif pos == "P":
                # Preposisi → attach ke V (root atau V terdekat di kiri)
                found = False
                for j in range(i - 1, -1, -1):
                    if pos_tags[j] == "V":
                        heads[i] = j
                        found = True
                        break
                if not found:
                    heads[i] = root_idx

            elif pos == "DET":
                # Determiner dalam BI: bisa pre-nominal (sebuah, para, beberapa)
                # maupun post-nominal demonstratif (ini, itu, tersebut).
                # 'ini'/'itu'/'tersebut' SELALU merujuk ke N di kiri (anaforik)
                POST_NOMINAL_DET = {"ini", "itu", "tersebut"}
                found = False
                if word in POST_NOMINAL_DET:
                    # Post-nominal: cari N di kiri
                    for j in range(i - 1, -1, -1):
                        if pos_tags[j] in ("N", "PRON", "ADJ"):
                            heads[i] = j
                            found = True
                            break
                else:
                    # Pre-nominal: cari N di kanan
                    for j in range(i + 1, n):
                        if pos_tags[j] == "N" or pos_tags[j] == "PRON":
                            heads[i] = j
                            found = True
                            break
                if not found:
                    heads[i] = root_idx

            elif pos == "NUM":
                # Numeralia → attach ke N terdekat
                found = False
                for j in range(i + 1, n):
                    if pos_tags[j] == "N":
                        heads[i] = j
                        found = True
                        break
                if not found:
                    for j in range(i - 1, -1, -1):
                        if pos_tags[j] == "N":
                            heads[i] = j
                            found = True
                            break
                if not found:
                    heads[i] = root_idx

            elif pos == "CONJ":
                # Konjungsi → attach ke token sesudahnya (conjoin-right)
                heads[i] = i + 1 if i + 1 < n else root_idx

            elif pos == "REL":
                # Relativizer 'yang' → attach ke N sebelumnya
                found = False
                for j in range(i - 1, -1, -1):
                    if pos_tags[j] == "N" or pos_tags[j] == "PRON":
                        heads[i] = j
                        found = True
                        break
                if not found:
                    heads[i] = root_idx

            else:
                # Fallback: attach ke token sebelumnya (projective)
                heads[i] = i - 1 if i > 0 else root_idx

        return heads

    def build_dep_mask(self, tokens: List[str], L: int) -> torch.Tensor:
        """
        Bangun dependency mask (L, L) berbasis dependency graph linguistik.

        Menggunakan _pos_tag() + _find_heads() untuk membangun
        dependency tree Bahasa Indonesia secara rule-based.

        Setiap edge (i → head[i]) menghasilkan koneksi dua arah
        dalam mask — karena f_syn attend ke neighbor dalam graph,
        bukan hanya ke atas/bawah tree.

        Kompleksitas: O(n²) worst-case tapi avg_degree jauh lebih kecil
        dari full attention karena setiap token punya tepat 1 head
        (dependency tree = O(n) edges).

        Args:
            tokens : List[str] — token yang sudah ditokenisasi
            L      : int — ukuran mask (setelah padding)

        Returns:
            mask : (L, L) bool tensor — True = ada dependency edge
        """
        mask = torch.zeros(L, L, dtype=torch.bool)
        n_real = min(len(tokens), L)

        if n_real == 0:
            return mask

        # Diagonal: setiap token attend ke dirinya sendiri
        for i in range(n_real):
            mask[i, i] = True

        # Jika hanya 1 token, tidak ada edge antar token
        if n_real == 1:
            return mask

        # Langkah 1: POS tagging semua token dengan multi-POS disambiguation
        # _pos_tag_sequence() lebih akurat untuk kata ambigu (ada >1 POS di KBBI)
        pos_tags = self._pos_tag_sequence(tokens[:n_real])

        # Langkah 2: Head-finding — bangun dependency tree
        heads = self._find_heads(tokens[:n_real], pos_tags)

        # Langkah 3: Konversi heads → dependency edges
        # Edge i→head[i] (anak ke head) dan head[i]→i (head ke anak)
        # Ini membangun undirected dependency graph
        for i, h in enumerate(heads):
            if h == -1:
                # Root tidak punya head — tetap ada self-loop (sudah di atas)
                continue
            if 0 <= h < n_real:
                mask[i, h] = True   # anak attend ke head-nya
                mask[h, i] = True   # head attend ke anak-nya (sibling info)

        # Langkah 4: Tambahkan sibling edges
        # Token yang berbagi head yang sama bisa saling attend
        # (misal: subjek dan objek keduanya attach ke verba yang sama)
        # Ini penting agar BSU bisa membandingkan argumen-argumen verba
        children_of: Dict[int, List[int]] = defaultdict(list)
        for i, h in enumerate(heads):
            if h != -1 and 0 <= h < n_real:
                children_of[h].append(i)
        for h_idx, siblings in children_of.items():
            for si in siblings:
                for sj in siblings:
                    if si != sj:
                        mask[si, sj] = True   # sibling attend ke sibling

        return mask

    def forward(
        self,
        texts: List[str],
        max_len: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict:
        """
        Proses batch teks → output siap untuk BahasaStateUnit.

        Args:
            texts   : List[str] — batch teks Indonesia
            max_len : panjang sequence maksimum (padding target)
            device  : torch device

        Returns dict:
            morpheme_ids : (B, L) tensor
            affix_ids    : (B, L) tensor
            dep_masks    : (B, L, L) tensor bool
            lengths      : (B,) tensor — panjang asli sebelum padding
            raw_tokens   : List[List[str]]
        """
        all_tokens = [self.tokenize(t) for t in texts]
        lengths = [len(t) for t in all_tokens]

        if max_len is None:
            max_len = max(lengths) if lengths else 1

        all_morpheme_ids = []
        all_affix_ids = []
        all_dep_masks = []
        all_raw = []

        for tokens in all_tokens:
            enc = self.encode_sequence(tokens, max_len=max_len)
            all_morpheme_ids.append(enc["morpheme_ids"])
            all_affix_ids.append(enc["affix_ids"])
            all_raw.append(enc["raw_tokens"])

            dep_mask = self.build_dep_mask(tokens, max_len)
            all_dep_masks.append(dep_mask)

        morpheme_tensor = torch.tensor(all_morpheme_ids, dtype=torch.long, device=device)
        affix_tensor = torch.tensor(all_affix_ids, dtype=torch.long, device=device)
        dep_masks_tensor = torch.stack(all_dep_masks).to(device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

        return {
            "morpheme_ids": morpheme_tensor,
            "affix_ids": affix_tensor,
            "dep_masks": dep_masks_tensor,
            "lengths": lengths_tensor,
            "raw_tokens": all_raw,
            "max_len": max_len,
        }


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """
    Load pre-built vocabulary dari file vocab_aksara.json.

    Format file:
      {"meta": {...}, "vocab": {"<PAD>": 0, "<UNK>": 1, ...}}

    Selalu memastikan special tokens ada di posisi yang benar:
      0=<PAD>, 1=<UNK>, 2=<BOS>, 3=<EOS>, 4=<MASK> (jika ada)

    Args:
        vocab_path: path ke vocab_aksara.json

    Returns:
        vocab: Dict[str, int]
    """
    import json
    from pathlib import Path

    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))

    # Support both formats: flat dict atau {meta, vocab}
    if isinstance(data, dict) and "vocab" in data:
        vocab = data["vocab"]
    else:
        vocab = data

    # Pastikan special tokens selalu ada
    REQUIRED = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for tok, expected_id in REQUIRED.items():
        if tok not in vocab:
            vocab[tok] = expected_id

    return vocab


def build_root_vocab(corpus: List[str], min_freq: int = 2,
                     vocab_path: Optional[str] = None) -> Dict[str, int]:
    """
    Bangun root vocabulary dari corpus Indonesia.
    Menggunakan MorfologiAnalyzer untuk mengekstrak root dari setiap kata.

    Args:
        corpus    : list of text strings
        min_freq  : frekuensi minimum token (untuk filter hapax)
        vocab_path: jika diisi, load dari file pre-built vocab (skip corpus scan)
                    — gunakan ini untuk vocab_aksara.json yang sudah dibangun
                    oleh tools/build_smart_vocab.py
    """
    if vocab_path is not None:
        return load_vocab(vocab_path)

    analyzer = MorfologiAnalyzer()
    freq: Dict[str, int] = {}

    for text in corpus:
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            root, _ = analyzer.best(word)
            freq[root] = freq.get(root, 0) + 1
            # Juga tambahkan kata asli sebagai fallback
            freq[word] = freq.get(word, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    for word, count in sorted(freq.items(), key=lambda x: -x[1]):
        if count >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab
