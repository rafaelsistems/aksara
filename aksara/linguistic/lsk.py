"""
LSK — Lapisan Semantik KBBI v2.0

Modul ini menyediakan semantic grounding dari KBBI (Kamus Besar Bahasa Indonesia)
untuk pipeline AKSARA. Menggunakan kbbi_core_v2.json (clean core schema).

Clean core schema per entry:
  {id, lemma, pos, sense_number, clean_definition}

Tidak ada semantic_vector/quantum_signature/quality_score di core.
Embedding dipelajari (learned) selama training.

Komponen:
  - LSKConfig      : Konfigurasi LSK
  - KBBIStore      : Loader dan index untuk kbbi_core_v2.json
  - LapisanSemantikKBBI : nn.Module — semantic embedding layer
"""

import os
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import difflib
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LSKConfig:
    kbbi_path: str = "kbbi_core_v2.json"
    kbbi_vector_dim: int = 16
    max_lemmas: int = 50000
    oov_strategy: str = "zero"
    freeze_embeddings: bool = False
    pretrained_path: str = "data/kbbi_pretrained.pt"


class KBBIStore:
    def __init__(self, kbbi_path: str = "kbbi_core_v2.json", max_lemmas: int = 50000):
        self.kbbi_path = kbbi_path
        self.max_lemmas = max_lemmas
        self.lemma_to_id: Dict[str, int] = {}
        self.id_to_lemma: Dict[int, str] = {}
        self.lemma_set: Set[str] = set()
        self.pos_freq: Dict[str, Counter] = defaultdict(Counter)
        self.pos_map: Dict[str, str] = {}
        self.definitions: Dict[str, List[str]] = {}
        self.slang_map = {}
        slang_path = "tools/slang_map.json"
        if os.path.exists(slang_path):
            with open(slang_path, "r", encoding="utf-8") as f:
                self.slang_map = json.load(f)
        self.entry_count: int = 0
        self.unique_lemmas: int = 0
        self.loaded: bool = False
        self._load()

    def _load(self):
        if not self.kbbi_path or not os.path.exists(self.kbbi_path):
            self.loaded = False
            return
        try:
            with open(self.kbbi_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.loaded = False
            return
        entries = data.get("entries", [])
        if not entries:
            self.loaded = False
            return
        self.entry_count = len(entries)
        lemma_id_counter = 1
        for entry in entries:
            lemma = entry.get("lemma", "").strip().lower()
            if not lemma:
                continue
            pos = str(entry.get("pos", "unk"))
            self.pos_freq[lemma][pos] += 1
            if lemma not in self.lemma_to_id:
                if lemma_id_counter >= self.max_lemmas:
                    break
                self.lemma_to_id[lemma] = lemma_id_counter
                self.id_to_lemma[lemma_id_counter] = lemma
                self.lemma_set.add(lemma)
                self.pos_map[lemma] = pos
                self.definitions[lemma] = []
                lemma_id_counter += 1
            defn = entry.get("clean_definition", "")
            if defn:
                self.definitions[lemma].append(defn)
        self.unique_lemmas = len(self.lemma_to_id)
        self.loaded = True

    def lookup(self, word: str, min_confidence: float = 0.75) -> Tuple[int, float]:
        clean = word.strip().lower()
        normalized = self.slang_map.get(clean, clean)
        if normalized in self.lemma_to_id:
            return self.lemma_to_id[normalized], 1.0
        if clean in self.lemma_to_id:
            return self.lemma_to_id[clean], 1.0
        if self.lemma_set and min_confidence <= 0.85:
            matches = difflib.get_close_matches(
                normalized,
                self.lemma_set,
                n=1,
                cutoff=max(min_confidence, 0.75),
            )
            if matches:
                best = matches[0]
                ratio = difflib.SequenceMatcher(None, normalized, best).ratio()
                if ratio >= min_confidence:
                    logger.warning("KBBIStore low-confidence fuzzy match: %s -> %s (%.2f)", word, best, ratio)
                    return self.lemma_to_id[best], ratio
        return 0, 0.0

    def lookup_exact(self, word: str) -> int:
        clean = word.strip().lower()
        normalized = self.slang_map.get(clean, clean)
        if normalized in self.lemma_to_id:
            return self.lemma_to_id[normalized]
        return self.lemma_to_id.get(clean, 0)

    def contains(self, word: str) -> bool:
        return word.strip().lower() in self.lemma_set

    def get_pos(self, word: str) -> str:
        clean_word = word.strip().lower()
        if clean_word in self.pos_freq:
            most_common = self.pos_freq[clean_word].most_common(1)
            return most_common[0][0] if most_common else ""
        return ""

    def get_pos_list(self, word: str, top_n: int = 5) -> List[str]:
        clean_word = word.strip().lower()
        if clean_word in self.pos_freq:
            return [pos for pos, _ in self.pos_freq[clean_word].most_common(top_n)]
        return []

    def get_pos_context(
        self,
        word: str,
        neighbor_pos: Optional[List[str]] = None,
        context_role_ids: Optional[torch.Tensor] = None,
    ) -> str:
        clean_word = word.strip().lower()
        if clean_word not in self.pos_freq:
            return ""
        pos_list = self.get_pos_list(clean_word, top_n=10)
        if not pos_list:
            return ""
        scores = {}
        for pos in pos_list:
            score = float(self.pos_freq[clean_word][pos])
            if neighbor_pos:
                ctx_bonus = sum(1.0 for npos in neighbor_pos if self._pos_compatible(pos, npos))
                score += ctx_bonus * 0.5
            if context_role_ids is not None:
                score += self._role_to_pos_bonus(pos, context_role_ids)
            scores[pos] = score
        if not scores:
            return ""
        best_pos = max(scores, key=scores.get)
        if len(scores) > 1:
            ordered_scores = sorted(scores.values(), reverse=True)
            runner_up = ordered_scores[1]
            if scores[best_pos] - runner_up < 0.5:
                logger.info("KBBIStore POS disambiguation low margin: %s -> %s", word, best_pos)
        return best_pos

    def _pos_compatible(self, pos1: str, pos2: str) -> bool:
        compat = {
            "v": ["n", "adj", "adv"],
            "n": ["v", "adj", "p"],
            "adj": ["n", "v", "adv"],
            "p": ["n", "pron"],
        }
        return pos2 in compat.get(pos1[:1], [])

    def _role_to_pos_bonus(self, pos: str, role_ids: torch.Tensor) -> float:
        role_mean = role_ids.float().mean().item()
        if "v" in pos and role_mean < 3:
            return 2.0
        return 0.0

    def get_definitions(self, word: str) -> List[str]:
        return self.definitions.get(word.strip().lower(), [])

    def __len__(self) -> int:
        return self.unique_lemmas

    def __repr__(self) -> str:
        status = "loaded" if self.loaded else "empty"
        return f"KBBIStore({status}, entries={self.entry_count}, lemmas={self.unique_lemmas}, path='{self.kbbi_path}', pos_freq={len(self.pos_freq)})"


class LapisanSemantikKBBI(nn.Module):
    def __init__(self, config: LSKConfig, root_vocab: Dict[str, int]):
        super().__init__()
        self.config = config
        self.root_vocab = root_vocab
        self.root_vocab_inv = {v: k for k, v in root_vocab.items()}
        self.kbbi_store = KBBIStore(kbbi_path=config.kbbi_path, max_lemmas=config.max_lemmas)
        self._build_vocab_kbbi_map()
        num_kbbi_entries = max(self.kbbi_store.unique_lemmas + 1, 2)
        self.kbbi_embeddings = nn.Embedding(
            num_embeddings=num_kbbi_entries,
            embedding_dim=config.kbbi_vector_dim,
            padding_idx=0,
        )
        self.proj = nn.Linear(config.kbbi_vector_dim, config.kbbi_vector_dim)
        self._sem_dim: Optional[int] = None
        self._pretrained_loaded = self._load_pretrained_embeddings()
        if not self._pretrained_loaded:
            nn.init.xavier_uniform_(self.kbbi_embeddings.weight[1:])
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        if config.freeze_embeddings:
            self.kbbi_embeddings.weight.requires_grad_(False)
        self._total_tokens: int = 0
        self._kbbi_hits: int = 0

    def _load_pretrained_embeddings(self) -> bool:
        path = self.config.pretrained_path
        if not path or not os.path.exists(path):
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning("Gagal load pre-trained embedding dari %s: %s", path, e)
            return False
        pretrained = payload.get("embeddings")
        if pretrained is None:
            logger.warning("Key 'embeddings' tidak ditemukan di '%s'", path)
            return False
        if pretrained.shape[1] != self.config.kbbi_vector_dim:
            logger.warning(
                "Dimensi pre-trained (%s) != kbbi_vector_dim (%s)",
                pretrained.shape[1],
                self.config.kbbi_vector_dim,
            )
            return False
        n_needed = self.kbbi_embeddings.weight.shape[0]
        n_have = pretrained.shape[0]
        if n_have < n_needed:
            logger.warning("Pre-trained punya %s baris, butuh %s", n_have, n_needed)
            return False
        with torch.no_grad():
            self.kbbi_embeddings.weight.copy_(pretrained[:n_needed])
            self.kbbi_embeddings.weight[0].zero_()
        return True

    def _build_vocab_kbbi_map(self):
        vocab_size = max(self.root_vocab.values()) + 1 if self.root_vocab else 1
        mapping = torch.zeros(vocab_size, dtype=torch.long)
        hits = 0
        total = 0
        for word, morpheme_id in self.root_vocab.items():
            if word in ("<PAD>", "<UNK>", "<BOS>", "<EOS>"):
                continue
            total += 1
            kbbi_id = self.kbbi_store.lookup_exact(word)
            if kbbi_id > 0:
                mapping[morpheme_id] = kbbi_id
                hits += 1
        self._initial_coverage = hits / max(total, 1)
        self.register_buffer("vocab_kbbi_map", mapping)

    def forward(self, morpheme_ids: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        max_id = self.vocab_kbbi_map.size(0) - 1
        safe_ids = morpheme_ids.clamp(0, max_id)
        kbbi_ids = self.vocab_kbbi_map[safe_ids]
        self._total_tokens += morpheme_ids.numel()
        self._kbbi_hits += (kbbi_ids > 0).sum().item()
        embeddings = self.kbbi_embeddings(kbbi_ids)
        if return_raw:
            return embeddings
        return self.proj(embeddings)

    def get_anchors(self, morpheme_ids: torch.Tensor) -> torch.Tensor:
        max_id = self.vocab_kbbi_map.size(0) - 1
        safe_ids = morpheme_ids.clamp(0, max_id)
        kbbi_ids = self.vocab_kbbi_map[safe_ids]
        return self.kbbi_embeddings(kbbi_ids)

    def set_sem_dim(self, d_sem: int):
        self._sem_dim = d_sem

    def get_anchors_to_sem(self, morpheme_ids: torch.Tensor) -> torch.Tensor:
        raw = self.get_anchors(morpheme_ids)
        d_kbbi = raw.shape[-1]
        d_target = self._sem_dim if self._sem_dim is not None else d_kbbi
        if d_target == d_kbbi:
            return raw
        if d_target < d_kbbi:
            return raw[..., :d_target]
        repeats = (d_target + d_kbbi - 1) // d_kbbi
        tiled = raw.repeat(1, 1, repeats)
        return tiled[..., :d_target]

    @property
    def kbbi_coverage(self) -> float:
        if self._total_tokens > 0:
            return self._kbbi_hits / self._total_tokens
        return self._initial_coverage

    def reset_coverage_stats(self):
        self._total_tokens = 0
        self._kbbi_hits = 0

    def get_stats(self) -> Dict:
        return {
            "kbbi_loaded": self.kbbi_store.loaded,
            "kbbi_entries": self.kbbi_store.entry_count,
            "kbbi_unique_lemmas": self.kbbi_store.unique_lemmas,
            "vocab_size": len(self.root_vocab),
            "initial_coverage": self._initial_coverage,
            "runtime_coverage": self.kbbi_coverage,
            "total_tokens_seen": self._total_tokens,
            "kbbi_hits": self._kbbi_hits,
            "embedding_dim": self.config.kbbi_vector_dim,
            "frozen": self.config.freeze_embeddings,
        }

    def __repr__(self) -> str:
        return f"LapisanSemantikKBBI(store={self.kbbi_store}, dim={self.config.kbbi_vector_dim}, coverage={self.kbbi_coverage:.1%})"
