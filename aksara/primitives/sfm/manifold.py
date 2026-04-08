"""
SemanticManifold — representasi kata sebagai state dinamis di Riemannian manifold.

OPOSISI TRANSFORMER:
  Transformer: embedding = vektor statis di Euclidean space ℝⁿ (belajar dari statistik)
  SFM:         representasi = distribusi di Riemannian manifold (dibangun dari KBBI)

Perbedaan fundamental:
  Euclidean embedding: kata adalah TITIK — jarak = norma vektor
  Riemannian manifold: kata adalah DISTRIBUSI — jarak = geodesic melewati konsep antara

Properti SFM:
  1. Context-sensitive: state kata berubah sesuai konteks kalimat
  2. Grounded: representasi dibangun dari relasi KBBI, bukan statistik distribusi token
  3. Interpretable: setiap dimensi punya makna (domain, register, relasi)
  4. Updateable: tambah kata baru = tambah ke leksikon, tidak perlu retrain
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from aksara.primitives.sfm.lexicon import LexiconLoader, EntriLeksikon
from aksara.primitives.sfm.geodesic import GeodesicDistance
from aksara.primitives.lps.morfem import Morfem, KelasKata


# ── Domain canonical untuk dimensi manifold ───────────────────────────────────
# Setiap domain menjadi satu dimensi dalam representasi medan semantik.
# Ini berbeda dari embedding arbitrary — setiap dimensi punya makna eksplisit.

DOMAIN_CANONICAL = [
    "kuliner", "kendaraan", "bangunan", "senjata", "busana",
    "pendidikan", "ekonomi", "kesehatan", "hukum", "seni",
    "olahraga", "teknologi", "alam", "sosial", "agama",
    "pemerintahan", "ilmu", "budaya", "bahasa", "waktu",
]

KELAS_CANONICAL = [
    "n", "v", "a", "adv", "num", "pron", "prep", "konj", "part",
]

REGISTER_FORMAL    = 1.0
REGISTER_INFORMAL  = 0.0
REGISTER_NETRAL    = 0.5


class SemanticState:
    """
    State semantik satu kata dalam konteks kalimat.

    Bukan tensor blackbox — setiap komponen punya makna linguistik eksplisit.
    """
    __slots__ = [
        "kata", "domain_vec", "kelas_vec", "register",
        "ada_di_leksikon", "ketidakpastian",
        "posisi_tensor",
    ]

    def __init__(
        self,
        kata: str,
        domain_vec: torch.Tensor,
        kelas_vec: torch.Tensor,
        register: float,
        ada_di_leksikon: bool,
        ketidakpastian: float,
    ):
        self.kata = kata
        self.domain_vec = domain_vec
        self.kelas_vec = kelas_vec
        self.register = register
        self.ada_di_leksikon = ada_di_leksikon
        self.ketidakpastian = ketidakpastian
        self.posisi_tensor: Optional[torch.Tensor] = None

    @property
    def vektor_lengkap(self) -> torch.Tensor:
        """
        Vektor representasi lengkap — konkatenasi semua komponen bermakna.
        Berbeda dari embedding arbitrary: setiap segmen punya makna eksplisit.
        """
        register_t = torch.tensor([self.register], dtype=torch.float32)
        pasti_t    = torch.tensor([1.0 - self.ketidakpastian], dtype=torch.float32)
        return torch.cat([self.domain_vec, self.kelas_vec, register_t, pasti_t])

    @property
    def dim(self) -> int:
        return len(DOMAIN_CANONICAL) + len(KELAS_CANONICAL) + 2


class SemanticManifold(nn.Module):
    """
    Semantic Field Manifold — representasi semantik native Indonesia.

    OPOSISI TRANSFORMER:
      Transformer: nn.Embedding(vocab_size, d_model) — vektor statis, dilatih dari
                   statistik distribusi token, setiap dimensi tidak bermakna.

      SFM: representasi dibangun dari struktur KBBI:
           - Domain    [20 dim]: posisi di ruang medan semantik (kuliner, senjata, dst)
           - Kelas kata [ 9 dim]: morfem role (nomina, verba, adjektiva, dst)
           - Register  [ 1 dim]: formal (1.0) / informal (0.0) / netral (0.5)
           - Kepastian [ 1 dim]: 1 - ketidakpastian semantik kata
           Total: 31 dimensi, setiap dimensi punya nama dan interpretasi linguistik.

    Prinsip:
      Representasi dibangun dari KBBI, bukan random init.
      Tidak ada komponen learned — tidak perlu training.
      Updateable: tambah kata = tambah ke leksikon, efektif langsung.

    Output encode_kalimat(): tensor (n_morfem, 31) — sinyal geometrik
      untuk CPE. Cosine similarity antar baris = kedekatan semantik dua morfem
      di ruang linguistik KBBI, bukan di ruang embedding statistik.
    """

    def __init__(
        self,
        leksikon: LexiconLoader,
    ):
        super().__init__()
        self.leksikon  = leksikon
        self.geodesic  = GeodesicDistance(leksikon)
        # Dict ringkas {kata: kelas} untuk dipakai LPSParser
        self.leksikon_dict: Dict[str, str] = {
            k: v.kelas for k, v in leksikon._entri.items()
        }

        # Dimensi representasi: setiap dimensi punya nama linguistik
        self.d_linguistik = len(DOMAIN_CANONICAL) + len(KELAS_CANONICAL) + 2
        # d_output = d_linguistik — tidak ada residual learned

        # Index untuk lookup cepat
        self._domain_idx = {d: i for i, d in enumerate(DOMAIN_CANONICAL)}
        self._kelas_idx  = {k: i for i, k in enumerate(KELAS_CANONICAL)}

        self._cache: Dict[str, SemanticState] = {}

    @classmethod
    def dari_kbbi(
        cls,
        kbbi_path: str,
    ) -> "SemanticManifold":
        """Factory: bangun manifold langsung dari file KBBI."""
        leksikon = LexiconLoader()
        leksikon.muat_kbbi(kbbi_path)
        return cls(leksikon)

    def encode_morfem(self, morfem: Morfem) -> SemanticState:
        """
        Encode satu morfem menjadi SemanticState.

        Berbeda dari embedding lookup Transformer:
        - Tidak hanya lookup tabel → bangun representasi dari properti linguistik
        - State mencerminkan posisi kata di ruang medan semantik KBBI
        """
        kata = morfem.root.lower()
        if kata in self._cache:
            return self._cache[kata]

        entri = self.leksikon.cari(kata)

        domain_vec = self._bangun_domain_vec(kata, entri)
        kelas_vec  = self._bangun_kelas_vec(morfem, entri)
        register   = self._inferensi_register(morfem, entri)
        ketidakpastian = self._hitung_ketidakpastian(kata, entri)

        state = SemanticState(
            kata=kata,
            domain_vec=domain_vec,
            kelas_vec=kelas_vec,
            register=register,
            ada_di_leksikon=entri is not None,
            ketidakpastian=ketidakpastian,
        )
        self._cache[kata] = state
        return state

    def encode_kalimat(
        self,
        morfem_list: List[Morfem],
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Encode seluruh kalimat menjadi tensor representasi linguistik.

        Returns:
            Tensor shape (n_morfem, d_linguistik) — setiap baris = satu morfem
            Setiap dimensi punya makna eksplisit:
              [0:20]  domain_vec   — keanggotaan di 20 domain KBBI
              [20:29] kelas_vec    — kelas kata (N, V, Adj, Adv, ...)
              [29]    register     — formal/informal/netral
              [30]    kepastian    — 1 - ketidakpastian semantik

        Dipakai oleh CPE untuk menghitung cosine similarity geometrik antar
        morfem sebagai sinyal tambahan — bukan menggantikan geodesic distance.
        Tidak ada komponen learned: representasi sepenuhnya dari KBBI.
        """
        states = [self.encode_morfem(m) for m in morfem_list]
        return torch.stack(
            [s.vektor_lengkap for s in states]
        ).to(device)  # (n_morfem, d_linguistik)

    def jarak_semantik(self, morfem_a: Morfem, morfem_b: Morfem) -> float:
        """
        Hitung jarak semantik antara dua morfem.
        Menggunakan geodesic distance, bukan Euclidean distance.
        """
        return self.geodesic.hitung(morfem_a.root, morfem_b.root)

    def kompatibel(
        self,
        morfem_a: Morfem,
        morfem_b: Morfem,
        threshold: float = 1.5,
    ) -> Tuple[bool, str]:
        """
        Apakah dua morfem semantically compatible?

        Returns:
            (compatible, penjelasan)
        """
        d = self.jarak_semantik(morfem_a, morfem_b)
        if d <= threshold:
            path = self.geodesic.path_semantik(morfem_a.root, morfem_b.root)
            return True, f"Kompatibel (jarak={d:.2f}, path={' → '.join(path)})"
        else:
            domain_a = self.leksikon.domain_kata(morfem_a.root) or "?"
            domain_b = self.leksikon.domain_kata(morfem_b.root) or "?"
            return False, (
                f"TIDAK kompatibel (jarak={d:.2f}): "
                f"domain '{morfem_a.root}'={domain_a} "
                f"≠ domain '{morfem_b.root}'={domain_b}"
            )

    # ── Private: bangun komponen representasi ────────────────────────────────

    def _bangun_domain_vec(
        self, kata: str, entri: Optional[EntriLeksikon]
    ) -> torch.Tensor:
        """
        Bangun vektor domain — setiap dimensi = keanggotaan di satu domain semantik.
        Ini bukan one-hot — kata bisa punya keanggotaan parsial di beberapa domain.
        """
        vec = torch.zeros(len(DOMAIN_CANONICAL))

        if entri and entri.domain:
            dom = entri.domain.lower()
            # Cari domain yang paling cocok
            for canonical in DOMAIN_CANONICAL:
                if canonical in dom or dom in canonical:
                    idx = self._domain_idx[canonical]
                    vec[idx] = 1.0
                    break
            # Keanggotaan parsial dari domain tetangga via relasi
            for rel_kata in (entri.sinonim + entri.hiponim)[:3]:
                rel_entri = self.leksikon.cari(rel_kata)
                if rel_entri and rel_entri.domain:
                    for canonical in DOMAIN_CANONICAL:
                        if canonical in rel_entri.domain.lower():
                            idx = self._domain_idx[canonical]
                            vec[idx] = max(vec[idx], 0.3)

        # Normalisasi ke [0, 1]
        if vec.sum() > 0:
            vec = vec / vec.sum()
        return vec

    def _bangun_kelas_vec(
        self, morfem: Morfem, entri: Optional[EntriLeksikon]
    ) -> torch.Tensor:
        """
        Bangun vektor kelas kata.
        Kelas kata dari LPS lebih reliabel dari leksikon untuk kata berimbuhan.
        """
        vec = torch.zeros(len(KELAS_CANONICAL))
        kelas_str = morfem.kelas_kata.value.lower()

        mapping = {
            "n": "n", "v": "v", "adj": "a", "adjektiva": "a",
            "adv": "adv", "adverbia": "adv",
            "num": "num", "pron": "pron",
            "prep": "prep", "konjungsi": "konj", "konj": "konj",
            "partikel": "part", "part": "part",
        }
        canonical_kelas = mapping.get(kelas_str, None)
        if canonical_kelas and canonical_kelas in self._kelas_idx:
            vec[self._kelas_idx[canonical_kelas]] = 1.0
        elif entri:
            pos = entri.kelas.lower()
            canonical = mapping.get(pos, None)
            if canonical and canonical in self._kelas_idx:
                vec[self._kelas_idx[canonical]] = 0.8

        return vec

    def _inferensi_register(
        self, morfem: Morfem, entri: Optional[EntriLeksikon]
    ) -> float:
        """
        Inferensi register kata: formal (1.0) vs informal (0.0).
        Ini adalah dimensi yang tidak ada di Transformer biasa.
        """
        if morfem.adalah_informal:
            return REGISTER_INFORMAL
        if morfem.adalah_serapan:
            return REGISTER_NETRAL
        if morfem.adalah_proper:
            return REGISTER_FORMAL
        if entri is not None:
            return REGISTER_FORMAL
        return REGISTER_NETRAL

    def _hitung_ketidakpastian(
        self, kata: str, entri: Optional[EntriLeksikon]
    ) -> float:
        """
        Hitung ketidakpastian semantik kata.
        Kata polisemik atau tidak ada di leksikon → ketidakpastian tinggi.
        """
        if entri is None:
            return 0.8  # tidak di leksikon → sangat tidak pasti
        # Kata dengan banyak relasi → lebih polisemik
        n_relasi = len(entri.sinonim) + len(entri.hiponim)
        if n_relasi > 10:
            return 0.6
        if n_relasi > 5:
            return 0.4
        return 0.2  # kata spesifik, sedikit relasi → pasti

    @property
    def d_output(self) -> int:
        return self.d_linguistik
