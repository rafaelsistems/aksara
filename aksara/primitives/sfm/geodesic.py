"""
GeodesicDistance — jarak semantik berbasis geodesic di Riemannian manifold.

OPOSISI TRANSFORMER:
  Transformer: jarak semantik = dot product vektor di Euclidean space (satu skala)
  SFM:         jarak semantik = geodesic di Riemannian manifold (melewati path bermakna)

Geodesic berbeda dari Euclidean:
  Euclidean("raja", "ratu") = ||v_raja - v_ratu|| — tidak melewati konsep antara
  Geodesic("raja", "ratu")  = path melalui "pemimpin" → "kerajaan" → "kekuasaan"
                              melewati konsep-konsep yang semantically relevant
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from aksara.primitives.sfm.lexicon import LexiconLoader


class GeodesicDistance:
    """
    Jarak semantik di ruang medan domain KBBI.

    OPOSISI TRANSFORMER:
    - Transformer: cosine_similarity(embed_a, embed_b) — Euclidean, statistik
    - GeodesicDistance: domain-based semantic distance — linguistik, deterministik

    Karena KBBI v2 hanya punya {lemma, pos, definisi}, relasi semantik
    dibangun dari:
      1. Domain yang sama (dari ekstraksi definisi) → jarak sangat dekat
      2. POS yang kompatibel → jarak sedang
      3. Domain berbeda yang diketahui → jarak jauh
      4. Salah satu/keduanya tidak diketahui → jarak menengah (open world)

    Semua keputusan punya justifikasi linguistik eksplisit — bukan blackbox.
    """

    # Matriks jarak antar domain — asimetri bisa dikodekan jika perlu
    DOMAIN_DISTANCE: Dict[Tuple[str, str], float] = {}

    # Jarak berdasarkan tipe relasi domain
    JARAK_SAMA_DOMAIN:      float = 0.2   # domain identik
    JARAK_DOMAIN_TERKAIT:   float = 0.8   # domain berkaitan (kuliner-kesehatan)
    JARAK_DOMAIN_BERBEDA:   float = 2.5   # domain jauh berbeda
    JARAK_SATU_TIDAK_DIKETAHUI: float = 1.2  # salah satu tanpa domain
    JARAK_KEDUANYA_TIDAK_DIKETAHUI: float = 1.0  # keduanya tanpa domain (netral)

    # Grup domain yang berkaitan — jarak lebih dekat antar sesama grup
    # Justifikasi linguistik: domain yang masuk grup sama = saling berkaitan
    # secara pragmatik dalam kehidupan sehari-hari (TBBBI, konteks pemakaian)
    GRUP_DOMAIN: Dict[str, str] = {
        # Grup AKTIVITAS SIPIL: ekonomi dan kuliner saling berkaitan
        # (belanja bahan makanan = aktivitas ekonomi yang paling umum)
        "kuliner":    "aktivitas_sipil",
        "ekonomi":    "aktivitas_sipil",
        "busana":     "aktivitas_sipil",
        # Grup SOSIAL-KELEMBAGAAN
        "hukum":      "sosial",
        "pendidikan": "sosial",
        # Grup INFRASTRUKTUR
        "bangunan":   "infrastruktur",
        "kendaraan":  "infrastruktur",
        # Grup KESEHATAN
        "kesehatan":  "kesehatan",
        # Grup SENI
        "alat_musik": "seni",
        # Grup KONFLIK — sengaja terpisah dari semua grup sipil
        "senjata":    "konflik",
        "militer":    "konflik",
    }

    # Jarak eksplisit antar pasangan domain yang punya relasi khusus
    # Ini meng-override kalkulasi grup — lebih presisi
    DOMAIN_DISTANCE_EKSPLISIT: Dict[Tuple[str, str], float] = {
        # ekonomi ↔ kuliner: sangat dekat (belanja makanan = aktivitas ekonomi)
        ("ekonomi", "kuliner"):    0.6,
        ("kuliner", "ekonomi"):    0.6,
        # ekonomi ↔ senjata: jauh (pembelian senjata bukan aktivitas sipil umum)
        ("ekonomi", "senjata"):    2.8,
        ("senjata", "ekonomi"):    2.8,
        # kuliner ↔ senjata: sangat jauh (tidak ada hubungan)
        ("kuliner", "senjata"):    3.0,
        ("senjata", "kuliner"):    3.0,
        # ekonomi ↔ sosial: cukup dekat
        ("ekonomi", "hukum"):      1.5,
        ("ekonomi", "pendidikan"): 1.5,
    }

    def __init__(self, leksikon: LexiconLoader, max_hop: int = 4):
        self.leksikon = leksikon
        self.max_hop  = max_hop
        self._cache: Dict[Tuple[str, str], float] = {}

    def hitung(self, kata_a: str, kata_b: str) -> float:
        """
        Hitung jarak semantik antara dua kata.

        Returns:
            float jarak ∈ [0, ~3.0]
            dekat = semantically compatible
            jauh  = semantically incompatible
        """
        a = kata_a.lower()
        b = kata_b.lower()

        if a == b:
            return 0.0

        cache_key = (min(a, b), max(a, b))
        if cache_key in self._cache:
            return self._cache[cache_key]

        jarak = self._hitung_domain_based(a, b)
        self._cache[cache_key] = jarak
        return jarak

    def kompatibel(self, kata_a: str, kata_b: str, threshold: float = 1.5) -> bool:
        return self.hitung(kata_a, kata_b) <= threshold

    def domain_distance(self, kata_a: str, kata_b: str) -> float:
        return self._hitung_domain_based(kata_a.lower(), kata_b.lower())

    def path_semantik(self, kata_a: str, kata_b: str) -> List[str]:
        """
        Kembalikan path semantik — untuk interpretabilitas.
        Dengan data KBBI yang ada, path adalah [a, domain_a, domain_b, b].
        """
        a, b = kata_a.lower(), kata_b.lower()
        if a == b:
            return [a]

        domain_a = self.leksikon.domain_kata(a)
        domain_b = self.leksikon.domain_kata(b)

        if domain_a and domain_b:
            if domain_a == domain_b:
                return [a, f"[{domain_a}]", b]
            else:
                return [a, f"[{domain_a}]", "↔", f"[{domain_b}]", b]
        elif domain_a:
            return [a, f"[{domain_a}]", "→?", b]
        elif domain_b:
            return [a, "?→", f"[{domain_b}]", b]
        else:
            return [a, "[?]", b]

    # ── Private ───────────────────────────────────────────────────────────────

    def _hitung_domain_based(self, a: str, b: str) -> float:
        """
        Hitung jarak berbasis domain — deterministik, linguistik.

        Hierarki keputusan:
        1. Kata identik → 0.0
        2. Domain sama → JARAK_SAMA_DOMAIN (sangat dekat)
        3. Domain berkaitan (grup sama) → JARAK_DOMAIN_TERKAIT
        4. Domain berbeda dan keduanya diketahui → JARAK_DOMAIN_BERBEDA
        5. Salah satu/keduanya tidak diketahui → jarak menengah
        6. POS compatibility bonus
        """
        domain_a = self.leksikon.domain_kata(a)
        domain_b = self.leksikon.domain_kata(b)
        entri_a  = self.leksikon.cari(a)
        entri_b  = self.leksikon.cari(b)

        # ── Kasus 1: keduanya punya domain ─────────────────────────────────
        if domain_a and domain_b:
            # Prioritas: cek jarak eksplisit antar domain (lebih presisi dari grup)
            eksplisit = self.DOMAIN_DISTANCE_EKSPLISIT.get((domain_a, domain_b))
            if eksplisit is not None:
                pos_bonus = self._pos_bonus(entri_a, entri_b)
                return max(0.0, eksplisit - pos_bonus)

            if domain_a == domain_b:
                base = self.JARAK_SAMA_DOMAIN
            elif self._satu_grup(domain_a, domain_b):
                base = self.JARAK_DOMAIN_TERKAIT
            else:
                base = self.JARAK_DOMAIN_BERBEDA

            # Bonus: POS sama → sedikit lebih dekat
            pos_bonus = self._pos_bonus(entri_a, entri_b)
            return max(0.0, base - pos_bonus)

        # ── Kasus 2: salah satu tidak punya domain ──────────────────────────
        if domain_a or domain_b:
            return self.JARAK_SATU_TIDAK_DIKETAHUI

        # ── Kasus 3: keduanya tidak punya domain ────────────────────────────
        # Gunakan POS compatibility sebagai signal lemah
        pos_bonus = self._pos_bonus(entri_a, entri_b)
        return self.JARAK_KEDUANYA_TIDAK_DIKETAHUI - pos_bonus * 0.3

    def _satu_grup(self, domain_a: str, domain_b: str) -> bool:
        """Apakah dua domain berada dalam grup yang sama?"""
        grup_a = self.GRUP_DOMAIN.get(domain_a)
        grup_b = self.GRUP_DOMAIN.get(domain_b)
        return (grup_a is not None) and (grup_a == grup_b)

    def _pos_bonus(
        self,
        entri_a: Optional[object],
        entri_b: Optional[object],
    ) -> float:
        """
        Bonus kecil jika POS sama — kata sifat dengan kata sifat lebih compatible
        dari kata sifat dengan kata kerja.
        """
        if entri_a is None or entri_b is None:
            return 0.0
        pos_a = getattr(entri_a, "kelas", "?").lower()
        pos_b = getattr(entri_b, "kelas", "?").lower()
        if pos_a == pos_b and pos_a != "?":
            return 0.2
        # Adj + Nomina biasa dalam frasa nominal → sedikit compatible
        if set([pos_a, pos_b]) == {"adj", "n"}:
            return 0.1
        return 0.0
