"""LPS — Linguistic Parse System (Primitif 1 AKSARA Framework)."""
from aksara.primitives.lps.morfem import Morfem, KelasKata, PeranGramatikal
from aksara.primitives.lps.parser import LPSParser
from aksara.primitives.lps.afiks import AfiksRules

__all__ = ["Morfem", "KelasKata", "PeranGramatikal", "LPSParser", "AfiksRules"]
