"""
AKSARA — Framework linguistik native untuk bahasa Indonesia.
Adaptive Knowledge & Semantic Architecture for Representation & Autonomy

Author: Emylton Leunufna

API Utama (framework baru):
    from aksara import AksaraFramework
    fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")
    state = fw.proses("Makanan tradisional khas Dompu sangat lezat.")

Lima Primitif:
    aksara.primitives.lps  — Linguistic Parse System
    aksara.primitives.sfm  — Semantic Field Manifold
    aksara.primitives.cpe  — Constraint Propagation Engine
    aksara.primitives.cmc  — Categorical Meaning Composer
    aksara.primitives.tda  — Topological Dependency Analyzer
"""

from aksara.framework import AksaraFramework
from aksara.config import AksaraConfig
from aksara.base.state import AksaraState, MorfemState, PelanggaranConstraint
from aksara.base.head import AksaraHead

from aksara.primitives.lps import LPSParser, Morfem, KelasKata, PeranGramatikal
from aksara.primitives.sfm import SemanticManifold, LexiconLoader
from aksara.primitives.cpe import CPEngine
from aksara.primitives.cmc import CMComposer
from aksara.primitives.tda import TDAnalyzer
from aksara.primitives.krl import KRLayer, KRLResult, Proposisi, Frame, FrameBank

__version__ = "1.0.0"
__author__  = "Emylton Leunufna"

__all__ = [
    # Framework API
    "AksaraFramework", "AksaraConfig",
    # Base
    "AksaraState", "MorfemState", "PelanggaranConstraint", "AksaraHead",
    # Primitif
    "LPSParser", "Morfem", "KelasKata", "PeranGramatikal",
    "SemanticManifold", "LexiconLoader",
    "CPEngine",
    "CMComposer",
    "TDAnalyzer",
    # KRL — Knowledge Representation Layer
    "KRLayer", "KRLResult", "Proposisi", "Frame", "FrameBank",
]
