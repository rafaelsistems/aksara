"""
KRL — Knowledge Representation Layer (Primitif 6 AKSARA Framework).

Menjembatani gap antara validasi linguistik (LPS/CPE/CMC)
dan pemahaman makna proporsional.
"""

from aksara.primitives.krl.proposition import Proposisi, SlotProposisi, TipeSlot
from aksara.primitives.krl.frame import Frame, SlotFrame, FrameBank
from aksara.primitives.krl.encoder import PropositionalEncoder
from aksara.primitives.krl.matcher import FrameMatcher
from aksara.primitives.krl.resolver import ReferenceResolver
from aksara.primitives.krl.layer import KRLayer, KRLResult

__all__ = [
    "Proposisi", "SlotProposisi", "TipeSlot",
    "Frame", "SlotFrame", "FrameBank",
    "PropositionalEncoder",
    "FrameMatcher",
    "ReferenceResolver",
    "KRLayer", "KRLResult",
]
