"""
aksara.heads — Head-head yang dibangun di atas AKSARA Framework.

Setiap head adalah implementasi konkret dari AksaraHead untuk task spesifik.
Head mengkonsumsi AksaraState dari pipeline primitif dan menghasilkan output
sesuai kebutuhan developer.

Head yang tersedia:
  CorrectnessEvaluatorHead  — evaluasi kebenaran kalimat (deterministik)
  LearnedCorrectnessHead    — evaluasi dengan parameter yang bisa di-fine-tune
"""

from aksara.heads.correctness import (
    CorrectnessEvaluatorHead,
    LearnedCorrectnessHead,
    HasilEvaluasi,
)

__all__ = [
    "CorrectnessEvaluatorHead",
    "LearnedCorrectnessHead",
    "HasilEvaluasi",
]
