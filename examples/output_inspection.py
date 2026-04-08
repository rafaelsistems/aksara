"""
AKSARA — Output Inspection (Autoregressive Edition)

Tujuan: Validasi generation autoregressive baru.

New Tests:
  1. Autoregressive vs single-pass comparison
  2. Fluency score (L_fluency)
  3. Sequence stability (tidak runtuh di tengah)
  4. Length control (stop dengan benar)
  5. Semantic coherence check

Cara jalankan:
    python examples/output_inspection.py --autoregressive
"""

import argparse
import sys
