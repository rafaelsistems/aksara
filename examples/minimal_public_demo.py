"""
minimal_public_demo.py
======================
Contoh publik minimal untuk verifikasi instansiasi, generasi, dan reasoning AKSARA.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aksara.framework import AksaraFramework
from aksara.config import AksaraConfig


def main() -> None:
    kbbi_path = Path("kbbi_core_v2.json")
    if not kbbi_path.exists():
        raise FileNotFoundError("kbbi_core_v2.json tidak tersedia di folder publik ini")

    fw = AksaraFramework.dari_kbbi(str(kbbi_path), config=AksaraConfig.default())

    kalimat = "Budi membeli beras di pasar. Dia membayar dengan tunai."
    state = fw.proses(kalimat)

    print("KALIMAT    :", kalimat)
    print("RINGKASAN  :", state.ringkasan())
    print("JELASKAN   :", state.jelaskan())
    if state.krl_result is not None:
        print("KRL        :", state.krl_result.jelaskan())


if __name__ == "__main__":
    main()
