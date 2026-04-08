"""
mini_cli.py
===========
CLI publik ringkas untuk verifikasi cepat model AKSARA.
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
    kalimat = "Saya pergi ke pasar."
    state = fw.proses(kalimat)

    print(state.ringkasan())
    print(state.jelaskan())


if __name__ == "__main__":
    main()
