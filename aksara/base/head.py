"""
AksaraHead — base class untuk semua head yang dibangun di atas AKSARA Framework.

Developer mendefinisikan head mereka sendiri untuk task spesifik.
Framework hanya menyediakan AksaraState sebagai input — head menentukan output.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch.nn as nn

from aksara.base.state import AksaraState


class AksaraHead(ABC, nn.Module):
    """
    Base class untuk semua head yang dibangun di atas AKSARA Framework.

    Framework menjamin:
    - Input selalu berupa AksaraState yang kaya informasi linguistik
    - Setiap field AksaraState punya makna eksplisit

    Developer tentukan sendiri:
    - Objective training
    - Format output
    - Task spesifik (evaluasi, klasifikasi, retrieval, dll.)

    LARANGAN di dalam head:
    - Tidak boleh menggunakan nn.MultiheadAttention
    - Tidak boleh menggunakan TransformerEncoder/Decoder
    - Tidak boleh melakukan next-token prediction
    """

    @abstractmethod
    def forward(self, state: AksaraState) -> Dict[str, Any]:
        """
        Proses AksaraState dan kembalikan output sesuai task.

        Args:
            state: AksaraState lengkap dari pipeline AKSARA

        Returns:
            dict output — format ditentukan oleh developer
        """
        ...

    def cek_prinsip(self) -> None:
        """
        Cek apakah implementasi head melanggar prinsip AKSARA.
        Dipanggil saat inisialisasi.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                raise RuntimeError(
                    f"⚠️  PELANGGARAN PRINSIP AKSARA\n"
                    f"   Modul '{name}' menggunakan nn.MultiheadAttention.\n"
                    f"   Ini melanggar prinsip: mekanisme utama harus constraint propagation,\n"
                    f"   bukan self-attention O(n²).\n"
                    f"   Ganti dengan CPE (Constraint Propagation Engine)."
                )
            cls_name = type(module).__name__
            if "Transformer" in cls_name:
                raise RuntimeError(
                    f"⚠️  PELANGGARAN PRINSIP AKSARA\n"
                    f"   Modul '{name}' adalah komponen Transformer ({cls_name}).\n"
                    f"   AKSARA harus 100% berlawanan paradigma dengan Transformer/Mamba.\n"
                    f"   Lihat docs/AKSARA_FRAMEWORK_SPEC.md untuk alternatif."
                )
