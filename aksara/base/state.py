"""
AksaraState — struktur data output dari pipeline AKSARA Framework.

Bukan tensor blackbox seperti hidden state Transformer.
Setiap field punya makna linguistik eksplisit.

Hierarki output:
  MorfemState          — state satu morfem (unit terkecil)
  PelanggaranConstraint — satu pelanggaran dengan token terlibat
  ViolationSpan        — lokasi presisi pelanggaran di teks asli
  ConstraintSatisfaction— ringkasan per-dimensi constraint
  AksaraState          — state lengkap kalimat (output pipeline)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import torch


@dataclass
class MorfemState:
    """
    State satu morfem setelah diproses oleh pipeline AKSARA.

    Bukan embedding vektor statis — ini adalah state dinamis
    yang berisi informasi linguistik eksplisit.
    """
    indeks: int
    teks: str
    root: str
    afiks: List[str]
    kelas_kata: str
    peran_gramatikal: str

    posisi_manifold: Optional[torch.Tensor] = None
    ketegangan: Optional[Dict[str, float]] = None
    pelanggaran: Optional[List[str]] = None


@dataclass
class PelanggaranConstraint:
    """
    Satu pelanggaran constraint yang terdeteksi.

    Setiap keputusan harus bisa dijelaskan — bukan blackbox.
    """
    tipe: str
    token_terlibat: List[str]
    dimensi: str
    severitas: float
    penjelasan: str


@dataclass
class ViolationSpan:
    """
    Lokasi presisi satu pelanggaran di teks asli — untuk violation localization.

    Berbeda dari PelanggaranConstraint yang menyebut root morfem,
    ViolationSpan menyebut posisi karakter di teks asli agar
    developer bisa highlight teks secara tepat.

    Contoh:
      teks  = "Dia pergi ke pasar untuk membeli senapan."
      span  = ViolationSpan(mulai=33, akhir=40, token="senapan",
                            dimensi="semantik", severitas=0.65,
                            penjelasan="domain senjata ≠ domain ekonomi")
    """
    mulai: int           # indeks karakter awal di teks_asli (inklusif)
    akhir: int           # indeks karakter akhir di teks_asli (eksklusif)
    token: str           # token teks asli yang melanggar
    root: str            # root morfem yang melanggar
    dimensi: str         # "morfologis" | "sintaktis" | "semantik" | "leksikal" | "topologis"
    severitas: float     # [0, 1] — 1 = pelanggaran maksimal
    penjelasan: str      # penjelasan dalam bahasa Indonesia


@dataclass
class ConstraintSatisfaction:
    """
    Ringkasan constraint satisfaction per dimensi untuk satu kalimat.

    Formula per dimensi:
      satisfaction = 1.0 - (energi_dimensi / energi_maks_teoritis)
      di mana energi_maks_teoritis = jumlah edge × severitas_maks (1.0)

    Interpretasi:
      1.0 = semua constraint di dimensi ini terpenuhi
      0.0 = semua constraint di dimensi ini dilanggar
    """
    morfologis: float = 1.0
    sintaktis: float  = 1.0
    semantik: float   = 1.0
    leksikal: float   = 1.0
    topologis: float  = 1.0
    animasi: float    = 1.0

    @property
    def rata_rata(self) -> float:
        return (self.morfologis + self.sintaktis + self.semantik
                + self.leksikal + self.topologis + self.animasi) / 6.0

    @property
    def terlemah(self) -> Tuple[str, float]:
        """Dimensi dengan satisfaction paling rendah."""
        dims = {
            "morfologis": self.morfologis,
            "sintaktis":  self.sintaktis,
            "semantik":   self.semantik,
            "leksikal":   self.leksikal,
            "topologis":  self.topologis,
            "animasi":    self.animasi,
        }
        k = min(dims, key=lambda d: dims[d])
        return k, dims[k]

    def ke_dict(self) -> Dict[str, float]:
        return {
            "morfologis": self.morfologis,
            "sintaktis":  self.sintaktis,
            "semantik":   self.semantik,
            "leksikal":   self.leksikal,
            "topologis":  self.topologis,
            "animasi":    self.animasi,
            "rata_rata":  self.rata_rata,
        }


@dataclass
class AksaraState:
    """
    State linguistik lengkap sebuah kalimat setelah diproses pipeline AKSARA.

    Ini adalah output primitif yang siap dikonsumsi oleh head apapun
    yang developer definisikan sendiri.

    Berbeda dari hidden state Transformer:
    - Setiap field punya makna linguistik eksplisit
    - Pelanggaran constraint tercatat beserta penjelasannya
    - Energi sistem terukur dan dapat diinterpretasi
    - Tidak ada dimensi yang "tersembunyi" tanpa makna
    - Violation spans memberi lokasi presisi di teks asli
    - skor_linguistik adalah satu angka yang bisa langsung dipakai head
    """

    teks_asli: str
    morfem_states: List[MorfemState]

    energi_total: float = 0.0
    energi_per_dimensi: Dict[str, float] = field(default_factory=dict)

    pelanggaran: List[PelanggaranConstraint] = field(default_factory=list)
    violation_spans: List[ViolationSpan] = field(default_factory=list)

    constraint_satisfaction: Optional[ConstraintSatisfaction] = None

    fitur_topologis: Optional[torch.Tensor] = None
    anomali_topologis: bool = False

    register: Optional[str] = None
    kelengkapan_struktur: float = 1.0

    # KRL — Knowledge Representation Layer (Primitif 6)
    # None jika KRL tidak diaktifkan atau belum diproses
    krl_result: Optional[object] = None   # type: KRLResult (hindari circular import)

    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def n_morfem(self) -> int:
        return len(self.morfem_states)

    @property
    def valid(self) -> bool:
        return self.energi_total < 0.5 and not self.anomali_topologis

    @property
    def ada_pelanggaran(self) -> bool:
        return len(self.pelanggaran) > 0

    @property
    def skor_linguistik(self) -> float:
        """
        Skor linguistik tunggal [0, 1] — makin tinggi makin valid.

        Formula (sesuai AKSARA_PRIMITIVES_MATH.md §6.2):
          skor = σ(-E) × w_energy
               + sat.rata_rata × w_sat
               + (1 - topo_penalty) × w_tda

        Bobot default:
          w_energy = 0.5  (energi CPE adalah sinyal terkuat)
          w_sat    = 0.3  (constraint satisfaction agregat)
          w_tda    = 0.2  (anomali topologis sebagai tiebreaker)

        Catatan: ini bukan output probabilistik Transformer —
        ini adalah fungsi deterministik dari energi constraint.
        """
        W_ENERGY = 0.5
        W_SAT    = 0.3
        W_TDA    = 0.2

        # Komponen 1: sigmoid(-E) — energi tinggi → skor rendah
        e_clamp = max(0.0, min(self.energi_total, 5.0))
        skor_energi = 1.0 / (1.0 + math.exp(e_clamp))  # σ(-E), E positif

        # Komponen 2: constraint satisfaction rata-rata
        sat = self.constraint_satisfaction
        skor_sat = sat.rata_rata if sat is not None else (1.0 - e_clamp / 5.0)

        # Komponen 3: topological coherence
        topo_penalty = 0.3 if self.anomali_topologis else 0.0
        skor_topo = max(0.0, 1.0 - topo_penalty)

        raw = (W_ENERGY * skor_energi
               + W_SAT   * skor_sat
               + W_TDA   * skor_topo)

        # Penalti per pelanggaran berat (severitas > 0.5)
        # Setiap pelanggaran berat mengurangi skor 0.08 — mencegah kalimat
        # dengan banyak pelanggaran morfologis/semantik lolos threshold.
        # Cap penalti di 0.4 agar skor tidak pernah negatif.
        n_berat = sum(1 for p in self.pelanggaran if p.severitas > 0.5)
        penalti_berat = min(0.4, n_berat * 0.08)
        raw = max(0.0, raw - penalti_berat)

        # Koreksi kelengkapan struktur: kalimat tidak lengkap diturunkan
        raw *= (0.5 + 0.5 * self.kelengkapan_struktur)

        return max(0.0, min(1.0, raw))

    def pelanggaran_per_dimensi(self, dimensi: str) -> List[PelanggaranConstraint]:
        return [p for p in self.pelanggaran if p.dimensi == dimensi]

    def spans_per_dimensi(self, dimensi: str) -> List[ViolationSpan]:
        return [s for s in self.violation_spans if s.dimensi == dimensi]

    def token_bermasalah(self) -> List[str]:
        """Daftar token (teks asli) yang terlibat dalam pelanggaran, deduplikasi."""
        seen = set()
        result = []
        for span in self.violation_spans:
            if span.token not in seen:
                seen.add(span.token)
                result.append(span.token)
        return result

    def ringkasan(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        n_p = len(self.pelanggaran)
        return (
            f"[{status}] skor={self.skor_linguistik:.3f} "
            f"energi={self.energi_total:.3f} "
            f"pelanggaran={n_p} "
            f"morfem={self.n_morfem}"
        )

    def jelaskan(self) -> str:
        """
        Penjelasan lengkap dalam bahasa Indonesia — mengapa kalimat ini valid/invalid.
        Ini adalah keunggulan utama AKSARA vs Transformer: setiap keputusan bisa dijelaskan.
        """
        lines = [f'Kalimat: "{self.teks_asli}"']
        lines.append(f"Skor linguistik : {self.skor_linguistik:.3f}")
        lines.append(f"Status          : {'VALID' if self.valid else 'TIDAK VALID'}")
        lines.append(f"Energi total    : {self.energi_total:.4f}")
        lines.append(f"Register        : {self.register or 'tidak diketahui'}")
        lines.append(f"Kelengkapan     : {self.kelengkapan_struktur:.0%}")

        if self.constraint_satisfaction:
            sat = self.constraint_satisfaction
            lines.append("\nConstraint Satisfaction:")
            for dim, val in sat.ke_dict().items():
                if dim == "rata_rata":
                    continue
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                lines.append(f"  {dim:<12} [{bar}] {val:.2f}")
            dim_lemah, val_lemah = sat.terlemah
            if val_lemah < 0.8:
                lines.append(f"  ⚠ Terlemah: {dim_lemah} ({val_lemah:.2f})")

        if self.pelanggaran:
            lines.append(f"\nPelanggaran ({len(self.pelanggaran)}):")
            for p in self.pelanggaran:
                lines.append(f"  [{p.dimensi}] sev={p.severitas:.2f} — {p.penjelasan}")

        if self.violation_spans:
            lines.append(f"\nLokasi Pelanggaran di Teks:")
            for s in self.violation_spans:
                snippet = self.teks_asli[s.mulai:s.akhir]
                lines.append(
                    f"  posisi [{s.mulai}:{s.akhir}] '{snippet}' "
                    f"({s.dimensi}, sev={s.severitas:.2f})"
                )
                lines.append(f"  → {s.penjelasan}")

        if self.anomali_topologis:
            lines.append("\n⚠ Anomali topologis terdeteksi — struktur relasi tidak koheren.")

        return "\n".join(lines)
