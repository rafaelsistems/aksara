"""
CPEngine — Constraint Propagation Engine (Primitif 3 AKSARA Framework).

OPOSISI TRANSFORMER:
  Transformer: weighted sum via self-attention O(n²) — fixed depth layers
  CPE:         energy minimization via constraint propagation — konvergen dinamis

Mekanisme:
  1. Bangun graf dependensi dari morfem list
  2. Evaluasi constraint tiap pasangan yang bertetangga
  3. Hitung energi total sistem
  4. Iterasi sampai energi konvergen (bukan fixed N layer)
  5. Output: state kesetimbangan + skor ketegangan per dimensi + pelanggaran

Kompleksitas: O(n × avg_degree × n_constraint) per iterasi — bukan O(n²)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from aksara.primitives.lps.morfem import Morfem, KelasKata, PeranGramatikal
from aksara.primitives.sfm.manifold import SemanticManifold
from aksara.primitives.sfm.lexicon import LexiconLoader
from aksara.primitives.sfm.geodesic import GeodesicDistance
from aksara.primitives.cpe.constraint import ConstraintSet, ConstraintResult, BOBOT_CONSTRAINT
from aksara.primitives.cpe.convergence import ConvergenceChecker
from aksara.base.state import AksaraState, MorfemState, PelanggaranConstraint
from aksara.config import AksaraConfig


@dataclass
class GrafDependensi:
    """
    Graf dependensi kalimat — representasi relasi antar morfem.

    OPOSISI TRANSFORMER:
    Transformer: semua pasangan token saling attend (dense O(n²))
    GrafDependensi: hanya pasangan yang bertetangga secara linguistik (sparse)
    """
    n: int
    edge: List[Tuple[int, int, str]]  # (idx_a, idx_b, tipe_relasi)

    @classmethod
    def dari_morfem_list(cls, morfem_list: List[Morfem]) -> "GrafDependensi":
        """
        Bangun graf dependensi dari list morfem.

        Strategi: window-based adjacency + relasi gramatikal
        Hanya hubungkan morfem yang bertetangga atau punya relasi eksplisit.
        """
        n = len(morfem_list)
        edges: List[Tuple[int, int, str]] = []

        # Window adjacency: setiap morfem terhubung ke 2 tetangga kiri/kanan
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):
                edges.append((i, j, "adjacent"))

        # Relasi gramatikal: subjek-predikat, predikat-objek
        subj_idx  = next((i for i, m in enumerate(morfem_list)
                          if m.peran_gramatikal == PeranGramatikal.SUBJEK), None)
        pred_idx  = next((i for i, m in enumerate(morfem_list)
                          if m.peran_gramatikal == PeranGramatikal.PREDIKAT), None)
        obj_idx   = next((i for i, m in enumerate(morfem_list)
                          if m.peran_gramatikal == PeranGramatikal.OBJEK), None)

        if subj_idx is not None and pred_idx is not None:
            if (subj_idx, pred_idx, "subj-pred") not in edges:
                edges.append((subj_idx, pred_idx, "subj-pred"))

        if pred_idx is not None and obj_idx is not None:
            if (pred_idx, obj_idx, "pred-obj") not in edges:
                edges.append((pred_idx, obj_idx, "pred-obj"))

        # Subjek↔komplemen nomina: untuk kalimat kopulatif "X [sangat] Y"
        # di mana Y adalah komplemen nomina/adjektiva dari X.
        # Justifikasi linguistik: "Makanan sangat meriam" — makanan (subj) dan
        # meriam (komplemen) harus punya domain compatible. Tanpa edge langsung,
        # CPE tidak bisa mendeteksi mismatch ini karena "sangat" tidak punya domain.
        if subj_idx is not None:
            for k, m in enumerate(morfem_list):
                if k == subj_idx:
                    continue
                # Komplemen: nomina/adjektiva di posisi akhir kalimat yang bukan
                # sudah terhubung sebagai objek atau subjek
                if (m.kelas_kata in (KelasKata.NOMINA, KelasKata.NOMINA_SERAPAN,
                                     KelasKata.ADJEKTIVA)
                        and m.peran_gramatikal not in (
                            PeranGramatikal.SUBJEK, PeranGramatikal.OBJEK)
                        and k > subj_idx
                        and (subj_idx, k, "subj-komplemen") not in edges):
                    edges.append((subj_idx, k, "subj-komplemen"))

        # Modifier-head: adjektiva/adverbia ke kata yang dimodifikasi
        # Adverbia temporal/aspektual hanya memodifikasi verba, bukan nomina
        # Justifikasi: 'sedang membangun' = Adv+V (valid), 'sedang pemerintah' = tidak lazim
        ADV_TEMPORAL = {"sedang", "sudah", "telah", "akan", "belum", "pernah",
                        "lagi", "masih", "baru", "sudah", "langsung", "segera"}
        for i, m in enumerate(morfem_list):
            if m.peran_gramatikal == PeranGramatikal.MODIFIER:
                is_adv_temporal = (
                    m.kelas_kata == KelasKata.ADVERBIA
                    and m.root.lower() in ADV_TEMPORAL
                )
                for j in range(max(0, i - 2), min(n, i + 3)):
                    if j == i:
                        continue
                    target = morfem_list[j]
                    # Adverbia temporal hanya mau terhubung ke verba
                    if is_adv_temporal and target.kelas_kata not in (
                        KelasKata.VERBA, KelasKata.VERBA_SERAPAN
                    ):
                        continue
                    if target.kelas_kata in (
                        KelasKata.NOMINA, KelasKata.VERBA,
                        KelasKata.NOMINA_PROPER, KelasKata.NOMINA_SERAPAN,
                        KelasKata.VERBA_SERAPAN,
                    ):
                        edges.append((i, j, "modifier"))
                        break

        # Deduplikasi
        edges = list(dict.fromkeys(edges))
        return cls(n=n, edge=edges)

    @property
    def avg_degree(self) -> float:
        if self.n == 0:
            return 0.0
        degree = [0] * self.n
        for i, j, _ in self.edge:
            degree[i] += 1
            degree[j] += 1
        return sum(degree) / self.n


class CPEngine(nn.Module):  # type: ignore[misc]
    """
    Constraint Propagation Engine — minimisasi energi linguistik.

    OPOSISI TRANSFORMER secara menyeluruh:

    1. Unit: bukan token id → morfem dengan metadata linguistik
    2. Representasi: bukan embedding lookup → SFM tensor dari KBBI
    3. Mekanisme: bukan attention → propagasi constraint di graf dependensi
    4. Depth: bukan fixed N layers → iterasi sampai energi konvergen
    5. Output: bukan hidden state → AksaraState dengan pelanggaran eksplisit
    6. Kompleksitas: bukan O(n²) → O(n × avg_degree) per iterasi

    Developer yang pakai framework ini mendapat:
    - state linguistik yang bisa diinspeksi per dimensi
    - list pelanggaran dengan penjelasan bahasa Indonesia
    - energi total yang mencerminkan "ketidakvalidan" kalimat
    """

    def __init__(
        self,
        manifold: SemanticManifold,
        max_iter: int = 10,
        convergence_delta: float = 1e-4,
        threshold_semantik: float = 1.5,
        config: Optional[AksaraConfig] = None,
    ):
        """
        Args:
            manifold:           SemanticManifold yang sudah dimuat dari KBBI
            max_iter:           batas atas iterasi jika tidak konvergen
            convergence_delta:  threshold konvergensi energi
            threshold_semantik: batas jarak semantik yang dianggap compatible
            config:             AksaraConfig untuk domain khusus (opsional)
        """
        super().__init__()
        self.manifold  = manifold
        self.max_iter  = max_iter
        self.threshold = threshold_semantik
        self.config    = config or AksaraConfig.default()

        self.constraint_set = ConstraintSet(
            leksikon=manifold.leksikon,
            geodesic=manifold.geodesic,
            threshold_semantik=threshold_semantik,
            config=self.config,
        )
        self.convergence = ConvergenceChecker(
            delta=convergence_delta, window=3
        )

        # Bobot per dimensi constraint — learnable tapi diinisialisasi dari prior linguistik
        # Bukan attention weight — ini bobot linguistik yang punya interpretasi
        self._bobot = nn.Parameter(torch.tensor([
            BOBOT_CONSTRAINT["morfologis"],
            BOBOT_CONSTRAINT["sintaktis"],
            BOBOT_CONSTRAINT["semantik"],
            BOBOT_CONSTRAINT["leksikal"],
            BOBOT_CONSTRAINT["animasi"],
        ], dtype=torch.float32))

    def forward(
        self,
        morfem_list: List[Morfem],
        sfm_tensor: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cpu"),
    ) -> AksaraState:
        """
        Proses list morfem → AksaraState via constraint propagation.

        OPOSISI TRANSFORMER:
        Transformer forward: token_ids → embedding → N × attention_layer → output
        CPEngine forward:    morfem_list → constraint_graph → iterasi_konvergen → AksaraState

        Args:
            morfem_list:  list Morfem dari LPS
            sfm_tensor:   (opsional) tensor SFM shape (n, d_sfm)
            device:       device untuk komputasi tensor

        Returns:
            AksaraState lengkap dengan pelanggaran, energi, dan state tiap morfem
        """
        if len(morfem_list) == 0:
            return self._state_kosong(device)

        # Bangun graf dependensi — sparse, bukan dense
        graf = GrafDependensi.dari_morfem_list(morfem_list)

        # Propagasi constraint — iterasi sampai konvergen
        # sfm_tensor diteruskan untuk sinyal cosine similarity geometrik
        semua_pelanggaran, energi_per_dim = self._propagasi(
            morfem_list, graf, sfm_tensor=sfm_tensor
        )

        # Hitung energi total
        bobot_norm = torch.softmax(self._bobot, dim=0)
        _DIMS = ["morfologis", "sintaktis", "semantik", "leksikal", "animasi"]
        energi_total = float(sum(
            bobot_norm[i].item() * energi_per_dim.get(dim, 0.0)
            for i, dim in enumerate(_DIMS)
        ))

        # Constraint kalimat-level: urutan kata
        pelanggaran_urutan = self._cek_urutan_kata(morfem_list)
        for p in pelanggaran_urutan:
            semua_pelanggaran.append(p)
        if pelanggaran_urutan:
            energi_total = min(1.0, energi_total + 0.15 * len(pelanggaran_urutan))

        # Bangun MorfemState list
        morfem_states = self._bangun_morfem_states(morfem_list, semua_pelanggaran)

        # Inferensi register kalimat
        register = self._inferensi_register_kalimat(morfem_list)

        # Kelengkapan struktur: ada S dan P?
        peran_ada = {m.peran_gramatikal for m in morfem_list}
        kelengkapan = 1.0

        ada_predikat = PeranGramatikal.PREDIKAT in peran_ada
        ada_subjek   = PeranGramatikal.SUBJEK in peran_ada

        # Kalimat nominal: subjek nomina + kata penanda komplemen dianggap lengkap
        # Contoh: "Jenis-jenis tarian antara lain tari saman dan tari kecak."
        PENANDA_KOMPLEMEN = {
            "antara", "lain", "yaitu", "yakni", "ialah",
            "meliputi", "terdiri", "misalnya", "contohnya",
        }
        ada_penanda = any(
            m.root.lower() in PENANDA_KOMPLEMEN for m in morfem_list
        )
        ada_nomina = any(
            m.kelas_kata in (KelasKata.NOMINA, KelasKata.NOMINA_PROPER,
                             KelasKata.NOMINA_SERAPAN)
            for m in morfem_list
        )

        # Kalimat nominal predikatif: Nomina + Adj/Adv+Adj (tanpa verba)
        # Contoh valid: "Makanan itu sangat lezat." — adj predikatif
        # Ini adalah konstruksi sah bahasa Indonesia (kalimat ekuatif/predikatif)
        ada_adj = any(
            m.kelas_kata == KelasKata.ADJEKTIVA for m in morfem_list
        )
        ada_adv_derajat = any(
            m.root.lower() in {"sangat", "amat", "terlalu", "cukup", "agak",
                                "paling", "lebih", "kurang", "sekali"}
            for m in morfem_list
        )

        if not ada_predikat:
            # Kalimat nominal dengan penanda komplemen — tetap lengkap
            if ada_penanda and ada_nomina:
                pass  # kelengkapan tetap 1.0
            # Kalimat nominal predikatif: N + Adj (valid konstruksi Indonesia)
            elif ada_nomina and ada_adj:
                pass  # "Makanan itu lezat." / "Udara sangat segar." = valid
            else:
                kelengkapan -= 0.5

        if not ada_subjek and not ada_penanda and not (ada_nomina and ada_adj):
            kelengkapan -= 0.3

        return AksaraState(
            teks_asli=" ".join(m.teks_asli for m in morfem_list),
            morfem_states=morfem_states,
            energi_total=energi_total,
            energi_per_dimensi=energi_per_dim,
            pelanggaran=semua_pelanggaran,
            register=register,
            kelengkapan_struktur=max(0.0, kelengkapan),
            metadata={
                "n_morfem":   len(morfem_list),
                "n_edge":     len(graf.edge),
                "avg_degree": graf.avg_degree,
                "n_iterasi":  self.convergence.n_iterasi(),
            },
        )

    def _propagasi(
        self,
        morfem_list: List[Morfem],
        graf: GrafDependensi,
        sfm_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[List[PelanggaranConstraint], Dict[str, float]]:
        """
        Propagasi constraint di graf dependensi.

        OPOSISI TRANSFORMER:
        - Bukan weighted sum dari semua token
        - Setiap iterasi: update state berdasarkan constraint dengan tetangga
        - Berhenti saat energi konvergen, bukan saat mencapai N iterasi

        sfm_tensor (opsional): tensor (n_morfem, d_linguistik) dari SemanticManifold.
          Jika tersedia, cosine similarity antar vektor morfem dipakai sebagai
          sinyal geometrik tambahan pada dimensi semantik.
          Justifikasi: dua morfem yang berdekatan di ruang linguistik KBBI
          (domain sama, kelas sama, register sama) seharusnya punya ketegangan rendah.
          Ini adalah sinyal geometri linguistik, bukan statistik token.
        """
        self.convergence.reset()

        energi_per_dim: Dict[str, float] = {
            "morfologis": 0.0,
            "sintaktis":  0.0,
            "semantik":   0.0,
            "leksikal":   0.0,
            "animasi":    0.0,
        }
        semua_hasil: List[ConstraintResult] = []

        for _iter in range(self.max_iter):
            energi_iter: Dict[str, float] = {k: 0.0 for k in energi_per_dim}
            hasil_iter: List[ConstraintResult] = []

            for i, j, relasi in graf.edge:
                ma = morfem_list[i]
                mb = morfem_list[j]
                hasil = self.constraint_set.evaluasi_pasangan(ma, mb, relasi)

                for r in hasil:
                    energi_iter[r.tipe] = energi_iter.get(r.tipe, 0.0) + r.ketegangan
                    if r.dilanggar:
                        hasil_iter.append(r)

                # Sinyal geometrik dari SFM: cosine similarity di ruang linguistik KBBI
                # Justifikasi: morfem yang jauh secara geometri (domain/kelas berbeda)
                # menambah ketegangan semantik, melengkapi geodesic distance.
                # cos_sim ∈ [-1, 1]: makin rendah = makin jauh = makin tegang
                # Bobot kecil (0.05) agar hanya menjadi sinyal tambahan, bukan dominan.
                if sfm_tensor is not None and i < sfm_tensor.shape[0] and j < sfm_tensor.shape[0]:
                    vi = sfm_tensor[i].unsqueeze(0)  # (1, d)
                    vj = sfm_tensor[j].unsqueeze(0)  # (1, d)
                    cos_sim = float(torch.nn.functional.cosine_similarity(vi, vj).item())
                    # Ketegangan geometrik: 0.0 jika identik, mendekati 0.05 jika ortogonal/berlawanan
                    keteg_geometrik = max(0.0, (1.0 - cos_sim) / 2.0) * 0.05
                    energi_iter["semantik"] = energi_iter.get("semantik", 0.0) + keteg_geometrik

            # Normalisasi per edge
            n_edge = max(len(graf.edge), 1)
            for k in energi_iter:
                energi_iter[k] /= n_edge

            energi_total_iter = sum(energi_iter.values()) / max(len(energi_iter), 1)
            self.convergence.update(energi_total_iter)

            # Update energi terbaik
            for k in energi_per_dim:
                energi_per_dim[k] = energi_iter[k]
            semua_hasil = hasil_iter

            if self.convergence.konvergen():
                break

        # Konversi ConstraintResult → PelanggaranConstraint
        pelanggaran = self._konversi_pelanggaran(semua_hasil)
        return pelanggaran, energi_per_dim

    def _konversi_pelanggaran(
        self, hasil: List[ConstraintResult]
    ) -> List[PelanggaranConstraint]:
        """
        Konversi hasil constraint ke format PelanggaranConstraint untuk AksaraState.
        Deduplikasi berdasarkan (tipe, morfem_a, morfem_b).
        """
        seen = set()
        pelanggaran = []
        for r in hasil:
            key = (r.tipe, r.morfem_a, r.morfem_b)
            if key in seen:
                continue
            seen.add(key)
            pelanggaran.append(PelanggaranConstraint(
                tipe=r.tipe,
                token_terlibat=[r.morfem_a, r.morfem_b],
                dimensi=r.tipe,
                severitas=r.ketegangan,
                penjelasan=r.penjelasan,
            ))
        return pelanggaran

    def _bangun_morfem_states(
        self,
        morfem_list: List[Morfem],
        pelanggaran: List[PelanggaranConstraint],
    ) -> List[MorfemState]:
        """Bangun MorfemState dari morfem dan pelanggaran yang ditemukan."""
        pelang_per_kata: Dict[str, List[str]] = {}
        for p in pelanggaran:
            for token in p.token_terlibat:
                if token not in pelang_per_kata:
                    pelang_per_kata[token] = []
                pelang_per_kata[token].append(p.penjelasan)

        states = []
        for m in morfem_list:
            states.append(MorfemState(
                indeks=m.indeks,
                teks=m.teks_asli,
                root=m.root,
                afiks=[a.bentuk for a in m.afiks_aktif],
                kelas_kata=m.kelas_kata.value,
                peran_gramatikal=m.peran_gramatikal.value,
                pelanggaran=pelang_per_kata.get(m.root, []),
            ))
        return states

    def _inferensi_register_kalimat(self, morfem_list: List[Morfem]) -> str:
        """
        Inferensi register keseluruhan kalimat dari register tiap morfem.
        formal / informal / mixed
        """
        n_informal = sum(1 for m in morfem_list if m.adalah_informal)
        n_serapan  = sum(1 for m in morfem_list if m.adalah_serapan)
        n_total    = max(len(morfem_list), 1)

        if n_informal / n_total > 0.3:
            return "informal"
        if n_serapan / n_total > 0.3:
            return "mixed"
        return "formal"

    def _cek_urutan_kata(
        self,
        morfem_list: List[Morfem],
    ) -> List[PelanggaranConstraint]:
        """
        Constraint urutan kata kalimat-level (word order).

        Justifikasi linguistik:
          Bahasa Indonesia adalah bahasa SVO (Subjek-Verba-Objek).
          Beberapa urutan kata adalah illegal secara topologis:
            1. Preposisi di akhir kalimat tanpa objek — stranded preposition
               tidak ada dalam bahasa Indonesia standar (berbeda dari bahasa Inggris).
            2. Dua preposisi berurutan tanpa nomina di antara keduanya.
            3. Kalimat dimulai preposisi dan diakhiri preposisi yang sama.
          Referensi: Alwi dkk. (2003) §8 Frasa Preposisional.

        Ini adalah constraint topologis kalimat-level, bukan pasangan morfem.
        Dipanggil dari forward() setelah propagasi constraint pasangan selesai.
        """
        pelanggaran = []
        n = len(morfem_list)
        if n == 0:
            return pelanggaran

        # ── Deteksi 1: Preposisi di posisi terakhir kalimat ──────────────────
        # "Di beli beras Budi pasar di." — preposisi penutup tanpa argumen
        terakhir = morfem_list[-1]
        if terakhir.kelas_kata == KelasKata.PREPOSISI:
            pelanggaran.append(PelanggaranConstraint(
                tipe="topologis",
                token_terlibat=[terakhir.root],
                dimensi="topologis",
                severitas=0.80,
                penjelasan=(
                    f"Pelanggaran urutan kata: preposisi '{terakhir.teks_asli}' "
                    f"di akhir kalimat tanpa argumen nomina. "
                    f"Bahasa Indonesia tidak mengenal stranded preposition."
                ),
            ))

        # ── Deteksi 2: Dua preposisi berurutan ───────────────────────────────
        # "kepada dari terdakwa" — tidak ada nomina di antara dua preposisi
        for i in range(n - 1):
            if (morfem_list[i].kelas_kata == KelasKata.PREPOSISI
                    and morfem_list[i + 1].kelas_kata == KelasKata.PREPOSISI):
                pelanggaran.append(PelanggaranConstraint(
                    tipe="topologis",
                    token_terlibat=[morfem_list[i].root, morfem_list[i + 1].root],
                    dimensi="topologis",
                    severitas=0.70,
                    penjelasan=(
                        f"Pelanggaran urutan kata: dua preposisi berurutan "
                        f"'{morfem_list[i].teks_asli} {morfem_list[i+1].teks_asli}' "
                        f"tanpa nomina di antara keduanya."
                    ),
                ))

        # ── Deteksi 3: Kalimat dimulai dan diakhiri preposisi yang sama ───────
        # "Di beli beras di." — preposisi framing tanpa makna
        pertama = morfem_list[0]
        if (pertama.kelas_kata == KelasKata.PREPOSISI
                and terakhir.kelas_kata == KelasKata.PREPOSISI
                and pertama.root == terakhir.root):
            pelanggaran.append(PelanggaranConstraint(
                tipe="topologis",
                token_terlibat=[pertama.root],
                dimensi="topologis",
                severitas=0.90,
                penjelasan=(
                    f"Pelanggaran urutan kata berat: preposisi '{pertama.teks_asli}' "
                    f"membuka dan menutup kalimat secara simetris — "
                    f"struktur kalimat kacau total."
                ),
            ))

        # ── Deteksi 4: Dative animacy — 'kepada' + nomina inanimate ───────────
        # Justifikasi (Alwi dkk. 2003 §8.4 Frasa Preposisional + theta-role dative):
        #   Preposisi 'kepada' adalah penanda dative: menunjuk penerima [±animate].
        #   Argumen 'kepada' yang berupa nomina benda mati / abstraksi bukan tindakan
        #   adalah pelanggaran selektivitas θ-dative.
        #   'kepada hakim' = valid (manusia/penerima)
        #   'kepada hukuman' = INVALID (hukuman bukan penerima, tidak bisa menerima)
        # Pengecualian: 'kepada' + nomina abstrak yang secara idiomatis sah:
        #   'berpegang kepada prinsip', 'tunduk kepada aturan' — ini relasi benefactive
        #   tapi tidak ada predikat aktif di sini, jadi diabaikan untuk saat ini.
        PREP_DATIVE = {"kepada", "terhadap"}
        # Justifikasi: penerima dative HARUS animate (manusia/hewan) atau NOMINA_PROPER.
        # Pengecualian berbasis domain (domain='hukum') terlalu lebar karena mencakup
        # abstraksi seperti 'hukuman', 'vonis', 'pasal' yang tidak bisa menjadi penerima.
        # Kata seperti 'terdakwa', 'hakim', 'direktur' sudah ada di KATA_ANIMATE_MANUSIA.
        for i in range(n - 1):
            tok = morfem_list[i]
            arg = morfem_list[i + 1]
            if (tok.kelas_kata == KelasKata.PREPOSISI
                    and tok.root.lower() in PREP_DATIVE
                    and arg.kelas_kata in (KelasKata.NOMINA, KelasKata.NOMINA_SERAPAN)):
                root_arg = arg.root.lower()
                arg_animate = (
                    arg.kelas_kata == KelasKata.NOMINA_PROPER
                    or self.constraint_set.leksikon.adalah_animate(root_arg)
                )
                if not arg_animate:
                    domain_arg = self.constraint_set.leksikon.domain_kata(root_arg)
                    pelanggaran.append(PelanggaranConstraint(
                        tipe="semantik",
                        token_terlibat=[tok.root, arg.root],
                        dimensi="semantik",
                        severitas=0.75,
                        penjelasan=(
                            f"Dative animacy violation: '{tok.teks_asli} {arg.teks_asli}' — "
                            f"preposisi '{tok.teks_asli}' menandai penerima [+animate], "
                            f"tapi '{arg.root}' adalah benda/abstraksi [{domain_arg or 'tanpa domain'}] "
                            f"yang tidak bisa menjadi penerima (dative recipient)."
                        ),
                    ))

        # ── Deteksi 5: Struktur kalimat pasif rusak ────────────────────────
        # Justifikasi (Alwi dkk. 2003 §7.3 Kalimat Pasif):
        #   Pola pasif sahih bahasa Indonesia:
        #     [SUBJEK] + di-V + oleh + [PELAKU]
        #     [SUBJEK] + di-V + oleh + [PELAKU] + kepada + [PENERIMA]
        #   Pola RUSAK yang terdeteksi di sini:
        #     oleh + [PELAKU] + di-V — kalimat pasif dimulai dengan agen ('oleh')
        #     tanpa subjek gramatikal yang mendahului predikat pasif.
        #   Ini bukan topikalisasi sah: topikalisasi pasif bahasa Indonesia
        #   memindahkan SUBJEK (bukan agen) ke depan.
        ada_predikat_pasif = any(
            m.kelas_kata == KelasKata.VERBA
            and m.root.lower() != m.teks_asli.lower()  # ada afiks
            and m.teks_asli.lower().startswith("di")
            for m in morfem_list
        )
        if ada_predikat_pasif:
            pertama = morfem_list[0] if morfem_list else None
            # Kalimat pasif rusak: dimulai dengan 'oleh' (agen) bukan subjek
            if (pertama is not None
                    and pertama.kelas_kata == KelasKata.PREPOSISI
                    and pertama.root.lower() == "oleh"):
                idx_pred = next(
                    (i for i, m in enumerate(morfem_list)
                     if m.kelas_kata == KelasKata.VERBA
                     and m.teks_asli.lower().startswith("di")),
                    None
                )
                # Identifikasi rentang frasa 'oleh': dari idx 0 sampai sebelum predikat
                # Nomina di dalam frasa 'oleh X' bukan subjek gramatikal
                # (LPS bisa salah label nomina ini sebagai SUBJEK)
                idx_oleh_arg = 1  # nomina langsung setelah 'oleh'
                ada_subj_sejati = (
                    idx_pred is not None
                    and any(
                        morfem_list[j].peran_gramatikal == PeranGramatikal.SUBJEK
                        and j != idx_oleh_arg  # bukan agen frasa 'oleh'
                        for j in range(idx_pred)
                        if morfem_list[j].kelas_kata not in (
                            KelasKata.PREPOSISI, KelasKata.KONJUNGSI
                        )
                    )
                )
                if not ada_subj_sejati:
                    agen = morfem_list[1].root if n > 1 else "?"
                    pelanggaran.append(PelanggaranConstraint(
                        tipe="sintaktis",
                        token_terlibat=["oleh", agen],
                        dimensi="sintaktis",
                        severitas=0.80,
                        penjelasan=(
                            f"Struktur pasif rusak: kalimat dimulai dengan 'oleh {agen}' "
                            f"(frasa agen) tanpa subjek gramatikal di awal. "
                            f"Pola pasif Indonesia: [Subjek] + di-V + oleh + [Pelaku]. "
                            f"Frasa agen tidak boleh mendahului subjek."
                        ),
                    ))

        return pelanggaran

    def _state_kosong(self, device: torch.device) -> AksaraState:
        return AksaraState(
            teks_asli="",
            morfem_states=[],
            energi_total=0.0,
            energi_per_dimensi={},
            pelanggaran=[],
            register="formal",
            kelengkapan_struktur=0.0,
        )
