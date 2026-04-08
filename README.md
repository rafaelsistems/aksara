# AKSARA
**Adaptive Knowledge & Semantic Architecture for Bahasa Representation & Autonomy**

> *"Kami tidak mengajarkan model bahasa Indonesia. Kami membuat model lahir sebagai bahasa Indonesia."*
>
> — Emylton Leunufna

---

## Apa itu AKSARA?

AKSARA adalah **pipeline pemahaman linguistik** untuk bahasa Indonesia. AKSARA **bukan model prediksi token**, bukan fine-tuned Transformer, dan bukan mesin yang tujuan utamanya menebak token berikutnya. AKSARA menganalisis, memvalidasi, dan meng-grounding struktur bahasa Indonesia dari prinsip linguistik, lalu menghasilkan state yang bisa dijelaskan.

AKSARA menghasilkan `AksaraState` — representasi linguistik lengkap yang bisa dikonsumsi oleh **head apapun** yang developer definisikan sendiri.

---

## Pipeline Aktual

Implementasi publik yang aktif saat ini berjalan melalui alur berikut:

```
Kalimat (string)
   ↓
[ LPS ] Linguistic Parse System        → dekomposisi morfem deterministik
   ↓
[ LSK ] Lapisan Semantik KBBI         → grounding semantik ke KBBI
   ↓
[ MEB ] Mesin Evolusi Bahasa          → evolusi state berbasis constraint
   ↓
[ AksaraState ]                       → output terstruktur + interpretasi
```

Catatan:
- Istilah lama seperti SFM/CPE/CMC/TDA/KRL masih muncul pada dokumen historis dan roadmap.
- Nama-nama itu **bukan berarti** implementasi publik saat ini sudah memiliki perilaku Transformer/Mamba atau mekanisme statistik token.
- Jika ada komentar internal yang menyebut "Transformer", "Mamba", "attention", atau istilah serupa, itu harus dibaca sebagai analogi historis/arsitektural, **bukan** kesetaraan perilaku.

---

## Oposisi terhadap Model Prediksi Token

| Aspek | Model prediksi token | AKSARA |
|---|---|---|
| Tujuan utama | Menebak token berikutnya | Memahami struktur linguistik |
| Unit dasar | Subword/token statistik | Morfem, root, afiks |
| Output utama | Logits / probabilitas next-token | `AksaraState` terstruktur |
| Grounding | Distribusional | KBBI + aturan linguistik |
| Penjelasan | Tersirat di bobot | Eksplisit di state dan pelanggaran |
| Perubahan pengetahuan | Retrain / finetune | Update leksikon / aturan |

---

## Cara Pakai

### Inisialisasi dari KBBI

Gunakan `AksaraFramework.dari_kbbi(...)` untuk membuat framework dari berkas leksikon KBBI, lalu panggil `proses(kalimat)` untuk menganalisis satu kalimat.

```python
from aksara import AksaraFramework

fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")
state = fw.proses("Dia makan nasi goreng.")
```

### Membaca hasil `AksaraState`

Hasil `proses(...)` adalah `AksaraState`. Dua cara baca yang paling umum adalah:

- `state.ringkasan()` untuk ringkasan singkat
- `state.jelaskan()` untuk penjelasan yang lebih lengkap

```python
print(state.ringkasan())
print(state.jelaskan())
```

Jika pipeline mengembalikan hasil KRL, hasil tersebut tetap bisa diakses dari `state.krl_result`.

```python
if state.krl_result is not None:
    print(state.krl_result.jelaskan())
```

### Contoh singkat lain

```python
from aksara import AksaraFramework

fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")
state = fw.proses("Hakim menjatuhkan hukuman.")
print(state.ringkasan())
```

```python
from aksara import AksaraFramework

fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")
state = fw.proses("Ibu memasak nasi.")
print(state.jelaskan())
```

---

## CLI

AKSARA menyediakan CLI via `py -m aksara` untuk inspeksi dan validasi.

Perintah yang tersedia:
- `audit` — jalankan native framework audit 10 poin
- `generate` — generate teks dari prompt menggunakan checkpoint
- `export` — export checkpoint ke direktori lain
- `info` — tampilkan info checkpoint
- `init` — buat template `aksara_config.yaml`
- `schema` — tampilkan schema config
- `diff` — bandingkan dua config YAML
- `merge` — merge config YAML berlapis

Contoh:
```bash
py -m aksara audit
py -m aksara info --checkpoint ./ckpt
py -m aksara generate --checkpoint ./ckpt --prompt "anak membaca"
py -m aksara export --checkpoint ./ckpt --output ./ckpt_export
```

Catatan penting:
- `generate` membutuhkan `--checkpoint` dan prompt yang tidak kosong.
- Jika checkpoint tidak lengkap atau tidak valid, CLI akan menampilkan pesan error yang jelas.
- Input kosong, whitespace-only, atau punctuation-only diperlakukan sebagai input kosong yang aman di boundary CLI.

---

## Evaluasi Korpus

Untuk evaluasi robustness dan smoke test, tersedia utilitas berikut:

```bash
py -3.11 tools/corpus_robustness_eval.py --output-json hasil.json
py -3.11 tools/corpus_robustness_eval_large.py --samples 5000 --output-json hasil_large.json
```

Ringkasan evaluasi mencakup:
- valid rate
- skor linguistik rata-rata
- rata-rata constraint
- rata-rata KRL
- ringkasan per kategori
- laporan JSON yang dapat dipakai tooling otomatis

---

## Arsitektur Prinsip

1. **Linguistik dulu, statistik kemudian** — struktur bahasa menjadi dasar, bukan frekuensi token.
2. **Deterministik dan dapat dijelaskan** — setiap keputusan komputasi harus bisa dilacak secara linguistik.
3. **State terstruktur, bukan logits mentah** — output utama adalah `AksaraState`, bukan probabilitas next-token.
4. **Grounding eksplisit ke leksikon Indonesia** — pengetahuan harus bisa diinspeksi dan diperbarui.
5. **Backward-compatible** — perubahan dokumentasi dan implementasi harus menjaga API yang sudah dipakai.

---

## Struktur Proyek

```
aksara/
  primitives/
    lps/     ← Linguistic Parse System
    sfm/     ← Semantic Field Manifold
    cpe/     ← Constraint Propagation Engine
    cmc/     ← Categorical Meaning Composer
    tda/     ← Topological Dependency Analyzer
    krl/     ← Knowledge Representation Layer
  base/
    state.py ← AksaraState (output pipeline)
    head.py  ← AksaraHead (base class untuk custom head)
  heads/
    correctness.py ← CorrectnessEvaluatorHead + LearnedCorrectnessHead
  config.py        ← AksaraConfig (domain-specific configuration)
  framework.py     ← AksaraFramework (orkestrator utama)
```

---

## Test Suite

```bash
# Semua test suite harus PASS sebelum deploy
py -3.11 tools/framework_diagnostic.py   # 6/6 PASS
py -3.11 tools/_test_integrasi.py        # PASS
py -3.11 tools/test_jalur_b.py           # 6/6 PASS
py -3.11 tools/test_krl.py              # 5/5 PASS
pytest tests/test_framework_end_to_end.py
pytest tests/test_large_evaluator_smoke.py
```

---

## Domain yang Didukung

```python
from aksara import AksaraFramework, AksaraConfig

# Domain hukum
fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json",
     config=AksaraConfig.untuk_domain("hukum"))

# Domain: hukum | kesehatan | militer | pertanahan | pendidikan | bisnis
```

---

## Data

- **KBBI:** `kbbi_core_v2.json` — 71,211 kata, 10 domain semantik
- **Python:** 3.11+, PyTorch 2.6+

---

## Penulis

Emylton Leunufna

---

## Instalasi

```bash
pip install -r requirements.txt
```

---

## Lisensi

MIT
