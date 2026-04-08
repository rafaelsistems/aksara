# AKSARA Framework — Spesifikasi Teknis
**Versi:** 4.0-draft  
**Status:** Dokumen Hidup — wajib diperbarui sebelum implementasi apapun  
**Penulis:** Emylton Leunufna  

---

## 1. Deklarasi Tujuan

AKSARA adalah **pipeline pemahaman linguistik native Indonesia**. AKSARA **bukan model prediksi token**, bukan sistem next-token prediction, dan bukan sekadar varian Transformer/Mamba yang diberi nama baru. AKSARA adalah framework yang memproses bahasa Indonesia sebagai struktur linguistik yang harus dianalisis, divalidasi, dan di-grounding secara eksplisit.

### 1.1 Prinsip Tidak Bisa Dilanggar

1. **Anti-prediksi token sebagai tujuan utama** — AKSARA tidak dirancang untuk menebak token berikutnya sebagai output primer
2. **Native Indonesia** — bahasa Indonesia bukan subset bahasa Inggris; framework ini dibangun dari struktur linguistik Indonesia, bukan dari statistik korpus generik
3. **Framework, bukan model** — semua komponen harus reusable, composable, dan task-agnostic
4. **Eksplisit, bukan implisit** — setiap keputusan komputasi harus punya justifikasi linguistik yang bisa dijelaskan, bukan blackbox statistical pattern
5. **Hukum, bukan statistik** — representasi dibangun dari hukum struktur bahasa, bukan dari frekuensi kemunculan token
6. **Backward-compatible** — kontrak publik yang sudah dipakai harus dipertahankan selama mungkin

### 1.2 Apa yang AKSARA Jamin

Framework AKSARA menjamin bahwa model apapun yang dibangun di atasnya:
- Memproses morfologi bahasa Indonesia secara deterministik dan benar
- Merepresentasikan makna kata dalam ruang yang grounded ke leksikon Indonesia
- Melakukan reasoning berbasis constraint linguistik, bukan statistik
- Menghasilkan output yang bisa dijelaskan per dimensi linguistik
- Mengembalikan state terstruktur (`AksaraState`) sebagai keluaran utama, bukan logits next-token

### 1.3 Apa yang AKSARA Tidak Tentukan

- Objective training (ditentukan developer)
- Format output head spesifik (ditentukan developer)
- Task spesifik (ditentukan developer)
- Arsitektur head (ditentukan developer)
- Strategi decoding token berikutnya sebagai tujuan primer

---

## 2. Posisi terhadap Transformer/Mamba

### 2.1 Oposisi Fundamental

| Dimensi | Transformer/Mamba | AKSARA |
|---|---|---|
| **Paradigma** | Statistik distribusi token | Hukum struktur linguistik |
| **Unit dasar** | Subword token (arbitrer) | Morfem (unit bermakna) |
| **Representasi** | Vektor statis, Euclidean space | State dinamis, ground linguistik |
| **Mekanisme utama** | Self-attention / SSM global | Evolusi constraint lokal dan grounding eksplisit |
| **Komposisi makna** | Implisit via stacking layers | Eksplisit via tahap linguistik berurutan |
| **Pengetahuan** | Tersimpan di bobot (tidak bisa diupdate tanpa retrain) | Tersimpan di leksikon dan state yang bisa diinspeksi |
| **Kegagalan** | Silent hallucination | Loud violation report |
| **Interpretabilitas** | Heatmap / hidden state | Pelanggaran dan state per dimensi linguistik |
| **Bahasa** | Agnostik — semua bahasa diperlakukan sama | Native Indonesia — constraint morfologi dikodekan hard |
| **Output primer** | Logits / probabilitas token | `AksaraState` terstruktur |

### 2.2 Bukan Sekedar Alternatif

AKSARA bukan alternatif Transformer yang lebih efisien. AKSARA adalah **paradigma berbeda** yang menjawab pertanyaan berbeda:

- Transformer menjawab: *"Token apa yang paling mungkin muncul berikutnya?"*
- AKSARA menjawab: *"Apakah struktur linguistik ini valid, dan mengapa?"*

Framework yang dibangun dari jawaban kedua menghasilkan model yang:
- Tidak menjadikan prediksi token sebagai tujuan primer
- Bisa dijelaskan karena reasoning transparan
- Tidak butuh retrain untuk update pengetahuan leksikal
- Spesifik dan akurat untuk bahasa Indonesia

---

## 3. Arsitektur Framework: Alur Implementasi Saat Ini

### Gambaran Umum

Implementasi publik yang aktif saat ini mengikuti alur berikut:

```
INPUT: teks bahasa Indonesia
         │
         ▼
┌─────────────────────┐
│  LPS                │  ← Linguistic Parse System
│  Morfem + peran     │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LSK                │  ← Lapisan Semantik KBBI
│  grounding semantik │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  MEB                │  ← Mesin Evolusi Bahasa
│  evolusi state      │
└─────────┬───────────┘
          │
          ▼
OUTPUT: AksaraState + interpretasi
```

### 3.1 Catatan tentang istilah historis

- Dokumen lama mungkin masih menyebut SFM/CPE/CMC/TDA/KRL.
- Istilah tersebut adalah **nama rancangan atau fase roadmap**, bukan kontrak perilaku wajib untuk implementasi publik saat ini.
- Komentar yang memakai istilah Transformer/Mamba, attention, encoder/decoder, atau analogi sejenis **tidak otomatis berarti** perilaku AKSARA setara dengan Transformer/Mamba.
- Yang menjadi kontrak adalah **perilaku proses**: analisis linguistik → grounding semantik → evolusi state → interpretasi.

---

## 4. Primitif 1: LPS — Linguistic Parse System

### 4.1 Status
Sudah ada di `aksara/linguistic/lps.py`. Perlu diperkuat tapi tidak digantikan.

### 4.2 Tanggung Jawab
- Tokenisasi berbasis morfem (bukan subword)
- Dekomposisi afiks (prefiks, sufiks, konfiks, reduplikasi)
- Assignment peran linguistik per morfem (root, prefiks, sufiks, klitika)
- Output: sequence morfem dengan metadata linguistik lengkap

### 4.3 Perbedaan dari Tokenizer Transformer
Transformer tokenizer (BPE/WordPiece): memotong kata berdasarkan frekuensi statistik.
```
"memasakan" → ["me", "##masa", "##kan"]  ← arbitrer, tidak bermakna
```
LPS: memotong berdasarkan struktur morfologis bahasa Indonesia.
```
"memasakan" → [{root:"masak", prefiks:"me-", sufiks:"-kan", kelas:"verba_aktif_transitif"}]
```

### 4.4 Yang Perlu Ditambahkan
- Support morfologi informal + serapan (di-ghosting, di-cancel)
- Deteksi reduplikasi semantik vs gramatikal
- Support kode-campur (code-mixing) Indonesia-Inggris

---

## 5. Primitif 2: LSK — Lapisan Semantik KBBI

### 5.1 Status
Sudah aktif di `aksara/linguistic/lsk.py`.

### 5.2 Tanggung Jawab
- Memuat dan mengindeks KBBI core
- Menyediakan lookup lemma, POS, dan konteks POS
- Menyediakan anchor semantik untuk grounding kata ke leksikon Indonesia
- Mengubah pengetahuan leksikal menjadi representasi yang dapat dipakai oleh tahap evolusi berikutnya

### 5.3 Kontrak Implementasi
LSK tidak memprediksi token. LSK:
- memetakan lemma ke entri KBBI,
- menyediakan konteks POS bila tersedia,
- memberi anchor semantik untuk grounding,
- menjaga kompatibilitas dengan `KBBIStore.lookup(...)`, `get_pos_list(...)`, dan `get_pos_context(...)`.

---

## 6. Primitif 3: MEB — Mesin Evolusi Bahasa

### 6.1 Status
Sudah ada di `aksara/core/meb.py`.

### 6.2 Motivasi
MEB adalah mesin evolusi state linguistik yang menerima input hasil LPS/LSK dan memperbarui state melalui evolusi eksplisit. Ini **bukan** sekadar mekanisme lokal berbasis constraint yang meniru Transformer, dan bukan SSM dalam arti perilaku statistik token.

### 6.3 Konsep Inti
Evolusi state dilakukan melalui pembaruan berlapis terhadap komponen morfologis, sintaktis, dan semantik, lalu diringkas menjadi state akhir yang bisa diinterpretasi.

### 6.4 Peran dalam Alur Aktual
Dalam implementasi publik saat ini, MEB adalah tahap yang mengubah hasil grounding dari LSK menjadi state yang siap dibaca sebagai `AksaraState` atau interpretasi lanjut.

---

## 7. Kontrak Output: AksaraState dan Interpretasi

### 7.1 Status
`AksaraState` adalah keluaran struktural utama yang dibaca developer.

### 7.2 Yang Harus Terlihat dari Output
- skor linguistik / kualitas state
- informasi pelanggaran constraint
- metadata proses
- hasil interpretasi atau representasi semantik yang bisa dibaca head berikutnya

### 7.3 Implikasi Kontrak
Tidak ada API publik yang boleh memperlakukan logits next-token sebagai output primer AKSARA. Jika ada head atau adaptor khusus, itu berada di lapisan pengguna, bukan kontrak inti framework.

---

## 8. Arsitektur Prinsip

1. **Linguistik lebih utama daripada prediksi token**
2. **Setiap tahap harus bisa dijelaskan**
3. **Grounding ke leksikon Indonesia wajib eksplisit**
4. **State terstruktur adalah kontrak utama**
5. **Komentar historis tidak boleh dibaca sebagai janji perilaku**
6. **Perubahan harus menjaga kompatibilitas API yang sudah dipakai**

---

## 9. API Framework

### 9.1 Prinsip Desain API
- **Composable:** setiap primitif bisa dipakai sendiri atau digabung
- **Extensible:** developer bisa tambah primitif baru atau gantikan yang ada
- **Transparent:** setiap output punya metadata linguistik yang bisa diinspeksi
- **Indonesian-first:** default behavior optimal untuk bahasa Indonesia

### 9.2 Interface Utama

```python
from aksara import AksaraFramework, LPS, LSK, MEB

# Inisialisasi framework
fw = AksaraFramework(
    lexicon=["kbbi", "wiktionary-id"],  # sumber leksikon
    language="id",                      # bahasa target
)

# Bangun pipeline sesuai kebutuhan
pipeline = fw.pipeline(
    parser=LPS(),
    grounding=LSK(),
    evolution=MEB(),
)

# Proses teks
state = pipeline.encode("Dia makan nasi goreng di warung.")

# State bisa digunakan oleh head apapun yang developer definisikan
output = my_custom_head(state)
```

### 9.3 Extension Points

```python
# Tambah sumber leksikon baru
fw.add_lexicon("bahasa_daerah_ntt", path="lexicon/ntt.json")

# Definisikan constraint baru
fw.add_constraint(
    name="temporal_consistency",
    fn=lambda tokens: check_tense_consistency(tokens),
)

# Sambungkan ke oracle faktual eksternal
fw.plug_oracle(
    name="wikipedia_id",
    fn=lambda query: wikipedia_lookup(query),
)

# Definisikan head untuk task spesifik
class MyEvaluatorHead(AksaraHead):
    def forward(self, state: AksaraState) -> dict:
        ...
```

---

## 10. Struktur Direktori Target

```
aksara/                          ← FRAMEWORK (publik)
  __init__.py                    ← public API entry point
  framework.py                   ← AksaraFramework class
  primitives/
    lps/                         ← Primitif 1
      __init__.py
      parser.py
      morpheme.py
      affix_rules.py
    sfm/                         ← roadmap historis
    cpe/                         ← roadmap historis
    cmc/                         ← roadmap historis
    tda/                         ← roadmap historis
    krl/                         ← roadmap historis
  base/
    head.py                      ← AksaraHead base class
    state.py                     ← AksaraState dataclass
    oracle.py                    ← Oracle interface

examples/                        ← CONTOH PENGGUNAAN (bukan framework)
  correctness_evaluator/         ← model evaluator kebenaran kalimat
  ...

docs/                            ← DOKUMENTASI
  AKSARA_FRAMEWORK_SPEC.md       ← dokumen ini
  AKSARA_PRIMITIVES_MATH.md      ← formalisasi matematis
  AKSARA_API_REFERENCE.md        ← referensi API lengkap
```

---

## 11. Urutan Implementasi

Implementasi mengikuti urutan dependency:

```
1. LPS (perkuat yang sudah ada)
   → Tidak ada dependency ke primitif lain

2. LSK (paket grounding leksikal)
   → Depends on: LPS output, KBBI data

3. MEB (evolusi state)
   → Depends on: LPS output, LSK state

4. Interpretasi / head
   → Depends on: AksaraState dan kebutuhan aplikasi

5. Framework API (assembly)
   → Integrasikan semua tahap aktif
   → Definisikan AksaraState, AksaraHead, AksaraFramework
```

**PENTING:** Implementasi baru harus mengikuti kontrak yang sudah didokumentasikan dan tidak boleh mengubah AKSARA menjadi model prediksi token.

---

## 12. Benchmark Keberhasilan Framework

Framework dianggap berhasil jika model yang dibangun di atasnya memenuhi:

| Metrik | Target |
|---|---|
| Akurasi evaluasi kalimat benar/salah | >90% |
| AUC pada corpus hard negative | >0.90 |
| Interpretabilitas: bisa jelaskan MENGAPA salah | 100% kasus |
| Update lexicon tanpa retrain | ✅ |
| Inference tanpa GPU (CPU-only) | ✅ untuk kalimat tunggal |
| Waktu inference per kalimat | <100ms CPU |

---

*Dokumen ini harus diperbarui setiap kali ada keputusan arsitektur baru sebelum implementasi dilanjutkan.*