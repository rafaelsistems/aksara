# AKSARA — Deep Evaluation Framework Design
**Tanggal:** 2026-04-06  
**Status:** DESAIN (belum implementasi)  
**Konteks:** Hasil benchmark epoch 10–20 dengan vocab rebuilt dari corpus_mixed_v1

---

## 1. Limitasi Benchmark Saat Ini

Benchmark sekarang (`action_benchmark.py`) mengukur:

```
hit_rate = token dari expected_set muncul di output (ya/tidak)
```

**Masalah:**
- Tidak membedakan *echo* (`makan makan makan`) vs *genuine usage* (`makan nasi di rumah`)
- Tidak mengukur apakah objek yang muncul *logis* untuk aksi tersebut
- Tidak mengukur urutan: apakah verb → objek atau acak
- Hit rate biner: `ikan` muncul = 1, meski kalimatnya nonsensical

---

## 2. Tiga Metrik Baru yang Perlu Didesain

### 2.1 SVO Correctness Score

**Definisi:** Apakah output mengandung struktur Subjek-Verb-Objek yang valid?

**Cara ukur (rencana):**
```
svo_score = 0
output_tokens = tokenize(output)

# Cek apakah ada verb domain di output
verb_present = any(t in DOMAIN_VERBS[domain] for t in output_tokens)

# Cek apakah ada objek domain SETELAH verb
if verb_present:
    verb_pos = first_verb_position(output_tokens)
    obj_after = any(t in DOMAIN_OBJS[domain]
                    for t in output_tokens[verb_pos:])
    if obj_after:
        svo_score = 1  # full SVO
    else:
        svo_score = 0.5  # hanya verb, tanpa objek

return svo_score
```

**Contoh:**
```
"saya makan"  → output "makan nasi di rumah"  → SVO = 1.0  ✅
"saya makan"  → output "makan makan kenyang"  → SVO = 0.5  ⚠️ (verb ada, obj echo)
"saya makan"  → output "pergi ke pasar"       → SVO = 0.0  ❌ (domain salah)
```

**Threshold:** SVO ≥ 0.5 per prompt = domain mulai dipahami

---

### 2.2 Verb-Object Pairing Score

**Definisi:** Apakah pasangan verb-objek yang dihasilkan *logis secara domain*?

**Cara ukur (rencana):**
```
# Definisi pasangan valid per domain
VALID_PAIRS = {
    "makan":     {("makan","nasi"), ("makan","roti"), ("minum","air"), ...},
    "membaca":   {("membaca","buku"), ("baca","artikel"), ...},
    "bekerja":   {("kerja","laporan"), ("buat","tugas"), ...},
    ...
}

# Cek sliding window bigram di output
bigrams = zip(output_tokens, output_tokens[1:])
pair_hits = sum(1 for b in bigrams if b in VALID_PAIRS[domain])
pairing_score = min(pair_hits / 2, 1.0)  # cap di 1.0
```

**Contoh:**
```
output "baca artikel dengan seksama" → ("baca","artikel") = valid → 1.0 ✅
output "baca baca baca artikel"      → ("baca","baca") = invalid → 0.5 ⚠️
output "ikan di pasar"               → tidak ada verb-obj pair → 0.0 ❌
```

**Catatan:** Ini akan mendeteksi echo learning secara langsung — echo selalu menghasilkan bigram (verb, verb) yang tidak valid.

---

### 2.3 Semantic Consistency Score (SCS v2)

**Definisi:** Apakah output dari 5 run untuk prompt yang sama *semantically consistent*?

**Cara ukur (rencana):**
```
# Dari 5 run, ambil semua token unik per run
token_sets = [set(tokenize(output)) for output in 5_runs]

# Hitung domain-token overlap antar run
domain_tokens_per_run = [s & DOMAIN_KEYWORDS[domain] for s in token_sets]
consistency = jaccard(union(domain_tokens_per_run), 
                      intersection(domain_tokens_per_run))

# Normalized: apakah rata-rata token domain yang sama muncul konsisten?
```

**Threshold:** SCS ≥ 0.3 = model punya preferensi domain yang konsisten

---

## 3. Baseline Weakness Profile

Berdasarkan hasil epoch 10–20 (vocab rebuilt, corpus_mixed_v1):

| Domain | Epoch 10 | Epoch 20 | Strength | Echo Pattern |
|--------|----------|----------|----------|--------------|
| membaca | 100% | 80% | **Strong** | `baca baca baca` |
| bekerja | 60% | 80% | **Strong** | `kerja mereka kerja` |
| makan | 0% | 20% | **Medium** | token ikan muncul sekali |
| formal | 0% | 0% | **Weak** | `teri pemerintah teri` — kata benar, echo |
| pergi | 0% | 0% | **Weak** | domain confusion |
| memasak | 0% | 0% | **Weak** | domain confusion |
| belajar | 0% | 0% | **Weak** | `anak bawah anak bawah` |
| memeriksa | 0% | 0% | **Weak** | `anak memeriksa anak` |

**Observasi penting:**
- `formal` output BENAR (`pemerintah menetapkan pemerintahan`) tapi tidak di-count karena echo — ini **false negative** di metric saat ini
- `memeriksa` output: `anak memeriksa anak dasar` — kata `memeriksa` muncul tapi objek salah
- `bekerja` diversity turun 20→5 di epoch 20 — tanda mulai collapse ke pola sempit

---

## 4. Hipotesis Formal (untuk divalidasi di epoch 50)

### H1: Domain Strength Berkorelasi dengan Coverage × Diversity Data

**Prediksi:** Domain yang kuat (`membaca`, `bekerja`) punya coverage tinggi DAN variasi objek tinggi di corpus.

**Test:** Hitung `coverage × unique_objects` per domain di corpus_mixed_v1.  
**Falsifikasi:** Jika domain lemah punya coverage tinggi tapi tetap gagal → H1 salah, masalah ada di tempat lain.

---

### H2: Echo Learning = Gejala Entropy Collapse, Bukan Data

**Prediksi:** Model echo bukan karena data jelek, tapi karena distribusi logit terkonsentrasi pada satu token setelah verb trigger.

**Test:** Print top-5 logit probability setelah token `makan` diproses.  
**Falsifikasi:** Jika distribusi logit flat (entropy tinggi) tapi output tetap echo → H2 salah.

---

### H3: Domain Lemah Karena Gradient Kalah dari Domain Kuat

**Prediksi:** `membaca` dan `bekerja` mendominasi gradient update karena frekuensi tinggi, mengorbankan domain lemah.

**Test:** Bandingkan loss per domain di akhir epoch vs awal.  
**Falsifikasi:** Jika semua domain loss turun merata → H3 salah, masalah ada di inference.

---

### H4: Prompt Token Suppression Menyebabkan Domain Confusion

**Prediksi (sudah sebagian divalidasi):** Token prompt di-suppress → model tidak bisa menggunakan konteks aksi → output acak.

**Status:** Sudah di-fix di GOS (KBBI boost nerf hanya untuk generated tokens). Epoch 30 akan mengkonfirmasi atau menolak fix ini.

**Test:** Bandingkan hit rate epoch 30 (post-fix) vs epoch 20 (pre-fix).

---

## 5. Keputusan Intervensi

**Tidak boleh intervensi sampai:**
1. Epoch 50 selesai
2. Pattern konsisten terlihat di 3+ snapshot
3. Hipotesis terkonfirmasi atau terbantah

**Baru boleh intervensi jika:**

| Kondisi | Intervensi |
|---------|-----------|
| Echo tetap ada sampai ep50 | Naikkan window suppression dari 6→8 |
| Domain lemah tetap 0% sampai ep50 | Curriculum learning: train domain lemah dulu |
| Over-smoothing > 30% | Tambah action corpus proportion ke 30% |
| makan masih 0% setelah vocab fix | Cek bigram "makan nasi" coverage di corpus |

---

## 6. Target Epoch 50

| Metrik | Minimum (OK) | Target (Bagus) | Breakthrough |
|--------|-------------|----------------|--------------|
| Pass count | 4/8 | 5-6/8 | 7-8/8 |
| Avg hit rate | 35% | 50% | >65% |
| Over-smoothing | <20% | <10% | <5% |
| Echo (SVO=0.5 instead of 1.0) | <50% output | <30% output | <10% |

**Jika epoch 50 menghasilkan < 4/8:**
→ Masalah bukan data, bukan vocab
→ Investigasi architecture: apakah semantic bias cukup kuat?
→ Pertimbangkan naikkan alpha 0.15 → 0.25

**Jika epoch 50 menghasilkan ≥ 5/8:**
→ AKSARA terbukti bisa belajar behavior dari data
→ Langkah berikutnya: longer training (epoch 100), lebih banyak action corpus
