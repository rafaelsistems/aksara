# AKSARA Framework — Formalisasi Matematis Enam Primitif
**Versi:** 5.0  
**Status:** Dokumen Hidup — wajib selesai sebelum implementasi primitif apapun  
**Penulis:** Emylton Leunufna  

---

## Notasi Umum

| Simbol | Definisi |
|---|---|
| `S` | Kalimat input: sequence morfem `[m₁, m₂, ..., mₙ]` |
| `M` | Riemannian manifold untuk representasi semantik |
| `g` | Metrik Riemannian di `M` |
| `μ(w)` | Distribusi probabilistik kata `w` di manifold `M` |
| `C` | Himpunan semua constraint linguistik |
| `G` | Graf dependensi kalimat: `G = (V, E)` |
| `𝒞` | Kategori makna: objek = satuan linguistik, morfisme = fungsi makna |
| `K` | Kompleks simplisial dari kalimat |
| `Π` | Proposisi: representasi makna propositional kalimat |
| `𝔽` | FrameBank: himpunan frame semantik |
| `φ` | Fungsi pencocokan frame: `Π → 𝔽` |
| `ρ` | Fungsi resolusi referensi: `pronoun × konteks → entitas` |

---

## Primitif 1: LPS — Linguistic Parse System

### 1.1 Definisi Formal

LPS adalah fungsi deterministik:

```
LPS: Σ* → M*

di mana:
  Σ* = semua string bahasa Indonesia
  M* = sequence morfem berstruktur
```

Setiap morfem `mᵢ ∈ M*` didefinisikan sebagai tuple:

```
mᵢ = (root, afiks, kelas, peran, posisi)

di mana:
  root   ∈ Leksikon              — bentuk dasar kata
  afiks  ∈ 2^AfiksIndonesia      — himpunan afiks aktif
  kelas  ∈ {N, V, Adj, Adv, ...} — kelas kata
  peran  ∈ {S, P, O, K, Pel}     — peran gramatikal dalam kalimat
  posisi ∈ ℕ                     — indeks dalam kalimat
```

### 1.2 Aturan Afiks (Hard Constraint)

Validitas kombinasi afiks bersifat **deterministik** — bukan probabilistik:

```
valid_afiks: AfiksIndonesia × KelasKata → {True, False}

Contoh:
  valid_afiks(me-, V_root) = True
  valid_afiks(me-, N_root) = True   (denominalisasi)
  valid_afiks(ke-an, V_root) = True (nominalisasi)
  valid_afiks(ke-an, N_root) = True (abstraksi)
  valid_afiks(me-, Adj_root) = False (umumnya)
```

Aturan ini dikodekan sebagai **finite state automaton** — bukan dipelajari dari data.

### 1.3 Reduplikasi

Reduplikasi adalah proses morfologis khusus Indonesia:

```
reduplikasi(w) = {
  penuh:   w-w         → "rumah-rumah"  (pluralitas)
  parsial: w[0:k]-w    → "tetamu"       (dari "tamu")
  berubah: w ~ w'      → "sayur-mayur"  (intensifikasi)
}
```

### 1.4 Output LPS

```
LPS(S) = [m₁, m₂, ..., mₙ] dengan metadata:
  - dependency_tree: pohon dependensi antar morfem
  - constituency: struktur konstituen (S, P, O, K, Pel)
  - affix_validity: {mᵢ: True/False} untuk setiap morfem
```

---

## Primitif 2: SFM — Semantic Field Manifold

### 2.1 Motivasi Matematis

Word embedding biasa mendefinisikan kata sebagai titik di Euclidean space ℝⁿ. Jarak antara dua kata adalah norma Euclidean:

```
d_E(a, b) = ||vₐ - v_b||₂
```

Ini memiliki keterbatasan fundamental: **jarak tidak melewati path semantik yang bermakna**. Kata "raja" dan "ratu" mungkin dekat di Euclidean space, tapi path di antara mereka tidak melewati konsep "pemimpin", "kerajaan", "kekuasaan".

SFM mendefinisikan kata sebagai titik di **Riemannian manifold** di mana jarak antar kata adalah **geodesic** — kurva terpendek di manifold yang melewati konsep-konsep antara.

### 2.2 Definisi Manifold Semantik

Manifold semantik `M` adalah ruang differentiable dengan metrik Riemannian `g`:

```
M = (ℝᵈ, g)

di mana:
  d   = dimensi representasi (hyperparameter)
  g   = metrik yang didefinisikan dari struktur KBBI
```

Metrik `g` pada titik `p ∈ M` didefinisikan dari **densitas relasi leksikal** di sekitar `p`:

```
g(p) = I + λ · ∇²ρ(p)

di mana:
  I    = matriks identitas (ruang "datar" sebagai baseline)
  ρ(p) = densitas relasi KBBI di sekitar titik p
  λ    = parameter kelengkungan (hyperparameter)
  ∇²ρ  = Hessian dari densitas relasi
```

Interpretasi: **di mana banyak kata KBBI berkerumun, ruang manifold lebih "bengkok"** — geodesic dipaksa melewati cluster tersebut.

### 2.3 Representasi Kata sebagai Distribusi

Setiap kata `w` direpresentasikan bukan sebagai titik, tapi sebagai **distribusi Gaussian di manifold**:

```
μ(w) = N_M(θ(w), Σ(w))

di mana:
  θ(w) ∈ M        — titik pusat kata w di manifold
  Σ(w) ∈ S²₊(d)  — matriks kovarians (ketidakpastian semantik)
```

Kata yang maknanya spesifik punya `Σ` kecil (distribusi sempit).  
Kata yang maknanya polisemik punya `Σ` besar (distribusi lebar).

```
Contoh:
  "nasi"     → θ dekat cluster kuliner, Σ kecil (makna spesifik)
  "ada"      → Σ besar (makna sangat polisemik: eksistensi, kepunyaan, dll.)
  "makanan"  → Σ sedang (lebih umum dari "nasi" tapi tidak sepolisemik "ada")
```

### 2.4 Jarak Semantik

Jarak antara dua kata adalah **Wasserstein distance** antar distribusinya di manifold:

```
d_SFM(a, b) = W₂(μ(a), μ(b))

di mana W₂ adalah 2-Wasserstein distance:
  W₂²(μ, ν) = inf_{γ ∈ Γ(μ,ν)} ∫ d_M(x,y)² dγ(x,y)
```

Ini memberikan **path optimal transportasi** dari distribusi `a` ke distribusi `b` — yang secara alami melewati konsep-konsep antara di manifold.

### 2.5 Context-Sensitivity

State kata berubah tergantung konteks kalimat. State kata `w` dalam kalimat `S` adalah:

```
θ(w | S) = θ(w) + Δ(w, S)

di mana:
  Δ(w, S) = Σⱼ weight(w, mⱼ) · (θ(mⱼ) - θ(w)) · α(dep(w, mⱼ))

  weight = bobot berdasarkan jarak dependensi
  α      = faktor decay berdasarkan tipe relasi dependensi
```

### 2.6 Membangun Manifold dari KBBI

Manifold dibangun melalui **propagasi relasi graf KBBI**:

```
Langkah 1: Inisialisasi
  Setiap lemma KBBI → titik awal θ₀(w) secara acak di ℝᵈ

Langkah 2: Propagasi relasi
  Untuk setiap relasi (w₁, rel, w₂) di KBBI:
    tarikan:  θ(w₁) ← θ(w₁) + η · (θ(w₂) - θ(w₁)) · f(rel)
    tolakan:  θ(w₁) ← θ(w₁) - η · (θ(w₂) - θ(w₁)) · f(antonim)

  di mana f(rel) = bobot per tipe relasi:
    f(sinonim)   = 1.0  (sangat dekat)
    f(hiponim)   = 0.7  (cukup dekat)
    f(domain)    = 0.5  (satu kelompok)
    f(derivasi)  = 0.4  (terkait morfologis)
    f(antonim)   = -0.8 (berlawanan arah)

Langkah 3: Metrik dari densitas
  Hitung ρ(p) = densitas titik KBBI di sekitar p
  Definisikan g(p) dari ∇²ρ(p)

Langkah 4: Konvergensi
  Iterasi sampai perubahan posisi < δ
```

---

## Primitif 3: CPE — Constraint Propagation Engine

### 3.1 Definisi Graf Dependensi

Graf dependensi kalimat `S`:

```
G = (V, E, τ)

di mana:
  V = {m₁, m₂, ..., mₙ}           — set morfem
  E ⊆ V × V                        — relasi dependensi
  τ: E → RelType                   — tipe relasi (subjek, objek, modifier, dll.)
```

### 3.2 Fungsi Ketegangan (Tension Function)

Ketegangan antara dua morfem yang terhubung:

```
tension(mᵢ, mⱼ, τᵢⱼ) = Σₖ wₖ · cₖ(mᵢ, mⱼ, τᵢⱼ)

di mana:
  k    ∈ {morfologis, sintaktis, semantik, leksikal}
  wₖ   = bobot per jenis constraint (learnable)
  cₖ   = skor constraint ke-k ∈ [0, 1]
          0 = tidak ada ketegangan (constraint terpenuhi)
          1 = ketegangan maksimal (constraint dilanggar)
```

### 3.3 Energi Total Kalimat

```
E(S) = Σ_{(i,j) ∈ E} tension(mᵢ, mⱼ, τᵢⱼ)

Kalimat valid   → E(S) rendah (mendekati 0)
Kalimat invalid → E(S) tinggi
```

### 3.4 Constraint per Jenis

**Constraint Morfologis:**
```
c_morph(mᵢ, mⱼ, τ) = 1 - valid_afiks(afiks(mᵢ), kelas(mⱼ)) 
                       jika τ adalah relasi head-dependent
```

**Constraint Sintaktis:**
```
c_synth(mᵢ, mⱼ, τ) = 1 - P(τ | kelas(mᵢ), kelas(mⱼ))
                       di mana P berasal dari tata bahasa Indonesia formal
```

**Constraint Semantik:**
```
c_sem(mᵢ, mⱼ, τ) = d_SFM(mᵢ, mⱼ) / d_max
                     dinormalisasi ke [0, 1]
                     tinggi = makna jauh = ketegangan semantik tinggi
```

**Constraint Leksikal:**
```
c_lex(mᵢ, mⱼ, τ) = |register(mᵢ) - register(mⱼ)|
                     di mana register ∈ [0, 1] (0=informal, 1=formal)
                     inkonsistensi register = ketegangan leksikal
```

### 3.5 Algoritma Propagasi

```
INPUT: G = (V, E, τ), state awal {θ(mᵢ)} dari SFM

ALGORITMA CPE:
  t = 0
  state⁰ = {θ(mᵢ) | i ∈ V}  ← dari SFM
  
  REPEAT:
    untuk setiap mᵢ ∈ V:
      tetangga_i = {mⱼ | (i,j) ∈ E atau (j,i) ∈ E}
      
      tension_i = Σⱼ∈tetangga tension(mᵢ, mⱼ, τᵢⱼ)
      
      state^{t+1}(mᵢ) = state^t(mᵢ) - η · ∇_{state(mᵢ)} tension_i
    
    δ = ||state^{t+1} - state^t||
    t = t + 1
  
  UNTIL δ < ε ATAU t = max_iter
  
OUTPUT:
  state_final   = state^t          ← state kesetimbangan
  E_total       = E(S)             ← energi total
  violations    = {(i,j, c) | tension(mᵢ,mⱼ,τ) > threshold}
```

### 3.6 Perbedaan Kritis dari Attention

Attention Transformer menghitung:
```
Attention(Q, K, V) = softmax(QKᵀ/√d) · V
```
Ini adalah **weighted average** — selalu menghasilkan kombinasi linear dari semua token.

CPE menghitung:
```
state_final = argmin_{state} E(S | G, constraints)
```
Ini adalah **minimisasi energi** — hasilnya adalah kesetimbangan sistem, bukan kombinasi linear. Depth reasoning tidak fixed — ia berhenti saat sistem konvergen.

---

## Primitif 4: CMC — Categorical Meaning Composer

### 4.1 Definisi Kategori Makna

Definisikan kategori 𝒞:

```
𝒞 = (Ob(𝒞), Hom(𝒞), ∘, id)

di mana:
  Ob(𝒞)         = {morfem, frasa, klausa, kalimat}   ← objek
  Hom(𝒞)(A, B)  = {f: A → B | f adalah morfisme makna} ← morfisme
  ∘              = komposisi morfisme
  id_A           = morfisme identitas untuk objek A
```

### 4.2 Hukum Kategori yang Harus Dipenuhi

**Asosiativitas:**
```
(f ∘ g) ∘ h = f ∘ (g ∘ h)

Artinya: cara kita menyusun makna dari kiri atau kanan
         harus menghasilkan makna yang sama.
```

**Identitas:**
```
id_B ∘ f = f = f ∘ id_A

Artinya: kata tanpa konteks punya morfisme identitas —
         tidak mengubah makna apapun.
```

### 4.3 Morfisme per Kelas Kata

Setiap kelas kata mendefinisikan tipe morfisme yang berbeda:

```
Nomina (N):
  f_N: Konteks → N → NP
  "nasi" dalam konteks kalimat aktif → frasa nominal subjek/objek

Verba (V):
  f_V: NP × NP → Klausa
  "makan" mengikat subjek dan objek → klausa utama

Adjektiva (Adj):
  f_Adj: N → N'  (modifikasi nomina)
  "lezat" memodifikasi nomina → nomina yang dipersempit maknanya

Adverbia (Adv):
  f_Adv: V → V'  (modifikasi verba)
  "cepat" memodifikasi verba → verba dengan cara berbeda
```

### 4.4 Komposisi Non-Commutative

Bahasa Indonesia memiliki urutan kata yang bermakna. Komposisi bersifat **non-commutative**:

```
f_Adj ∘ makna(N) ≠ makna(N) ∘ f_Adj   (secara umum)

"makanan lezat" = f_lezat ∘ makna("makanan")  ✅
"lezat makanan" = makna("makanan") ∘ f_lezat  → tidak natural ⚠️
```

Ini dikodekan sebagai constraint di kategori: beberapa morfisme **tidak commutatif**.

### 4.5 Deteksi Inkompatibilitas Makna

Kalimat invalid ketika **morfisme tidak bisa dikomposisi secara valid**:

```
"makanan meriam"
  f_meriam: tipe domain(senjata) → tidak punya morfisme valid ke domain(kuliner)
  → komposisi f_meriam ∘ makna("makanan") = UNDEFINED ❌
  → ini adalah indikator kalimat invalid

"makanan lezat"
  f_lezat: tipe domain(atribut_rasa) → punya morfisme ke domain(kuliner)
  → komposisi f_lezat ∘ makna("makanan") = DEFINED ✅
```

### 4.6 Implementasi sebagai Typed Morphism Graph

Secara komputasional, morfisme diimplementasikan sebagai **typed graph**:

```
Node: satuan linguistik (morfem, frasa, klausa)
Edge: morfisme dengan tipe (domain_source, domain_target, valid: bool)

Komposisi valid jika:
  domain_target(f₁) = domain_source(f₂)
  
Deteksi: DFS/BFS pada typed graph untuk mencari path valid
Kompleksitas: O(|E|) — linear terhadap jumlah relasi
```

---

## Primitif 5: TDA — Topological Dependency Analyzer

### 5.1 Membangun Kompleks Simplisial

Dari kalimat `S = [m₁, ..., mₙ]` dan state SFM, bangun kompleks simplisial `K`:

```
K = ∪_{k=0}^{p} Kₖ

di mana:
  K₀ = {mᵢ}                          — 0-simplex: setiap morfem
  K₁ = {(mᵢ, mⱼ) | d_SFM(mᵢ,mⱼ) < ε₁}  — 1-simplex: pasangan dekat
  K₂ = {(mᵢ,mⱼ,mₖ) | semua pasangan < ε₂} — 2-simplex: triple
  ...hingga dimensi p (biasanya p=2 cukup untuk analisis kalimat)
```

Threshold `ε` bervariasi dari kecil ke besar — ini yang menghasilkan **persistence**.

### 5.2 Persistent Homology

Saat threshold `ε` meningkat dari 0 ke ∞, fitur topologis **lahir dan mati**:

```
β₀ = jumlah komponen terhubung   (cluster kata)
β₁ = jumlah loop independen      (siklus makna)
β₂ = jumlah rongga/void          (kekosongan makna)
```

**Barcode diagram:** setiap fitur punya interval [lahir, mati]:
```
Fitur persisten (interval panjang) → relasi makna yang kuat
Fitur sementara (interval pendek)  → noise / relasi lemah
```

### 5.3 Aplikasi untuk Deteksi Inkoherensi

**Kalimat valid:**
```
"Anak itu makan nasi goreng di warung."
  → β₀ = 1 (semua kata terhubung dalam satu komponen)
  → β₁ = 0 (tidak ada loop anomali)
  → persistent diagram: semua fitur lahir awal, mati terlambat ✅
```

**Kalimat invalid (domain swap):**
```
"Anak itu makan meriam goreng di warung."
  → β₀ = 2 sementara (ada kata yang terisolasi secara semantik)
  → "meriam" tidak masuk komponen utama sampai ε sangat besar
  → persistent diagram: ada fitur yang lahir terlambat ❌
```

**Kalimat invalid (negasi kompleks):**
```
"Tidak ada yang tidak tidak hadir."
  → β₁ > 0 (ada loop — negasi ganda menghasilkan siklus)
  → persistent diagram: ada loop yang tidak biasa ⚠️
```

### 5.4 Fitur Topologis sebagai Representasi

Dari persistent homology, ekstrak **persistence diagram** sebagai vektor fitur:

```
PD(S) = {(b, d) | b = birth, d = death, untuk setiap fitur}

Ubah ke vektor via persistence image atau persistence landscape:
  PI(S) = vektor fitur topologis kalimat S
```

Ini memberikan **representasi invariant** terhadap perubahan kecil di state SFM — robust terhadap noise.

### 5.5 Kompleksitas Komputasi

```
Membangun kompleks:    O(n²) terhadap jumlah morfem
Persistent homology:   O(n³) worst case, O(n·α(n)) average dengan struktur sparse

Optimasi untuk kalimat pendek-menengah (n < 50):
  Batasi ke p=2 (2-simplex)
  Gunakan Vietoris-Rips dengan threshold adaptif
  Kompleksitas praktis: O(n² log n)
```

---

## 6. Integrasi: Bagaimana Lima Primitif Bekerja Bersama

### 6.1 Flow Lengkap

```
S = "Makanan tradisional khas Dompu sangat meriam."

Step 1 — LPS:
  parse → [{root:"makan", sufiks:"-an", kelas:N},
            {root:"tradisional", kelas:Adj},
            {root:"khas", kelas:Adj},
            {root:"Dompu", kelas:N_proper},
            {root:"sangat", kelas:Adv},
            {root:"meriam", kelas:N}]
  dependency: [Adj→N, Adj→N, Adv→Adj, N→P]
  catatan: tidak ada predikat eksplisit → ketidaklengkapan

Step 2 — SFM:
  θ("makanan") → cluster kuliner, Σ sedang
  θ("meriam")  → cluster senjata, Σ kecil
  d_SFM(makanan, meriam) = BESAR → jauh di manifold

Step 3 — CPE:
  tension(makanan, meriam, modifier) = tinggi
  → c_sem = 0.95 (domain kuliner ≠ domain senjata)
  E(S) = tinggi → kalimat tidak stabil

Step 4 — CMC:
  f_meriam: domain(senjata) → tidak ada morfisme valid ke domain(kuliner)
  → komposisi UNDEFINED pada pair (sangat, meriam)

Step 5 — TDA:
  "meriam" tidak masuk komponen utama sampai ε besar
  → persistent diagram menunjukkan anomali topologis

OUTPUT yang digabungkan:
  valid      = False
  confidence = 0.97
  violations = [
    {type: "semantic_mismatch", tokens: ["makanan", "meriam"],
     dimension: "semantic", severity: 0.95,
     explanation: "domain kuliner ≠ domain senjata"},
    {type: "morphism_undefined", tokens: ["sangat", "meriam"],
     dimension: "lexical", severity: 0.80,
     explanation: "adverbia intensitas tidak valid untuk nomina benda"},
  ]
  topological_anomaly = True
  energy = 0.847
```

### 6.2 Formula Skor Akhir

```
skor_linguistik(S) = σ(-E(S)) · w_energy
                   + coherence_TDA(S) · w_tda  
                   + morphism_validity_CMC(S) · w_cmc

di mana σ = sigmoid, dan w adalah bobot yang bisa di-tune
```

---

## 7. Properti Matematis yang Harus Dipertahankan Selama Implementasi

### 7.1 SFM
- [ ] Metrik `g` harus positive definite di semua titik
- [ ] Geodesic harus unique (manifold convex secara lokal)
- [ ] Wasserstein distance harus memenuhi triangle inequality

### 7.2 CPE
- [ ] Algoritma propagasi harus konvergen (energi monoton turun)
- [ ] State akhir harus unik untuk input yang sama (deterministic)
- [ ] Kompleksitas harus O(n · d) per iterasi, bukan O(n²)

### 7.3 CMC
- [ ] Hukum asosiativitas harus terpenuhi untuk semua komposisi
- [ ] Morfisme identitas harus ada untuk setiap kelas kata
- [ ] Non-commutativity harus dikodekan secara eksplisit

### 7.4 TDA
- [ ] Persistent homology harus stable (kecil perubahan input → kecil perubahan diagram)
- [ ] Persistence diagram harus invariant terhadap isometri manifold
- [ ] Kompleksitas komputasi harus feasible untuk kalimat panjang

---

### 7.5 KRL
- [ ] Encoder harus deterministik: input morfem yang sama → proposisi yang sama
- [ ] FrameMatcher harus idempoten: cocokkan ulang → skor identik
- [ ] ReferenceResolver harus konsisten: urutan kalimat sama → ikatan sama
- [ ] Kelengkapan pemahaman monoton: lebih banyak slot terisi → skor lebih tinggi

---

*Setiap properti di atas harus diverifikasi dengan unit test sebelum primitif dianggap selesai diimplementasi.*

---

## Primitif 6: KRL — Knowledge Representation Layer

### 6.1 Motivasi

Lima primitif (LPS/SFM/CPE/CMC/TDA) menghasilkan **validasi** kalimat — apakah kalimat benar secara linguistik. Namun belum ada komponen yang menghasilkan **pemahaman** — representasi makna yang bisa di-reasoning.

KRL menjembatani gap ini dengan tiga fungsi:

```
KRL: M* → (Π, Frame, R)

di mana:
  M*    = sequence morfem dari LPS
  Π     = Proposisi — representasi propositional kalimat
  Frame = frame semantik yang paling cocok dari FrameBank
  R     = himpunan ikatan referensi (anafor → anteseden)
```

### 6.2 Proposisi

Proposisi adalah struktur `(aksi, slot)` di mana:

```
Π = ⟨aksi, {τ₁: v₁, τ₂: v₂, ..., τₖ: vₖ}⟩

di mana:
  aksi   ∈ Σ_verba      — root verba predikat
  τᵢ    ∈ TipeSlot      — peran tematik (AGEN, PASIEN, LOKASI, ...)
  vᵢ    ∈ Σ*            — nilai slot (token/frasa)
```

Mapping dari peran gramatikal TBBBI ke peran tematik:

```
SUBJEK  →  AGEN   (kalimat aktif)  |  PASIEN (kalimat pasif di-)
PREDIKAT → aksi   (verba utama)
OBJEK   →  PASIEN (kalimat aktif)  |  AGEN   (kalimat pasif)
KET+di  →  LOKASI
KET+ke  →  TUJUAN
KET+dari → ASAL
KET+dengan → CARA
KET+karena → SEBAB
```

**Kelengkapan proposisi:**

```
κ(Π) = Σᵢ wᵢ · 𝟙[τᵢ ∈ Π.slot]

w_aksi   = 0.25
w_agen   = 0.25
w_pasien = 0.25
w_ket    = 0.25  (salah satu keterangan cukup)
```

### 6.3 FrameBank

FrameBank adalah himpunan frame semantik:

```
𝔽 = {f₁, f₂, ..., f₁₂}

Setiap frame fᵢ = ⟨nama, V_pemicu, S_wajib, S_opsional, domain⟩
  V_pemicu  = himpunan verba root yang mengaktifkan frame
  S_wajib   = slot yang wajib ada untuk situasi lengkap
  S_opsional = slot opsional
  domain    = domain semantik utama
```

**12 Frame inti bahasa Indonesia:**

| Frame | Domain | Slot Wajib |
|---|---|---|
| JUAL_BELI | ekonomi | pembeli, barang |
| PERJALANAN | aktivitas | pelaku, tujuan |
| KOMUNIKASI | aktivitas | pengirim, pesan |
| PENDIDIKAN | pendidikan | pelajar, materi |
| KESEHATAN | kesehatan | pasien, tindakan |
| HUKUM_PIDANA | hukum | terdakwa, perbuatan |
| PEMBUATAN | aktivitas | pembuat, hasil |
| KEPEMILIKAN | ekonomi | pemilik, objek |
| ATRIBUSI | deskriptif | entitas, atribut |
| KONFLIK | sosial | pihak_1, pihak_2 |
| PEMERINTAHAN | hukum | otoritas, kebijakan |
| EKSISTENSI | deskriptif | entitas |

### 6.4 FrameMatcher

Fungsi skor kecocokan proposisi terhadap frame:

```
skor(Π, f) = α · 𝟙[aksi(Π) ∈ V_pemicu(f) ∨ strip(aksi) ∈ V_pemicu(f)]
           + β · coverage_wajib(Π, f)
           + γ · 𝟙[domain(aksi) = domain(f)]

di mana:
  α = 0.6   (bobot verba pemicu)
  β = 0.4   (bobot coverage slot wajib)
  γ = 0.1   (bonus domain cocok)

coverage_wajib(Π, f) = |{s ∈ S_wajib(f) : ∃τ ∈ Π.slot, τ ↦ s}| / |S_wajib(f)|
```

Frame terbaik:
```
f* = argmax_{f ∈ 𝔽} skor(Π, f)
```

Threshold minimal: `skor(Π, f*) ≥ 0.3` — jika tidak terpenuhi, tidak ada frame yang cocok.

### 6.5 ReferenceResolver

Resolusi anafor berbasis aturan — bukan attention:

```
ρ(a, K) = argmax_{e ∈ Entitas(K)} kompatibilitas(a, e) · recency(e, K)

di mana:
  a        = token anaforis (pronomina/demonstrativa)
  K        = konteks wacana (proposisi sebelumnya, window=5)
  Entitas(K) = semua slot entitas dari proposisi dalam K

kompatibilitas(a, e):
  pronomina persona tunggal → e harus PERSONA/ORGANISASI (skor 0.85)
  pronomina persona jamak   → e harus PERSONA/ORGANISASI jamak (skor 0.70)
  demonstrativa anaforis    → e = entitas NP terakhir apapun (skor 0.80)

recency(e, K) = 1 / (1 + jarak_kalimat(e, K_sekarang))
```

**Tipe anafor yang ditangani:**
- Pronomina persona: `dia`, `ia`, `beliau`, `mereka`, `-nya`
- Demonstrativa anaforis: `itu`, `tersebut`, `tadi`, `dimaksud`

### 6.6 Kelengkapan Pemahaman

```
Ω(Π, f*) = 0.4 · 𝟙[Π ≠ ∅]
          + 0.2 · 𝟙[κ(Π) ≥ 0.75]
          + 0.3 · 𝟙[f* ≠ ∅]
          + 0.1 · 𝟙[coverage_wajib(Π, f*) = 1.0]
```

`Ω ∈ [0, 1]` — makin tinggi, makin lengkap pemahaman framework terhadap kalimat.

### 6.7 Oposisi terhadap Transformer

| Aspek | Transformer | AKSARA KRL |
|---|---|---|
| Proposisi | Implisit di vektor, perlu probing | Eksplisit, bisa dibaca langsung |
| Frame | Tidak ada — distribusi token | 12 frame deterministik, verifiable |
| Referensi | Attention weight, tidak transparan | Aturan kompatibilitas + recency |
| Kelengkapan | Tidak terukur | Terukur [0,1] via `Ω` |
| Reasoning | Perlu fine-tuning khusus | Langsung dari struktur proposisi |
