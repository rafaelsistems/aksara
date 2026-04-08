# Definisi Formal CLM

## Cognitive Language Model (CLM)

**Cognitive Language Model (CLM)** adalah kelas model bahasa yang dirancang untuk:
- memahami konteks secara eksplisit
- membangun representasi internal berbasis struktur dan makna
- melakukan reasoning, bukan sekadar next-token prediction
- menghasilkan output generatif yang diturunkan dari pemahaman, grounding semantik, dan constraint struktural

## Karakteristik Utama

CLM biasanya memiliki ciri:
1. **Understanding-centric**
   - inti prosesnya adalah pemahaman
2. **Reasoning-enabled**
   - sistem mampu menalar hubungan antar elemen bahasa
3. **Semantically grounded**
   - representasi terikat pada makna dan domain knowledge
4. **Generative**
   - mampu menghasilkan keluaran bahasa yang koheren
5. **Portable**
   - dapat diadaptasi ke bahasa atau domain lain

## Posisi AKSARA

AKSARA adalah framework yang menghasilkan model **Cognitive Language Model (CLM)** untuk bahasa Indonesia.
Implementasi publik saat ini menunjukkan:
- generasi teks
- reasoning berbasis struktur
- grounding semantik
- evaluasi kualitas output
- penolakan terhadap paradigma token-prediction murni sebagai inti utama

## Batasan Istilah

CLM **bukan** sekadar:
- tokenizer
- classifier
- next-token predictor murni

CLM juga **bukan harus** Transformer atau Mamba.
CLM adalah kategori yang lebih luas, dengan fokus pada pemahaman dan penalaran.

## Ringkasan

Jika diringkas:
> CLM adalah model bahasa yang memusatkan pemrosesan pada pemahaman dan reasoning, lalu menggunakan pemahaman itu untuk menghasilkan bahasa secara generatif.
