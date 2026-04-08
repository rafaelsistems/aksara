# AKSARA
**Adaptive Knowledge & Semantic Architecture for Bahasa Representation & Autonomy**

> *"Kami tidak mengajarkan model bahasa Indonesia. Kami membuat model lahir sebagai bahasa Indonesia."*

## Status Rilis Publik

Repositori ini adalah baseline publik yang sudah disanitasi:
- hanya berisi allowlist file/folder aman
- tidak menyertakan dataset internal
- tidak menyertakan checkpoint atau artefak training
- tidak menyertakan file audit/debug internal

## Cara Verifikasi Cepat

```bash
pytest tests/test_framework_end_to_end.py -q
pytest tests/test_large_evaluator_smoke.py -q
pytest tests/test_framework_generation_reasoning.py -q
```

## Fokus Proyek

AKSARA adalah pipeline pemahaman linguistik untuk bahasa Indonesia yang:
- dapat dibuat sebagai model
- dapat menghasilkan generasi interpretif
- dapat melakukan reasoning berbasis struktur bahasa
- mengutamakan pemahaman linguistik, bukan prediksi token ala Transformer/Mamba

## Contoh Penggunaan

```python
from aksara.framework import AksaraFramework

fw = AksaraFramework.dari_kbbi("kbbi_core_v2.json")
state = fw.proses("Budi membeli beras di pasar.")
print(state.ringkasan())
print(state.jelaskan())
```

## Catatan Publik

Semua file sensitif dan internal sudah dikeluarkan dari baseline publik.
