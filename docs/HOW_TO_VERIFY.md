# How to Verify the Model

Panduan singkat untuk memverifikasi baseline publik AKSARA.

## 1. Jalankan test publik

```bash
pytest tests/test_public_generation_output.py -q
```

## 2. Jalankan demo minimal

```bash
python examples/minimal_public_demo.py
```

## 3. Jalankan CLI ringkas

```bash
python examples/mini_cli.py
```

## 4. Cek hasil

Pastikan:
- program berjalan tanpa error
- `ringkasan()` mengembalikan string
- `jelaskan()` mengembalikan string
- output tetap konsisten untuk kalimat sederhana

## Catatan

Jika `kbbi_core_v2.json` tidak tersedia di folder publik ini, beberapa demo akan berhenti lebih awal dengan pesan yang jelas.
