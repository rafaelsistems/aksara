"""
AKSARA — Master Run Script (KBBI Aktif)

Menjalankan semua test suite dengan KBBI semantic grounding aktif.
Ini adalah bukti bahwa arsitektur AKSARA bekerja end-to-end
dengan semua komponen termasuk LSK/KBBI.

Urutan:
  1. Validate Mini Loop (KBBI aktif)
  2. Output Inspection (KBBI aktif)
  3. Stress Test (KBBI aktif)
  4. Ablation Study (KBBI aktif)
  5. Baseline Comparison (KBBI aktif)
  6. Generalization Test (KBBI aktif)

Cara jalankan:
    python examples/run_all_with_kbbi.py
    python examples/run_all_with_kbbi.py --quick     # mode cepat (epoch lebih sedikit)
    python examples/run_all_with_kbbi.py --skip-heavy # skip ablation & baseline (lama)
"""

import argparse
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def auto_detect_kbbi() -> str:
    """Auto-detect KBBI path."""
    candidates = [
        "kbbi_true_clean_production.json",
        Path(__file__).parent.parent / "kbbi_true_clean_production.json",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p.resolve())
    return ""


def run_test(name, func, **kwargs):
    """Jalankan satu test dengan error handling."""
    print(f"\n{'#'*80}")
    print(f"#  {name}")
    print(f"{'#'*80}")

    t0 = time.time()
    try:
        result = func(**kwargs)
        elapsed = time.time() - t0
        print(f"\n  ✅ {name} SELESAI ({elapsed:.1f}s)")
        return {"status": "PASS", "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ {name} GAGAL: {e}")
        traceback.print_exc()
        return {"status": "FAIL", "time": elapsed, "error": str(e)}


def run_validate_mini_loop(kbbi_path, epochs):
    """Jalankan validate_mini_loop dengan KBBI."""
    from examples.validate_mini_loop import main as validate_main

    class Args:
        pass
    args = Args()
    args.kbbi = kbbi_path
    args.epochs = epochs

    # validate_mini_loop doesn't take args directly, so we use subprocess
    import subprocess
    cmd = [sys.executable, "examples/validate_mini_loop.py"]
    if kbbi_path:
        cmd.extend(["--kbbi", kbbi_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])
    return {"returncode": result.returncode}


def run_output_inspection(kbbi_path, epochs):
    """Jalankan output inspection."""
    import subprocess
    cmd = [sys.executable, "examples/output_inspection.py", "--epochs", str(epochs)]
    if kbbi_path:
        cmd.extend(["--kbbi", kbbi_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])
    return {"returncode": result.returncode}


def run_stress_test(kbbi_path, epochs):
    """Jalankan stress test."""
    import subprocess
    cmd = [sys.executable, "examples/stress_test.py", "--epochs", str(epochs)]
    if kbbi_path:
        cmd.extend(["--kbbi", kbbi_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])
    return {"returncode": result.returncode}


def run_ablation_study(kbbi_path, epochs):
    """Jalankan ablation study."""
    import subprocess
    cmd = [sys.executable, "examples/ablation_study.py", "--epochs", str(epochs)]
    if kbbi_path:
        cmd.extend(["--kbbi", kbbi_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])
    return {"returncode": result.returncode}


def run_baseline_comparison(kbbi_path, epochs):
    """Jalankan baseline comparison."""
    import subprocess
    cmd = [sys.executable, "examples/baseline_comparison.py", "--epochs", str(epochs)]
    if kbbi_path:
        cmd.extend(["--kbbi", kbbi_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])
    return {"returncode": result.returncode}


def run_generalization_test(kbbi_path, epochs):
    """Jalankan generalization test."""
    import subprocess
    cmd = [sys.executable, "examples/generalization_test.py", "--epochs", str(epochs)]
    if kbbi_path:
        cmd.extend(["--kbbi", kbbi_path])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(result.stdout)
    if result.stderr:
        print(result.stderr[-500:])
    return {"returncode": result.returncode}


def main():
    parser = argparse.ArgumentParser(description="AKSARA Master Run (KBBI Aktif)")
    parser.add_argument("--kbbi", type=str, default="", help="Path ke KBBI JSON")
    parser.add_argument("--quick", action="store_true", help="Mode cepat (epoch sedikit)")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip ablation & baseline (lama)")
    args = parser.parse_args()

    kbbi_path = args.kbbi or auto_detect_kbbi()

    # Epoch settings
    if args.quick:
        ep_validate = 5
        ep_inspect = 6
        ep_stress = 4
        ep_ablation = 5
        ep_baseline = 8
        ep_general = 5
    else:
        ep_validate = 10
        ep_inspect = 12
        ep_stress = 8
        ep_ablation = 10
        ep_baseline = 15
        ep_general = 10

    print(f"\n{'='*80}")
    print(f"  AKSARA — Master Run Suite (KBBI {'AKTIF' if kbbi_path else 'TIDAK AKTIF'})")
    print(f"{'='*80}")
    print(f"  KBBI Path : {kbbi_path if kbbi_path else '(none)'}")
    print(f"  Mode      : {'QUICK' if args.quick else 'FULL'}")
    print(f"  Skip Heavy: {args.skip_heavy}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not kbbi_path:
        print(f"\n  ⚠️  KBBI tidak ditemukan! Jalankan dari root proyek AKSARA")
        print(f"     atau gunakan --kbbi <path>")

    all_results = {}
    total_start = time.time()

    # 1. Validate Mini Loop
    all_results["validate"] = run_test(
        "1. Validate Mini Loop",
        run_validate_mini_loop,
        kbbi_path=kbbi_path, epochs=ep_validate,
    )

    # 2. Output Inspection
    all_results["inspection"] = run_test(
        "2. Output Inspection",
        run_output_inspection,
        kbbi_path=kbbi_path, epochs=ep_inspect,
    )

    # 3. Stress Test
    all_results["stress"] = run_test(
        "3. Stress Test",
        run_stress_test,
        kbbi_path=kbbi_path, epochs=ep_stress,
    )

    if not args.skip_heavy:
        # 4. Ablation Study
        all_results["ablation"] = run_test(
            "4. Ablation Study",
            run_ablation_study,
            kbbi_path=kbbi_path, epochs=ep_ablation,
        )

        # 5. Baseline Comparison
        all_results["baseline"] = run_test(
            "5. Baseline Comparison",
            run_baseline_comparison,
            kbbi_path=kbbi_path, epochs=ep_baseline,
        )

    # 6. Generalization Test
    all_results["generalization"] = run_test(
        "6. Generalization Test",
        run_generalization_test,
        kbbi_path=kbbi_path, epochs=ep_general,
    )

    total_time = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  MASTER RUN SUMMARY")
    print(f"{'='*80}\n")

    print(f"  {'Test':<30} | {'Status':>8} | {'Time':>8}")
    print(f"  {'-'*55}")

    pass_count = 0
    fail_count = 0
    for name, result in all_results.items():
        status = result["status"]
        elapsed = result["time"]
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {name:<30} | {icon} {status:>5} | {elapsed:>6.1f}s")
        if status == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    print(f"\n  Total: {pass_count} PASS, {fail_count} FAIL")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  KBBI: {'AKTIF ✅' if kbbi_path else 'TIDAK AKTIF ⚠️'}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save results
    output_path = Path("aksara_output/master_run_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "kbbi_active": bool(kbbi_path),
        "mode": "quick" if args.quick else "full",
        "total_time_seconds": total_time,
        "results": {k: {"status": v["status"], "time": v["time"]}
                    for k, v in all_results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print()


if __name__ == "__main__":
    main()
