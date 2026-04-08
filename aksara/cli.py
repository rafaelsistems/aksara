"""aksara.cli — Command Line Interface untuk AKSARA framework.

Penggunaan:
    py -m aksara <command> [opsi]

Commands:
    audit     — Jalankan native framework audit (10-point checklist)
    generate  — Generate teks dari prompt menggunakan checkpoint
    export    — Export model ke direktori checkpoint
    info      — Tampilkan info checkpoint (versi, params, vocab)

Contoh:
    py -m aksara audit
    py -m aksara info --checkpoint ./ckpt
    py -m aksara generate --checkpoint ./ckpt --prompt "anak membaca"
    py -m aksara export --checkpoint ./ckpt --output ./ckpt_export
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _is_blank_text(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip() == ""


def _normalize_prompt_text(value: str) -> str:
    return " ".join(value.strip().split())


def load_yaml_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file tidak ditemukan: '{path}'")
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _parse_yaml_minimal(path)


def _parse_yaml_minimal(path: str) -> dict:
    result: dict = {}
    stack: list[tuple[int, dict]] = [(-1, result)]
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if not line or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key_raw, _, val_raw = stripped.partition(":")
            key = key_raw.strip()
            val_str = val_raw.strip()
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if val_str == "" or val_str.startswith("#"):
                new_dict: dict = {}
                parent[key] = new_dict
                stack.append((indent, new_dict))
            else:
                val_clean = val_str.split("#")[0].strip()
                parent[key] = _parse_yaml_scalar(val_clean)
    return result


def _parse_yaml_scalar(s: str):
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s.lower() in ("null", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


YAML_CONFIG_VERSION = "1.0"
SUPPORTED_CONFIG_VERSIONS = {"1.0"}
_YAML_SCHEMA: dict[tuple, dict] = {
    ("bsu_config", "d_morpheme"): {"type": int, "min": 8, "max": 2048, "req": False},
    ("bsu_config", "d_semantic"): {"type": int, "min": 8, "max": 2048, "req": False},
    ("bsu_config", "d_role"): {"type": int, "min": 4, "max": 512, "req": False},
    ("bsu_config", "d_context"): {"type": int, "min": 8, "max": 2048, "req": False},
    ("meb_config", "n_layers"): {"type": int, "min": 1, "max": 64, "req": False},
    ("meb_config", "n_dep_heads"): {"type": int, "min": 1, "max": 32, "req": False},
    ("meb_config", "dropout"): {"type": float, "min": 0.0, "max": 0.9, "req": False},
    ("gos_config", "teacher_forcing"): {"type": bool, "min": None, "max": None, "req": False},
    ("lps_config", "dep_window"): {"type": int, "min": 1, "max": 20, "req": False},
    ("lps_config", "min_root_length"): {"type": int, "min": 1, "max": 10, "req": False},
    ("lsk_config", "kbbi_path"): {"type": str, "min": None, "max": None, "req": False},
    ("lsk_config", "kbbi_vector_dim"): {"type": int, "min": 4, "max": 512, "req": False},
    ("lsk_config", "max_lemmas"): {"type": int, "min": 1000, "max": 200000, "req": False},
    ("lsk_config", "pretrained_path"): {"type": str, "min": None, "max": None, "req": False},
    (None, "label_smoothing"): {"type": float, "min": 0.0, "max": 0.5, "req": False},
    (None, "lambda_root"): {"type": float, "min": 0.0, "max": 10.0, "req": False},
    (None, "lambda_fluency"): {"type": float, "min": 0.0, "max": 5.0, "req": False},
}
_KNOWN_SECTIONS = {"bsu_config", "meb_config", "gos_config", "lps_config", "lsk_config"}
_KNOWN_TOP_LEVEL = {"config_version", "label_smoothing", "lambda_root", "lambda_fluency"} | _KNOWN_SECTIONS


class YAMLConfigError(ValueError):
    pass


def validate_yaml_config(raw: dict, strict: bool = False) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    cfg_ver = raw.get("config_version")
    if cfg_ver is not None:
        cfg_ver_str = str(cfg_ver)
        if cfg_ver_str not in SUPPORTED_CONFIG_VERSIONS:
            errors.append(
                f"config_version '{cfg_ver_str}' tidak didukung. Versi yang didukung: {SUPPORTED_CONFIG_VERSIONS}. "
                "Jalankan `py -m aksara init --force` untuk generate config terbaru."
            )
    for (section, field), spec in _YAML_SCHEMA.items():
        if section is None:
            val = raw.get(field)
            loc = field
        else:
            sec_dict = raw.get(section, {})
            val = sec_dict.get(field) if isinstance(sec_dict, dict) else None
            loc = f"{section}.{field}"
        if val is None:
            if spec["req"]:
                errors.append(f"Field wajib '{loc}' tidak ditemukan di config.")
            continue
        expected_type = spec["type"]
        if expected_type is float and isinstance(val, int):
            val = float(val)
        if not isinstance(val, expected_type):
            errors.append(f"'{loc}': tipe tidak valid. Diharapkan {expected_type.__name__}, dapat {type(val).__name__} (nilai: {val!r}).")
            continue
        lo, hi = spec["min"], spec["max"]
        if lo is not None and val < lo:
            errors.append(f"'{loc}': nilai {val!r} terlalu kecil (minimum: {lo}).")
        if hi is not None and val > hi:
            errors.append(f"'{loc}': nilai {val!r} terlalu besar (maksimum: {hi}).")
    for key in raw:
        if key not in _KNOWN_TOP_LEVEL:
            warnings.append(f"Field tidak dikenal di top-level: '{key}'. Mungkin typo? Field yang valid: {sorted(_KNOWN_TOP_LEVEL)}.")
    for sec in _KNOWN_SECTIONS:
        sec_dict = raw.get(sec)
        if sec_dict is None or not isinstance(sec_dict, dict):
            continue
        known_in_sec = {f for (s, f) in _YAML_SCHEMA if s == sec}
        for key in sec_dict:
            if key not in known_in_sec:
                warnings.append(f"Field tidak dikenal di '{sec}': '{key}'. Mungkin typo? Field yang valid: {sorted(known_in_sec)}.")
    if errors:
        err_block = "\n".join(f"  ✗ {e}" for e in errors)
        raise YAMLConfigError(f"Config YAML tidak valid ({len(errors)} error):\n{err_block}")
    if strict and warnings:
        warn_block = "\n".join(f"  ⚠ {w}" for w in warnings)
        raise YAMLConfigError(
            f"Config YAML tidak lolos strict mode ({len(warnings)} warning dijadikan error):\n{warn_block}\nPerbaiki field yang tidak dikenal, atau jalankan tanpa strict mode."
        )
    return warnings


def merge_config(base_path: str, *override_paths: str, strict: bool = False, resolve: str = "override") -> dict:
    merged, _ = merge_config_with_report(base_path, *override_paths, strict=strict, resolve=resolve)
    return merged


def merge_config_with_report(base_path: str, *override_paths: str, strict: bool = False, resolve: str = "override") -> tuple:
    valid_strategies = {"override", "base", "critical-safe"}
    if resolve not in valid_strategies:
        raise YAMLConfigError(f"resolve strategy tidak valid: '{resolve}'. Pilihan: {sorted(valid_strategies)}")
    import copy
    def _flatten(d: dict, prefix: str = "") -> dict:
        result = {}
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, full))
            else:
                result[full] = v
        return result
    def _deep_merge(base: dict, override: dict) -> dict:
        result = copy.deepcopy(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = _deep_merge(result[key], val)
            else:
                result[key] = val
        return result
    merged = load_yaml_config(base_path)
    validate_yaml_config(merged, strict=strict)
    all_paths = [base_path] + list(override_paths)
    conflicts: list[dict] = []
    for i, path in enumerate(override_paths):
        layer = load_yaml_config(path)
        validate_yaml_config(layer, strict=strict)
        flat_merged = _flatten(merged)
        flat_layer = _flatten(layer)
        prev_name = os.path.basename(all_paths[i])
        winner_name = os.path.basename(path)
        layer_filtered = {}
        for full_key, new_val in flat_layer.items():
            old_val = flat_merged.get(full_key)
            if old_val is not None and old_val != new_val:
                sev = _DIFF_SEVERITY.get(full_key, "minor")
                if resolve == "base":
                    resolved_val = old_val
                    strategy_winner = prev_name
                elif resolve == "critical-safe" and sev == "critical":
                    resolved_val = old_val
                    strategy_winner = prev_name
                else:
                    resolved_val = new_val
                    strategy_winner = winner_name
                conflicts.append({
                    "field": full_key,
                    "winner": winner_name,
                    "loser": prev_name,
                    "old_value": old_val,
                    "new_value": new_val,
                    "severity": sev,
                    "resolved": resolved_val,
                    "strategy_winner": strategy_winner,
                    "explanation": _FIELD_EXPLANATION.get(full_key, ""),
                })
                if strategy_winner == prev_name:
                    parts = full_key.split(".", 1)
                    if len(parts) == 2:
                        sec, field = parts
                        layer_filtered.setdefault(sec, {})
                        layer_filtered[sec][field] = old_val
                    else:
                        layer_filtered[full_key] = old_val
                    continue
            parts = full_key.split(".", 1)
            if len(parts) == 2:
                sec, field = parts
                layer_filtered.setdefault(sec, {})
                layer_filtered[sec][field] = new_val
            else:
                layer_filtered[full_key] = new_val
        merged = _deep_merge(merged, layer_filtered)
    validate_yaml_config(merged, strict=strict)
    return merged, conflicts


def config_from_yaml(yaml_path: str, strict: bool = False, verbose: bool = False) -> "AksaraConfig":
    from aksara.core.bsu import BSUConfig
    from aksara.core.meb import MEBConfig
    from aksara.core.gos import GOSConfig
    from aksara.linguistic.lps import LPSConfig
    from aksara.linguistic.lsk import LSKConfig
    from aksara.core.model import AksaraConfig
    raw = load_yaml_config(yaml_path)
    warnings_list = validate_yaml_config(raw, strict=strict)
    for w in warnings_list:
        print(f"[AKSARA Config] Warning: {w}")
    def section(key: str) -> dict:
        return raw.get(key) or {}
    bsu_d = section("bsu_config")
    meb_d = section("meb_config")
    gos_d = section("gos_config")
    lps_d = section("lps_config")
    lsk_d = section("lsk_config")
    bsu = BSUConfig(d_morpheme=bsu_d.get("d_morpheme", 64), d_semantic=bsu_d.get("d_semantic", 64), d_role=bsu_d.get("d_role", 32), d_context=bsu_d.get("d_context", 64))
    meb = MEBConfig(n_layers=meb_d.get("n_layers", 6), n_dep_heads=meb_d.get("n_dep_heads", 4), dropout=meb_d.get("dropout", 0.1))
    meb.bsu_config = bsu
    gos = GOSConfig(teacher_forcing=gos_d.get("teacher_forcing", True))
    gos.bsu_config = bsu
    lps = LPSConfig(dep_window=lps_d.get("dep_window", 4), min_root_length=lps_d.get("min_root_length", 3))
    lsk = LSKConfig(kbbi_path=lsk_d.get("kbbi_path", "kbbi_core_v2.json"), kbbi_vector_dim=lsk_d.get("kbbi_vector_dim", 16), max_lemmas=lsk_d.get("max_lemmas", 50000), pretrained_path=lsk_d.get("pretrained_path", "data/kbbi_pretrained.pt"))
    return AksaraConfig(bsu_config=bsu, meb_config=meb, gos_config=gos, lps_config=lps, lsk_config=lsk, label_smoothing=raw.get("label_smoothing", 0.1), lambda_root=raw.get("lambda_root", 2.0), lambda_fluency=raw.get("lambda_fluency", 0.1))


def cmd_audit(args) -> int:
    audit_script = os.path.join(os.path.dirname(__file__), "..", "tools", "audit_verdict.py")
    audit_script = os.path.abspath(audit_script)
    if not os.path.exists(audit_script):
        print(f"[AKSARA] tools/audit_verdict.py tidak ditemukan di '{audit_script}'")
        return 1
    import subprocess
    result = subprocess.run([sys.executable, audit_script], cwd=os.getcwd())
    return result.returncode


def cmd_info(args) -> int:
    path = args.checkpoint
    if not path:
        print("[AKSARA] --checkpoint wajib untuk perintah 'info'")
        return 1
    ckpt_file = os.path.join(path, "checkpoint.json")
    vocab_file = os.path.join(path, "vocab.json")
    config_file = os.path.join(path, "config.json")
    if not os.path.exists(ckpt_file):
        print(f"[AKSARA] checkpoint.json tidak ditemukan di '{path}'")
        return 1
    print("=" * 60)
    print("  AKSARA CHECKPOINT INFO")
    print("=" * 60)
    with open(ckpt_file, encoding="utf-8") as f:
        ckpt = json.load(f)
    print(f"  Versi          : {ckpt.get('aksara_version', '?')}")
    print(f"  Disimpan       : {ckpt.get('saved_at', '?')}")
    print(f"  Vocab size     : {ckpt.get('vocab_size', '?'):,}")
    print(f"  Total params   : {ckpt.get('n_params_total', '?'):,}")
    print(f"  Trainable      : {ckpt.get('n_params_trainable', '?'):,}")
    print(f"  KBBI pre-seeded: {ckpt.get('pretrained_kbbi', '?')}")
    sha = ckpt.get('model_sha256', '')
    if sha:
        print(f"  SHA-256        : {sha[:24]}...")
    if os.path.exists(config_file):
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        bsu = cfg.get("bsu_config", {})
        meb = cfg.get("meb_config", {})
        print()
        print("  BSU: d_morpheme={d_morpheme}, d_semantic={d_semantic}, d_role={d_role}, d_context={d_context}".format(**{"d_morpheme": bsu.get("d_morpheme", "?"), "d_semantic": bsu.get("d_semantic", "?"), "d_role": bsu.get("d_role", "?"), "d_context": bsu.get("d_context", "?")}))
        print(f"  MEB: n_layers={meb.get('n_layers','?')}, n_dep_heads={meb.get('n_dep_heads','?')}")
    if os.path.exists(vocab_file):
        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)
        specials = [k for k in vocab if k.startswith("<")]
        print(f"\n  Special tokens : {specials}")
    print("=" * 60)
    return 0


def cmd_generate(args) -> int:
    if not args.checkpoint:
        print("[AKSARA] --checkpoint wajib untuk perintah 'generate'")
        return 1
    if _is_blank_text(args.prompt):
        print("[AKSARA] Prompt tidak boleh kosong. Berikan teks input yang valid untuk perintah 'generate'.")
        return 1
    prompt = _normalize_prompt_text(args.prompt)
    print(f"[AKSARA] Memuat checkpoint dari '{args.checkpoint}'...")
    from aksara.core.model import AksaraModel
    model_config = None
    if hasattr(args, "config") and args.config:
        strict = getattr(args, "strict", False)
        verbose = getattr(args, "verbose", False)
        model_config = config_from_yaml(args.config, strict=strict, verbose=verbose)
    try:
        model = AksaraModel.from_pretrained(args.checkpoint, config=model_config)
    except FileNotFoundError:
        print(f"[AKSARA] Checkpoint tidak ditemukan atau belum lengkap di '{args.checkpoint}'.")
        return 1
    except ValueError as exc:
        print(f"[AKSARA] Checkpoint tidak valid: {exc}")
        return 1
    model.eval()
    print(f"[AKSARA] Prompt  : '{prompt}'")
    print(f"[AKSARA] Max len : {args.max_length}")
    print(f"[AKSARA] Temp    : {args.temperature}")
    print()
    result = model.generate(texts=[prompt], max_length=args.max_length, temperature=args.temperature)
    texts = result.get("generated_texts", [])
    for i, txt in enumerate(texts):
        print(f"  [{i}] {txt}")
    return 0


def _print_config_resolution(bsu_d: dict, meb_d: dict, gos_d: dict, lps_d: dict, lsk_d: dict, raw: dict):
    defaults = {
        "bsu_config.d_morpheme": 64, "bsu_config.d_semantic": 64, "bsu_config.d_role": 32, "bsu_config.d_context": 64,
        "meb_config.n_layers": 6, "meb_config.n_dep_heads": 4, "meb_config.dropout": 0.1,
        "gos_config.teacher_forcing": True,
        "lps_config.dep_window": 4, "lps_config.min_root_length": 3,
        "lsk_config.kbbi_path": "kbbi_core_v2.json", "lsk_config.kbbi_vector_dim": 16, "lsk_config.max_lemmas": 50000, "lsk_config.pretrained_path": "data/kbbi_pretrained.pt",
        "label_smoothing": 0.1, "lambda_root": 2.0, "lambda_fluency": 0.1,
    }
    sections_map = {"bsu_config": bsu_d, "meb_config": meb_d, "gos_config": gos_d, "lps_config": lps_d, "lsk_config": lsk_d}
    print("\n[AKSARA Config] Resolusi field (YAML → aktif):")
    print(f"  {'Field':<40} {'Sumber':<8} {'Nilai Aktif'}")
    print(f"  {'-'*40} {'-'*8} {'-'*20}")
    for full_key, default_val in defaults.items():
        if "." in full_key:
            sec, field = full_key.split(".", 1)
            sec_d = sections_map.get(sec, {})
            yaml_val = sec_d.get(field)
        else:
            yaml_val = raw.get(full_key)
        if yaml_val is not None:
            source = "YAML"
            active = yaml_val
        else:
            source = "default"
            active = default_val
        print(f"  {full_key:<40} {source:<8} {active!r}")
    print()


def cmd_schema(args) -> int:
    schema_data = {"config_version": YAML_CONFIG_VERSION, "supported_versions": sorted(SUPPORTED_CONFIG_VERSIONS), "description": "AKSARA YAML Config Schema — semua field optional kecuali ditandai required=true", "fields": {}}
    defaults = {
        "bsu_config.d_morpheme": 64, "bsu_config.d_semantic": 64, "bsu_config.d_role": 32, "bsu_config.d_context": 64,
        "meb_config.n_layers": 6, "meb_config.n_dep_heads": 4, "meb_config.dropout": 0.1,
        "gos_config.teacher_forcing": True, "lps_config.dep_window": 4, "lps_config.min_root_length": 3,
        "lsk_config.kbbi_path": "kbbi_core_v2.json", "lsk_config.kbbi_vector_dim": 16, "lsk_config.max_lemmas": 50000, "lsk_config.pretrained_path": "data/kbbi_pretrained.pt",
        "label_smoothing": 0.1, "lambda_root": 2.0, "lambda_fluency": 0.1,
    }
    for (section, field), spec in _YAML_SCHEMA.items():
        full_key = f"{section}.{field}" if section else field
        schema_data["fields"][full_key] = {"type": spec["type"].__name__, "min": spec["min"], "max": spec["max"], "required": spec["req"], "default": defaults.get(full_key)}
    fmt = getattr(args, "format", "text") or "text"
    if fmt == "json" or args.output:
        if args.output:
            output_str = json.dumps(schema_data, ensure_ascii=False, indent=2)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)
            print(f"[AKSARA] Schema diekspor ke '{args.output}'")
        else:
            print(json.dumps(schema_data, ensure_ascii=True, indent=2))
    else:
        print(f"AKSARA Config Schema (versi {YAML_CONFIG_VERSION})")
        print("=" * 60)
        print(f"  {'Field':<40} {'Type':<7} {'Min':>6} {'Max':>7}  {'Default'}")
        print(f"  {'-'*40} {'-'*7} {'-'*6} {'-'*7}  {'-'*15}")
        for full_key, info in schema_data["fields"].items():
            lo = "-" if info["min"] is None else str(info["min"])
            hi = "-" if info["max"] is None else str(info["max"])
            req = " *" if info["required"] else ""
            print(f"  {full_key+req:<40} {info['type']:<7} {lo:>6} {hi:>7}  {info['default']!r}")
        print()
        print("  * = field wajib")
        print("  Semua field lain optional — jika tidak ada, pakai nilai default.")
        print("  Gunakan --format json untuk output yang bisa diparse programmatically.")
    return 0


def cmd_diff(args) -> int:
    path_a = args.config_a
    path_b = args.config_b
    fmt = getattr(args, "format", "text") or "text"
    try:
        res_a = resolve_config(path_a, strict=False)
        res_b = resolve_config(path_b, strict=False)
    except (YAMLConfigError, FileNotFoundError) as e:
        print(f"[AKSARA] Error membaca config: {e}")
        return 1
    diffs = {}
    for key in res_a:
        val_a = res_a[key]["value"]
        val_b = res_b.get(key, {}).get("value", None)
        if val_a != val_b:
            sev = _DIFF_SEVERITY.get(key, "minor")
            diffs[key] = {"severity": sev, "a": {"source": res_a[key]["source"], "value": val_a}, "b": {"source": res_b.get(key, {}).get("source", "?"), "value": val_b}}
    if not diffs:
        print("[AKSARA] Kedua config identik (tidak ada perbedaan)")
        return 0
    sorted_diffs = sorted(diffs.items(), key=lambda kv: (_SEVERITY_ORDER.get(kv[1]["severity"], 9), kv[0]))
    sev_counts = {"critical": 0, "major": 0, "minor": 0}
    for _, d in diffs.items():
        sev_counts[d["severity"]] = sev_counts.get(d["severity"], 0) + 1
    if fmt == "json":
        print(json.dumps({"config_a": path_a, "config_b": path_b, "n_diffs": len(diffs), "severity_summary": sev_counts, "diffs": {k: v for k, v in sorted_diffs}}, ensure_ascii=True, indent=2))
    else:
        name_a = os.path.basename(path_a)
        name_b = os.path.basename(path_b)
        print(f"[AKSARA] Config diff: '{name_a}' vs '{name_b}'")
        sev_parts = []
        if sev_counts["critical"]:
            sev_parts.append(f"{sev_counts['critical']} critical")
        if sev_counts["major"]:
            sev_parts.append(f"{sev_counts['major']} major")
        if sev_counts["minor"]:
            sev_parts.append(f"{sev_counts['minor']} minor")
        print(f"  {len(diffs)} field berbeda — " + ", ".join(sev_parts))
        if sev_counts["critical"]:
            print("  ⛔ PERHATIAN: ada perubahan CRITICAL yang membuat model tidak kompatibel!")
        print()
        w = 40
        print(f"  {'Severity':<12} {'Field':<{w}} {name_a:<18}  {name_b}")
        print(f"  {'-'*12} {'-'*w} {'-'*18}  {'-'*18}")
        for key, d in sorted_diffs:
            sev_lbl = _SEVERITY_LABEL.get(d["severity"], d["severity"])
            v_a = repr(d["a"]["value"])
            v_b = repr(d["b"]["value"])
            print(f"  {sev_lbl:<12} {key:<{w}} {v_a:<18}  {v_b}")
    return 0


def cmd_merge(args) -> int:
    base = args.base
    layers = args.override or []
    out = args.output
    verbose = getattr(args, "verbose", False)
    dry_run = getattr(args, "dry_run", False)
    fmt = getattr(args, "format", "yaml") or "yaml"
    resolve = getattr(args, "resolve", "override") or "override"
    report = getattr(args, "report", None)
    try:
        merged_raw, conflicts = merge_config_with_report(base, *layers, resolve=resolve)
    except (YAMLConfigError, FileNotFoundError) as e:
        print(f"[AKSARA] Error merge config: {e}")
        return 1
    layer_names = [os.path.basename(base)] + [os.path.basename(l) for l in layers]
    print(f"[AKSARA] Merge {len(layer_names)} layer: " + " ← ".join(reversed(layer_names)))
    print(f"  Resolve strategy: '{resolve}'")
    if conflicts:
        conflicts_sorted = sorted(conflicts, key=lambda c: (_SEVERITY_ORDER.get(c["severity"], 9), c["field"]))
        critical_conflicts = [c for c in conflicts if c["severity"] == "critical"]
        print(f"  {len(conflicts)} conflict terdeteksi" + (f" — {len(critical_conflicts)} CRITICAL" if critical_conflicts else ""))
        if critical_conflicts:
            print("  ⛔ PERHATIAN: override field CRITICAL bisa mengubah arsitektur model!")
        print()
        print(f"  {'Severity':<12} {'Field':<40} {'Loser (lama)':<18} → {'Winner (baru)'}")
        print(f"  {'-'*12} {'-'*40} {'-'*18}   {'-'*18}")
        for c in conflicts_sorted:
            sev_lbl = _SEVERITY_LABEL.get(c["severity"], c["severity"])
            sw = c.get("strategy_winner", c["winner"])
            resolved = c.get("resolved", c["new_value"])
            override_marker = " ✓" if sw == c["winner"] else " ✖ (ditolak)"
            expl = c.get("explanation", "")
            print(f"  {sev_lbl:<12} {c['field']:<40} {repr(c['old_value']):<18} → {repr(resolved)} [{sw}{override_marker}]")
            if expl:
                print(f"  {'':12} {'':40} └─ {expl}")
        print()
    else:
        print("  Tidak ada conflict — semua override field baru (tidak menimpa nilai existing)")
    if report and conflicts:
        conflict_data = {"layers": layer_names, "resolve_strategy": resolve, "n_conflicts": len(conflicts), "severity_summary": {"critical": sum(1 for c in conflicts if c["severity"] == "critical"), "major": sum(1 for c in conflicts if c["severity"] == "major"), "minor": sum(1 for c in conflicts if c["severity"] == "minor")}, "conflicts": conflicts}
        with open(report, "w", encoding="utf-8") as f:
            json.dump(conflict_data, f, ensure_ascii=False, indent=2)
        print(f"  Conflict report diekspor ke '{report}'")
    elif report and not conflicts:
        print(f"  Tidak ada conflict untuk diekspor ke '{report}'")
    if dry_run:
        print("[AKSARA] --dry-run: tidak ada file yang ditulis.")
        if verbose:
            import tempfile as _tf
            import os as _os
            with _tf.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                for sk, sv in merged_raw.items():
                    if isinstance(sv, dict):
                        tmp.write(f"{sk}:\n")
                        for k, v in sv.items():
                            tmp.write(f"  {k}: {v!r}\n")
                    else:
                        tmp.write(f"{sk}: {sv!r}\n")
                tmp_path = tmp.name
            try:
                config_from_yaml(tmp_path, verbose=True)
            finally:
                _os.unlink(tmp_path)
        return 0
    if fmt == "json":
        output_str = json.dumps(merged_raw, ensure_ascii=True, indent=2)
    else:
        lines = ["# AKSARA merged config — " + " + ".join(layer_names)]
        for sec_key, sec_val in merged_raw.items():
            if isinstance(sec_val, dict):
                lines.append(f"{sec_key}:")
                for k, v in sec_val.items():
                    lines.append(f"  {k}: {v!r}")
            else:
                lines.append(f"{sec_key}: {sec_val!r}")
        output_str = "\n".join(lines) + "\n"
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"[AKSARA] Merged config ditulis ke '{out}'")
        if verbose:
            config_from_yaml(out, verbose=True)
    else:
        print(output_str)
    return 0


def cmd_export(args) -> int:
    if not args.checkpoint:
        print("[AKSARA] --checkpoint wajib untuk perintah 'export'")
        return 1
    if not args.output:
        print("[AKSARA] --output wajib untuk perintah 'export'")
        return 1
    import shutil
    src = args.checkpoint
    dst = args.output
    if os.path.abspath(src) == os.path.abspath(dst):
        print("[AKSARA] --checkpoint dan --output tidak boleh sama")
        return 1
    print(f"[AKSARA] Export '{src}' → '{dst}'")
    os.makedirs(dst, exist_ok=True)
    for fname in ("model.pt", "vocab.json", "config.json", "checkpoint.json"):
        fpath = os.path.join(src, fname)
        if os.path.exists(fpath):
            shutil.copy2(fpath, os.path.join(dst, fname))
            print(f"  Copied: {fname}")
        else:
            print(f"  Skip  : {fname} (tidak ada)")
    ckpt_dst = os.path.join(dst, "checkpoint.json")
    model_dst = os.path.join(dst, "model.pt")
    if os.path.exists(ckpt_dst) and os.path.exists(model_dst):
        from aksara.core.model import AksaraModel
        saved_sha = json.load(open(ckpt_dst)).get("model_sha256", "")
        actual_sha = AksaraModel._compute_checksum(model_dst)
        if saved_sha and saved_sha == actual_sha:
            print("\n  [OK] Integritas terverifikasi (SHA-256 cocok)")
        else:
            print("\n  [WARN] SHA-256 tidak cocok — export mungkin tidak lengkap")
    print(f"\n[AKSARA] Export selesai → '{dst}'")
    return 0


def cmd_init(args) -> int:
    output_path = args.output or "aksara_config.yaml"
    if os.path.exists(output_path) and not args.force:
        print(f"[AKSARA] '{output_path}' sudah ada. Gunakan --force untuk menimpa.")
        return 1
    template = f"""# AKSARA Config — Format YAML
config_version: "{YAML_CONFIG_VERSION}"
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"[AKSARA] Template config dibuat: '{output_path}'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aksara", description="AKSARA — Native Indonesian LLM Framework", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version="AKSARA 2.0 (native Indonesian LLM framework)")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True
    p_audit = sub.add_parser("audit", help="Jalankan native framework audit (10-point checklist)")
    p_audit.set_defaults(func=cmd_audit)
    p_info = sub.add_parser("info", help="Tampilkan info checkpoint")
    p_info.add_argument("--checkpoint", "-c", metavar="DIR")
    p_info.set_defaults(func=cmd_info)
    p_gen = sub.add_parser("generate", help="Generate teks dari prompt")
    p_gen.add_argument("--checkpoint", "-c", metavar="DIR")
    p_gen.add_argument("--config", metavar="YAML")
    p_gen.add_argument("--prompt", "-p", metavar="TEXT", required=True)
    p_gen.add_argument("--max-length", type=int, default=30, metavar="N")
    p_gen.add_argument("--temperature", type=float, default=0.8, metavar="T")
    p_gen.add_argument("--strict", action="store_true")
    p_gen.add_argument("--verbose", action="store_true")
    p_gen.set_defaults(func=cmd_generate)
    p_exp = sub.add_parser("export", help="Export checkpoint ke direktori baru")
    p_exp.add_argument("--checkpoint", "-c", metavar="DIR", required=True)
    p_exp.add_argument("--output", "-o", metavar="DIR", required=True)
    p_exp.set_defaults(func=cmd_export)
    p_init = sub.add_parser("init", help="Buat template aksara_config.yaml")
    p_init.add_argument("--output", "-o", metavar="FILE", default="aksara_config.yaml")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)
    p_schema = sub.add_parser("schema", help="Tampilkan schema config AKSARA (field, type, range, default)")
    p_schema.add_argument("--output", "-o", metavar="FILE")
    p_schema.add_argument("--format", choices=["text", "json"], default="text")
    p_schema.set_defaults(func=cmd_schema)
    p_diff = sub.add_parser("diff", help="Bandingkan dua config YAML (dengan severity)")
    p_diff.add_argument("--config-a", "-a", metavar="YAML", required=True)
    p_diff.add_argument("--config-b", "-b", metavar="YAML", required=True)
    p_diff.add_argument("--format", choices=["text", "json"], default="text")
    p_diff.set_defaults(func=cmd_diff)
    p_merge = sub.add_parser("merge", help="Merge config YAML berlapis (layer composition)")
    p_merge.add_argument("--base", "-b", metavar="YAML", required=True)
    p_merge.add_argument("--override", "-o", metavar="YAML", action="append")
    p_merge.add_argument("--output", metavar="FILE")
    p_merge.add_argument("--format", choices=["yaml", "json"], default="yaml")
    p_merge.add_argument("--verbose", action="store_true")
    p_merge.add_argument("--dry-run", action="store_true")
    p_merge.add_argument("--resolve", choices=["override", "base", "critical-safe"], default="override")
    p_merge.add_argument("--report", metavar="FILE")
    p_merge.set_defaults(func=cmd_merge)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())