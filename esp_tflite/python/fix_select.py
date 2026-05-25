#!/usr/bin/env python3
"""
Replace SELECT opcode with SELECT_V2 in TFLite FlatBuffer models.

This script converts a TFLite model to JSON format, replaces SELECT (opcode 64)
with SELECT_V2 (opcode 123) in the operator_codes array, and rebuilds the binary.
This transformation is required for tflite-micro compatibility.

Usage:
  python strip_names.py --schema schema.fbs --tflite model_int8.tflite
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None):
    subprocess.check_call(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def to_json(schema, tflite_path, out_dir=None):
    tflite_path = Path(tflite_path).resolve()
    out_dir = Path(out_dir or tflite_path.parent)
    schema = Path(schema).resolve()
    cmd = ["flatc", "-t", "--strict-json", "-o", str(out_dir), str(schema), "--", str(tflite_path)]
    run(cmd, cwd=out_dir)
    return out_dir / (tflite_path.stem + ".json")


def rebuild_binary(schema, json_path, out_dir=None):
    json_path = Path(json_path).resolve()
    out_dir = Path(out_dir or json_path.parent)
    schema = Path(schema).resolve()
    cmd = ["flatc", "-b", str(schema), str(json_path)]
    run(cmd, cwd=out_dir)

    candidates = [
        out_dir / (json_path.stem + ".bin"),
        out_dir / (json_path.stem + ".tflite"),
        out_dir / json_path.stem,
    ]
    for c in candidates:
        if c.exists():
            return c
    files = sorted(out_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        if f.is_file() and f.suffix.lower() in (".bin", ".tflite", ""):
            return f
    raise FileNotFoundError("Could not find rebuilt binary produced by flatc")


def replace_select_builtin_code(json_path):
    """Replace SELECT builtin_code with SELECT_V2 in operator_codes."""
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False
    operator_codes = data.get("operator_codes")
    if isinstance(operator_codes, list):
        for op_code in operator_codes:
            if isinstance(op_code, dict) and op_code.get("builtin_code") == "SELECT":
                op_code["builtin_code"] = "SELECT_V2"
                if "deprecated_builtin_code" in op_code:
                    op_code["deprecated_builtin_code"] = 123
                modified = True

    if modified:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return modified

def main():
    parser = argparse.ArgumentParser(description="Replace SELECT with SELECT_V2 in TFLite models")
    parser.add_argument("--schema", type=str, default="schema.fbs", help="FlatBuffers schema file")
    parser.add_argument("--tflite", type=str, required=True, help="Input .tflite file")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    schema = Path(args.schema)
    if not schema.exists():
        print(f"Error: Schema file not found: {schema}", file=sys.stderr)
        sys.exit(1)

    tflite_path = Path(args.tflite)
    if not tflite_path.exists():
        print(f"Error: TFLite file not found: {tflite_path}", file=sys.stderr)
        sys.exit(1)

    json_path = to_json(schema, tflite_path, out_dir=args.out_dir)
    
    if replace_select_builtin_code(json_path):
        print(f"✓ SELECT -> SELECT_V2 replaced")
    else:
        print("  (no SELECT operations found)")

    rebuild_binary(schema, json_path, out_dir=args.out_dir)

if __name__ == "__main__":
    main()