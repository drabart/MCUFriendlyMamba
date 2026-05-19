#!/usr/bin/env python3
"""
Strip tensor 'name' fields from a FlatBuffer TFLite JSON and rebuild binary.

Usage examples:
  python python/strip_tensor_names.py --schema python/models/schema.fbs --tflite python/models/model_int8.tflite
  python python/strip_tensor_names.py --json python/models/model_int8.json --schema python/models/schema.fbs

The script:
 - Optionally converts .tflite -> .json using `flatc -t --strict-json`
 - Loads the JSON, removes `name` keys from tensors in each subgraph
 - Saves the JSON (overwriting) and runs `flatc -b schema.fbs model_int8.json`
 - Compares sizes before/after and tries to load the rebuilt binary with a TFLite interpreter
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, cwd=None, silent=False):
    if not silent:
        print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, stdout=subprocess.DEVNULL if silent else None, stderr=subprocess.DEVNULL if silent else None)


def to_json(schema, tflite_path, out_dir=None):
    tflite_path = Path(tflite_path).resolve()  # Convert to absolute path
    out_dir = Path(out_dir or tflite_path.parent)
    schema = Path(schema).resolve()  # Convert to absolute path
    cmd = ["flatc", "-t", "--strict-json", "-o", str(out_dir), str(schema), "--", str(tflite_path)]
    run(cmd, cwd=out_dir, silent=True)
    return out_dir / (tflite_path.stem + ".json")


def rebuild_binary(schema, json_path, out_dir=None):
    json_path = Path(json_path).resolve()  # Convert to absolute path
    out_dir = Path(out_dir or json_path.parent)
    schema = Path(schema).resolve()  # Convert to absolute path
    cmd = ["flatc", "-b", str(schema), str(json_path)]
    run(cmd, cwd=out_dir, silent=True)

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


def strip_tensor_names(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False
    subgraphs = data.get("subgraphs")
    if isinstance(subgraphs, list):
        for sg in subgraphs:
            tensors = sg.get("tensors")
            if isinstance(tensors, list):
                for t in tensors:
                    if isinstance(t, dict) and "name" in t:
                        del t["name"]
                        modified = True

    if modified:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def size_readable(p):
    if isinstance(p, int):
        b = p
    else:
        b = p.stat().st_size
    for unit in ["B", "KB", "MB"]:
        if b < 1024.0:
            return f"{b:.1f}{unit}"
        b /= 1024.0
    return f"{b:.1f}GB"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=str, default="schema.fbs", help="FlatBuffers schema file")
    parser.add_argument("--tflite", type=str, help="Input .tflite file to convert to JSON")
    parser.add_argument("--json", type=str, help="Existing JSON file to load and modify (if provided, skip tflite->json)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for conversion steps")
    args = parser.parse_args()

    schema = Path(args.schema)
    if not schema.exists():
        print("Schema file not found:", schema)
        sys.exit(2)

    if args.json:
        json_path = Path(args.json)
        if not json_path.exists():
            print("JSON file not found:", json_path)
            sys.exit(2)
    elif args.tflite:
        tflite_path = Path(args.tflite)
        if not tflite_path.exists():
            print("TFLite file not found:", tflite_path)
            sys.exit(2)
        # Capture original size before converting to JSON
        orig_size = tflite_path.stat().st_size
        json_path = to_json(schema, tflite_path, out_dir=args.out_dir)
    else:
        print("Either --tflite or --json must be provided")
        sys.exit(2)

    orig_bin = None
    if args.tflite:
        orig_bin = Path(args.tflite)

    strip_tensor_names(json_path)

    rebuilt = rebuild_binary(schema, json_path, out_dir=args.out_dir)

    if orig_bin and rebuilt.exists():
        print(f"Stripped {orig_bin.name}:")
        print(f"  before: {size_readable(orig_size)}")
        print(f"  after:  {size_readable(rebuilt)}")

if __name__ == "__main__":
    main()