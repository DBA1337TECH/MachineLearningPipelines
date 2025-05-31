#!/usr/bin/env python3

import os
import gzip
import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path("data/ghidra_databases")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ghidra_file(file_path: Path) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to load {file_path}: {e}")
        return {}


def extract_features(db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example structure:
    {
        "functions": [...],
        "strings": [...],
        "instructions": [...],
        ...
    }
    """
    features = {
        "function_names": [],
        "string_constants": [],
        "instruction_mnemonics": [],
        "xrefs": [],
    }

    for fn in db.get("functions", []):
        features["function_names"].append(fn.get("name", "unknown"))

    for s in db.get("strings", []):
        features["string_constants"].append(s.get("value", ""))

    for instr in db.get("instructions", []):
        features["instruction_mnemonics"].append(instr.get("mnemonic", ""))

    for ref in db.get("xrefs", []):
        features["xrefs"].append(ref.get("to", ""))

    return features


def compress_and_save(features: Dict[str, Any], out_path: Path):
    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        json.dump(features, gz)


def main():
    ghidra_files = list(DATA_DIR.glob("*.json"))
    print(f"[+] Found {len(ghidra_files)} files to process.")

    for ghidra_file in ghidra_files:
        print(f"[*] Processing {ghidra_file.name}")
        db = load_ghidra_file(ghidra_file)
        if not db:
            continue

        features = extract_features(db)
        out_file = OUTPUT_DIR / f"{ghidra_file.stem}.features.json.gz"
        compress_and_save(features, out_file)
        print(f"[+] Saved: {out_file}")


if __name__ == "__main__":
    main()

