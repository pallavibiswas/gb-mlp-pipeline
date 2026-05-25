#!/volume/NFS/qz161/ovito-pro-3.7.8-x86_64/bin/ovitos
# -*- coding: utf-8 -*-
"""
Convert all LAMMPS .dump files in the current folder to LAMMPS .data using OVITO.
Naming:
  - Primary:   T{T}_seed{seed}_position.data
  - If exists: T{T}_seed{seed}_1_position.data (or _2_, _3_, ... until free)
If T missing -> Tunknown; if seed missing -> seed0.
"""

import os
import re
from ovito.io import import_file, export_file

OUTPUT_DIR = "."    # ???????;????
ATOM_STYLE = "atomic"

def parse_T_seed(name: str):
    mT = re.search(r'T(\d+)', name)
    mSeed = re.search(r'(?i)seed[_-]?(\d+)', name)
    T_val = mT.group(1) if mT else "unknown"
    seed_val = mSeed.group(1) if mSeed else "0"
    return T_val, seed_val

def build_out_path(T_val: str, seed_val: str, idx: int = 0) -> str:
    """
    idx = 0  -> T{T}_seed{seed}_position.data
    idx >=1  -> T{T}_seed{seed}_{idx}_position.data
    """
    base = f"T{T_val}_seed{seed_val}"
    if idx == 0:
        fname = f"{base}_position.data"
    else:
        fname = f"{base}_{idx}_position.data"
    return os.path.join(OUTPUT_DIR, fname)

def next_available_path(T_val: str, seed_val: str) -> str:
    """
    ????????????????
    ?? _position.data;???,???? _1_/_2_...
    """
    idx = 0
    while True:
        candidate = build_out_path(T_val, seed_val, idx)
        if not os.path.exists(candidate):
            return candidate
        idx += 1

def main():
    files = [f for f in os.listdir(".") if os.path.isfile(f) and f.endswith(".dump")]
    if not files:
        print("No .dump files found in current directory.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_found = total_done = total_fail = 0

    for dump_file in sorted(files):
        total_found += 1
        T_val, seed_val = parse_T_seed(dump_file)
        out_path = next_available_path(T_val, seed_val)

        print(f"[Processing] {dump_file} -> {out_path}")
        try:
            pipeline = import_file(dump_file)
            export_file(pipeline, out_path, "lammps/data", atom_style=ATOM_STYLE)
            print(f"[Done] {out_path}")
            total_done += 1
        except Exception as e:
            print(f"[Error] {dump_file} -> {e}")
            total_fail += 1

    print("\n===== Summary =====")
    print(f"Found   : {total_found}")
    print(f"Exported: {total_done}")
    print(f"Failed  : {total_fail}")

if __name__ == "__main__":
    main()
