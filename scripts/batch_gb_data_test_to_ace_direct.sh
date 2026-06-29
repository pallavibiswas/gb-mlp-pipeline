#!/bin/bash
set -euo pipefail

DENOISED_ROOT="$HOME/MD_GB_sims/gb_data_test_denoised"
ACE_ROOT="$HOME/ACE"
SAMPLE_DIR="$ACE_ROOT/sample_data"
ARCHIVE_ROOT="$HOME/ACE/runs2/gb_data_test_ace"
OVITO="/volume/NFS/qz161/ovito-pro-3.7.8-x86_64/bin/ovitos"

FIXED_INPUT="$SAMPLE_DIR/T2300_fcc-vac.denoised.restored.dump"
FIXED_DATA="$SAMPLE_DIR/T2300_seed0_position.data"
LMP_EXEC="/volume/NFS/yf245/lammps-static/bin/lmp"
POTENTIAL="$ACE_ROOT/output_potential.yace"
TEMPLATE="$ACE_ROOT/in.template"

mkdir -p "$ARCHIVE_ROOT"

process_one() {
    local subset="$1"
    local src="$2"
    local base stem run_dir dest

    base="$(basename "$src")"
    stem="${base%.extxyz}"
    run_dir="$ACE_ROOT/runs2/run_${stem}"
    dest="$ARCHIVE_ROOT/$subset/$stem"
    run_base="$(basename "$run_dir")"

    if [[ -f "$dest/pace_row_output.txt" || -f "$dest/$run_base/pace_row_output.txt" ]]; then
        echo "Skipping existing ACE: $dest"
        return 0
    fi


    echo
    echo "=================================================="
    echo "Processing: $src"
    echo "Run dir: $run_dir"
    echo "Archive dir: $dest"
    echo "=================================================="

    mkdir -p "$dest"
    rm -rf "$run_dir"
    mkdir -p "$run_dir"
    rm -f "$FIXED_INPUT" "$FIXED_DATA"

    cp "$src" "$FIXED_INPUT"

    echo "[1/5] Running OVITO conversion..."
    (
        cd "$SAMPLE_DIR"
        "$OVITO" ovitodumptodata.py
    )

    if [[ ! -f "$FIXED_DATA" ]]; then
        echo "ERROR: missing converted data file: $FIXED_DATA"
        return 1
    fi

    echo "[2/5] Staging run directory..."
    cp "$FIXED_DATA" "$run_dir/input.data"
    cp "$TEMPLATE" "$run_dir/in.template"
    cp "$POTENTIAL" "$run_dir/"

    python3 - <<PY
from pathlib import Path
run_dir = Path("$run_dir")
template = Path("$TEMPLATE").read_text()
template = template.replace("structure.data", "input.data")
(run_dir / "in.run").write_text(template)
PY

    echo "[3/5] Running LAMMPS directly..."
    (
        cd "$run_dir"
        "$LMP_EXEC" -in in.run > lammps_stdout.txt 2> lammps_stderr.txt
    )

    echo "[4/5] Checking output..."
    if [[ ! -f "$run_dir/pace_row_output.txt" ]]; then
        echo "ERROR: pace_row_output.txt not found for $src"
        echo "--- lammps_stdout.txt ---"
        tail -n 50 "$run_dir/lammps_stdout.txt" || true
        echo "--- lammps_stderr.txt ---"
        tail -n 50 "$run_dir/lammps_stderr.txt" || true
        return 1
    fi

    echo "[5/5] Archiving..."
    mkdir -p "$dest"
    rm -rf "$dest/$run_base"
    mv "$run_dir" "$dest/"
    printf "%s\n" "$src" > "$dest/source_file.txt"

    echo "Done: $src"
}


echo
echo "=============================="
echo "GBAL denoised input root:"
echo "$DENOISED_ROOT"
echo "=============================="

mapfile -t FILES < <(find "$DENOISED_ROOT" -type f -name "*.denoised_.extxyz" | sort -V)

echo "Found ${#FILES[@]} denoised EXTXYZ files"

for f in "${FILES[@]}"; do
    rel="${f#$DENOISED_ROOT/}"
    subset="$(dirname "$rel")"

    echo
    echo "Processing GBAL folder: $subset"
    echo "File: $f"

    process_one "$subset" "$f"
done

echo
echo "All GBAL sample files processed."
echo "Archived outputs are under:"
echo "$ARCHIVE_ROOT"
