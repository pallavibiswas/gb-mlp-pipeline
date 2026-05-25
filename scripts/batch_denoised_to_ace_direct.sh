#!/bin/bash
set -euo pipefail

DENOISED_ROOT="$HOME/KiteL_synthetic_0K/denoised"
ACE_ROOT="$HOME/ACE"
SAMPLE_DIR="$ACE_ROOT/sample_data"
ARCHIVE_ROOT="$ACE_ROOT/runs2/batch_archived_outputs_direct"
OVITO="/volume/NFS/qz161/ovito-pro-3.7.8-x86_64/bin/ovitos"

FIXED_INPUT="$SAMPLE_DIR/T2300_fcc-vac.denoised.restored.dump"
FIXED_DATA="$SAMPLE_DIR/T2300_seed0_position.data"
LMP_EXEC="/volume/NFS/yf245/lammps-static/bin/lmp"
POTENTIAL="$ACE_ROOT/output_potential.yace"
TEMPLATE="$ACE_ROOT/in.template"

mkdir -p "$ARCHIVE_ROOT/kite" "$ARCHIVE_ROOT/L"

process_one() {
    local subset="$1"
    local src="$2"
    local base stem run_dir dest

    base="$(basename "$src")"
    stem="${base%.extxyz}"
    run_dir="$ACE_ROOT/runs2/run_${stem}"
    dest="$ARCHIVE_ROOT/$subset/$stem"

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
    cp -r "$run_dir" "$dest/"
    printf "%s\n" "$src" > "$dest/source_file.txt"

    echo "Done: $src"
}

for subset in kite L; do
    echo
    echo "###############"
    echo "SUBSET: $subset"
    echo "###############"

    find "$DENOISED_ROOT/$subset" -maxdepth 1 -type f -name "*.extxyz" | sort | while read -r f; do
        process_one "$subset" "$f"
    done
done

echo
echo "All files processed."
echo "Archived outputs are under:"
echo "$ARCHIVE_ROOT"
