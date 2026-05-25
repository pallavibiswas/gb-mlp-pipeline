#!/bin/bash
set -euo pipefail

SRC="$HOME/KiteL_synthetic_0K/denoised/kite/dump.gb.0.sigma_0p010.rep_0.denoised_.extxyz"

ACE_ROOT="$HOME/ACE"
SAMPLE_DIR="$ACE_ROOT/sample_data"
OVITO="/volume/NFS/qz161/ovito-pro-3.7.8-x86_64/bin/ovitos"
RUN_DIR="$ACE_ROOT/runs2/test_direct_run"
ARCHIVE_DIR="$ACE_ROOT/runs2/test_direct_archived"

FIXED_INPUT="$SAMPLE_DIR/T2300_fcc-vac.denoised.restored.dump"
FIXED_DATA="$SAMPLE_DIR/T2300_seed0_position.data"
LMP_EXEC="/volume/NFS/yf245/lammps-static/bin/lmp"
POTENTIAL="$ACE_ROOT/output_potential.yace"
TEMPLATE="$ACE_ROOT/in.template"

mkdir -p "$ARCHIVE_DIR"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
rm -f "$FIXED_INPUT" "$FIXED_DATA"

echo "Processing: $SRC"

cp "$SRC" "$FIXED_INPUT"

echo "[1/5] Running OVITO conversion..."
(
  cd "$SAMPLE_DIR"
  "$OVITO" ovitodumptodata.py
)

if [[ ! -f "$FIXED_DATA" ]]; then
  echo "ERROR: missing $FIXED_DATA"
  exit 1
fi

echo "[2/5] Staging run directory..."
cp "$FIXED_DATA" "$RUN_DIR/input.data"
cp "$TEMPLATE" "$RUN_DIR/in.template"
cp "$POTENTIAL" "$RUN_DIR/"

python3 - <<PY
from pathlib import Path
run_dir = Path("$RUN_DIR")
template = Path("$TEMPLATE").read_text()
template = template.replace("structure.data", "input.data")
(run_dir / "in.run").write_text(template)
PY

echo "[3/5] Running LAMMPS directly..."
(
  cd "$RUN_DIR"
  "$LMP_EXEC" -in in.run > lammps_stdout.txt 2> lammps_stderr.txt
)

echo "[4/5] Checking output..."
if [[ ! -f "$RUN_DIR/pace_row_output.txt" ]]; then
  echo "ERROR: pace_row_output.txt not found"
  echo "--- stdout ---"
  tail -n 50 "$RUN_DIR/lammps_stdout.txt" || true
  echo "--- stderr ---"
  tail -n 50 "$RUN_DIR/lammps_stderr.txt" || true
  exit 1
fi

echo "[5/5] Archiving..."
cp -r "$RUN_DIR" "$ARCHIVE_DIR/"
echo "$SRC" > "$ARCHIVE_DIR/source_file.txt"

echo "Done."
