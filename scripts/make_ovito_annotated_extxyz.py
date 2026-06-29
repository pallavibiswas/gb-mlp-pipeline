from pathlib import Path
import re
import numpy as np
import pandas as pd
from ase.io import read, write

DENOISED_ROOT = Path.home() / "MD_GB_sims/gb_data_test_denoised"
ACE_ROOT = Path.home() / "ACE/runs2/gb_data_test_ace"
PRED_CSV = Path.home() / "MD_GB_sims/gb_data_classifier_outputs/gb_data_test_predictions.csv"
OUT_ROOT = Path.home() / "MD_GB_sims/gb_data_ovito_annotated"

# Only make transition frames first to keep files manageable.
TRANSITION_PAIRS = {
    300: [120000, 123000],
    450: [111000, 114000],
    600: [105000, 108000],
    750: [96000, 99000],
    900: [87000, 90000],
}

LABEL_TO_NUM = {
    "Kite": 0,
    "L": 1,
}

def find_denoised_frame(temp, timestep):
    folder = DENOISED_ROOT / f"GBALTemp{temp}_1_Alt"
    patterns = [
        f"dump.gbstep*.{timestep}.cfg.denoised_.extxyz",
        f"*{timestep}*.denoised_.extxyz",
        f"*{timestep}*.extxyz",
    ]

    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"No denoised frame found for {temp} K timestep {timestep}")

def find_ace_file(temp, timestep):
    folder = ACE_ROOT / f"GBALTemp{temp}_1_Alt"
    matches = sorted(folder.glob(f"**/*{timestep}*/pace_row_output.txt"))

    if not matches:
        matches = sorted(folder.glob(f"**/*{timestep}*.denoised_*/**/pace_row_output.txt"))

    if not matches:
        raise FileNotFoundError(f"No ACE pace_row_output.txt found for {temp} K timestep {timestep}")

    return matches[0]

def load_ace_rows(path, n_atoms):
    """
    Load ACE per-atom rows and return exactly one descriptor vector per atom.

    Some pace_row_output.txt files contain repeated descriptor rows per atom.
    We use the first column as atom ID, group duplicate rows by atom ID,
    and average descriptors for each atom.
    """
    rows = []
    widths = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Skip short metadata/header rows such as "4000 326"
            if len(parts) < 10:
                continue

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(vals)
            widths.append(len(vals))

    if not rows:
        raise ValueError(f"No ACE rows found in {path}")

    unique_widths, counts_widths = np.unique(widths, return_counts=True)
    target_width = int(unique_widths[np.argmax(counts_widths)])

    arr = np.asarray([r for r in rows if len(r) == target_width], dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]

    if arr.size == 0:
        raise ValueError(f"No finite ACE rows found in {path}")

    ids_float = arr[:, 0]
    ids_int = np.rint(ids_float).astype(int)

    # Try both common atom-ID conventions:
    # LAMMPS-style: 1..N
    # Python-style : 0..N-1
    for offset in (1, 0):
        valid = (
            np.isclose(ids_float, ids_int, atol=1e-6)
            & (ids_int >= offset)
            & (ids_int < offset + n_atoms)
        )

        sub = arr[valid]
        sub_ids = ids_int[valid] - offset

        if len(sub) == 0:
            continue

        unique_ids = np.unique(sub_ids)

        if len(unique_ids) == n_atoms:
            desc = sub[:, 1:]  # drop atom ID column

            out = np.zeros((n_atoms, desc.shape[1]), dtype=float)
            counts = np.zeros(n_atoms, dtype=float)

            np.add.at(out, sub_ids, desc)
            np.add.at(counts, sub_ids, 1)

            if np.any(counts == 0):
                missing = np.where(counts == 0)[0][:10]
                raise ValueError(f"Missing ACE descriptors for atom IDs: {missing}")

            out = out / counts[:, None]

            print(
                f"  ACE rows grouped by atom ID: raw_rows={len(arr)}, "
                f"unique_atoms={n_atoms}, descriptors={out.shape[1]}"
            )

            return out

    raise ValueError(
        f"Could not map ACE rows to atoms for {path}. "
        f"atoms={n_atoms}, ACE rows={arr.shape[0]}, width={arr.shape[1]}. "
        "First column did not contain a complete set of atom IDs."
    )

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED_CSV)
    pred["temperature"] = pred["temperature"].astype(int)
    pred["timestep"] = pred["timestep"].astype(int)

    written = []

    for temp, timesteps in TRANSITION_PAIRS.items():
        for timestep in timesteps:
            row = pred[(pred["temperature"] == temp) & (pred["timestep"] == timestep)]

            if row.empty:
                raise ValueError(f"No prediction row for {temp} K timestep {timestep}")

            row = row.iloc[0]

            label = row["pred_label"]
            class_num = LABEL_TO_NUM[label]

            denoised_file = find_denoised_frame(temp, timestep)
            ace_file = find_ace_file(temp, timestep)

            print(f"\nProcessing {temp} K timestep {timestep}")
            print(f"  label: {label} -> {class_num}")
            print(f"  denoised: {denoised_file}")
            print(f"  ACE: {ace_file}")

            atoms = read(denoised_file)
            ace = load_ace_rows(ace_file, n_atoms=len(atoms))

            if len(atoms) != ace.shape[0]:
                raise ValueError(
                    f"Atom count mismatch for {temp} K {timestep}: "
                    f"atoms={len(atoms)}, ACE rows={ace.shape[0]}"
                )

            n = len(atoms)

            # Frame-level classifier output repeated per atom.
            atoms.arrays["ML_Class"] = np.full(n, class_num, dtype=int)
            atoms.arrays["ML_ProbKite"] = np.full(n, float(row["prob_Kite"]))
            atoms.arrays["ML_ProbL"] = np.full(n, float(row["prob_L"]))
            atoms.arrays["ML_Confidence"] = np.full(n, float(row["confidence"]))

            # Per-atom ACE descriptors.
            for j in range(ace.shape[1]):
                atoms.arrays[f"ACE_{j:03d}"] = ace[:, j]

            atoms.info["temperature"] = temp
            atoms.info["timestep"] = timestep
            atoms.info["pred_label"] = str(label)
            atoms.info["prob_Kite"] = float(row["prob_Kite"])
            atoms.info["prob_L"] = float(row["prob_L"])
            atoms.info["confidence"] = float(row["confidence"])

            out_dir = OUT_ROOT / f"GBALTemp{temp}_1_Alt"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"{temp}K_t{timestep}_{label}_ML_ACE.extxyz"
            write(out_file, atoms, format="extxyz")

            written.append(out_file)
            print(f"  wrote: {out_file}")

    print("\nDone. Wrote OVITO annotated files:")
    for f in written:
        print(f)

if __name__ == "__main__":
    main()
