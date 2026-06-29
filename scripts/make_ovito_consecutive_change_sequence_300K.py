from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read, write
import shutil

TEMP = 300
TMIN = 90000
TMAX = 150000

GB_AXIS = "x"
GB_HALF_WIDTH = 14.0

DENOISED_ROOT = Path.home() / "MD_GB_sims/gb_data_test_denoised"
ACE_ROOT = Path.home() / "ACE/runs2/gb_data_test_ace"
PRED_CSV = Path.home() / "MD_GB_sims/gb_data_classifier_outputs/gb_data_test_predictions.csv"
OUT_ROOT = Path.home() / "MD_GB_sims/gb_data_ovito_consecutive_change_sequence_300K"

AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

def find_denoised_frame(temp, timestep):
    folder = DENOISED_ROOT / f"GBALTemp{temp}_1_Alt"
    matches = sorted(folder.glob(f"*{timestep}*.denoised_.extxyz"))
    if not matches:
        matches = sorted(folder.glob(f"*{timestep}*.extxyz"))
    if not matches:
        raise FileNotFoundError(f"No denoised frame found for {temp} K timestep {timestep}")
    return matches[0]

def find_ace_file(temp, timestep):
    folder = ACE_ROOT / f"GBALTemp{temp}_1_Alt"
    matches = sorted(folder.glob(f"**/*{timestep}*/pace_row_output.txt"))
    if not matches:
        matches = sorted(folder.glob(f"**/*{timestep}*.denoised_*/**/pace_row_output.txt"))
    if not matches:
        raise FileNotFoundError(f"No ACE file found for {temp} K timestep {timestep}")
    return matches[0]

def load_ace_rows(path, n_atoms):
    rows, widths = [], []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
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

    ids_float = arr[:, 0]
    ids_int = np.rint(ids_float).astype(int)

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

        if len(np.unique(sub_ids)) == n_atoms:
            desc = sub[:, 1:]

            out = np.zeros((n_atoms, desc.shape[1]), dtype=float)
            counts = np.zeros(n_atoms, dtype=float)

            np.add.at(out, sub_ids, desc)
            np.add.at(counts, sub_ids, 1)

            if np.any(counts == 0):
                raise ValueError("Some atoms are missing ACE descriptors.")

            return out / counts[:, None]

    raise ValueError(f"Could not map ACE rows to atoms for {path}")

def central_gb_mask(atoms, axis="x", half_width=14.0):
    idx = AXIS_TO_INDEX[axis]
    coord = atoms.positions[:, idx]
    center = 0.5 * (coord.min() + coord.max())
    dist = np.abs(coord - center)
    return (dist <= half_width).astype(int), dist, center

def normalize_by_max(x, max_val):
    x = np.asarray(x, dtype=float)
    if max_val < 1e-12:
        return np.zeros_like(x)
    return np.clip(x / max_val, 0, 1)

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(PRED_CSV)
    pred["temperature"] = pred["temperature"].astype(int)
    pred["timestep"] = pred["timestep"].astype(int)

    g = pred[
        (pred["temperature"] == TEMP)
        & (pred["timestep"] >= TMIN)
        & (pred["timestep"] <= TMAX)
    ].copy()

    g = g.sort_values("timestep").reset_index(drop=True)

    if len(g) < 2:
        raise ValueError("Need at least 2 frames for consecutive change.")

    print(f"Frames: {len(g)}")
    print(f"Range : {g['timestep'].min()} to {g['timestep'].max()}")

    # First pass: compute consecutive changes and global scaling.
    records = []

    print("\nFirst pass: computing consecutive frame changes...")
    prev_atoms = None
    prev_ace = None
    prev_timestep = None

    all_disp = []
    all_ace_delta = []

    for i, row in g.iterrows():
        timestep = int(row["timestep"])
        atoms = read(find_denoised_frame(TEMP, timestep))
        n = len(atoms)
        ace = load_ace_rows(find_ace_file(TEMP, timestep), n)

        if prev_atoms is None:
            disp = np.zeros(n)
            ace_delta = np.zeros(n)
        else:
            if len(prev_atoms) != n:
                raise ValueError(f"Atom count mismatch: {prev_timestep} vs {timestep}")
            disp = np.linalg.norm(atoms.positions - prev_atoms.positions, axis=1)
            ace_delta = np.linalg.norm(ace - prev_ace, axis=1)

        records.append({
            "i": i,
            "row": row,
            "atoms": atoms,
            "ace": ace,
            "disp": disp,
            "ace_delta": ace_delta,
            "prev_timestep": prev_timestep,
        })

        all_disp.append(disp)
        all_ace_delta.append(ace_delta)

        prev_atoms = atoms
        prev_ace = ace
        prev_timestep = timestep

    max_disp = float(np.max(np.concatenate(all_disp)))
    max_ace_delta = float(np.max(np.concatenate(all_ace_delta)))

    print(f"Global max consecutive displacement: {max_disp}")
    print(f"Global max consecutive ACE delta: {max_ace_delta}")

    # Second pass: write sequence.
    print("\nSecond pass: writing OVITO sequence...")

    for rec in records:
        i = rec["i"]
        row = rec["row"]
        atoms = rec["atoms"]
        timestep = int(row["timestep"])
        label = str(row["pred_label"])
        prev_timestep = rec["prev_timestep"]

        n = len(atoms)
        disp_norm = normalize_by_max(rec["disp"], max_disp)
        ace_delta_norm = normalize_by_max(rec["ace_delta"], max_ace_delta)

        # Consecutive transition score = change since previous frame.
        consecutive_score = 0.5 * disp_norm + 0.5 * ace_delta_norm

        is_gb, gb_dist, gb_center = central_gb_mask(atoms, GB_AXIS, GB_HALF_WIDTH)
        is_gb_bool = is_gb.astype(bool)

        consecutive_score_gb = np.zeros(n)
        consecutive_score_gb[is_gb_bool] = consecutive_score[is_gb_bool]

        atoms.arrays["Is_GB"] = is_gb.astype(int)
        atoms.arrays["GB_Distance"] = gb_dist.astype(float)

        atoms.arrays["Consecutive_Displacement_Norm"] = disp_norm.astype(float)
        atoms.arrays["Consecutive_ACE_Delta_Norm"] = ace_delta_norm.astype(float)
        atoms.arrays["Consecutive_Transition_Score"] = consecutive_score.astype(float)
        atoms.arrays["Consecutive_Transition_Score_GB"] = consecutive_score_gb.astype(float)

        atoms.arrays["ML_ProbKite"] = np.full(n, float(row["prob_Kite"]))
        atoms.arrays["ML_ProbL"] = np.full(n, float(row["prob_L"]))
        atoms.arrays["ML_Confidence"] = np.full(n, float(row["confidence"]))

        atoms.info["temperature"] = TEMP
        atoms.info["previous_timestep"] = -1 if prev_timestep is None else int(prev_timestep)
        atoms.info["timestep"] = timestep
        atoms.info["pred_label"] = label
        atoms.info["prob_Kite"] = float(row["prob_Kite"])
        atoms.info["prob_L"] = float(row["prob_L"])
        atoms.info["confidence"] = float(row["confidence"])

        out = OUT_ROOT / (
            f"frame_{i:04d}_t{timestep}_{label}_"
            f"consecScore{consecutive_score_gb.max():.3f}.extxyz"
        )
        write(out, atoms, format="extxyz")

        print(
            f"{i:04d} prev={prev_timestep} -> t={timestep} label={label:4s} "
            f"P(Kite)={row['prob_Kite']:.3f} P(L)={row['prob_L']:.3f} "
            f"max_GB_consec_score={consecutive_score_gb.max():.3f}"
        )

    print("\nDone. Output folder:")
    print(OUT_ROOT)

if __name__ == "__main__":
    main()
