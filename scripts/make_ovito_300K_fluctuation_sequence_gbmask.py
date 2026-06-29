from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import read, write
import shutil

TEMP = 300
TMIN = 90000
TMAX = 150000

# Adjust if the selected GB band is too narrow/wide in OVITO.
GB_AXIS = "x"
GB_HALF_WIDTH = 14.0

DENOISED_ROOT = Path.home() / "MD_GB_sims/gb_data_test_denoised"
PRED_CSV = Path.home() / "MD_GB_sims/gb_data_classifier_outputs/gb_data_test_predictions.csv"
OUT_ROOT = Path.home() / "MD_GB_sims/gb_data_ovito_300K_fluctuation_sequence_gbmask"

LABEL_TO_NUM = {"Kite": 0, "L": 1}
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

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

def central_gb_mask(atoms, axis="x", half_width=14.0):
    idx = AXIS_TO_INDEX[axis]
    coord = atoms.positions[:, idx]

    # Central slab mask: selects atoms near the middle GB band.
    center = 0.5 * (coord.min() + coord.max())
    dist = np.abs(coord - center)
    mask = dist <= half_width

    return mask.astype(int), dist, center

def main():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PRED_CSV)
    df["temperature"] = df["temperature"].astype(int)
    df["timestep"] = df["timestep"].astype(int)

    g = df[
        (df["temperature"] == TEMP)
        & (df["timestep"] >= TMIN)
        & (df["timestep"] <= TMAX)
    ].copy()

    g = g.sort_values("timestep").reset_index(drop=True)

    if g.empty:
        raise ValueError("No 300 K frames found in selected timestep range.")

    g["prev_label"] = g["pred_label"].shift()
    g["ML_SwitchFrame"] = (g["pred_label"] != g["prev_label"]).astype(int)
    g.loc[0, "ML_SwitchFrame"] = 0
    g["ML_LowConfidence"] = (g["confidence"] < 0.70).astype(int)

    print(f"Writing {len(g)} annotated 300 K frames")
    print(f"Timestep range: {g['timestep'].min()} to {g['timestep'].max()}")
    print(f"GB mask: central slab along {GB_AXIS}, half-width = {GB_HALF_WIDTH}")

    for i, row in g.iterrows():
        timestep = int(row["timestep"])
        label = str(row["pred_label"])
        class_num = LABEL_TO_NUM[label]

        src = find_denoised_frame(TEMP, timestep)
        atoms = read(src)
        n = len(atoms)

        is_gb, gb_dist, gb_center = central_gb_mask(atoms, GB_AXIS, GB_HALF_WIDTH)
        is_gb_bool = is_gb.astype(bool)

        # Frame-level ML values repeated on all atoms.
        atoms.arrays["ML_Class"] = np.full(n, class_num, dtype=int)
        atoms.arrays["ML_ProbKite"] = np.full(n, float(row["prob_Kite"]))
        atoms.arrays["ML_ProbL"] = np.full(n, float(row["prob_L"]))
        atoms.arrays["ML_Confidence"] = np.full(n, float(row["confidence"]))
        atoms.arrays["ML_LowConfidence"] = np.full(n, int(row["ML_LowConfidence"]), dtype=int)
        atoms.arrays["ML_SwitchFrame"] = np.full(n, int(row["ML_SwitchFrame"]), dtype=int)

        # New OVITO-friendly GB properties.
        atoms.arrays["Is_GB"] = is_gb.astype(int)
        atoms.arrays["GB_Distance"] = gb_dist.astype(float)

        ml_class_gb = np.full(n, -1, dtype=int)
        ml_class_gb[is_gb_bool] = class_num
        atoms.arrays["ML_Class_GB"] = ml_class_gb

        low_conf_gb = np.zeros(n, dtype=int)
        low_conf_gb[is_gb_bool] = int(row["ML_LowConfidence"])
        atoms.arrays["ML_LowConfidence_GB"] = low_conf_gb

        switch_gb = np.zeros(n, dtype=int)
        switch_gb[is_gb_bool] = int(row["ML_SwitchFrame"])
        atoms.arrays["ML_SwitchFrame_GB"] = switch_gb

        atoms.info["temperature"] = TEMP
        atoms.info["timestep"] = timestep
        atoms.info["pred_label"] = label
        atoms.info["prob_Kite"] = float(row["prob_Kite"])
        atoms.info["prob_L"] = float(row["prob_L"])
        atoms.info["confidence"] = float(row["confidence"])
        atoms.info["gb_axis"] = GB_AXIS
        atoms.info["gb_half_width"] = GB_HALF_WIDTH
        atoms.info["gb_center"] = float(gb_center)

        out = OUT_ROOT / f"frame_{i:04d}_t{timestep}_{label}_pKite{row['prob_Kite']:.3f}_pL{row['prob_L']:.3f}.extxyz"
        write(out, atoms, format="extxyz")

        print(
            f"{i:04d}  t={timestep}  label={label:4s}  "
            f"P(Kite)={row['prob_Kite']:.3f}  P(L)={row['prob_L']:.3f}  "
            f"GB atoms={int(is_gb.sum())}/{n}"
        )

    print("\nDone. Output folder:")
    print(OUT_ROOT)

if __name__ == "__main__":
    main()
