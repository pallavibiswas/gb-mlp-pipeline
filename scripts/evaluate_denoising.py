import re
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd


CLEAN_DIR = Path.home() / "Kite_L"
SYNTH_ROOT = Path.home() / "KiteL_synthetic_0K"
DENOISED_ROOT = SYNTH_ROOT / "denoised"
OUT_DIR = Path.home() / "graphite" / "evaluation_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"dump\.gb\.(?P<idx>\d+)\.sigma_(?P<sigma>0p\d+)\.rep_(?P<rep>\d+)"
)


def parse_metadata(path: Path):
    m = pattern.search(path.name)
    if not m:
        raise ValueError(f"Could not parse filename: {path.name}")
    idx = int(m.group("idx"))
    sigma = float(m.group("sigma").replace("p", "."))
    rep = int(m.group("rep"))
    return idx, sigma, rep


def get_last_frame(path: Path):
    frames = ase.io.read(str(path), index=":")
    if isinstance(frames, list):
        return frames[-1]
    return frames


def displacement_stats(reference_atoms, test_atoms):
    ref = reference_atoms.positions
    test = test_atoms.positions

    if ref.shape != test.shape:
        raise ValueError(
            f"Shape mismatch: reference {ref.shape}, test {test.shape}"
        )

    disp = np.linalg.norm(test - ref, axis=1)
    return {
        "avg_disp_A": float(np.mean(disp)),
        "max_disp_A": float(np.max(disp)),
        "std_disp_A": float(np.std(disp)),
    }


rows = []

for subset in ["kite", "L"]:
    noisy_dir = SYNTH_ROOT / subset
    den_dir = DENOISED_ROOT / subset

    noisy_files = sorted([p for p in noisy_dir.rglob("*") if p.is_file()])

    for noisy_path in noisy_files:
        try:
            idx, sigma, rep = parse_metadata(noisy_path)
            clean_path = CLEAN_DIR / f"dump.gb.{idx}"
            denoised_path = den_dir / f"{noisy_path.stem}.denoised_.extxyz"

            if not clean_path.exists():
                print(f"Missing clean file: {clean_path}")
                continue

            if not denoised_path.exists():
                print(f"Missing denoised file: {denoised_path}")
                continue

            clean_atoms = get_last_frame(clean_path)
            noisy_atoms = get_last_frame(noisy_path)
            denoised_atoms = get_last_frame(denoised_path)

            noisy_stats = displacement_stats(clean_atoms, noisy_atoms)
            denoised_stats = displacement_stats(clean_atoms, denoised_atoms)

            row = {
                "subset": subset,
                "source_idx": idx,
                "source_file": f"dump.gb.{idx}",
                "sigma": sigma,
                "rep": rep,
                "noisy_file": str(noisy_path),
                "denoised_file": str(denoised_path),
                "noisy_avg_disp_A": noisy_stats["avg_disp_A"],
                "noisy_max_disp_A": noisy_stats["max_disp_A"],
                "noisy_std_disp_A": noisy_stats["std_disp_A"],
                "denoised_avg_disp_A": denoised_stats["avg_disp_A"],
                "denoised_max_disp_A": denoised_stats["max_disp_A"],
                "denoised_std_disp_A": denoised_stats["std_disp_A"],
            }

            row["avg_disp_improvement_A"] = (
                row["noisy_avg_disp_A"] - row["denoised_avg_disp_A"]
            )
            row["max_disp_improvement_A"] = (
                row["noisy_max_disp_A"] - row["denoised_max_disp_A"]
            )
            row["std_disp_improvement_A"] = (
                row["noisy_std_disp_A"] - row["denoised_std_disp_A"]
            )

            row["avg_disp_improvement_pct"] = (
                100.0
                * row["avg_disp_improvement_A"]
                / row["noisy_avg_disp_A"]
                if row["noisy_avg_disp_A"] > 0
                else np.nan
            )
            row["max_disp_improvement_pct"] = (
                100.0
                * row["max_disp_improvement_A"]
                / row["noisy_max_disp_A"]
                if row["noisy_max_disp_A"] > 0
                else np.nan
            )
            row["std_disp_improvement_pct"] = (
                100.0
                * row["std_disp_improvement_A"]
                / row["noisy_std_disp_A"]
                if row["noisy_std_disp_A"] > 0
                else np.nan
            )

            rows.append(row)

        except Exception as e:
            print(f"Failed on {noisy_path}: {e}")

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No evaluation rows were created.")

detail_csv = OUT_DIR / "denoising_eval_detail.csv"
df.to_csv(detail_csv, index=False)

summary = (
    df.groupby(["subset", "sigma"])
    .agg(
        noisy_avg_disp_A_mean=("noisy_avg_disp_A", "mean"),
        noisy_avg_disp_A_std=("noisy_avg_disp_A", "std"),
        denoised_avg_disp_A_mean=("denoised_avg_disp_A", "mean"),
        denoised_avg_disp_A_std=("denoised_avg_disp_A", "std"),
        noisy_max_disp_A_mean=("noisy_max_disp_A", "mean"),
        noisy_max_disp_A_std=("noisy_max_disp_A", "std"),
        denoised_max_disp_A_mean=("denoised_max_disp_A", "mean"),
        denoised_max_disp_A_std=("denoised_max_disp_A", "std"),
        noisy_std_disp_A_mean=("noisy_std_disp_A", "mean"),
        noisy_std_disp_A_std=("noisy_std_disp_A", "std"),
        denoised_std_disp_A_mean=("denoised_std_disp_A", "mean"),
        denoised_std_disp_A_std=("denoised_std_disp_A", "std"),
        avg_disp_improvement_A_mean=("avg_disp_improvement_A", "mean"),
        avg_disp_improvement_A_std=("avg_disp_improvement_A", "std"),
        avg_disp_improvement_pct_mean=("avg_disp_improvement_pct", "mean"),
        avg_disp_improvement_pct_std=("avg_disp_improvement_pct", "std"),
        max_disp_improvement_A_mean=("max_disp_improvement_A", "mean"),
        max_disp_improvement_A_std=("max_disp_improvement_A", "std"),
        std_disp_improvement_A_mean=("std_disp_improvement_A", "mean"),
        std_disp_improvement_A_std=("std_disp_improvement_A", "std"),
        n=("source_idx", "count"),
    )
    .reset_index()
)

summary_csv = OUT_DIR / "denoising_eval_summary.csv"
summary.to_csv(summary_csv, index=False)

print("\nSaved detail CSV to:")
print(detail_csv)

print("\nSaved summary CSV to:")
print(summary_csv)

print("\n=== Summary by subset and sigma ===")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(summary.to_string(index=False))
