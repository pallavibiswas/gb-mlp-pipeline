#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ACE_ROOT = Path.home() / "ACE" / "runs2" / "batch_archived_outputs_direct"
OUT_DIR = Path.home() / "graphite" / "mlp_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Labels
LABEL_MAP = {
    "L": 0,
    "kite": 1,
}
LABEL_NAME = {0: "L", 1: "kite"}

# Split / model config
TEST_SIZE = 0.20
RANDOM_STATE = 42

HIDDEN = (256, 128)
ALPHA = 1e-4
LR_INIT = 1e-3
MAX_ITER = 300
PCA_KEEP = 0.95


def find_ace_files(root: Path) -> list[Path]:
    return sorted(root.rglob("pace_row_output.txt"))


def parse_metadata_from_path(p: Path) -> dict:
    """
    Example path:
    .../batch_archived_outputs_direct/kite/dump.gb.2.sigma_0p010.rep_2.denoised_/run_dump.gb.2.sigma_0p010.rep_2.denoised_/pace_row_output.txt
    """
    s = str(p)

    subset = None
    if "/kite/" in s:
        subset = "kite"
    elif "/L/" in s:
        subset = "L"
    else:
        raise ValueError(f"Could not infer subset from path: {p}")

    m = re.search(
        r"dump\.gb\.(?P<idx>\d+)\.sigma_(?P<sigma>0p\d+)\.rep_(?P<rep>\d+)",
        s
    )
    if not m:
        raise ValueError(f"Could not parse dump/sigma/rep from path: {p}")

    source_idx = int(m.group("idx"))
    sigma = float(m.group("sigma").replace("p", "."))
    rep = int(m.group("rep"))

    # Group by original source structure to avoid leakage across reps/sigmas of same base file
    group_id = f"{subset}_dump.gb.{source_idx}"

    return {
        "subset": subset,
        "y": LABEL_MAP[subset],
        "source_idx": source_idx,
        "sigma": sigma,
        "rep": rep,
        "group_id": group_id,
        "path": str(p),
    }


def load_matrix_robust(p: Path) -> np.ndarray:
    """
    Read numeric rows only. Keep rows with the most common column count.
    This is robust to comments/header lines in pace_row_output.txt.
    """
    rows = []
    lengths = []

    with p.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                nums = [float(x) for x in parts]
            except ValueError:
                continue
            rows.append(nums)
            lengths.append(len(nums))

    if not rows:
        raise ValueError(f"No numeric rows found in {p}")

    mode_len = Counter(lengths).most_common(1)[0][0]
    rows2 = [r for r in rows if len(r) == mode_len]

    X = np.asarray(rows2, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def featureize(X: np.ndarray) -> np.ndarray:
    """
    Condense atom-wise ACE matrix into a fixed-length vector:
    [mean(feature_1..d), std(feature_1..d)]
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.concatenate([mu, sd], axis=0)


def build_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        meta = parse_metadata_from_path(p)
        X = load_matrix_robust(p)
        feat = featureize(X)

        row = {
            **meta,
            "n_rows": X.shape[0],
            "n_cols": X.shape[1],
        }
        row.update({f"f{i}": feat[i] for i in range(feat.shape[0])})
        rows.append(row)

        print(
            f"[OK] {meta['subset']} "
            f"dump.gb.{meta['source_idx']} "
            f"sigma={meta['sigma']:.3f} rep={meta['rep']} "
            f"matrix={X.shape} feat_dim={feat.shape[0]}",
            flush=True,
        )
    return rows


def main() -> None:
    ace_files = find_ace_files(ACE_ROOT)
    if not ace_files:
        raise FileNotFoundError(f"No pace_row_output.txt files found under {ACE_ROOT}")

    print(f"Found {len(ace_files)} ACE files.", flush=True)

    rows = build_rows(ace_files)
    df = pd.DataFrame(rows)

    csv_path = OUT_DIR / "ace_denoised_kite_vs_L.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset CSV to: {csv_path}", flush=True)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    y = df["y"].values
    groups = df["group_id"].values

    print(f"Dataset shape: X={X.shape}, y={y.shape}", flush=True)
    print("Class counts:", df["subset"].value_counts().to_dict(), flush=True)
    print("Unique groups:", len(np.unique(groups)), flush=True)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    print(
        f"Train size={len(train_df)}, Test size={len(test_df)}",
        flush=True,
    )
    print(
        "Train groups:",
        len(train_df["group_id"].unique()),
        "Test groups:",
        len(test_df["group_id"].unique()),
        flush=True,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_KEEP, random_state=RANDOM_STATE)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            alpha=ALPHA,
            learning_rate_init=LR_INIT,
            max_iter=MAX_ITER,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )),
    ])

    print("Fitting pipeline...", flush=True)
    pipe.fit(Xtr, ytr)

    pca = pipe.named_steps["pca"]
    mlp = pipe.named_steps["mlp"]

    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    cm = confusion_matrix(yte, yhat)

    print(f"\nTest Accuracy: {acc:.4f}", flush=True)
    print("\nConfusion Matrix [rows=true, cols=pred]:", flush=True)
    print(cm, flush=True)

    report = classification_report(
        yte,
        yhat,
        target_names=["L", "kite"],
        digits=4,
    )
    print("\nClassification Report:", flush=True)
    print(report, flush=True)

    # Save predictions
    pred_df = test_df[[
        "subset", "source_idx", "sigma", "rep", "group_id", "path"
    ]].copy()
    pred_df["y_true"] = yte
    pred_df["y_pred"] = yhat
    pred_df["true_label"] = [LABEL_NAME[v] for v in yte]
    pred_df["pred_label"] = [LABEL_NAME[v] for v in yhat]
    pred_path = OUT_DIR / "ace_denoised_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # Save summary text
    summary_path = OUT_DIR / "ace_denoised_mlp_summary.txt"
    with summary_path.open("w") as f:
        f.write(f"ACE root: {ACE_ROOT}\n")
        f.write(f"Dataset CSV: {csv_path}\n")
        f.write(f"Dataset shape: X={X.shape}, y={y.shape}\n")
        f.write(f"Class counts: {df['subset'].value_counts().to_dict()}\n")
        f.write(f"Unique groups: {len(np.unique(groups))}\n")
        f.write(f"Train size: {len(train_df)}\n")
        f.write(f"Test size: {len(test_df)}\n")
        f.write(f"Train groups: {len(train_df['group_id'].unique())}\n")
        f.write(f"Test groups: {len(test_df['group_id'].unique())}\n")
        f.write(f"PCA retained components: {pca.n_components_}\n")
        f.write(f"MLP iterations: {mlp.n_iter_}\n")
        f.write(f"Best validation score: {getattr(mlp, 'best_validation_score_', 'NA')}\n")
        f.write(f"Test Accuracy: {acc:.6f}\n\n")
        f.write("Confusion Matrix [rows=true, cols=pred]:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print(f"\nSaved predictions to: {pred_path}", flush=True)
    print(f"Saved summary to: {summary_path}", flush=True)
    print(f"PCA retained components: {pca.n_components_}", flush=True)
    print(f"MLP iterations: {mlp.n_iter_}", flush=True)


if __name__ == "__main__":
    main()
