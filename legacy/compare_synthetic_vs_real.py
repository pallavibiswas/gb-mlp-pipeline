#!/usr/bin/env python3
"""
compare_synthetic_vs_real_clean.py

Purpose:
- Compare BASELINE vs SYNTHETIC augmentation for Kite(0) vs L(1)
- Use GROUP-AWARE train/test splits by file path
- Skip invalid splits where the TEST set contains only one class
- Report mean/std over valid splits only

Run:
  module purge
  module load python39
  source ~/venvs/mlpace/bin/activate
  cd ~/MLP_ACE
  python3 compare_synthetic_vs_real_clean.py
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------
# Config
# -------------------------------------------------
CSV = "data/ace_kite_vs_L_atom_optionA_equalFiles.csv"

# Best baseline config so far
HIDDEN = (128, 64)
ALPHA = 0.007
PCA_KEEP = 0.95

TEST_SIZE = 0.20
SEEDS = list(range(30))   # use more seeds to get enough valid splits

# Synthetic augmentation settings
AUGMENT_CLASS = 0   # 0=Kite, 1=L
NOISE_STD = 0.005

# -------------------------------------------------
# Synthetic generator
# -------------------------------------------------
def make_synthetic(X_class, n_new, rng, noise_std=0.005, a_low=0.2, a_high=0.8):
    """
    Create synthetic samples by interpolation between same-class points + Gaussian noise.
    """
    n = len(X_class)
    if n < 2 or n_new <= 0:
        return np.empty((0, X_class.shape[1]), dtype=X_class.dtype)

    i1 = rng.integers(0, n, size=n_new)
    i2 = rng.integers(0, n, size=n_new)

    X1 = X_class[i1]
    X2 = X_class[i2]

    lam = rng.uniform(a_low, a_high, size=(n_new, 1))
    X_syn = lam * X1 + (1.0 - lam) * X2

    if noise_std > 0:
        X_syn = X_syn + rng.normal(0.0, noise_std, size=X_syn.shape)

    return X_syn

# -------------------------------------------------
# Train/eval helper
# -------------------------------------------------
def train_model(Xtr, ytr, Xte, yte, seed):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_KEEP, random_state=seed)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            alpha=ALPHA,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed
        ))
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=[0, 1])

    # cm = [[K->K, K->L],
    #       [L->K, L->L]]
    kk, kl = cm[0, 0], cm[0, 1]
    lk, ll = cm[1, 0], cm[1, 1]

    kite_recall = kk / (kk + kl) if (kk + kl) > 0 else np.nan
    l_recall = ll / (lk + ll) if (lk + ll) > 0 else np.nan

    return {
        "acc": acc,
        "kite_recall": kite_recall,
        "L_recall": l_recall,
        "L_to_Kite": int(lk),
        "Kite_to_L": int(kl),
        "cm": cm
    }

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(CSV)

feat_cols = [c for c in df.columns if c.startswith("f")]
X = df[feat_cols].values
y = df["y"].values.astype(int)
groups = df["path"].astype(str).values

print("Total rows:", len(df))
print("Class counts:", dict(pd.Series(y).value_counts().sort_index()))
print("Unique groups:", len(np.unique(groups)))
print("Num features:", len(feat_cols))

baseline_rows = []
synthetic_rows = []

valid_splits = 0
skipped_splits = 0

# -------------------------------------------------
# Multi-split comparison
# -------------------------------------------------
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # Skip invalid test splits that contain only one class
    if len(np.unique(yte)) < 2:
        skipped_splits += 1
        print(f"\nSeed {seed}: skipped (single-class test set)")
        continue

    valid_splits += 1
    print(f"\n==============================")
    print(f"Seed {seed}")
    print("Train counts:", dict(pd.Series(ytr).value_counts().sort_index()))
    print("Test counts :", dict(pd.Series(yte).value_counts().sort_index()))

    # ---------------- BASELINE ----------------
    base = train_model(Xtr, ytr, Xte, yte, seed)
    baseline_rows.append({
        "seed": seed,
        "accuracy": base["acc"],
        "kite_recall": base["kite_recall"],
        "L_recall": base["L_recall"],
        "L_to_Kite": base["L_to_Kite"],
        "Kite_to_L": base["Kite_to_L"],
    })

    print("BASELINE")
    print(" ACC:", round(base["acc"], 4))
    print(" Kite recall:", round(base["kite_recall"], 4))
    print(" L recall:", round(base["L_recall"], 4))
    print(" L->Kite:", base["L_to_Kite"], " Kite->L:", base["Kite_to_L"])
    print(" CM:\n", base["cm"])

    # ---------------- SYNTHETIC ----------------
    rng = np.random.default_rng(seed)

    class_counts = pd.Series(ytr).value_counts()
    target = class_counts.max()
    current = np.sum(ytr == AUGMENT_CLASS)
    n_new = target - current

    X_aug = Xtr[ytr == AUGMENT_CLASS]
    X_syn = make_synthetic(X_aug, n_new=n_new, rng=rng, noise_std=NOISE_STD)
    y_syn = np.full(len(X_syn), AUGMENT_CLASS, dtype=int)

    Xtr_aug = np.vstack([Xtr, X_syn])
    ytr_aug = np.concatenate([ytr, y_syn])

    syn = train_model(Xtr_aug, ytr_aug, Xte, yte, seed)
    synthetic_rows.append({
        "seed": seed,
        "accuracy": syn["acc"],
        "kite_recall": syn["kite_recall"],
        "L_recall": syn["L_recall"],
        "L_to_Kite": syn["L_to_Kite"],
        "Kite_to_L": syn["Kite_to_L"],
    })

    print("SYNTHETIC")
    print(" Added synthetic:", len(X_syn), "for class", AUGMENT_CLASS)
    print(" ACC:", round(syn["acc"], 4))
    print(" Kite recall:", round(syn["kite_recall"], 4))
    print(" L recall:", round(syn["L_recall"], 4))
    print(" L->Kite:", syn["L_to_Kite"], " Kite->L:", syn["Kite_to_L"])
    print(" CM:\n", syn["cm"])

# -------------------------------------------------
# Aggregate results
# -------------------------------------------------
print("\n\n========================================")
print("VALID SPLITS:", valid_splits)
print("SKIPPED SPLITS:", skipped_splits)

if valid_splits == 0:
    print("No valid splits with both classes in test set.")
    raise SystemExit

baseline_df = pd.DataFrame(baseline_rows)
synthetic_df = pd.DataFrame(synthetic_rows)

print("\n==============================")
print("BASELINE SUMMARY")
print(baseline_df[["accuracy", "kite_recall", "L_recall", "L_to_Kite", "Kite_to_L"]].agg(["mean", "std", "min", "max"]))

print("\n==============================")
print("SYNTHETIC SUMMARY")
print(synthetic_df[["accuracy", "kite_recall", "L_recall", "L_to_Kite", "Kite_to_L"]].agg(["mean", "std", "min", "max"]))

# Save CSVs
baseline_df.to_csv("baseline_multi_split_results.csv", index=False)
synthetic_df.to_csv("synthetic_multi_split_results.csv", index=False)

print("\nSaved:")
print(" baseline_multi_split_results.csv")
print(" synthetic_multi_split_results.csv")
