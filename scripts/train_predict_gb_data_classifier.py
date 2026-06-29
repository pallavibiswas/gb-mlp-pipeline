from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import joblib


TRAIN_ROOT = Path.home() / "ACE/runs2/gb_data_original_train_ace"
TEST_ROOT  = Path.home() / "ACE/runs2/gb_data_test_ace"
OUT_DIR    = Path.home() / "MD_GB_sims/gb_data_classifier_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ace_feature(path: Path):
    # pace_row_output.txt may start with a short metadata/header row.
    # Keep only numeric descriptor rows with the dominant wide column count.
    rows = []
    widths = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Skip metadata rows like: "4000 326"
            if len(parts) < 10:
                continue

            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            rows.append(vals)
            widths.append(len(vals))

    if not rows:
        raise ValueError(f"No numeric ACE descriptor rows found in {path}")

    # Use the dominant descriptor width.
    # This avoids crashes if one malformed/header row sneaks through.
    unique, counts = np.unique(widths, return_counts=True)
    target_width = int(unique[np.argmax(counts)])

    arr = np.asarray([r for r in rows if len(r) == target_width], dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]

    if arr.size == 0:
        raise ValueError(f"No finite ACE rows in {path}")

    # Auto-drop atom-id-like first column if present
    first = arr[:, 0]
    n = arr.shape[0]
    if (
        np.allclose(first, np.round(first), atol=1e-6)
        and len(np.unique(first)) > 0.8 * n
        and np.nanmin(first) >= 0
        and np.nanmax(first) <= 2 * n + 10
        and arr.shape[1] > 2
    ):
        arr = arr[:, 1:]

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return np.concatenate([mean, std])


def train_label_from_path(path: Path):
    rel = path.relative_to(TRAIN_ROOT)
    label = rel.parts[0]  # Kite or L
    sample = rel.parts[1].replace(".denoised_", "")
    return label, sample


def test_meta_from_path(path: Path):
    rel = path.relative_to(TEST_ROOT)
    folder = rel.parts[0]
    sample_dir = rel.parts[1]
    sample = sample_dir.replace(".denoised_", "")

    m_temp = re.search(r"Temp(\d+)", folder)
    temperature = int(m_temp.group(1)) if m_temp else None

    m_step = re.search(r"gbstep\d+\.(\d+)", sample)
    timestep = int(m_step.group(1)) if m_step else None

    return folder, sample, temperature, timestep


def collect_train():
    rows, X, y = [], [], []

    files = sorted(TRAIN_ROOT.glob("**/pace_row_output.txt"))
    print(f"Train ACE files: {len(files)}", flush=True)

    for i, path in enumerate(files, 1):
        label, sample = train_label_from_path(path)
        feat = load_ace_feature(path)

        X.append(feat)
        y.append(label)
        rows.append({
            "sample": sample,
            "label": label,
            "path": str(path),
        })

        print(f"[train {i}/{len(files)}] {label} {sample}", flush=True)

    return np.vstack(X), np.array(y), pd.DataFrame(rows)


def collect_test():
    rows, X = [], []

    files = sorted(TEST_ROOT.glob("**/pace_row_output.txt"))
    print(f"Test ACE files: {len(files)}", flush=True)

    for i, path in enumerate(files, 1):
        folder, sample, temperature, timestep = test_meta_from_path(path)
        feat = load_ace_feature(path)

        X.append(feat)
        rows.append({
            "folder": folder,
            "sample": sample,
            "temperature": temperature,
            "timestep": timestep,
            "path": str(path),
        })

        print(f"[test {i}/{len(files)}] {folder} {sample}", flush=True)

    return np.vstack(X), pd.DataFrame(rows)


print("Loading train features...", flush=True)
X_train, y_train, train_df = collect_train()

print("Loading test features...", flush=True)
X_test, test_df = collect_test()

print("Train shape:", X_train.shape, flush=True)
print("Test shape :", X_test.shape, flush=True)
print("Train label counts:", dict(pd.Series(y_train).value_counts()), flush=True)

clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=3000,
        random_state=42
    )),
])

# Cross-validation on original train set
min_class = pd.Series(y_train).value_counts().min()
n_splits = min(5, int(min_class))

summary_lines = []
summary_lines.append(f"Train shape: {X_train.shape}")
summary_lines.append(f"Test shape: {X_test.shape}")
summary_lines.append(f"Train label counts: {dict(pd.Series(y_train).value_counts())}")

if n_splits >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_cv = cross_val_predict(clf, X_train, y_train, cv=cv)

    summary_lines.append("")
    summary_lines.append(f"Stratified CV folds: {n_splits}")
    summary_lines.append("CV confusion matrix labels [Kite, L]:")
    labels = ["Kite", "L"]
    summary_lines.append(str(confusion_matrix(y_train, y_cv, labels=labels)))
    summary_lines.append("")
    summary_lines.append(classification_report(y_train, y_cv))
else:
    summary_lines.append("Not enough samples for CV.")

# Fit final model
clf.fit(X_train, y_train)
joblib.dump(clf, OUT_DIR / "gb_data_classifier_logreg.joblib")

# Predict train
train_prob = clf.predict_proba(X_train)
classes = list(clf.named_steps["model"].classes_)
train_pred = clf.predict(X_train)

train_df["pred_label"] = train_pred
train_df["confidence"] = train_prob.max(axis=1)
for c in classes:
    train_df[f"prob_{c}"] = train_prob[:, classes.index(c)]
train_df.to_csv(OUT_DIR / "gb_data_train_predictions.csv", index=False)

# Predict test
test_prob = clf.predict_proba(X_test)
test_pred = clf.predict(X_test)

test_df["pred_label"] = test_pred
test_df["confidence"] = test_prob.max(axis=1)
for c in classes:
    test_df[f"prob_{c}"] = test_prob[:, classes.index(c)]

if "prob_Kite" not in test_df:
    test_df["prob_Kite"] = np.nan
if "prob_L" not in test_df:
    test_df["prob_L"] = np.nan

test_df = test_df.sort_values(["temperature", "timestep", "sample"])
test_df.to_csv(OUT_DIR / "gb_data_test_predictions.csv", index=False)

# Transition summary
summary_rows = []
for temp, g in test_df.groupby("temperature"):
    g = g.sort_values("timestep")
    preds = g["pred_label"].tolist()
    timesteps = g["timestep"].tolist()

    switches = []
    for i in range(1, len(g)):
        if preds[i] != preds[i - 1]:
            switches.append(f"{timesteps[i-1]}->{timesteps[i]}:{preds[i-1]}->{preds[i]}")

    min_conf_idx = g["confidence"].idxmin()
    max_kite_idx = g["prob_Kite"].idxmax()

    summary_rows.append({
        "temperature": temp,
        "n_frames": len(g),
        "first_pred": preds[0],
        "last_pred": preds[-1],
        "n_switches": len(switches),
        "switches": "; ".join(switches),
        "min_confidence": g.loc[min_conf_idx, "confidence"],
        "min_conf_timestep": g.loc[min_conf_idx, "timestep"],
        "max_prob_Kite": g.loc[max_kite_idx, "prob_Kite"],
        "max_prob_Kite_timestep": g.loc[max_kite_idx, "timestep"],
    })

transition_df = pd.DataFrame(summary_rows).sort_values("temperature")
transition_df.to_csv(OUT_DIR / "transition_summary_by_temperature.csv", index=False)

# Summary text
summary_lines.append("")
summary_lines.append("Final train accuracy:")
summary_lines.append(str((train_df["label"] == train_df["pred_label"]).mean()))
summary_lines.append("")
summary_lines.append("Transition summary:")
summary_lines.append(transition_df.to_string(index=False))

(OUT_DIR / "gb_data_classifier_summary.txt").write_text("\n".join(summary_lines))

# Plots
plt.figure(figsize=(10, 6))
for temp, g in test_df.groupby("temperature"):
    g = g.sort_values("timestep")
    plt.plot(g["timestep"], g["prob_Kite"], marker="o", linewidth=1, markersize=2, label=f"{temp}K")
plt.xlabel("Timestep")
plt.ylabel("P(Kite)")
plt.title("P(Kite) vs timestep by temperature")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "prob_Kite_vs_timestep_by_temperature.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 6))
for temp, g in test_df.groupby("temperature"):
    g = g.sort_values("timestep")
    plt.plot(g["timestep"], g["prob_L"], marker="o", linewidth=1, markersize=2, label=f"{temp}K")
plt.xlabel("Timestep")
plt.ylabel("P(L)")
plt.title("P(L) vs timestep by temperature")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "prob_L_vs_timestep_by_temperature.png", dpi=200)
plt.close()

pivot = test_df.pivot_table(index="temperature", columns="timestep", values="prob_Kite", aggfunc="mean")
plt.figure(figsize=(12, 5))
plt.imshow(pivot.values, aspect="auto", origin="lower")
plt.yticks(range(len(pivot.index)), pivot.index)
step_cols = list(pivot.columns)
tick_idx = np.linspace(0, len(step_cols)-1, min(10, len(step_cols))).astype(int)
plt.xticks(tick_idx, [step_cols[i] for i in tick_idx], rotation=45)
plt.colorbar(label="P(Kite)")
plt.xlabel("Timestep")
plt.ylabel("Temperature")
plt.title("P(Kite) heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "prob_Kite_heatmap.png", dpi=200)
plt.close()

pred_map = {"L": 0, "Kite": 1}
test_df["pred_numeric"] = test_df["pred_label"].map(pred_map)
pivot_pred = test_df.pivot_table(index="temperature", columns="timestep", values="pred_numeric", aggfunc="mean")
plt.figure(figsize=(12, 5))
plt.imshow(pivot_pred.values, aspect="auto", origin="lower")
plt.yticks(range(len(pivot_pred.index)), pivot_pred.index)
step_cols = list(pivot_pred.columns)
tick_idx = np.linspace(0, len(step_cols)-1, min(10, len(step_cols))).astype(int)
plt.xticks(tick_idx, [step_cols[i] for i in tick_idx], rotation=45)
plt.colorbar(label="Predicted class: L=0, Kite=1")
plt.xlabel("Timestep")
plt.ylabel("Temperature")
plt.title("Predicted class heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "predicted_class_heatmap.png", dpi=200)
plt.close()

print("\nDONE")
print("Outputs written to:", OUT_DIR)
print((OUT_DIR / "gb_data_classifier_summary.txt").read_text())
