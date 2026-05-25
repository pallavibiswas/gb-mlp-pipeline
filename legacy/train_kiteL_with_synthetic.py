#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV = "data/ace_kite_vs_L_atom_optionA_equalFiles.csv"
RANDOM_STATE = 42

# choose which class to augment:
# 0 = Kite, 1 = L
AUGMENT_CLASS = 0   

# how many synthetic samples to create
# "match_majority" makes augmented class reach the size of the larger class
SYNTH_MODE = "match_majority"

# interpolation / noise settings
NOISE_STD = 0.02
ALPHA_LOW = 0.2
ALPHA_HIGH = 0.8

# model settings
PCA_KEEP = 0.95
ALPHA = 0.005

def make_synthetic(X_class, n_new, rng, noise_std=0.02, a_low=0.2, a_high=0.8):
    """
    Create synthetic samples by interpolating between random same-class pairs
    and adding small Gaussian noise.
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

    noise = rng.normal(loc=0.0, scale=noise_std, size=X_syn.shape)
    X_syn = X_syn + noise
    return X_syn


df = pd.read_csv(CSV)

feat_cols = [c for c in df.columns if c.startswith("f")]
X = df[feat_cols].values
y = df["y"].values.astype(int)
groups = df["path"].astype(str).values

print("Total rows:", len(df))
print("Class counts:", dict(pd.Series(y).value_counts().sort_index()))
print("Unique groups:", len(np.unique(groups)))

# group-aware split
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

Xtr, ytr = X[train_idx], y[train_idx]
Xte, yte = X[test_idx], y[test_idx]

print("Train groups:", len(np.unique(groups[train_idx])), "Test groups:", len(np.unique(groups[test_idx])))
print("Train counts before synthesis:", dict(pd.Series(ytr).value_counts().sort_index()))
print("Test counts:", dict(pd.Series(yte).value_counts().sort_index()))

rng = np.random.default_rng(RANDOM_STATE)

# decide how many synthetic samples to make
count0 = np.sum(ytr == 0)
count1 = np.sum(ytr == 1)

if SYNTH_MODE == "match_majority":
    target = max(count0, count1)
    current = np.sum(ytr == AUGMENT_CLASS)
    n_new = target - current
else:
    n_new = 0

X_aug = Xtr[ytr == AUGMENT_CLASS]
X_syn = make_synthetic(
    X_aug,
    n_new=n_new,
    rng=rng,
    noise_std=NOISE_STD,
    a_low=ALPHA_LOW,
    a_high=ALPHA_HIGH
)
y_syn = np.full(len(X_syn), AUGMENT_CLASS, dtype=int)

Xtr_aug = np.vstack([Xtr, X_syn]) if len(X_syn) > 0 else Xtr
ytr_aug = np.concatenate([ytr, y_syn]) if len(y_syn) > 0 else ytr

print("Synthetic samples added:", len(X_syn), "for class", AUGMENT_CLASS)
print("Train counts after synthesis:", dict(pd.Series(ytr_aug).value_counts().sort_index()))

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=PCA_KEEP, random_state=RANDOM_STATE)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=ALPHA,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    ))
])

pipe.fit(Xtr_aug, ytr_aug)
pred = pipe.predict(Xte)

acc = accuracy_score(yte, pred)
cm = confusion_matrix(yte, pred, labels=[0, 1])

print("\nTEST ACCURACY:", acc)
print("TEST CM [ [K->K, K->L], [L->K, L->L] ]:\n", cm)
print("\nClassification report:")
print(classification_report(yte, pred, target_names=["KITE(0)", "L(1)"], digits=4))
