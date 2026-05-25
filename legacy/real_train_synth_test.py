#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CSV = "data/ace_kite_vs_L_atom_optionA_equalFiles.csv"

# best config so far
HIDDEN = (128, 64)
ALPHA = 0.007
PCA_KEEP = 0.95
RANDOM_STATE = 42

# synthetic generation settings
SYNTH_CLASS = 0   # 0=Kite, 1=L
NOISE_STD = 0.005
N_SYNTH = 3000

def make_synthetic(X_class, n_new, rng, noise_std=0.005, a_low=0.2, a_high=0.8):
    n = len(X_class)
    if n < 2:
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

df = pd.read_csv(CSV)

feat_cols = [c for c in df.columns if c.startswith("f")]
X = df[feat_cols].values
y = df["y"].values.astype(int)

print("Total real rows:", len(df))
print("Class counts:", dict(pd.Series(y).value_counts().sort_index()))

# train on ALL real data
X_train = X
y_train = y

# make synthetic test data from one class
rng = np.random.default_rng(RANDOM_STATE)
X_class = X[y == SYNTH_CLASS]
X_syn = make_synthetic(X_class, n_new=N_SYNTH, rng=rng, noise_std=NOISE_STD)
y_syn = np.full(len(X_syn), SYNTH_CLASS, dtype=int)

print("Synthetic test shape:", X_syn.shape)
print("Synthetic class:", SYNTH_CLASS)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=PCA_KEEP, random_state=RANDOM_STATE)),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        alpha=ALPHA,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    ))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_syn)

acc = accuracy_score(y_syn, pred)
cm = confusion_matrix(y_syn, pred, labels=[0, 1])

print("\nSynthetic test accuracy:", acc)
print("Confusion matrix:\n", cm)
print(classification_report(y_syn, pred, target_names=["KITE(0)", "L(1)"], digits=4))
