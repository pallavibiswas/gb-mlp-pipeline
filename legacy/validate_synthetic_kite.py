#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

CSV = "data/ace_kite_vs_L_atom_optionA_equalFiles.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

AUGMENT_CLASS = 0   # 0 = Kite
NOISE_STD = 0.005

def make_synthetic(X_class, n_new, rng, noise_std=0.005, a_low=0.2, a_high=0.8):
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
        X_syn += rng.normal(0.0, noise_std, size=X_syn.shape)

    return X_syn

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(CSV)
feat_cols = [c for c in df.columns if c.startswith("f")]

X = df[feat_cols].values
y = df["y"].values.astype(int)
groups = df["path"].astype(str).values

# fixed group split so train/test are separated
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

Xtr, ytr = X[train_idx], y[train_idx]
Xte, yte = X[test_idx], y[test_idx]

print("Train counts:", dict(pd.Series(ytr).value_counts().sort_index()))
print("Test counts:", dict(pd.Series(yte).value_counts().sort_index()))

# generate synthetic Kite to match majority
rng = np.random.default_rng(RANDOM_STATE)
count0 = np.sum(ytr == 0)
count1 = np.sum(ytr == 1)
target = max(count0, count1)
current = np.sum(ytr == AUGMENT_CLASS)
n_new = target - current

X_real_class = Xtr[ytr == AUGMENT_CLASS]
X_syn = make_synthetic(X_real_class, n_new=n_new, rng=rng, noise_std=NOISE_STD)

print("Synthetic shape:", X_syn.shape)

# -------------------------
# Standardize using train only
# -------------------------
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xsyn_s = scaler.transform(X_syn)

# -------------------------
# 1) Mean/std comparison
# -------------------------
real_mean = Xtr_s[ytr == AUGMENT_CLASS].mean(axis=0).mean()
syn_mean = Xsyn_s.mean(axis=0).mean()

real_std = Xtr_s[ytr == AUGMENT_CLASS].std(axis=0).mean()
syn_std = Xsyn_s.std(axis=0).mean()

print("\n=== Distribution summary ===")
print("Real class mean(avg over features):", real_mean)
print("Syn  class mean(avg over features):", syn_mean)
print("Real class std (avg over features):", real_std)
print("Syn  class std (avg over features):", syn_std)

# -------------------------
# 2) PCA location check
# -------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
Xtr_p = pca.fit_transform(Xtr_s)
Xsyn_p = pca.transform(Xsyn_s)

real_p = Xtr_p[ytr == AUGMENT_CLASS]

real_center = real_p.mean(axis=0)
syn_center = Xsyn_p.mean(axis=0)

dist_centers = np.linalg.norm(real_center - syn_center)

print("\n=== PCA check ===")
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Distance between real-class center and synthetic center (2D PCA):", dist_centers)

# -------------------------
# 3) Nearest-neighbor check
# -------------------------
nn = NearestNeighbors(n_neighbors=5)
nn.fit(Xtr_s)

distances, indices = nn.kneighbors(Xsyn_s)

neighbor_labels = ytr[indices]  # shape: (n_syn, 5)
same_class_frac = (neighbor_labels == AUGMENT_CLASS).mean()

print("\n=== Nearest-neighbor check ===")
print("Average 5-NN distance:", distances.mean())
print("Fraction of 5-NN labels matching intended class:", same_class_frac)

# -------------------------
# 4) Compare against real test class proximity
# -------------------------
Xte_class = Xte[yte == AUGMENT_CLASS]
if len(Xte_class) > 0:
    Xte_class_s = scaler.transform(Xte_class)
    dist_real_test, _ = nn.kneighbors(Xte_class_s)
    print("\n=== Real-vs-synthetic neighbor distance ===")
    print("Avg 5-NN distance for REAL held-out class atoms:", dist_real_test.mean())
    print("Avg 5-NN distance for SYNTHETIC atoms:", distances.mean())
