#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV = "data/ace_kite_vs_L_atom_optionA_equalFiles.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

AUGMENT_CLASS = 0   # 0 = Kite, 1 = L
NOISE_STD = 0.01    # noise in SCALED space
A_LOW = 0.2
A_HIGH = 0.8

# best baseline config
HIDDEN = (128, 64)
ALPHA = 0.007
PCA_KEEP = 0.95

def make_synthetic_scaled(X_class_scaled, n_new, rng, noise_std=0.01, a_low=0.2, a_high=0.8):
    """
    Create synthetic samples in SCALED feature space by interpolation
    between same-class points + small Gaussian noise.
    """
    n = len(X_class_scaled)
    if n < 2 or n_new <= 0:
        return np.empty((0, X_class_scaled.shape[1]), dtype=X_class_scaled.dtype)

    i1 = rng.integers(0, n, size=n_new)
    i2 = rng.integers(0, n, size=n_new)

    X1 = X_class_scaled[i1]
    X2 = X_class_scaled[i2]

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

# Fixed group split
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

Xtr, ytr = X[train_idx], y[train_idx]
Xte, yte = X[test_idx], y[test_idx]

print("Train counts:", dict(pd.Series(ytr).value_counts().sort_index()))
print("Test counts:", dict(pd.Series(yte).value_counts().sort_index()))

# -------------------------
# Scale TRAIN data first
# -------------------------
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# Generate enough synthetic samples to match majority class
rng = np.random.default_rng(RANDOM_STATE)
count0 = np.sum(ytr == 0)
count1 = np.sum(ytr == 1)
target = max(count0, count1)
current = np.sum(ytr == AUGMENT_CLASS)
n_new = target - current

X_real_class_s = Xtr_s[ytr == AUGMENT_CLASS]
X_syn_s = make_synthetic_scaled(
    X_real_class_s,
    n_new=n_new,
    rng=rng,
    noise_std=NOISE_STD,
    a_low=A_LOW,
    a_high=A_HIGH
)

print("Synthetic shape:", X_syn_s.shape)

# -------------------------
# 1) Distribution check in scaled space
# -------------------------
real_mean = X_real_class_s.mean(axis=0).mean()
syn_mean = X_syn_s.mean(axis=0).mean()

real_std = X_real_class_s.std(axis=0).mean()
syn_std = X_syn_s.std(axis=0).mean()

print("\n=== Distribution summary (scaled space) ===")
print("Real class mean(avg over features):", real_mean)
print("Syn  class mean(avg over features):", syn_mean)
print("Real class std (avg over features):", real_std)
print("Syn  class std (avg over features):", syn_std)

# -------------------------
# 2) PCA location check
# -------------------------
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
Xtr_p = pca2.fit_transform(Xtr_s)
Xsyn_p = pca2.transform(X_syn_s)

real_p = Xtr_p[ytr == AUGMENT_CLASS]
real_center = real_p.mean(axis=0)
syn_center = Xsyn_p.mean(axis=0)
dist_centers = np.linalg.norm(real_center - syn_center)

print("\n=== PCA check ===")
print("Explained variance ratio:", pca2.explained_variance_ratio_)
print("Distance between real-class center and synthetic center (2D PCA):", dist_centers)

# -------------------------
# 3) Nearest-neighbor check
# -------------------------
nn = NearestNeighbors(n_neighbors=5)
nn.fit(Xtr_s)

distances, indices = nn.kneighbors(X_syn_s)
neighbor_labels = ytr[indices]
same_class_frac = (neighbor_labels == AUGMENT_CLASS).mean()

print("\n=== Nearest-neighbor check ===")
print("Average 5-NN distance:", distances.mean())
print("Fraction of 5-NN labels matching intended class:", same_class_frac)

Xte_class_s = Xte_s[yte == AUGMENT_CLASS]
if len(Xte_class_s) > 0:
    dist_real_test, _ = nn.kneighbors(Xte_class_s)
    print("\n=== Real-vs-synthetic neighbor distance ===")
    print("Avg 5-NN distance for REAL held-out class atoms:", dist_real_test.mean())
    print("Avg 5-NN distance for SYNTHETIC atoms:", distances.mean())

# -------------------------
# 4) Train baseline on real scaled train data
# -------------------------
baseline_pipe = Pipeline([
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

baseline_pipe.fit(Xtr_s, ytr)
pred_base = baseline_pipe.predict(Xte_s)

print("\n=== BASELINE TEST ===")
print("ACC:", accuracy_score(yte, pred_base))
print("CM:\n", confusion_matrix(yte, pred_base, labels=[0, 1]))
print(classification_report(yte, pred_base, target_names=["KITE(0)", "L(1)"], digits=4))

# -------------------------
# 5) Train augmented model (real train + synthetic in scaled space)
# -------------------------
y_syn = np.full(len(X_syn_s), AUGMENT_CLASS, dtype=int)

Xtr_aug_s = np.vstack([Xtr_s, X_syn_s])
ytr_aug = np.concatenate([ytr, y_syn])

aug_pipe = Pipeline([
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

aug_pipe.fit(Xtr_aug_s, ytr_aug)
pred_aug = aug_pipe.predict(Xte_s)

print("\n=== AUGMENTED TEST ===")
print("ACC:", accuracy_score(yte, pred_aug))
print("CM:\n", confusion_matrix(yte, pred_aug, labels=[0, 1]))
print(classification_report(yte, pred_aug, target_names=["KITE(0)", "L(1)"], digits=4))
