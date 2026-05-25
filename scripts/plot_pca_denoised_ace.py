#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CSV = Path.home() / "graphite" / "mlp_outputs" / "ace_denoised_kite_vs_L.csv"
OUT = Path.home() / "graphite" / "mlp_outputs" / "pca_denoised_ace.png"

df = pd.read_csv(CSV)

feat_cols = [c for c in df.columns if c.startswith("f")]
X = df[feat_cols].values
y = df["subset"].values

X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
for label, marker in [("L", "o"), ("kite", "s")]:
    mask = (y == label)
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=label,
        marker=marker,
        alpha=0.8,
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Denoised ACE Features")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT, dpi=300)
print(f"Saved PCA plot to: {OUT}")
