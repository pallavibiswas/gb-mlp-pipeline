#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

CSV = "data/ace_kite_vs_L_atom_optionA_equalFiles.csv"

HIDDEN = (128,64)
ALPHA = 0.007
PCA_KEEP = 0.95

TEST_SIZE = 0.20
SEEDS = list(range(10))

AUGMENT_CLASS = 0
NOISE_STD = 0.005

def make_synthetic(X_class, n_new, rng):
    n = len(X_class)
    i1 = rng.integers(0,n,size=n_new)
    i2 = rng.integers(0,n,size=n_new)

    lam = rng.uniform(0.2,0.8,size=(n_new,1))

    X_syn = lam*X_class[i1] + (1-lam)*X_class[i2]
    X_syn += rng.normal(0,NOISE_STD,size=X_syn.shape)

    return X_syn

def train_model(Xtr,ytr,Xte,yte):

    pipe = Pipeline([
        ("scaler",StandardScaler()),
        ("pca",PCA(n_components=PCA_KEEP)),
        ("mlp",MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            alpha=ALPHA,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20
        ))
    ])

    pipe.fit(Xtr,ytr)
    pred = pipe.predict(Xte)

    acc = accuracy_score(yte,pred)
    cm = confusion_matrix(yte,pred,labels=[0,1])

    kk,kl = cm[0]
    lk,ll = cm[1]

    kite_recall = kk/(kk+kl)
    L_recall = ll/(lk+ll)

    return acc,kite_recall,L_recall,lk,kl


df = pd.read_csv(CSV)

feat_cols=[c for c in df.columns if c.startswith("f")]

X=df[feat_cols].values
y=df["y"].values.astype(int)
groups=df["path"].values


baseline=[]
synthetic=[]


for seed in SEEDS:

    gss=GroupShuffleSplit(n_splits=1,test_size=TEST_SIZE,random_state=seed)
    train_idx,test_idx=next(gss.split(X,y,groups))

    Xtr,Xte=X[train_idx],X[test_idx]
    ytr,yte=y[train_idx],y[test_idx]

    # baseline
    baseline.append(train_model(Xtr,ytr,Xte,yte))


    # synthetic augmentation
    rng=np.random.default_rng(seed)

    class_counts=pd.Series(ytr).value_counts()

    target=class_counts.max()
    current=np.sum(ytr==AUGMENT_CLASS)

    n_new=target-current

    X_aug=Xtr[ytr==AUGMENT_CLASS]

    X_syn=make_synthetic(X_aug,n_new,rng)
    y_syn=np.full(len(X_syn),AUGMENT_CLASS)

    Xtr_aug=np.vstack([Xtr,X_syn])
    ytr_aug=np.concatenate([ytr,y_syn])

    synthetic.append(train_model(Xtr_aug,ytr_aug,Xte,yte))


baseline=np.array(baseline)
synthetic=np.array(synthetic)

print("\n==============================")
print("BASELINE MEAN")
print("accuracy:",baseline[:,0].mean())
print("kite recall:",baseline[:,1].mean())
print("L recall:",baseline[:,2].mean())
print("L->Kite:",baseline[:,3].mean())

print("\n==============================")
print("SYNTHETIC MEAN")
print("accuracy:",synthetic[:,0].mean())
print("kite recall:",synthetic[:,1].mean())
print("L recall:",synthetic[:,2].mean())
print("L->Kite:",synthetic[:,3].mean())
