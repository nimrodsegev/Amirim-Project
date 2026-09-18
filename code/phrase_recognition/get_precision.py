"""Precision, recall, and accuracy for balanced MLP probe, imbalanced test, all 8 layers, pooled i=2-5."""
import warnings, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json

warnings.filterwarnings("ignore")
SEED = 42
LAYERS = [5, 7, 10, 13, 15, 20, 25, 30]
I_VALUES = [2, 3, 4, 5]

data = np.load("data/probing_features.npz")
hidden_states = data["hidden_states"]
labels = data["labels"]
i_values = data["i_values"]
layer_indices = data["layer_indices"].tolist()
with open("data/probing_metadata.json") as f:
    meta = json.load(f)
phrases = np.array([r["phrase"] for r in meta["rows"]])

def downsample(X, y, g, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    n = min(len(pos), len(neg))
    keep = np.concatenate([pos, rng.choice(neg, n, replace=False)])
    return X[keep], y[keep], g[keep]

print(f"{'layer':<8}{'n':<7}{'pos%':<7}{'acc':<8}{'precision':<12}{'recall':<8}{'TP':<7}{'FP':<7}{'FN':<7}{'TN'}")
print("-"*80)

for L in LAYERS:
    slot = layer_indices.index(L)
    pool_mask = np.isin(i_values, I_VALUES)
    X = hidden_states[pool_mask, slot, :]
    y = labels[pool_mask, slot].astype(int)
    g = phrases[pool_mask]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(X, y, g))
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    gtr = g[tr_idx]

    Xtr_b, ytr_b, _ = downsample(Xtr, ytr, gtr, SEED+L)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_b.astype(np.float32))
    Xte_s = sc.transform(Xte.astype(np.float32))

    clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                        early_stopping=True, validation_fraction=0.15,
                        random_state=SEED, n_iter_no_change=10)
    clf.fit(Xtr_s, ytr_b)
    proba = clf.predict_proba(Xte_s)[:, 1]
    pred = (proba >= ytr_b.mean()).astype(int)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    TP = int(((pred==1) & (yte==1)).sum())
    FP = int(((pred==1) & (yte==0)).sum())
    FN = int(((pred==0) & (yte==1)).sum())
    TN = int(((pred==0) & (yte==0)).sum())

    print(f"L{L:<7}{len(yte):<7}{100*yte.mean():<7.1f}{acc*100:<8.1f}{prec*100:<12.1f}{rec*100:<8.1f}{TP:<7}{FP:<7}{FN:<7}{TN}")
