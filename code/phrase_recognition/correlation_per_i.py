"""
Per-phrase correlation per layer, for each i value (i=2, 3, 4, 5).
Same setup as correlation_all_layers.py but breaks down by i in the test set.
"""
import json, warnings
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")

TEST_FRAC = 0.2
SEED = 42
LAYERS = [5, 7, 10, 13, 15, 20, 25, 30]
I_VALUES = [2, 3, 4, 5]

data = np.load("data/probing_features.npz")
hidden_states = data["hidden_states"]; labels = data["labels"]; i_values = data["i_values"]
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

print(f"{'layer':<8}" + "".join(f"i={i:<10}" for i in I_VALUES) + f"{'all_i':<10}")
print(f"{'':<8}" + "".join(f"corr(n)':<10" for _ in I_VALUES) + f"{'corr(n)':<10}")
print("-"*70)

results = {}  # layer -> {i: (corr, n)}
for L in LAYERS:
    slot = layer_indices.index(L)
    pool_mask = np.isin(i_values, I_VALUES)
    X = hidden_states[pool_mask, slot, :]
    y = labels[pool_mask, slot].astype(int)
    g = phrases[pool_mask]
    iv = i_values[pool_mask]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(X, y, g))
    Xtr_full, Xte = X[tr_idx], X[te_idx]
    ytr_full, yte = y[tr_idx], y[te_idx]
    gtr_full, gte = g[tr_idx], g[te_idx]
    ite = iv[te_idx]
    Xtr, ytr, _ = downsample(Xtr_full, ytr_full, gtr_full, SEED+L)
    sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr.astype(np.float32))
    Xte_s = sc.transform(Xte.astype(np.float32))
    clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                        early_stopping=True, validation_fraction=0.15,
                        random_state=SEED, n_iter_no_change=10)
    clf.fit(Xtr_s, ytr)
    proba = clf.predict_proba(Xte_s)[:, 1]

    results[L] = {}
    row = f"L{L:<7}"
    for i_val in I_VALUES:
        keep = ite == i_val
        proba_i = proba[keep]; yte_i = yte[keep]; gte_i = gte[keep]
        pp = defaultdict(lambda: {"p":[], "y":[]})
        for ph, p, lab in zip(gte_i, proba_i, yte_i):
            pp[ph]["p"].append(p); pp[ph]["y"].append(lab)
        rows = [(np.mean(d["p"]), np.mean(d["y"])) for d in pp.values() if len(d["p"]) >= 3]
        if len(rows) < 2:
            results[L][i_val] = (float("nan"), len(rows))
            row += f"{'-':<10}"
            continue
        preds = np.array([r[0] for r in rows]); actuals = np.array([r[1] for r in rows])
        corr = float(np.corrcoef(preds, actuals)[0, 1])
        results[L][i_val] = (corr, len(rows))
        row += f"{corr:.3f}({len(rows):>3})  "

    # All i pooled
    pp_all = defaultdict(lambda: {"p":[], "y":[]})
    for ph, p, lab in zip(gte, proba, yte):
        pp_all[ph]["p"].append(p); pp_all[ph]["y"].append(lab)
    rows = [(np.mean(d["p"]), np.mean(d["y"])) for d in pp_all.values() if len(d["p"]) >= 3]
    preds_all = np.array([r[0] for r in rows]); actuals_all = np.array([r[1] for r in rows])
    corr_all = float(np.corrcoef(preds_all, actuals_all)[0, 1])
    results[L]["all"] = (corr_all, len(rows))
    row += f"{corr_all:.3f}({len(rows):>3})"
    print(row)

with open("results/correlation_per_i.json", "w") as f:
    out = {str(L): {str(k): v for k, v in d.items()} for L, d in results.items()}
    json.dump(out, f, indent=2)
print("\nSaved to results/correlation_per_i.json")
