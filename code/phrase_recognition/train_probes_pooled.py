"""
Pooled per-layer probes (not per-i).

For each layer in {5, 7, 10, 13, 15, 20, 25, 30}:
  - Train one logistic and one MLP probe on data pooled across i=2 to i=5.
  - Train one probe on imbalanced data, one on balanced data (downsample negatives).
  - Test set: held-out phrases (80/20 split by phrase).
  - Report metrics:
      Overall (test set pooled across i)
      Per-i breakdown (test set sliced by i)
      Per-category breakdown (test set sliced by buildings/idioms/movies)
  - Metrics: AUC-ROC, AUC-PR (with positive-rate baseline), balanced accuracy, n
"""
import json, warnings
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

TEST_FRAC = 0.2
SEED = 42
I_VALUES_USED = [2, 3, 4, 5]
LAYERS_TO_PROBE = [5, 7, 10, 13, 15, 20, 25, 30]

print("Loading features...")
data = np.load("data/probing_features.npz")
hidden_states = data["hidden_states"]
labels = data["labels"]
i_values = data["i_values"]
layer_indices = data["layer_indices"].tolist()

with open("data/probing_metadata.json") as f:
    meta = json.load(f)
phrases = np.array([r["phrase"] for r in meta["rows"]])
categories = np.array([r["category"] for r in meta["rows"]])
print(f"  N rows: {len(phrases)}, layers: {layer_indices}\n")

def fit_score(Xtr, ytr, Xte, yte, kind):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr.astype(np.float32))
    Xte_s = sc.transform(Xte.astype(np.float32))
    if kind == "log":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                            early_stopping=True, validation_fraction=0.15,
                            random_state=SEED, n_iter_no_change=10)
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)[:, 1]
        pred = (proba >= ytr.mean()).astype(int)
    return proba, pred

def score_metrics(yte, proba, pred):
    if len(set(yte.tolist())) < 2:
        return {"auc_roc": float("nan"), "auc_pr": float("nan"),
                "bal_acc": float("nan"), "pos_rate": float(yte.mean()), "n": len(yte)}
    return {
        "auc_roc": float(roc_auc_score(yte, proba)),
        "auc_pr": float(average_precision_score(yte, proba)),
        "bal_acc": float(balanced_accuracy_score(yte, pred)),
        "pos_rate": float(yte.mean()),
        "n": len(yte),
    }

def downsample_balanced(X, y, groups, seed):
    """Keep all positives, randomly pick equal number of negatives."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y, groups
    n_pos = len(pos_idx)
    if n_pos > len(neg_idx): n_pos = len(neg_idx)
    sampled_neg = rng.choice(neg_idx, size=n_pos, replace=False)
    keep = np.concatenate([pos_idx, sampled_neg])
    return X[keep], y[keep], groups[keep]

# Build pooled dataset: only i=2..5 rows
pool_mask = np.isin(i_values, I_VALUES_USED)
print(f"Rows in pooled dataset (i in {I_VALUES_USED}): {pool_mask.sum()}\n")

all_results = []  # one record per (layer, probe, mode), each contains overall + per-i + per-cat

for L in LAYERS_TO_PROBE:
    slot = layer_indices.index(L)
    print(f"\n===== Layer {L} =====")

    # Pull all (X, y, groups, i, cat) for this layer
    X = hidden_states[pool_mask, slot, :]
    y = labels[pool_mask, slot].astype(int)
    g = phrases[pool_mask]
    i_v = i_values[pool_mask]
    c_v = categories[pool_mask]

    # One shared train/test split, by phrase
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(X, y, g))
    Xtr_full, Xte = X[tr_idx], X[te_idx]
    ytr_full, yte = y[tr_idx], y[te_idx]
    gtr_full = g[tr_idx]
    i_te, c_te = i_v[te_idx], c_v[te_idx]

    # For each (mode, kind), train probe on the appropriate train set
    for mode in ["imbalanced", "balanced"]:
        if mode == "balanced":
            Xtr, ytr, _ = downsample_balanced(Xtr_full, ytr_full, gtr_full, SEED + L)
        else:
            Xtr, ytr = Xtr_full, ytr_full

        for kind in ["log", "mlp"]:
            proba, pred = fit_score(Xtr, ytr, Xte, yte, kind)

            # Overall metrics
            overall = score_metrics(yte, proba, pred)
            overall["n_train"] = len(ytr)
            overall["train_pos_rate"] = float(ytr.mean())

            # Per-i metrics
            per_i = {}
            for i_val in I_VALUES_USED:
                m_i = i_te == i_val
                if m_i.sum() < 5: continue
                per_i[int(i_val)] = score_metrics(yte[m_i], proba[m_i], pred[m_i])

            # Per-category metrics
            per_cat = {}
            for cat in ["building", "idiom", "movie"]:
                m_c = c_te == cat
                if m_c.sum() < 5: continue
                per_cat[cat] = score_metrics(yte[m_c], proba[m_c], pred[m_c])

            all_results.append({
                "layer": L, "kind": kind, "mode": mode,
                "overall": overall, "per_i": per_i, "per_cat": per_cat,
            })
            print(f"  {mode:<11} {kind:<4} | overall: AUC={overall['auc_roc']:.3f} "
                  f"AUC-PR={overall['auc_pr']:.3f} bal_acc={overall['bal_acc']:.3f} "
                  f"n_train={overall['n_train']} n_test={overall['n']}")

# --- save raw ---
with open("results/probing_pooled.json", "w") as f:
    json.dump(all_results, f, indent=2)


def fmt(v): return f"{v:.3f}" if not np.isnan(v) else "  -  "

def print_grid(title, mode, kind, metric):
    print("\n" + "=" * 100)
    print(f"{title}: {mode} data, {kind}, metric={metric}")
    print("=" * 100)
    print(f"{'layer':<6}{'overall':<12}{'n_test':<10}{'pos_rate':<10}"
          + "".join(f"i={i:<10}" for i in I_VALUES_USED) + f"{'n_i_sum':<10}")
    print("-" * 100)
    for L in LAYERS_TO_PROBE:
        r = next((x for x in all_results if x["layer"]==L and x["kind"]==kind and x["mode"]==mode), None)
        if r is None: continue
        ov = r["overall"]
        line = f"{L:<6}{fmt(ov[metric]):<12}{ov['n']:<10}{ov['pos_rate']:<10.2f}"
        n_sum = 0
        for i_val in I_VALUES_USED:
            if i_val in r["per_i"]:
                line += f"{fmt(r['per_i'][i_val][metric]):<12}"
                n_sum += r["per_i"][i_val]["n"]
            else:
                line += f"{'  -  ':<12}"
        line += f"{n_sum:<10}"
        print(line)

def print_cat_grid(title, mode, kind, metric):
    print("\n" + "=" * 100)
    print(f"{title}: {mode} data, {kind}, metric={metric}")
    print("=" * 100)
    print(f"{'layer':<6}{'overall':<12}"
          + "".join(f"{c:<24}" for c in ["building (n)", "idiom (n)", "movie (n)"]))
    print("-" * 100)
    for L in LAYERS_TO_PROBE:
        r = next((x for x in all_results if x["layer"]==L and x["kind"]==kind and x["mode"]==mode), None)
        if r is None: continue
        line = f"{L:<6}{fmt(r['overall'][metric]):<12}"
        for cat in ["building", "idiom", "movie"]:
            if cat in r["per_cat"]:
                m = r["per_cat"][cat]
                line += f"{fmt(m[metric]):<8}({m['n']:<5})       "
            else:
                line += f"{'  -  ':<24}"
        print(line)

# Print everything
for metric in ["auc_roc", "auc_pr", "bal_acc"]:
    for kind in ["log", "mlp"]:
        for mode in ["imbalanced", "balanced"]:
            print_grid(f"Overall + per-i", mode, kind, metric)

print("\n\n")
print("#" * 100)
print("CATEGORY ANALYSIS")
print("#" * 100)
for metric in ["auc_roc", "auc_pr"]:
    for kind in ["log", "mlp"]:
        for mode in ["imbalanced", "balanced"]:
            print_cat_grid(f"Category-wise", mode, kind, metric)

print("\nSaved to results/probing_pooled.json")
