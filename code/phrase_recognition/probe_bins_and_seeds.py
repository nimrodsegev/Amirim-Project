"""
Two things in one script:

PART 1: Per-phrase bin analysis (the Yuval-approved view).
  For probes at L13 and L25 (logistic + MLP, balanced + imbalanced training data):
    - Average probe prediction per phrase
    - Group by actual success rate bin
    - Show probe prediction per bin
  Done on: (a) all pooled data, (b) i=3 only.

PART 2: Multi-seed sanity check on the pooled per-layer probes.
  Run the same pooled probe at L13, L25, L15 (best layers from before) across 5 seeds.
  Report mean ± std for AUC-ROC, AUC-PR. Also break down per-category.
  This addresses: was L13's "buildings work here" finding a fluke?
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
I_VALUES_USED = [2, 3, 4, 5]

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

def fit_score(Xtr, ytr, Xte, kind, seed=42):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr.astype(np.float32))
    Xte_s = sc.transform(Xte.astype(np.float32))
    if kind == "log":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    else:
        clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                            early_stopping=True, validation_fraction=0.15,
                            random_state=seed, n_iter_no_change=10)
    clf.fit(Xtr_s, ytr)
    return clf.predict_proba(Xte_s)[:, 1]

def downsample_balanced(X, y, groups, seed):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y, groups
    n = min(len(pos_idx), len(neg_idx))
    keep = np.concatenate([pos_idx[:n] if len(pos_idx)==n else rng.choice(pos_idx, n, replace=False),
                           rng.choice(neg_idx, n, replace=False)])
    return X[keep], y[keep], groups[keep]

# ============================================================
# PART 1: BIN ANALYSIS
# ============================================================
print("\n" + "#" * 80)
print("PART 1: Bin analysis for L13 and L25")
print("#" * 80)

def bin_analysis(layer, kind, mode, filter_i_3=False, seed=42):
    """Train pooled probe, compute per-phrase predictions, bin by actual success rate."""
    slot = layer_indices.index(layer)
    pool_mask = np.isin(i_values, I_VALUES_USED)

    X = hidden_states[pool_mask, slot, :]
    y = labels[pool_mask, slot].astype(int)
    g = phrases[pool_mask]
    iv = i_values[pool_mask]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    tr_idx, te_idx = next(splitter.split(X, y, g))
    Xtr_full, Xte = X[tr_idx], X[te_idx]
    ytr_full, yte = y[tr_idx], y[te_idx]
    gtr_full = g[tr_idx]
    gte, ite = g[te_idx], iv[te_idx]

    if mode == "balanced":
        Xtr, ytr, _ = downsample_balanced(Xtr_full, ytr_full, gtr_full, seed + layer)
    else:
        Xtr, ytr = Xtr_full, ytr_full

    proba = fit_score(Xtr, ytr, Xte, kind, seed)

    # Optionally filter test set to i=3 only
    if filter_i_3:
        keep = ite == 3
        proba_use = proba[keep]
        yte_use = yte[keep]
        gte_use = gte[keep]
    else:
        proba_use = proba
        yte_use = yte
        gte_use = gte

    # Per-phrase: avg pred + actual rate
    per_phrase = defaultdict(lambda: {"preds": [], "labels": []})
    for ph, p, lab in zip(gte_use, proba_use, yte_use):
        per_phrase[ph]["preds"].append(p)
        per_phrase[ph]["labels"].append(lab)

    rows = []
    for ph, d in per_phrase.items():
        if len(d["preds"]) < 3: continue
        rows.append({
            "phrase": ph,
            "avg_pred": float(np.mean(d["preds"])),
            "actual_rate": float(np.mean(d["labels"])),
            "n": len(d["preds"]),
        })
    preds_arr = np.array([r["avg_pred"] for r in rows])
    actuals_arr = np.array([r["actual_rate"] for r in rows])
    corr = float(np.corrcoef(preds_arr, actuals_arr)[0, 1]) if len(rows) > 1 else float("nan")
    return rows, corr

def print_bins(title, rows, corr):
    print(f"\n--- {title} ---")
    print(f"  n_phrases in test = {len(rows)}, Pearson correlation = {corr:.3f}")
    bins = [(0.0, 0.0, "exactly 0"),
            (0.01, 0.25, "0.01 - 0.25"),
            (0.25, 0.50, "0.25 - 0.50"),
            (0.50, 0.75, "0.50 - 0.75"),
            (0.75, 0.99, "0.75 - 0.99"),
            (1.0, 1.0, "exactly 1")]
    print(f"  {'actual rate bin':<20}{'n phrases':<12}{'avg probe pred':<16}")
    print("  " + "-"*48)
    for lo, hi, label in bins:
        if lo == hi:
            members = [r for r in rows if r["actual_rate"] == lo]
        else:
            members = [r for r in rows if lo <= r["actual_rate"] <= hi
                       and r["actual_rate"] != 0.0 and r["actual_rate"] != 1.0]
        if not members: continue
        avg = np.mean([r["avg_pred"] for r in members])
        print(f"  {label:<20}{len(members):<12}{avg:<16.3f}")

# Run all combinations
combos = []
for layer in [13, 25]:
    for kind in ["log", "mlp"]:
        for mode in ["imbalanced", "balanced"]:
            for filt, filt_name in [(False, "all_i"), (True, "i=3_only")]:
                rows, corr = bin_analysis(layer, kind, mode, filter_i_3=filt)
                title = f"L{layer} {kind.upper()} {mode} {filt_name}"
                combos.append({"title": title, "layer": layer, "kind": kind,
                               "mode": mode, "scope": filt_name,
                               "corr": corr, "n_phrases": len(rows)})
                print_bins(title, rows, corr)

# Summary table of correlations
print("\n\n" + "=" * 90)
print("CORRELATION SUMMARY")
print("=" * 90)
print(f"{'layer':<8}{'kind':<6}{'mode':<14}{'scope':<14}{'correlation':<14}{'n_phrases':<12}")
print("-" * 90)
for c in combos:
    print(f"L{c['layer']:<7}{c['kind']:<6}{c['mode']:<14}{c['scope']:<14}"
          f"{c['corr']:<14.3f}{c['n_phrases']:<12}")

# ============================================================
# PART 2: MULTI-SEED POOLED PROBES
# ============================================================
print("\n\n" + "#" * 80)
print("PART 2: Multi-seed pooled probes (5 seeds)")
print("#" * 80)

LAYERS_FOR_SEEDS = [13, 15, 25]
SEEDS = [0, 5, 26, 42, 63]

def pooled_probe(layer, kind, mode, seed):
    """Returns (overall_auc_roc, overall_auc_pr, per_cat_auc_roc dict)."""
    slot = layer_indices.index(layer)
    pool_mask = np.isin(i_values, I_VALUES_USED)
    X = hidden_states[pool_mask, slot, :]
    y = labels[pool_mask, slot].astype(int)
    g = phrases[pool_mask]
    c_arr = categories[pool_mask]

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    tr_idx, te_idx = next(splitter.split(X, y, g))
    Xtr_full, Xte = X[tr_idx], X[te_idx]
    ytr_full, yte = y[tr_idx], y[te_idx]
    gtr_full = g[tr_idx]
    c_te = c_arr[te_idx]

    if mode == "balanced":
        Xtr, ytr, _ = downsample_balanced(Xtr_full, ytr_full, gtr_full, seed + layer)
    else:
        Xtr, ytr = Xtr_full, ytr_full

    proba = fit_score(Xtr, ytr, Xte, kind, seed)
    auc_roc = float(roc_auc_score(yte, proba)) if len(set(yte.tolist())) > 1 else float("nan")
    auc_pr = float(average_precision_score(yte, proba)) if len(set(yte.tolist())) > 1 else float("nan")

    per_cat = {}
    for cat in ["building", "idiom", "movie"]:
        m_c = c_te == cat
        if m_c.sum() < 5 or len(set(yte[m_c].tolist())) < 2:
            per_cat[cat] = (float("nan"), m_c.sum())
            continue
        per_cat[cat] = (float(roc_auc_score(yte[m_c], proba[m_c])), int(m_c.sum()))
    return auc_roc, auc_pr, per_cat

results_seeds = defaultdict(list)  # (layer, kind, mode) -> list of dicts

for layer in LAYERS_FOR_SEEDS:
    for kind in ["log", "mlp"]:
        for mode in ["balanced", "imbalanced"]:
            for seed in SEEDS:
                auc_roc, auc_pr, per_cat = pooled_probe(layer, kind, mode, seed)
                results_seeds[(layer, kind, mode)].append({
                    "seed": seed, "auc_roc": auc_roc, "auc_pr": auc_pr,
                    "building_auc": per_cat["building"][0],
                    "idiom_auc": per_cat["idiom"][0],
                    "movie_auc": per_cat["movie"][0],
                    "building_n": per_cat["building"][1],
                })

# Print summary
print(f"\n{'layer':<6}{'kind':<5}{'mode':<13}"
      f"{'AUC-ROC mean±std':<22}{'AUC-PR mean±std':<22}"
      f"{'building AUC':<18}{'idiom AUC':<14}{'movie AUC':<14}")
print("-" * 110)
for layer in LAYERS_FOR_SEEDS:
    for kind in ["log", "mlp"]:
        for mode in ["balanced", "imbalanced"]:
            rs = results_seeds[(layer, kind, mode)]
            roc_vals = np.array([r["auc_roc"] for r in rs])
            pr_vals = np.array([r["auc_pr"] for r in rs])
            b_vals = np.array([r["building_auc"] for r in rs if not np.isnan(r["building_auc"])])
            i_vals = np.array([r["idiom_auc"] for r in rs if not np.isnan(r["idiom_auc"])])
            m_vals = np.array([r["movie_auc"] for r in rs if not np.isnan(r["movie_auc"])])

            def fmt(arr):
                if len(arr) == 0: return "  -  "
                return f"{arr.mean():.3f}±{arr.std():.3f}"
            print(f"L{layer:<5}{kind:<5}{mode:<13}"
                  f"{fmt(roc_vals):<22}{fmt(pr_vals):<22}"
                  f"{fmt(b_vals):<18}{fmt(i_vals):<14}{fmt(m_vals):<14}")

# Save raw
with open("results/probing_bins_seeds.json", "w") as f:
    json.dump({
        "bin_combos": combos,
        "seed_results": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in results_seeds.items()},
    }, f, indent=2)
print("\nSaved to results/probing_bins_seeds.json")
