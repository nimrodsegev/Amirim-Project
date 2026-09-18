"""
Sanity check: train the same probes (logistic and MLP) on real hidden states
but with RANDOMLY SHUFFLED labels. If the probe still gets above-chance AUC,
something is leaking. Expectation: AUC should sit right at 0.5.

Done for L13 and L25, at i=2 through i=5.
"""
import json, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

TEST_FRAC = 0.2
SEED = 42

print("Loading features...")
data = np.load("data/probing_features.npz")
hidden_states = data["hidden_states"]
labels = data["labels"]
i_values = data["i_values"]
layer_indices = data["layer_indices"].tolist()

with open("data/probing_metadata.json") as f:
    meta = json.load(f)
phrases = np.array([r["phrase"] for r in meta["rows"]])

print(f"\n{'layer':<8}{'i':<4}{'pos %':<8}{'real log AUC':<14}{'shuf log AUC':<14}"
      f"{'real MLP AUC':<14}{'shuf MLP AUC':<14}")
print("-" * 80)

for L in [13, 25]:
    slot = layer_indices.index(L)
    for i_val in [2, 3, 4, 5]:
        mask = i_values == i_val
        if mask.sum() < 200: continue
        X = hidden_states[mask, slot, :].astype(np.float32)
        y_real = labels[mask, slot].astype(int)
        groups = phrases[mask]

        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=SEED)
        tr_idx, te_idx = next(splitter.split(X, y_real, groups))
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr_real, yte_real = y_real[tr_idx], y_real[te_idx]
        if yte_real.sum() < 5 or yte_real.sum() == len(yte_real): continue

        # Random labels: shuffle the training labels (keep test labels real for evaluation)
        rng = np.random.default_rng(SEED + L*100 + i_val)
        ytr_shuf = rng.permutation(ytr_real)

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)

        # Logistic — real labels
        log_real = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        log_real.fit(Xtr_s, ytr_real)
        log_real_auc = roc_auc_score(yte_real, log_real.predict_proba(Xte_s)[:, 1])

        # Logistic — shuffled labels
        log_shuf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        log_shuf.fit(Xtr_s, ytr_shuf)
        log_shuf_auc = roc_auc_score(yte_real, log_shuf.predict_proba(Xte_s)[:, 1])

        # MLP — real labels
        mlp_real = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                                  early_stopping=True, validation_fraction=0.15,
                                  random_state=SEED, n_iter_no_change=10)
        mlp_real.fit(Xtr_s, ytr_real)
        mlp_real_auc = roc_auc_score(yte_real, mlp_real.predict_proba(Xte_s)[:, 1])

        # MLP — shuffled labels
        mlp_shuf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                                  early_stopping=True, validation_fraction=0.15,
                                  random_state=SEED, n_iter_no_change=10)
        mlp_shuf.fit(Xtr_s, ytr_shuf)
        mlp_shuf_auc = roc_auc_score(yte_real, mlp_shuf.predict_proba(Xte_s)[:, 1])

        print(f"L{L:<7}{i_val:<4}{100*ytr_real.mean():<8.1f}"
              f"{log_real_auc:<14.3f}{log_shuf_auc:<14.3f}"
              f"{mlp_real_auc:<14.3f}{mlp_shuf_auc:<14.3f}")

print("\nInterpretation:")
print("  Expected: 'shuf' columns should sit at AUC ~0.50 (no signal when labels are random).")
print("  If shuf AUC is notably above 0.5, something is leaking and we need to investigate.")
print("  If shuf AUC ~ 0.5 across the board, the original results are clean.")
