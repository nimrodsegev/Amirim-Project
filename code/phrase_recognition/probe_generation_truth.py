"""
Probe trained on hidden states, predicting REAL GENERATION success
(not patchscopes). Same setup as the original probing script for
direct comparison: balanced MLP, pooled i=2-5, GroupShuffleSplit by phrase.
"""
import warnings, numpy as np, json
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix
warnings.filterwarnings("ignore")

SEED = 42

data = np.load("data/generation_probing_features.npz")
hidden_states = data["hidden_states"]   # (24740, 8, 4096)
success       = data["success"]         # (24740,)
i_values      = data["i_values"]
layer_indices = data["layer_indices"].tolist()

with open("data/generation_probing_metadata.json") as f:
    meta = json.load(f)
phrases = np.array([r["phrase"] for r in meta["rows"]])

def downsample(X, y, g, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    n = min(len(pos), len(neg))
    keep = np.concatenate([pos, rng.choice(neg, n, replace=False)])
    return X[keep], y[keep], g[keep]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(splitter.split(hidden_states, success, phrases))

print(f"{'layer':<8}{'n':<7}{'pos%':<7}{'acc':<8}{'prec':<7}{'rec':<7}{'AUC':<8}{'AUC-PR':<8}{'bal_acc'}")
print("-"*70)

for slot, L in enumerate(layer_indices):
    X_feat = hidden_states[:, slot, :]

    Xtr = X_feat[tr_idx]; ytr = success[tr_idx]; gtr = phrases[tr_idx]
    Xte = X_feat[te_idx]; yte = success[te_idx]

    Xtr_b, ytr_b, _ = downsample(Xtr, ytr, gtr, SEED+L)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_b.astype(np.float32))
    Xte_s = sc.transform(Xte.astype(np.float32))

    clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                        early_stopping=True, validation_fraction=0.15,
                        random_state=SEED, n_iter_no_change=10)
    clf.fit(Xtr_s, ytr_b)
    proba = clf.predict_proba(Xte_s)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    acc  = accuracy_score(yte, pred)*100
    prec = precision_score(yte, pred, zero_division=0)*100
    rec  = recall_score(yte, pred, zero_division=0)*100
    auc  = roc_auc_score(yte, proba)
    aucpr= average_precision_score(yte, proba)
    bacc = balanced_accuracy_score(yte, pred)

    print(f"L{L:<7}{len(yte):<7}{100*yte.mean():<7.1f}{acc:<8.1f}{prec:<7.1f}{rec:<7.1f}{auc:<8.3f}{aucpr:<8.3f}{bacc:.3f}")

print("\n=== Per i, using L25 as example ===")
slot25 = layer_indices.index(25)
X_feat = hidden_states[:, slot25, :]
Xtr = X_feat[tr_idx]; ytr = success[tr_idx]; gtr = phrases[tr_idx]
Xte = X_feat[te_idx]; yte = success[te_idx]; ite = i_values[te_idx]
Xtr_b, ytr_b, _ = downsample(Xtr, ytr, gtr, SEED+25)
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr_b.astype(np.float32))
Xte_s = sc.transform(Xte.astype(np.float32))
clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                    early_stopping=True, validation_fraction=0.15,
                    random_state=SEED, n_iter_no_change=10)
clf.fit(Xtr_s, ytr_b)
proba = clf.predict_proba(Xte_s)[:, 1]
pred  = (proba >= 0.5).astype(int)
for iv in [2,3,4,5]:
    m = ite == iv
    if m.sum() < 5: continue
    acc_i = accuracy_score(yte[m], pred[m])*100
    auc_i = roc_auc_score(yte[m], proba[m])
    apr_i = average_precision_score(yte[m], proba[m])
    print(f"  i={iv}: n={m.sum()}, pos%={100*yte[m].mean():.1f}, acc={acc_i:.1f}, AUC={auc_i:.3f}, AUC-PR={apr_i:.3f}")
