"""
Task 4: deep dive into false positives of the generation-label probe.
"""
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
LAYERS = [5, 7, 10, 13, 15, 20, 25, 30]
PROBE_LAYER = 25

print("Loading generation probing features...")
data_gen = np.load("data/generation_probing_features.npz")
hidden_gen = data_gen["hidden_states"]
success_gen = data_gen["success"]
with open("data/generation_probing_metadata.json") as f:
    meta_gen = json.load(f)["rows"]
phrases_gen = np.array([r["phrase"] for r in meta_gen])

print("Loading confidence + generated text data...")
with open("data/generation_confidence.json") as f:
    conf_rows = json.load(f)
conf_lookup = {}
for r in conf_rows:
    key = (r["phrase"], r["instance_idx"], r["i"])
    conf_lookup[key] = r

def downsample_idx(y, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])

slot = LAYERS.index(PROBE_LAYER)
X = hidden_gen[:, slot, :]
y = success_gen.astype(int)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(splitter.split(X, y, phrases_gen))
Xtr, ytr = X[tr_idx], y[tr_idx]
Xte, yte = X[te_idx], y[te_idx]

bal_idx = downsample_idx(ytr, SEED + PROBE_LAYER)
Xtr_b, ytr_b = Xtr[bal_idx], ytr[bal_idx]
sc = StandardScaler()
Xtr_s = sc.fit_transform(Xtr_b.astype(np.float32))
Xte_s = sc.transform(Xte.astype(np.float32))

clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                     early_stopping=True, validation_fraction=0.15,
                     random_state=SEED, n_iter_no_change=10)
clf.fit(Xtr_s, ytr_b)
pred = clf.predict(Xte_s)
proba = clf.predict_proba(Xte_s)[:, 1]

fp_mask = (pred == 1) & (yte == 0)
fp_indices_in_test = np.where(fp_mask)[0]
print(f"\nFalse positives at L{PROBE_LAYER}: {len(fp_indices_in_test)} out of {len(yte)} test rows "
      f"({100*len(fp_indices_in_test)/len(yte):.1f}% of test set)")

meta_te = [meta_gen[i] for i in te_idx]

fp_rows = []
for local_i in fp_indices_in_test:
    m = meta_te[local_i]
    key = (m["phrase"], m["instance_idx"], m["i"])
    conf_info = conf_lookup.get(key)
    if conf_info is None:
        continue
    fp_rows.append({
        "phrase": m["phrase"], "i": m["i"], "probe_confidence": float(proba[local_i]),
        "model_confidence": conf_info["confidence"], "target": conf_info["target"],
        "generated": conf_info["generated"], "context_tail": conf_info["context_tail"],
    })

print(f"Joined with confidence data: {len(fp_rows)} rows\n")

model_confs = np.array([r["model_confidence"] for r in fp_rows if r["model_confidence"] is not None])
print(f"Among false positives, model's own confidence in its (wrong) output:")
print(f"  mean: {model_confs.mean():.4f}, median: {np.median(model_confs):.4f}")
print(f"  fraction with model confidence > 0.7: {100*np.mean(model_confs > 0.7):.1f}%")
print(f"  fraction with model confidence < 0.3: {100*np.mean(model_confs < 0.3):.1f}%")

fp_sorted_by_model_conf = sorted(fp_rows, key=lambda r: -(r["model_confidence"] or 0))
print("\n--- False positives where the MODEL was also highly confident (double-confident, still wrong) ---")
for r in fp_sorted_by_model_conf[:8]:
    print(f"\nPhrase: {r['phrase']} (i={r['i']})")
    print(f"  probe confidence: {r['probe_confidence']:.3f}, model's own confidence: {r['model_confidence']:.3f}")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

print("\n--- False positives where the MODEL itself was unsure (probe was fooled, model knew it might be wrong) ---")
fp_sorted_low_model_conf = sorted(fp_rows, key=lambda r: (r["model_confidence"] or 1))
for r in fp_sorted_low_model_conf[:8]:
    print(f"\nPhrase: {r['phrase']} (i={r['i']})")
    print(f"  probe confidence: {r['probe_confidence']:.3f}, model's own confidence: {r['model_confidence']:.3f}")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

print("\nDone.")
