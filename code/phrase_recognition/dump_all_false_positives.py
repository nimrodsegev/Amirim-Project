"""
Dumps ALL 412 false positives (target + generated text) in a clean, compact
format, ready to paste into an LLM prompt for semantic judgment. Same
probe/split/seed as before, so this is the identical set of 412 cases.
"""
import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
LAYERS = [5, 7, 10, 13, 15, 20, 25, 30]
PROBE_LAYER = 25

data_gen = np.load("data/generation_probing_features.npz")
hidden_gen = data_gen["hidden_states"]
success_gen = data_gen["success"]
with open("data/generation_probing_metadata.json") as f:
    meta_gen = json.load(f)["rows"]
phrases_gen = np.array([r["phrase"] for r in meta_gen])

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

fp_mask = (pred == 1) & (yte == 0)
fp_indices = np.where(fp_mask)[0]
meta_te = [meta_gen[i] for i in te_idx]

with open("all_false_positives.txt", "w") as out:
    n = 0
    for local_i in fp_indices:
        m = meta_te[local_i]
        key = (m["phrase"], m["instance_idx"], m["i"])
        conf_info = conf_lookup.get(key)
        if conf_info is None:
            continue
        n += 1
        target = conf_info["target"].strip()
        generated = conf_info["generated"].strip()
        out.write(f"{n}. PHRASE: {m['phrase']} | TARGET: {target} | GENERATED: {generated}\n")

print(f"Wrote {n} false positives to all_false_positives.txt")
print("Run: cat all_false_positives.txt")
