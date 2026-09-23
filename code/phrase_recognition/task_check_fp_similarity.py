"""
For all 412 false positives of the generation-label probe at L25, compute
a real text-similarity score between the target phrase and what the model
actually generated (trimmed to a comparable length), then bucket into
close / somewhat similar / unrelated. Uses difflib's SequenceMatcher,
a standard character-level similarity ratio, no GPU needed.
"""
import json
import difflib
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

fp_mask = (pred == 1) & (yte == 0)
fp_indices = np.where(fp_mask)[0]
meta_te = [meta_gen[i] for i in te_idx]
print(f"\nFalse positives: {len(fp_indices)}")

def similarity(target, generated):
    target = target.strip().lower()
    generated = generated.strip().lower()
    target_words = target.split()
    gen_words = generated.split()
    # trim generated to roughly the target's length, so trailing
    # continuation text doesn't unfairly drag the score down
    trimmed_gen = " ".join(gen_words[:len(target_words) + 1])
    return difflib.SequenceMatcher(None, target, trimmed_gen).ratio()

results = []
for local_i in fp_indices:
    m = meta_te[local_i]
    key = (m["phrase"], m["instance_idx"], m["i"])
    conf_info = conf_lookup.get(key)
    if conf_info is None:
        continue
    sim = similarity(conf_info["target"], conf_info["generated"])
    results.append({
        "phrase": m["phrase"], "target": conf_info["target"],
        "generated": conf_info["generated"], "similarity": sim,
        "model_confidence": conf_info["confidence"],
    })

sims = np.array([r["similarity"] for r in results])
print(f"\nMean similarity: {sims.mean():.3f}")
print(f"Median similarity: {np.median(sims):.3f}")

close = (sims >= 0.5).sum()
mid = ((sims >= 0.25) & (sims < 0.5)).sum()
far = (sims < 0.25).sum()
n = len(sims)
print(f"\nClose (similarity >= 0.50): {close} ({100*close/n:.1f}%)")
print(f"Somewhat similar (0.25-0.50): {mid} ({100*mid/n:.1f}%)")
print(f"Unrelated (< 0.25): {far} ({100*far/n:.1f}%)")

print("\n--- Most similar (closest matches) ---")
for r in sorted(results, key=lambda r: -r["similarity"])[:6]:
    print(f"\n{r['phrase']}  (similarity={r['similarity']:.3f}, model conf={r['model_confidence']:.3f})")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

print("\n--- Least similar (most unrelated) ---")
for r in sorted(results, key=lambda r: r["similarity"])[:6]:
    print(f"\n{r['phrase']}  (similarity={r['similarity']:.3f}, model conf={r['model_confidence']:.3f})")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

print("\nDone.")
