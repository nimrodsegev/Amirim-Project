"""
Extends the same patchscopes methodology to the very earliest layers:
L0 = raw token embedding (before any transformer block)
L1 = after transformer block 1
L2 = after transformer block 2
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score

MODEL_NAME = "allenai/OLMo-2-1124-7B"
PROBE_PROMPT = "Repeat this: X"
REPLACE_TOKEN = "X"
BUFFER = 3
LAYER_INDICES = [0, 1, 2]
MIN_I = 2
SEED = 42

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

def process_instance(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start = 1 + len(context_ids)
    n_phrase = len(phrase_ids)
    if n_phrase < MIN_I + 1:
        return []

    rows = []
    with torch.no_grad():
        mi = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        out = model(input_ids=mi, output_hidden_states=True)

        pi = tokenizer.encode(PROBE_PROMPT)
        pt = torch.tensor(pi).to(model.device)
        ri = tokenizer.encode(REPLACE_TOKEN, add_special_tokens=False)
        rm = pt == ri[0]
        si = tokenizer.encode(" " + REPLACE_TOKEN, add_special_tokens=False)
        if len(si) == 1:
            rm = rm | (pt == si[0])
        if not rm.any():
            return []
        rp = rm.nonzero(as_tuple=False).flatten()
        be = model.get_input_embeddings()(pt).unsqueeze(0)

        for tok_idx in range(phrase_start, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start
            tr = n_phrase - phrase_pos - 1
            if tr < MIN_I or tr > 5:
                continue
            rest = tokenizer.decode(phrase_ids[-tr:], skip_special_tokens=True)

            les = torch.stack([out.hidden_states[L][:, tok_idx].squeeze(0) for L in LAYER_INDICES], dim=0)
            bie = be.repeat(len(les), 1, 1)
            for b in range(len(les)):
                for p in rp:
                    bie[b, p] = les[b]

            try:
                g = model.generate(
                    inputs_embeds=bie, max_new_tokens=tr + BUFFER,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                layer_success = []
                for ln in range(len(les)):
                    txt = tokenizer.decode(g[ln], skip_special_tokens=True)
                    layer_success.append(rest.lower() in txt.lower())

                rows.append({
                    "hidden_states": les.cpu().to(torch.float32).numpy().astype(np.float16),
                    "labels": np.array(layer_success, dtype=bool),
                    "i": tr, "phrase_pos": phrase_pos,
                })
            except Exception:
                continue
    return rows


print("Loading context data...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

work = []
for phrase, matches in ctx_data.items():
    for j, m in enumerate(matches):
        work.append({"phrase": phrase, "category": m["category"], "context_before": m["context_before"], "instance_idx": j})
print(f"  {len(work)} instances\n")

all_states, all_labels, all_i, metadata = [], [], [], []
start = time.time()
for idx, item in enumerate(work):
    if idx % 200 == 0:
        elapsed = time.time() - start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (len(work) - idx) / rate if rate > 0 else 0
        print(f"  {idx}/{len(work)} | {elapsed:.0f}s | ETA {eta/60:.0f} min | {len(all_states)} rows")
    rows = process_instance(item["context_before"], item["phrase"])
    for r in rows:
        all_states.append(r["hidden_states"])
        all_labels.append(r["labels"])
        all_i.append(r["i"])
        metadata.append({"phrase": item["phrase"], "category": item["category"],
                          "instance_idx": item["instance_idx"], "phrase_pos": r["phrase_pos"], "i": r["i"]})

print("\nStacking arrays...")
hidden_states = np.stack(all_states, axis=0)
labels = np.stack(all_labels, axis=0)
i_values = np.array(all_i, dtype=np.int32)
phrases_arr = np.array([m["phrase"] for m in metadata])

Path("data").mkdir(exist_ok=True)
np.savez_compressed("data/probing_features_early.npz", hidden_states=hidden_states, labels=labels,
                     i_values=i_values, layer_indices=np.array(LAYER_INDICES, dtype=np.int32))
with open("data/probing_metadata_early.json", "w") as f:
    json.dump({"layer_indices": LAYER_INDICES, "rows": metadata}, f, indent=2)

print(f"\nTotal rows: {len(metadata)}")
print("\n" + "=" * 70)
print("PATCHSCOPES SUCCESS RATE AT EARLY LAYERS (pooled i=2-5)")
print("=" * 70)
any_success = labels.any(axis=1)
print(f"{'layer':<10}{'success%'}")
for k, L in enumerate(LAYER_INDICES):
    print(f"L{L:<9}{100*labels[:,k].mean():.1f}")
print(f"{'any of 3':<10}{100*any_success.mean():.1f}")

print("\n" + "=" * 70)
print("PROBE RESULTS AT EARLY LAYERS (same method as the other 8 layers)")
print("=" * 70)

def downsample_idx(y, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
print(f"{'layer':<8}{'n_test':<9}{'pos%_test':<12}{'acc_imbal':<12}{'acc_bal':<10}{'prec_imbal'}")
print("-" * 60)
for k, L in enumerate(LAYER_INDICES):
    X = hidden_states[:, k, :]
    y = labels[:, k].astype(int)
    tr_idx, te_idx = next(splitter.split(X, y, phrases_arr))
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xte, yte = X[te_idx], y[te_idx]
    bal_idx = downsample_idx(ytr, SEED + L)
    Xtr_b, ytr_b = Xtr[bal_idx], ytr[bal_idx]
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_b.astype(np.float32))
    Xte_s = sc.transform(Xte.astype(np.float32))
    clf = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=200,
                         early_stopping=True, validation_fraction=0.15,
                         random_state=SEED, n_iter_no_change=10)
    clf.fit(Xtr_s, ytr_b)
    pred = clf.predict(Xte_s)
    acc_i = accuracy_score(yte, pred) * 100
    acc_b = balanced_accuracy_score(yte, pred) * 100
    prec_i = precision_score(yte, pred, zero_division=0) * 100
    print(f"L{L:<7}{len(yte):<9}{100*yte.mean():<12.1f}{acc_i:<12.1f}{acc_b:<10.1f}{prec_i:.1f}")

print("\nDone. Copy this entire output back to Claude.")
