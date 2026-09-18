"""
New source of truth: natural greedy generation (no patchscopes).
For each phrase, at each position i (tokens remaining), take the real prefix
(BOS + phrase tokens up to that position) and greedily generate i+BUFFER tokens.
Success = the true remaining i tokens appear in the generated continuation.

Mirrors the patchscopes pipeline exactly except for the core check:
  - same BOS handling
  - same position logic: tokens_remaining = n_tokens - phrase_pos - 1
  - same i range (1 .. n_tokens-1)
  - same success criterion: case-insensitive substring match, buffer=3
"""
import csv, json, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
BUFFER = 3

DATASETS = [
    ("data/buildings_clean.csv", "Famous Buildings", "building"),
    ("data/idioms.csv",           "Idioms",           "idiom"),
    ("data/imdb_top250_multiword.csv", "IMDB Movies", "movie"),
]

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
HAS_BOS = tokenizer.bos_token_id is not None
print(f"Loaded. has_bos={HAS_BOS}\n")

def analyze_phrase(phrase):
    input_ids = tokenizer.encode(phrase)
    if HAS_BOS and input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    if HAS_BOS:
        phrase_tokens = input_ids[1:]
        start_idx = 1
    else:
        phrase_tokens = input_ids
        start_idx = 0
    n_tokens = len(phrase_tokens)
    if n_tokens < 3:
        return {}
    results = {i: {"success": False} for i in range(1, n_tokens)}

    with torch.no_grad():
        # For each position inside the phrase, greedily generate forward from the real prefix
        for tok_idx in range(start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - start_idx
            tokens_remaining = n_tokens - phrase_pos - 1
            if tokens_remaining < 1:
                continue
            rest = tokenizer.decode(phrase_tokens[-tokens_remaining:], skip_special_tokens=True)

            # Real prefix: everything up to and including the current token
            prefix_ids = input_ids[:tok_idx + 1]
            prefix = torch.tensor(prefix_ids).unsqueeze(0).to(model.device)
            g = model.generate(
                input_ids=prefix,
                max_new_tokens=tokens_remaining + BUFFER,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Only decode the newly generated part
            gen_ids = g[0][len(prefix_ids):]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if rest.lower() in gen_text.lower():
                results[tokens_remaining]["success"] = True
    return results

def n_tok(p):
    ids = tokenizer.encode(p)
    if HAS_BOS and ids[0] != tokenizer.bos_token_id:
        ids = [tokenizer.bos_token_id] + ids
    return len(ids) - (1 if HAS_BOS else 0)

all_entries = []
for path, name, cat in DATASETS:
    print(f"\n=== {name} ===")
    phrases = []
    with open(path) as f:
        reader = csv.DictReader(f)
        col = reader.fieldnames[0]
        for row in reader:
            phrases.append(row[col])
    print(f"  {len(phrases)} phrases")
    for idx, p in enumerate(phrases):
        if idx % 50 == 0: print(f"    {idx}/{len(phrases)}")
        r = analyze_phrase(p)
        all_entries.append({"phrase": p, "category": cat, "n_tokens": n_tok(p), "per_i": r})

# Aggregate
agg = defaultdict(lambda: {"tested": 0, "success": 0})
for e in all_entries:
    for i, d in e["per_i"].items():
        agg[i]["tested"] += 1
        if d["success"]:
            agg[i]["success"] += 1
agg = dict(sorted(agg.items()))

Path("results").mkdir(exist_ok=True)
with open("results/generation_truth_naked.json", "w") as f:
    json.dump({"overall": agg, "model": MODEL_NAME, "buffer": BUFFER,
               "per_phrase": [{"phrase": e["phrase"], "category": e["category"],
                               "n_tokens": e["n_tokens"],
                               "per_i": {str(k): v for k, v in e["per_i"].items()}}
                              for e in all_entries]}, f, indent=2)

# Print, and compare to patchscopes naked results if available
print("\n=== GENERATION-TRUTH NAKED, OVERALL ===")
print(f"{'i':<5}{'tested':<10}{'success':<10}{'rate':<10}")
print("-" * 35)
for i in sorted(agg.keys()):
    d = agg[i]
    rate = 100 * d["success"] / d["tested"] if d["tested"] else 0
    print(f"{i:<5}{d['tested']:<10}{d['success']:<10}{rate:<10.1f}")

try:
    with open("results/lookahead_analysis_fixed.json") as f:
        ps = json.load(f)["overall"]
    ps = {int(k): v for k, v in ps.items()} if isinstance(list(ps.keys())[0], str) else ps
    print("\n=== PATCHSCOPES vs GENERATION (both naked) ===")
    print(f"{'i':<5}{'PS rate':<10}{'GEN rate':<10}{'delta':<10}")
    print("-" * 35)
    for i in sorted(set(list(agg.keys()) + list(ps.keys()))):
        a = ps.get(i, {"tested": 0, "success": 0})
        b = agg.get(i, {"tested": 0, "success": 0})
        ar = 100*a["success"]/a["tested"] if a["tested"] else 0
        br = 100*b["success"]/b["tested"] if b["tested"] else 0
        print(f"{i:<5}{ar:<10.1f}{br:<10.1f}{br-ar:+<10.1f}")
except Exception as e:
    print(f"\nCould not load patchscopes comparison: {e}")

print("\nDone.")
