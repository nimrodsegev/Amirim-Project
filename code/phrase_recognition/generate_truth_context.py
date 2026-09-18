"""
New source of truth (generation, not patchscopes), now WITH natural context.
For each phrase instance (context_before + phrase), at each position i inside
the phrase, take the real prefix (BOS + context + phrase tokens up to that
position) and greedily generate i+BUFFER tokens. Success = the true remaining
i tokens appear in the generated continuation.

Same success criterion and position logic as the naked generation run and as
the with-context patchscopes run, so all three are directly comparable.
"""
import json, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
BUFFER = 3

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

def analyze_with_context(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start_idx = 1 + len(context_ids)
    n_phrase_tokens = len(phrase_ids)
    if n_phrase_tokens < 3:
        return {}
    results = {i: {"success": False} for i in range(1, n_phrase_tokens)}

    with torch.no_grad():
        for tok_idx in range(phrase_start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start_idx
            tokens_remaining = n_phrase_tokens - phrase_pos - 1
            if tokens_remaining < 1:
                continue
            rest = tokenizer.decode(phrase_ids[-tokens_remaining:], skip_special_tokens=True)

            prefix_ids = input_ids[:tok_idx + 1]
            prefix = torch.tensor(prefix_ids).unsqueeze(0).to(model.device)
            g = model.generate(
                input_ids=prefix,
                max_new_tokens=tokens_remaining + BUFFER,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = g[0][len(prefix_ids):]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if rest.lower() in gen_text.lower():
                results[tokens_remaining]["success"] = True
    return results

print("Loading context data (v2, up to 10 contexts per phrase)...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

# Use ALL contexts per phrase (not just the first), same as the full patchscopes with-context run
work = []
for phrase, matches in ctx_data.items():
    for m in matches:
        work.append({"phrase": phrase, "category": m["category"], "context_before": m["context_before"]})
print(f"  {len(work)} (phrase, context) instances to process\n")

all_entries = []
for idx, item in enumerate(work):
    if idx % 200 == 0:
        print(f"  {idx}/{len(work)}")
    r = analyze_with_context(item["context_before"], item["phrase"])
    item["per_i"] = r
    all_entries.append(item)

agg = defaultdict(lambda: {"tested": 0, "success": 0})
for e in all_entries:
    for i, d in e["per_i"].items():
        agg[i]["tested"] += 1
        if d["success"]:
            agg[i]["success"] += 1
agg = dict(sorted(agg.items()))

Path("results").mkdir(exist_ok=True)
with open("results/generation_truth_context.json", "w") as f:
    json.dump({
        "overall": agg,
        "per_phrase": [{"phrase": e["phrase"], "category": e["category"],
                        "per_i": {str(k): v for k, v in e["per_i"].items()}}
                       for e in all_entries],
    }, f, indent=2)

print("\n=== GENERATION-TRUTH WITH CONTEXT, OVERALL ===")
print(f"{'i':<5}{'tested':<10}{'success':<10}{'rate':<10}")
print("-" * 35)
for i in sorted(agg.keys()):
    d = agg[i]
    rate = 100 * d["success"] / d["tested"] if d["tested"] else 0
    print(f"{i:<5}{d['tested']:<10}{d['success']:<10}{rate:<10.1f}")

# Compare to naked generation
try:
    with open("results/generation_truth_naked.json") as f:
        naked = json.load(f)["overall"]
    naked = {int(k): v for k, v in naked.items()} if isinstance(list(naked.keys())[0], str) else naked
    print("\n=== NAKED vs WITH-CONTEXT (both generation-truth) ===")
    print(f"{'i':<5}{'naked rate':<12}{'ctx rate':<12}{'delta':<10}")
    print("-" * 40)
    for i in sorted(set(list(agg.keys()) + list(naked.keys()))):
        a = naked.get(i, {"tested": 0, "success": 0})
        b = agg.get(i, {"tested": 0, "success": 0})
        ar = 100*a["success"]/a["tested"] if a["tested"] else 0
        br = 100*b["success"]/b["tested"] if b["tested"] else 0
        print(f"{i:<5}{ar:<12.1f}{br:<12.1f}{br-ar:+<10.1f}")
except Exception as e:
    print(f"\nCould not load naked comparison: {e}")

print("\nDone.")
