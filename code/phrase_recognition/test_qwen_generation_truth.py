"""
Same naked + with-context generation-truth methodology used for OLMo-2-7B,
run on Qwen2.5-14B instead, so the numbers land directly next to what we
already have.
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-14B"
BUFFER = 3

print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

def analyze_naked(phrase_text):
    phrase_ids = tokenizer.encode(phrase_text, add_special_tokens=False)
    input_ids = ([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []) + phrase_ids
    phrase_start_idx = len(input_ids) - len(phrase_ids)
    n_phrase_tokens = len(phrase_ids)
    if n_phrase_tokens < 3:
        return []
    rows = []
    with torch.no_grad():
        for tok_idx in range(phrase_start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start_idx
            tokens_remaining = n_phrase_tokens - phrase_pos - 1
            if tokens_remaining < 1 or tokens_remaining > 8:
                continue
            rest = tokenizer.decode(phrase_ids[-tokens_remaining:], skip_special_tokens=True)
            prefix_ids = input_ids[:tok_idx + 1]
            prefix = torch.tensor(prefix_ids).unsqueeze(0).to(model.device)
            g = model.generate(
                input_ids=prefix, max_new_tokens=tokens_remaining + BUFFER,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(g[0][len(prefix_ids):], skip_special_tokens=True)
            success = rest.lower() in gen_text.lower()
            rows.append({"i": tokens_remaining, "success": success})
    return rows

def analyze_with_context(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = ([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []) + context_ids + phrase_ids
    phrase_start_idx = len(input_ids) - len(phrase_ids)
    n_phrase_tokens = len(phrase_ids)
    if n_phrase_tokens < 3:
        return []
    rows = []
    with torch.no_grad():
        for tok_idx in range(phrase_start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start_idx
            tokens_remaining = n_phrase_tokens - phrase_pos - 1
            if tokens_remaining < 2 or tokens_remaining > 5:
                continue
            rest = tokenizer.decode(phrase_ids[-tokens_remaining:], skip_special_tokens=True)
            prefix_ids = input_ids[:tok_idx + 1]
            prefix = torch.tensor(prefix_ids).unsqueeze(0).to(model.device)
            g = model.generate(
                input_ids=prefix, max_new_tokens=tokens_remaining + BUFFER,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.decode(g[0][len(prefix_ids):], skip_special_tokens=True)
            success = rest.lower() in gen_text.lower()
            rows.append({"i": tokens_remaining, "success": success})
    return rows

print("Loading phrase list...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)
phrases = list(ctx_data.keys())

print(f"Running NAKED generation truth on {len(phrases)} phrases...")
naked_by_i = {}
for idx, phrase in enumerate(phrases):
    if idx % 100 == 0:
        print(f"  {idx}/{len(phrases)}")
    for r in analyze_naked(phrase):
        naked_by_i.setdefault(r["i"], []).append(r["success"])

print("\nNAKED results:")
print(f"{'i':<5}{'n':<8}{'success%'}")
for i in sorted(naked_by_i):
    vals = naked_by_i[i]
    print(f"{i:<5}{len(vals):<8}{100*sum(vals)/len(vals):.1f}")

print(f"\nRunning WITH-CONTEXT generation truth...")
ctx_by_i = {}
work = [(p, m["context_before"]) for p, matches in ctx_data.items() for m in matches]
print(f"{len(work)} (phrase, context) instances")
for idx, (phrase, context_before) in enumerate(work):
    if idx % 500 == 0:
        print(f"  {idx}/{len(work)}")
    for r in analyze_with_context(context_before, phrase):
        ctx_by_i.setdefault(r["i"], []).append(r["success"])

print("\nWITH-CONTEXT results:")
print(f"{'i':<5}{'n':<8}{'success%'}")
for i in sorted(ctx_by_i):
    vals = ctx_by_i[i]
    print(f"{i:<5}{len(vals):<8}{100*sum(vals)/len(vals):.1f}")

print("\nDone.")
