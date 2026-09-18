"""
Task 2: model's confidence in its own generated output, and whether it's
lower when wrong.
Task 4: false-positive deep dive for later, cross-referenced with the
model's own confidence and what it actually generated.
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
BUFFER = 3

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

def analyze_with_confidence(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start_idx = 1 + len(context_ids)
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
                input_ids=prefix,
                max_new_tokens=tokens_remaining + BUFFER,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            gen_ids = g.sequences[0][len(prefix_ids):]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            success = rest.lower() in gen_text.lower()

            n_conf_steps = min(tokens_remaining, len(g.scores))
            log_probs = []
            for step in range(n_conf_steps):
                logits = g.scores[step][0]
                probs = torch.softmax(logits, dim=-1)
                chosen_token_id = gen_ids[step] if step < len(gen_ids) else None
                if chosen_token_id is None:
                    continue
                p = probs[chosen_token_id].item()
                log_probs.append(np.log(max(p, 1e-12)))
            confidence = float(np.exp(np.mean(log_probs))) if log_probs else None

            rows.append({
                "i": tokens_remaining, "target": rest, "generated": gen_text,
                "success": success, "confidence": confidence,
                "context_tail": context_before[-140:],
            })
    return rows


print("Loading context data...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

work = []
for phrase, matches in ctx_data.items():
    for j, m in enumerate(matches):
        work.append({"phrase": phrase, "category": m["category"], "context_before": m["context_before"], "instance_idx": j})
print(f"  {len(work)} instances\n")

all_rows = []
start = time.time()
for idx, item in enumerate(work):
    if idx % 200 == 0:
        elapsed = time.time() - start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (len(work) - idx) / rate if rate > 0 else 0
        print(f"  {idx}/{len(work)} | {elapsed:.0f}s | ETA {eta/60:.0f} min | {len(all_rows)} rows")
    rows = analyze_with_confidence(item["context_before"], item["phrase"])
    for r in rows:
        r["phrase"] = item["phrase"]
        r["category"] = item["category"]
        r["instance_idx"] = item["instance_idx"]
        all_rows.append(r)

print(f"\nTotal rows: {len(all_rows)}")

with open("data/generation_confidence.json", "w") as f:
    json.dump(all_rows, f, indent=2)
print("Saved to data/generation_confidence.json")

confidences = np.array([r["confidence"] for r in all_rows if r["confidence"] is not None])
successes = np.array([r["success"] for r in all_rows if r["confidence"] is not None])

print("\n" + "=" * 70)
print("TASK 2: CONFIDENCE vs SUCCESS")
print("=" * 70)
print(f"Mean confidence when SUCCESS: {confidences[successes].mean():.4f}")
print(f"Mean confidence when FAILURE: {confidences[~successes].mean():.4f}")
print(f"Median confidence when SUCCESS: {np.median(confidences[successes]):.4f}")
print(f"Median confidence when FAILURE: {np.median(confidences[~successes]):.4f}")

low_conf_success = sorted([r for r in all_rows if r["success"] and r["confidence"] is not None],
                           key=lambda r: r["confidence"])[:5]
low_conf_fail = sorted([r for r in all_rows if not r["success"] and r["confidence"] is not None],
                        key=lambda r: r["confidence"])[:8]

print("\n--- Low-confidence SUCCESSES (model was right despite being unsure) ---")
for r in low_conf_success:
    print(f"\nPhrase: {r['phrase']} (i={r['i']}, confidence={r['confidence']:.4f})")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

print("\n--- Low-confidence FAILURES (model was unsure AND wrong) ---")
for r in low_conf_fail:
    print(f"\nPhrase: {r['phrase']} (i={r['i']}, confidence={r['confidence']:.4f})")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

high_conf_fail = sorted([r for r in all_rows if not r["success"] and r["confidence"] is not None],
                         key=lambda r: -r["confidence"])[:8]
print("\n--- HIGH-confidence FAILURES (model was SURE but wrong) ---")
for r in high_conf_fail:
    print(f"\nPhrase: {r['phrase']} (i={r['i']}, confidence={r['confidence']:.4f})")
    print(f"  target: {r['target']!r}")
    print(f"  generated: {r['generated']!r}")

print("\nDone. Copy this entire output back to Claude.")
