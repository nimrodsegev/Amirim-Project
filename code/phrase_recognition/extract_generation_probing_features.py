"""
Combined extraction for probing against the NEW ground truth (real generation).
For each (phrase, context instance), do ONE forward pass to get hidden states
at all positions/layers (cheap), then for each position with i in [2,3,4,5],
run real greedy generation from that exact prefix to get the true success label.

Saves the same layer set as before [5,7,10,13,15,20,25,30] so all our existing
probe code can be reused directly, just swapping in this new labels array.
"""
import json, torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
BUFFER = 3
LAYERS = [5, 7, 10, 13, 15, 20, 25, 30]
TARGET_I = [2, 3, 4, 5]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

print("Loading context data...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

all_hidden = []   # list of (8,) arrays -> one per row
all_success = []  # list of 0/1 -> one per row (generation success)
all_meta = []     # list of dicts

work = []
for phrase, matches in ctx_data.items():
    for inst_idx, m in enumerate(matches):
        work.append((phrase, m["category"], inst_idx, m["context_before"]))
print(f"{len(work)} (phrase, context) instances to process\n")

for idx, (phrase, category, inst_idx, context_before) in enumerate(work):
    if idx % 100 == 0:
        print(f"  {idx}/{len(work)}")

    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start_idx = 1 + len(context_ids)
    n_phrase_tokens = len(phrase_ids)
    if n_phrase_tokens < 3:
        continue

    # ONE forward pass gets hidden states at every position for every layer
    with torch.no_grad():
        mi = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        out = model(input_ids=mi, output_hidden_states=True)

        for tok_idx in range(phrase_start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start_idx
            tokens_remaining = n_phrase_tokens - phrase_pos - 1
            if tokens_remaining not in TARGET_I:
                continue
            rest = tokenizer.decode(phrase_ids[-tokens_remaining:], skip_special_tokens=True)

            # hidden states at the 8 target layers, this position
            layer_vecs = []
            for L in LAYERS:
                vec = out.hidden_states[L][:, tok_idx].squeeze(0).float().cpu().numpy()
                layer_vecs.append(vec)
            layer_vecs = np.stack(layer_vecs, axis=0)  # (8, 4096)

            # REAL generation success from this exact prefix
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
            success = int(rest.lower() in gen_text.lower())

            all_hidden.append(layer_vecs)
            all_success.append(success)
            all_meta.append({
                "phrase": phrase, "category": category, "instance_idx": inst_idx,
                "i": tokens_remaining
            })

print(f"\nTotal rows collected: {len(all_hidden)}")

hidden_arr = np.stack(all_hidden, axis=0)  # (N, 8, 4096)
success_arr = np.array(all_success, dtype=np.int64)  # (N,)
i_values_arr = np.array([m["i"] for m in all_meta], dtype=np.int64)

Path("data").mkdir(exist_ok=True)
np.savez("data/generation_probing_features.npz",
         hidden_states=hidden_arr, success=success_arr,
         i_values=i_values_arr, layer_indices=np.array(LAYERS))

with open("data/generation_probing_metadata.json", "w") as f:
    json.dump({"rows": all_meta}, f, indent=2)

print("Saved to data/generation_probing_features.npz and data/generation_probing_metadata.json")

# Quick summary
print(f"\nOverall success rate: {success_arr.mean()*100:.1f}%")
for i in TARGET_I:
    m = i_values_arr == i
    if m.sum() > 0:
        print(f"  i={i}: {m.sum()} rows, {100*success_arr[m].mean():.1f}% success")
