"""
Extract hidden states and per-layer success labels for probing classifier training.
For each (phrase, context-instance, position with i>=2), saves:
  - hidden states at 8 selected layers
  - success label at each of those 8 layers
  - metadata (phrase, category, i, instance_idx, position)

Output:
  data/probing_features.npz   - hidden_states, labels, i_values, layer_indices
  data/probing_metadata.json  - per-row metadata (phrase, category, instance_idx)
"""
import json, time
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
PROBE_PROMPT = "Repeat this: X"
REPLACE_TOKEN = "X"
BUFFER = 3
LAYER_INDICES = [5, 7, 10, 13, 15, 20, 25, 30]
MIN_I = 2  # only positions where i >= 2

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
    """
    Run forward + patchscopes once. Returns list of dicts, one per qualifying
    position. Each dict has hidden_states (8 layers x 4096), labels (8-bool),
    and metadata (i, position).
    """
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start = 1 + len(context_ids)
    n_phrase = len(phrase_ids)
    if n_phrase < MIN_I + 1:
        return []

    def hook_fn(m, inp):
        m.captured_input = inp[0].clone()
        return inp
    hook = model.model.norm.register_forward_pre_hook(hook_fn)

    rows = []
    with torch.no_grad():
        mi = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        out = model(input_ids=mi, output_hidden_states=True)
        last_pn = model.model.norm.captured_input.clone()
        hook.remove()

        pi = tokenizer.encode(PROBE_PROMPT)
        pt = torch.tensor(pi).to(model.device)
        ri = tokenizer.encode(REPLACE_TOKEN, add_special_tokens=False)
        rm = pt == ri[0]
        si = tokenizer.encode(" " + REPLACE_TOKEN, add_special_tokens=False)
        if len(si) == 1:
            rm = rm | (pt == si[0])
        if not rm.any(): return []
        rp = rm.nonzero(as_tuple=False).flatten()
        be = model.get_input_embeddings()(pt).unsqueeze(0)

        for tok_idx in range(phrase_start, len(input_ids) - 1):
            phrase_pos = tok_idx - phrase_start
            tr = n_phrase - phrase_pos - 1
            if tr < MIN_I: continue
            rest = tokenizer.decode(phrase_ids[-tr:], skip_special_tokens=True)

            # Collect hidden states at all 32 layers (we need full set for patchscopes).
            # The pre-norm value goes in the slot right after the last transformer layer.
            les = []
            for li in range(1, len(out.hidden_states)):
                le = out.hidden_states[li][:, tok_idx] if li < len(out.hidden_states)-1 else last_pn[:, tok_idx]
                les.append(le.squeeze(0))
            les = torch.stack(les, dim=0)  # [32, 4096]
            bie = be.repeat(len(les), 1, 1)
            for b in range(len(les)):
                for p in rp:
                    bie[b, p] = les[b]

            try:
                g = model.generate(
                    inputs_embeds=bie,
                    max_new_tokens=tr + BUFFER,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # Per-layer success: layer index l (0-indexed in `les`) corresponds to
                # transformer layer l+1 (1-indexed for our reporting).
                layer_success = [False] * 32
                for ln in range(len(les)):
                    txt = tokenizer.decode(g[ln], skip_special_tokens=True)
                    if rest.lower() in txt.lower():
                        layer_success[ln] = True

                # Pull out only the layers we want (1-indexed -> 0-indexed)
                selected_states = np.zeros((len(LAYER_INDICES), les.shape[1]), dtype=np.float16)
                selected_labels = np.zeros(len(LAYER_INDICES), dtype=bool)
                for k, layer_n in enumerate(LAYER_INDICES):
                    selected_states[k] = les[layer_n - 1].cpu().to(torch.float32).numpy().astype(np.float16)
                    selected_labels[k] = layer_success[layer_n - 1]

                rows.append({
                    "hidden_states": selected_states,  # (8, 4096)
                    "labels": selected_labels,         # (8,)
                    "i": tr,
                    "phrase_pos": phrase_pos,
                })
            except Exception:
                continue
    return rows


# Load context data
print("Loading context data...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

work = []
for phrase, matches in ctx_data.items():
    for j, m in enumerate(matches):
        work.append({
            "phrase": phrase, "category": m["category"],
            "context_before": m["context_before"], "instance_idx": j,
        })
print(f"  {len(work)} instances\n")

# Process and accumulate
all_states = []
all_labels = []
all_i = []
metadata = []

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
        metadata.append({
            "phrase": item["phrase"],
            "category": item["category"],
            "instance_idx": item["instance_idx"],
            "phrase_pos": r["phrase_pos"],
            "i": r["i"],
        })

# Stack and save
print("\nStacking arrays...")
hidden_states = np.stack(all_states, axis=0)   # (N, 8, 4096)
labels = np.stack(all_labels, axis=0)           # (N, 8)
i_values = np.array(all_i, dtype=np.int32)      # (N,)

print(f"  hidden_states: {hidden_states.shape}, dtype={hidden_states.dtype}")
print(f"  labels:        {labels.shape}, dtype={labels.dtype}")
print(f"  i_values:      {i_values.shape}")

Path("data").mkdir(exist_ok=True)
np.savez_compressed(
    "data/probing_features.npz",
    hidden_states=hidden_states,
    labels=labels,
    i_values=i_values,
    layer_indices=np.array(LAYER_INDICES, dtype=np.int32),
)
with open("data/probing_metadata.json", "w") as f:
    json.dump({"layer_indices": LAYER_INDICES, "rows": metadata}, f, indent=2)

# Summary
print("\nDone.")
print(f"Total rows: {len(metadata)}")
print(f"\nLabel distribution per layer (positive rate at each i):")
print(f"{'i':<4}{'rows':<8}" + "".join(f"L{L:<8}" for L in LAYER_INDICES))
print("-" * (12 + 9*len(LAYER_INDICES)))
for i_val in sorted(set(i_values.tolist())):
    mask = i_values == i_val
    n = mask.sum()
    if n == 0: continue
    row = f"{i_val:<4}{n:<8}"
    for k in range(len(LAYER_INDICES)):
        rate = labels[mask, k].mean() * 100
        row += f"{rate:<9.1f}"
    print(row)

print(f"\nSaved to data/probing_features.npz and data/probing_metadata.json")
