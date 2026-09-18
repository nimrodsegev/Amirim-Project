"""
Same rerun but with 'Repeat this: X X X X X' instead of 'Repeat this: X'.
Budget bug fixed (buffer=3) so the comparison isolates the prompt effect.
"""
import csv, json, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
PROBE_PROMPT = "Repeat this: X X X X X"  # <-- only change from rerun_fixed.py
REPLACE_TOKEN = "X"
BUFFER = 3

DATASETS = [
    ("data/buildings_clean.csv", "Famous Buildings", "building"),
    ("data/idioms.csv",           "Idioms",           "idiom"),
    ("data/imdb_top250_multiword.csv", "IMDB Movies", "movie"),
]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print("Loaded.\n")

def analyze_phrase(phrase):
    input_ids = tokenizer.encode(phrase)
    if tokenizer.bos_token_id and input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    if len(input_ids) < 3: return {}
    phrase_tokens = input_ids[1:]
    n = len(phrase_tokens)
    results = {i: {"success": False, "first_layer": None, "layer_third": None}
               for i in range(1, n)}

    def hook_fn(m, inp):
        m.captured_input = inp[0].clone()
        return inp
    hook = model.model.norm.register_forward_pre_hook(hook_fn)

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
        rp = rm.nonzero(as_tuple=False).flatten()
        be = model.get_input_embeddings()(pt).unsqueeze(0)

        for tok_idx in range(1, len(input_ids) - 1):
            pp = tok_idx - 1
            tr = n - pp - 1
            rest = tokenizer.decode(phrase_tokens[-tr:], skip_special_tokens=True)
            les = []
            for li in range(1, len(out.hidden_states)):
                le = out.hidden_states[li][:, tok_idx] if li < len(out.hidden_states)-1 else last_pn[:, tok_idx]
                les.append(le.squeeze(0))
            les = torch.stack(les, dim=0)
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
                for ln in range(len(les)):
                    txt = tokenizer.decode(g[ln], skip_special_tokens=True)
                    if rest.lower() in txt.lower():
                        al = ln + 1
                        third = "early" if al <= 10 else ("middle" if al <= 21 else "late")
                        if not results[tr]["success"]:
                            results[tr] = {"success": True, "first_layer": al, "layer_third": third}
                        break
            except Exception:
                continue
    return results

def n_tok(p):
    ids = tokenizer.encode(p)
    if tokenizer.bos_token_id and ids[0] != tokenizer.bos_token_id:
        ids = [tokenizer.bos_token_id] + ids
    return len(ids) - 1

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

agg = defaultdict(lambda: {"tested": 0, "success": 0, "early": 0, "middle": 0, "late": 0})
for e in all_entries:
    for i, d in e["per_i"].items():
        agg[i]["tested"] += 1
        if d["success"]:
            agg[i]["success"] += 1
            agg[i][d["layer_third"]] += 1
agg = dict(sorted(agg.items()))

Path("results").mkdir(exist_ok=True)
with open("results/lookahead_5x_fixed.json", "w") as f:
    json.dump({"overall": agg, "prompt": PROBE_PROMPT, "buffer": BUFFER}, f, indent=2)

# 1X results for comparison
with open("results/lookahead_analysis_fixed.json") as f:
    one_x = json.load(f)["overall"]
one_x = {int(k): v for k, v in one_x.items()} if isinstance(list(one_x.keys())[0], str) else one_x

print("\n" + "="*90)
print("HEAD-TO-HEAD: 1X vs 5X (both with buffer=3)")
print("="*90)
print(f"{'i':<5}{'1X tested':<12}{'1X succ':<10}{'1X rate':<10}{'5X tested':<12}{'5X succ':<10}{'5X rate':<10}{'delta':<10}")
print("-"*90)
all_i = sorted(set(list(agg.keys()) + list(one_x.keys())))
for i in all_i:
    o = one_x.get(i, {"tested": 0, "success": 0})
    f_ = agg.get(i, {"tested": 0, "success": 0})
    or_ = 100*o["success"]/o["tested"] if o["tested"] else 0
    fr_ = 100*f_["success"]/f_["tested"] if f_["tested"] else 0
    d = fr_ - or_
    print(f"{i:<5}{o['tested']:<12}{o['success']:<10}{or_:<10.1f}{f_['tested']:<12}{f_['success']:<10}{fr_:<10.1f}{d:+<10.1f}")

print("\nLayer distribution comparison at i where both have successes:")
print(f"{'i':<5}{'1X E/M/L%':<25}{'5X E/M/L%':<25}")
print("-"*60)
for i in all_i:
    o = one_x.get(i, {})
    f_ = agg.get(i, {})
    os_ = o.get("success", 0)
    fs = f_.get("success", 0)
    if os_ == 0 and fs == 0: continue
    o_str = f"{100*o.get('early',0)/os_:.0f}/{100*o.get('middle',0)/os_:.0f}/{100*o.get('late',0)/os_:.0f}" if os_ else "—"
    f_str = f"{100*f_.get('early',0)/fs:.0f}/{100*f_.get('middle',0)/fs:.0f}/{100*f_.get('late',0)/fs:.0f}" if fs else "—"
    print(f"{i:<5}{o_str:<25}{f_str:<25}")

print("\nDone. Saved to results/lookahead_5x_fixed.json")
