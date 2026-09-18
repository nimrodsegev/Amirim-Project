"""
Full rerun on OLMo-3-7B with the same bug fix (buffer=3) as OLMo-2.
Adjusted BOS handling: OLMo-3 has no BOS token, so we analyze positions
starting from 0 instead of 1.
"""
import csv, json, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/Olmo-3-1025-7B"
PROBE_PROMPT = "Repeat this: X"
REPLACE_TOKEN = "X"
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
    # phrase_tokens = tokens that are actually part of the phrase
    # start_idx = where in input_ids the phrase begins
    if HAS_BOS:
        phrase_tokens = input_ids[1:]
        start_idx = 1
    else:
        phrase_tokens = input_ids
        start_idx = 0
    n_tokens = len(phrase_tokens)
    if n_tokens < 3:
        return {}
    results = {i: {"success": False, "first_layer": None, "layer_third": None}
               for i in range(1, n_tokens)}

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
        if not rm.any():
            return results
        rp = rm.nonzero(as_tuple=False).flatten()
        be = model.get_input_embeddings()(pt).unsqueeze(0)

        # Loop positions: for OLMo-2 this is 1..len-2, for OLMo-3 this is 0..len-2
        for tok_idx in range(start_idx, len(input_ids) - 1):
            phrase_pos = tok_idx - start_idx
            tokens_remaining = n_tokens - phrase_pos - 1
            if tokens_remaining < 1:
                continue
            rest = tokenizer.decode(phrase_tokens[-tokens_remaining:], skip_special_tokens=True)

            les = []
            for li in range(1, len(out.hidden_states)):
                if li < len(out.hidden_states) - 1:
                    le = out.hidden_states[li][:, tok_idx]
                else:
                    le = last_pn[:, tok_idx]
                les.append(le.squeeze(0))
            les = torch.stack(les, dim=0)
            bie = be.repeat(len(les), 1, 1)
            for b in range(len(les)):
                for p in rp:
                    bie[b, p] = les[b]
            try:
                g = model.generate(
                    inputs_embeds=bie,
                    max_new_tokens=tokens_remaining + BUFFER,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                for ln in range(len(les)):
                    txt = tokenizer.decode(g[ln], skip_special_tokens=True)
                    if rest.lower() in txt.lower():
                        al = ln + 1
                        third = "early" if al <= 10 else ("middle" if al <= 21 else "late")
                        if not results[tokens_remaining]["success"]:
                            results[tokens_remaining] = {"success": True, "first_layer": al, "layer_third": third}
                        break
            except Exception:
                continue
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

agg = defaultdict(lambda: {"tested": 0, "success": 0, "early": 0, "middle": 0, "late": 0})
for e in all_entries:
    for i, d in e["per_i"].items():
        agg[i]["tested"] += 1
        if d["success"]:
            agg[i]["success"] += 1
            agg[i][d["layer_third"]] += 1
agg = dict(sorted(agg.items()))

Path("results").mkdir(exist_ok=True)
out_json = {"overall": agg, "model": MODEL_NAME, "buffer": BUFFER, "has_bos": HAS_BOS}
with open("results/lookahead_olmo3_fixed.json", "w") as f:
    json.dump(out_json, f, indent=2)

# Per-phrase save
serial = [{"phrase": e["phrase"], "category": e["category"], "n_tokens": e["n_tokens"],
           "per_i": {str(k): v for k, v in e["per_i"].items()}}
          for e in all_entries]
with open("results/layer_analysis_olmo3_fixed.json", "w") as f:
    json.dump(serial, f, indent=2)

# Overall print
print("\n=== OLMo-3-7B OVERALL ===")
print(f"{'i':<5}{'tested':<10}{'success':<10}{'rate':<10}{'early':<8}{'middle':<8}{'late':<8}")
print("-" * 60)
for i in sorted(agg.keys()):
    d = agg[i]
    rate = 100 * d["success"] / d["tested"] if d["tested"] else 0
    print(f"{i:<5}{d['tested']:<10}{d['success']:<10}{rate:<10.1f}{d['early']:<8}{d['middle']:<8}{d['late']:<8}")

# Head-to-head with OLMo-2 fixed
try:
    with open("results/lookahead_analysis_fixed.json") as f:
        o2 = json.load(f)["overall"]
    o2 = {int(k): v for k, v in o2.items()} if isinstance(list(o2.keys())[0], str) else o2
    print("\n=== HEAD-TO-HEAD: OLMo-2 vs OLMo-3 (both buffer=3) ===")
    print(f"{'i':<5}{'O2 n':<8}{'O2 succ':<10}{'O2 rate':<10}{'O3 n':<8}{'O3 succ':<10}{'O3 rate':<10}{'delta':<10}")
    print("-" * 80)
    all_i = sorted(set(list(agg.keys()) + list(o2.keys())))
    for i in all_i:
        a = o2.get(i, {"tested": 0, "success": 0})
        b = agg.get(i, {"tested": 0, "success": 0})
        ar = 100*a["success"]/a["tested"] if a["tested"] else 0
        br = 100*b["success"]/b["tested"] if b["tested"] else 0
        print(f"{i:<5}{a['tested']:<8}{a['success']:<10}{ar:<10.1f}{b['tested']:<8}{b['success']:<10}{br:<10.1f}{br-ar:+<10.1f}")

    print("\nLayer distribution (where both have successes):")
    print(f"{'i':<5}{'O2 E/M/L%':<25}{'O3 E/M/L%':<25}")
    print("-" * 60)
    for i in all_i:
        a = o2.get(i, {}); b = agg.get(i, {})
        as_ = a.get("success", 0); bs = b.get("success", 0)
        if as_ == 0 and bs == 0: continue
        a_str = f"{100*a.get('early',0)/as_:.0f}/{100*a.get('middle',0)/as_:.0f}/{100*a.get('late',0)/as_:.0f}" if as_ else "-"
        b_str = f"{100*b.get('early',0)/bs:.0f}/{100*b.get('middle',0)/bs:.0f}/{100*b.get('late',0)/bs:.0f}" if bs else "-"
        print(f"{i:<5}{a_str:<25}{b_str:<25}")
except Exception as e:
    print(f"\nCould not load OLMo-2 comparison: {e}")

print("\nDone.")
