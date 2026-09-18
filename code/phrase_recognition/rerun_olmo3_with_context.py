"""
Run the with-context analysis on OLMo-3-7B.
Uses the same FineWeb-Edu contexts collected for OLMo-2.
Adjusts for OLMo-3 having no BOS token.
"""
import json, torch, time
from pathlib import Path
from collections import defaultdict
from statistics import mean
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/Olmo-3-1025-7B"
PROBE_PROMPT = "Repeat this: X"
REPLACE_TOKEN = "X"
BUFFER = 3

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
HAS_BOS = tokenizer.bos_token_id is not None
print(f"Loaded. has_bos={HAS_BOS}\n")

def analyze(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    if HAS_BOS:
        input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
        phrase_start = 1 + len(context_ids)
    else:
        input_ids = context_ids + phrase_ids
        phrase_start = len(context_ids)
    n_phrase = len(phrase_ids)
    if n_phrase < 3: return {}
    results = {i: {"success": False, "first_layer": None, "layer_third": None}
               for i in range(1, n_phrase)}

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
        if not rm.any(): return results
        rp = rm.nonzero(as_tuple=False).flatten()
        be = model.get_input_embeddings()(pt).unsqueeze(0)

        for tok_idx in range(phrase_start, len(input_ids) - 1):
            pp = tok_idx - phrase_start
            tr = n_phrase - pp - 1
            if tr < 1: continue
            rest = tokenizer.decode(phrase_ids[-tr:], skip_special_tokens=True)
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


print("Loading context data (collected with OLMo-2 tokenizer, reused as raw text)...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

work = []
for phrase, matches in ctx_data.items():
    for j, m in enumerate(matches):
        work.append({
            "phrase": phrase, "category": m["category"],
            "context_before": m["context_before"], "instance_idx": j,
        })
print(f"  {len(work)} instances across {len(ctx_data)} phrases\n")

start = time.time()
all_results = []
for idx, item in enumerate(work):
    if idx % 200 == 0:
        elapsed = time.time() - start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (len(work) - idx) / rate if rate > 0 else 0
        print(f"  {idx}/{len(work)} | {elapsed:.0f}s | ETA {eta/60:.0f} min")
    r = analyze(item["context_before"], item["phrase"])
    item["per_i"] = r
    all_results.append(item)

Path("results").mkdir(exist_ok=True)
serial = [{**{k: v for k, v in r.items() if k != "per_i"},
           "per_i": {str(k): v for k, v in r["per_i"].items()}} for r in all_results]
with open("results/lookahead_olmo3_with_context.json", "w") as f:
    json.dump(serial, f, indent=2)

# Aggregate
inst_agg = defaultdict(lambda: {"tested": 0, "success": 0, "early": 0, "middle": 0, "late": 0})
for r in all_results:
    for i, d in r["per_i"].items():
        inst_agg[i]["tested"] += 1
        if d["success"]:
            inst_agg[i]["success"] += 1
            inst_agg[i][d["layer_third"]] += 1
inst_agg = dict(sorted(inst_agg.items()))

# Compare to OLMo-3 naked + OLMo-2 with context
with open("results/lookahead_olmo3_fixed.json") as f:
    o3_naked = json.load(f)["overall"]
o3_naked = {int(k): v for k, v in o3_naked.items()} if isinstance(list(o3_naked.keys())[0], str) else o3_naked

# OLMo-2 with context aggregate
o2_ctx_agg = defaultdict(lambda: {"tested": 0, "success": 0})
with open("results/lookahead_with_context_full.json") as f:
    o2_ctx = json.load(f)
for r in o2_ctx:
    for i_str, d in r["per_i"].items():
        i = int(i_str)
        o2_ctx_agg[i]["tested"] += 1
        if d["success"]: o2_ctx_agg[i]["success"] += 1

print("\n" + "="*100)
print("OLMo-3: naked vs with-context, plus OLMo-2 with-context for reference")
print("="*100)
print(f"{'i':<4}{'O3 naked %':<12}{'O3 ctx n':<10}{'O3 ctx %':<12}{'O3 boost':<12}{'O2 ctx %':<12}{'O3-O2 ctx':<12}")
print("-"*100)
all_i = sorted(set(list(inst_agg.keys()) + list(o3_naked.keys())))
for i in all_i:
    a = o3_naked.get(i, {"tested":0,"success":0})
    b = inst_agg.get(i, {"tested":0,"success":0})
    c = o2_ctx_agg.get(i, {"tested":0,"success":0})
    ar = 100*a["success"]/a["tested"] if a["tested"] else 0
    br = 100*b["success"]/b["tested"] if b["tested"] else 0
    cr = 100*c["success"]/c["tested"] if c["tested"] else 0
    boost = br - ar if a["tested"] and b["tested"] else 0
    diff = br - cr if b["tested"] and c["tested"] else 0
    print(f"{i:<4}{ar:<12.1f}{b['tested']:<10}{br:<12.1f}{boost:+<12.1f}{cr:<12.1f}{diff:+<12.1f}")

print("\nIf O3 boost is similar to OLMo-2's (we saw +30 at i=3, +30 at i=5):")
print("  -> the OLMo-2 vs OLMo-3 naked gap is mostly the BOS confound.")
print("If O3 boost is much smaller:")
print("  -> there's a real OLMo-3 capability gap independent of BOS handling.")
print("\nDone.")
