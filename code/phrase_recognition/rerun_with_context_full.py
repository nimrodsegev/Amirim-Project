"""
Full with-context analysis on all matches in data/phrases_with_context_v2.json.
Produces:
  1. Instance-level success rates (comparable to naked)
  2. Phrase-level averaged rates (each phrase counts equally)
  3. Layer distribution (E/M/L %, instance-level)
  4. Paired first-layer comparison (naked vs context, per-phrase avg)
  5. Per-phrase stability distribution (how often phrase X succeeds across its contexts)
"""
import json, torch, time
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
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
print("Loaded.\n")


def analyze(context_before, phrase_text):
    context_ids = tokenizer.encode(context_before, add_special_tokens=False) if context_before.strip() else []
    join_str = " " if context_ids else ""
    phrase_ids = tokenizer.encode(join_str + phrase_text, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + context_ids + phrase_ids
    phrase_start = 1 + len(context_ids)
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


print("Loading context data...")
with open("data/phrases_with_context_v2.json") as f:
    ctx_data = json.load(f)

# Build work list: (phrase, category, context_before, instance_idx)
work = []
for phrase, matches in ctx_data.items():
    for j, m in enumerate(matches):
        work.append({
            "phrase": phrase,
            "category": m["category"],
            "context_before": m["context_before"],
            "instance_idx": j,
        })
print(f"  {len(work)} total instances across {len(ctx_data)} phrases")
print(f"  Starting analysis...\n")

start = time.time()
all_results = []  # list of {phrase, category, instance_idx, per_i}
for idx, item in enumerate(work):
    if idx % 200 == 0:
        elapsed = time.time() - start
        rate = idx / elapsed if elapsed > 0 else 0
        eta = (len(work) - idx) / rate if rate > 0 else 0
        print(f"  {idx}/{len(work)} | {elapsed:.0f}s elapsed | ETA {eta/60:.0f} min")
    r = analyze(item["context_before"], item["phrase"])
    item["per_i"] = r
    all_results.append(item)

# Save raw results
Path("results").mkdir(exist_ok=True)
serial = [{**{k: v for k, v in r.items() if k != "per_i"},
           "per_i": {str(k): v for k, v in r["per_i"].items()}} for r in all_results]
with open("results/lookahead_with_context_full.json", "w") as f:
    json.dump(serial, f, indent=2)

# =================== TABLE 1: instance-level aggregate ===================
inst_agg = defaultdict(lambda: {"tested": 0, "success": 0, "early": 0, "middle": 0, "late": 0})
for r in all_results:
    for i, d in r["per_i"].items():
        inst_agg[i]["tested"] += 1
        if d["success"]:
            inst_agg[i]["success"] += 1
            inst_agg[i][d["layer_third"]] += 1
inst_agg = dict(sorted(inst_agg.items()))

with open("results/lookahead_analysis_fixed.json") as f:
    naked = json.load(f)["overall"]
naked = {int(k): v for k, v in naked.items()} if isinstance(list(naked.keys())[0], str) else naked

print("\n" + "="*90)
print("TABLE 1: Instance-level success rates (naked vs with-context)")
print("="*90)
print(f"{'i':<4}{'naked n':<10}{'naked %':<10}{'ctx n':<10}{'ctx succ':<10}{'ctx %':<10}{'delta':<10}")
print("-"*90)
for i in sorted(set(list(inst_agg.keys()) + list(naked.keys()))):
    a = naked.get(i, {"tested":0,"success":0})
    b = inst_agg.get(i, {"tested":0,"success":0})
    ar = 100*a["success"]/a["tested"] if a["tested"] else 0
    br = 100*b["success"]/b["tested"] if b["tested"] else 0
    print(f"{i:<4}{a['tested']:<10}{ar:<10.1f}{b['tested']:<10}{b['success']:<10}{br:<10.1f}{br-ar:+<10.1f}")

# =================== TABLE 2: phrase-level averaged rates ===================
# For each (phrase, i), compute success rate across its instances. Then average across phrases.
phrase_per_i = defaultdict(lambda: defaultdict(list))  # phrase -> i -> [0/1,...]
for r in all_results:
    for i, d in r["per_i"].items():
        phrase_per_i[r["phrase"]][i].append(1 if d["success"] else 0)

print("\n" + "="*75)
print("TABLE 2: Phrase-level averaged rates (each phrase counts equally)")
print("="*75)
print(f"{'i':<4}{'phrases':<10}{'avg rate':<12}{'phrases always':<18}{'phrases never':<16}")
print("-"*75)
for i in sorted(inst_agg.keys()):
    per_phrase_rates = []
    always = never = 0
    for phrase, by_i in phrase_per_i.items():
        if i in by_i and len(by_i[i]) > 0:
            rate = sum(by_i[i]) / len(by_i[i])
            per_phrase_rates.append(rate)
            if rate == 1.0: always += 1
            if rate == 0.0: never += 1
    if not per_phrase_rates: continue
    avg = 100 * mean(per_phrase_rates)
    print(f"{i:<4}{len(per_phrase_rates):<10}{avg:<12.1f}{always:<18}{never:<16}")

# =================== TABLE 3: layer distribution (instance-level) ===================
print("\n" + "="*65)
print("TABLE 3: Layer distribution with context (instance-level successes)")
print("="*65)
print(f"{'i':<4}{'successes':<12}{'Early %':<10}{'Middle %':<10}{'Late %':<10}")
print("-"*65)
for i in sorted(inst_agg.keys()):
    d = inst_agg[i]
    s = d["success"]
    if s == 0: continue
    print(f"{i:<4}{s:<12}{100*d['early']/s:<10.1f}{100*d['middle']/s:<10.1f}{100*d['late']/s:<10.1f}")

# =================== TABLE 4: paired first-layer comparison ===================
# For each phrase that succeeded in BOTH naked and (any context instance) at each i,
# use naked's first_layer vs the MEDIAN of the ctx instances' first_layers.
try:
    with open("results/layer_analysis_fixed.json") as f:
        naked_pp_raw = json.load(f)
    naked_pp = {}
    for ds, entries in naked_pp_raw.items():
        for e in entries:
            naked_pp[e["phrase"]] = e["per_i"]

    ctx_first_layers = defaultdict(lambda: defaultdict(list))  # phrase -> i -> [first_layers]
    for r in all_results:
        for i, d in r["per_i"].items():
            if d["success"] and d["first_layer"] is not None:
                ctx_first_layers[r["phrase"]][i].append(d["first_layer"])

    print("\n" + "="*80)
    print("TABLE 4: Paired first-layer comparison (same phrases in both conditions)")
    print("="*80)
    print(f"{'i':<4}{'matched':<10}{'naked avg_L':<14}{'ctx med_L':<14}{'shift':<10}{'earlier':<10}{'later':<10}")
    print("-"*80)
    for i in sorted(inst_agg.keys()):
        i_str = str(i)
        pairs = []
        for phrase, n_per_i in naked_pp.items():
            if phrase not in ctx_first_layers: continue
            if i not in ctx_first_layers[phrase]: continue
            n_d = n_per_i.get(i_str)
            if not n_d or not n_d["success"] or n_d["first_layer"] is None: continue
            n_L = n_d["first_layer"]
            c_Ls = ctx_first_layers[phrase][i]
            if not c_Ls: continue
            c_L = median(c_Ls)
            pairs.append((n_L, c_L))
        if not pairs: continue
        n_avg = mean(p[0] for p in pairs)
        c_avg = mean(p[1] for p in pairs)
        earlier = sum(1 for n, c in pairs if c < n)
        later = sum(1 for n, c in pairs if c > n)
        print(f"{i:<4}{len(pairs):<10}{n_avg:<14.1f}{c_avg:<14.1f}{c_avg-n_avg:+<10.1f}{earlier:<10}{later:<10}")
except FileNotFoundError:
    print("\n  Skipping paired comparison (naked layer file not found)")

# =================== TABLE 5: per-phrase stability ===================
print("\n" + "="*95)
print("TABLE 5: Per-phrase stability (distribution of success counts across contexts)")
print("="*95)
print(f"  For each i, histogram of 'X of N contexts succeeded' across phrases with >=5 contexts.")
print()
print(f"  {'i':<4}{'phrases':<10}{'mean/N':<10}{'0/N':<6}{'1-2':<6}{'3-4':<6}{'5-6':<6}{'7-8':<6}{'9-10':<6}{'N/N':<6}")
print("-"*95)
for i in sorted(inst_agg.keys()):
    buckets = {"0": 0, "1-2": 0, "3-4": 0, "5-6": 0, "7-8": 0, "9-10": 0, "N/N": 0}
    rates = []
    n_phrases = 0
    for phrase, by_i in phrase_per_i.items():
        if i not in by_i: continue
        outcomes = by_i[i]
        if len(outcomes) < 5: continue  # need enough data to assess stability
        n_phrases += 1
        s = sum(outcomes)
        n = len(outcomes)
        rate = s / n
        rates.append(rate)
        if s == 0: buckets["0"] += 1
        elif s == n: buckets["N/N"] += 1
        elif s <= 2: buckets["1-2"] += 1
        elif s <= 4: buckets["3-4"] += 1
        elif s <= 6: buckets["5-6"] += 1
        elif s <= 8: buckets["7-8"] += 1
        else: buckets["9-10"] += 1
    if n_phrases == 0: continue
    m = mean(rates)
    print(f"  {i:<4}{n_phrases:<10}{m:<10.2f}{buckets['0']:<6}{buckets['1-2']:<6}{buckets['3-4']:<6}"
          f"{buckets['5-6']:<6}{buckets['7-8']:<6}{buckets['9-10']:<6}{buckets['N/N']:<6}")

print("\n  Bimodal (many 0/N and N/N, few in middle) = stable representations.")
print("  Spread (many in the middle buckets) = context effect is unreliable per-phrase.")

print("\nDone.")
