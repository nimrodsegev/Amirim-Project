"""
Full rerun with the token-budget bug fixed.
Uses dynamic budget: max_new_tokens = i + 3.
Reruns lookahead + layer analysis + all PDF analyses in one pass.
Saves everything to results/*_fixed.json and prints comparison tables.
"""
import csv, json, torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-1124-7B"
PROBE_PROMPT = "Repeat this: X"
REPLACE_TOKEN = "X"
BUFFER = 3  # max_new_tokens = i + BUFFER

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
print("Model loaded.\n")


def analyze_phrase(phrase):
    """
    Returns dict {i: {'success': bool, 'first_layer': int or None, 'layer_third': str or None}}
    Uses dynamic max_new_tokens = i + BUFFER.
    """
    input_ids = tokenizer.encode(phrase)
    if tokenizer.bos_token_id and input_ids[0] != tokenizer.bos_token_id:
        input_ids = [tokenizer.bos_token_id] + input_ids
    if len(input_ids) < 3:
        return {}

    phrase_tokens = input_ids[1:]
    n_tokens = len(phrase_tokens)
    results = {i: {"success": False, "first_layer": None, "layer_third": None}
               for i in range(1, n_tokens)}

    def pre_norm_capture_hook(module, inp):
        module.captured_input = inp[0].clone()
        return inp
    hook = model.model.norm.register_forward_pre_hook(pre_norm_capture_hook)

    with torch.no_grad():
        mi = torch.tensor(input_ids).unsqueeze(0).to(model.device)
        outputs = model(input_ids=mi, output_hidden_states=True)
        last_pre_norm = model.model.norm.captured_input.clone()
        hook.remove()

        probe_ids = tokenizer.encode(PROBE_PROMPT)
        probe_tensor = torch.tensor(probe_ids).to(model.device)
        rep_ids = tokenizer.encode(REPLACE_TOKEN, add_special_tokens=False)
        rep_mask = probe_tensor == rep_ids[0]
        sp_ids = tokenizer.encode(" " + REPLACE_TOKEN, add_special_tokens=False)
        if len(sp_ids) == 1:
            rep_mask = rep_mask | (probe_tensor == sp_ids[0])
        if not rep_mask.any():
            return results
        rep_positions = rep_mask.nonzero(as_tuple=False).flatten()
        base_embeds = model.get_input_embeddings()(probe_tensor).unsqueeze(0)

        for token_idx in range(1, len(input_ids) - 1):
            phrase_pos = token_idx - 1
            tokens_remaining = n_tokens - phrase_pos - 1
            last_i = phrase_tokens[-tokens_remaining:]
            rest = tokenizer.decode(last_i, skip_special_tokens=True)

            layer_embs = []
            for li in range(1, len(outputs.hidden_states)):
                if li < len(outputs.hidden_states) - 1:
                    le = outputs.hidden_states[li][:, token_idx]
                else:
                    le = last_pre_norm[:, token_idx]
                layer_embs.append(le.squeeze(0))
            layer_embs = torch.stack(layer_embs, dim=0)
            bie = base_embeds.repeat(len(layer_embs), 1, 1)
            for b in range(len(layer_embs)):
                for p in rep_positions:
                    bie[b, p] = layer_embs[b]

            try:
                gen = model.generate(
                    inputs_embeds=bie,
                    max_new_tokens=tokens_remaining + BUFFER,  # THE FIX
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                for ln in range(len(layer_embs)):
                    txt = tokenizer.decode(gen[ln], skip_special_tokens=True)
                    if rest.lower() in txt.lower():
                        actual_layer = ln + 1
                        third = "early" if actual_layer <= 10 else ("middle" if actual_layer <= 21 else "late")
                        # keep best (first found) — don't overwrite if already set
                        if not results[tokens_remaining]["success"]:
                            results[tokens_remaining] = {
                                "success": True,
                                "first_layer": actual_layer,
                                "layer_third": third,
                            }
                        break
            except Exception:
                continue

    return results


def phrase_n_tokens(phrase):
    ids = tokenizer.encode(phrase)
    if tokenizer.bos_token_id and ids[0] != tokenizer.bos_token_id:
        ids = [tokenizer.bos_token_id] + ids
    return len(ids) - 1


# ---------- Run on all datasets ----------
all_phrase_results = {}  # {dataset_name: [(phrase, category, n_tokens, per_i_results), ...]}
for path, name, cat in DATASETS:
    print(f"\n{'='*70}\nProcessing: {name}\n{'='*70}")
    phrases = []
    with open(path) as f:
        reader = csv.DictReader(f)
        col = reader.fieldnames[0]
        for row in reader:
            phrases.append(row[col])
    print(f"  {len(phrases)} phrases")
    entries = []
    for idx, p in enumerate(phrases):
        if idx % 50 == 0:
            print(f"    {idx}/{len(phrases)}")
        r = analyze_phrase(p)
        entries.append({"phrase": p, "category": cat, "n_tokens": phrase_n_tokens(p), "per_i": r})
    all_phrase_results[name] = entries

# ---------- Aggregate ----------
def aggregate(entries):
    agg = defaultdict(lambda: {"tested": 0, "success": 0, "early": 0, "middle": 0, "late": 0})
    for e in entries:
        for i, d in e["per_i"].items():
            agg[i]["tested"] += 1
            if d["success"]:
                agg[i]["success"] += 1
                agg[i][d["layer_third"]] += 1
    return dict(sorted(agg.items()))

per_dataset = {name: aggregate(entries) for name, entries in all_phrase_results.items()}
overall_entries = [e for es in all_phrase_results.values() for e in es]
overall = aggregate(overall_entries)

# ---------- Save raw ----------
Path("results").mkdir(exist_ok=True)
with open("results/lookahead_analysis_fixed.json", "w") as f:
    json.dump({"per_dataset": per_dataset, "overall": overall, "buffer": BUFFER}, f, indent=2)
# full per-phrase for layer + follow-up
serial = {name: [{"phrase": e["phrase"], "category": e["category"], "n_tokens": e["n_tokens"],
                  "per_i": {str(k): v for k, v in e["per_i"].items()}}
                 for e in entries]
          for name, entries in all_phrase_results.items()}
with open("results/layer_analysis_fixed.json", "w") as f:
    json.dump(serial, f, indent=2)

# ---------- Print main table ----------
def print_table(agg, title):
    print(f"\n{title}")
    print(f"{'i':<5}{'tested':<10}{'success':<10}{'rate':<10}{'early':<8}{'middle':<8}{'late':<8}")
    print("-" * 60)
    for i in sorted(agg.keys()):
        d = agg[i]
        rate = 100 * d["success"] / d["tested"] if d["tested"] else 0
        print(f"{i:<5}{d['tested']:<10}{d['success']:<10}{rate:<10.1f}{d['early']:<8}{d['middle']:<8}{d['late']:<8}")

for name, agg in per_dataset.items():
    print_table(agg, f"=== {name} (FIXED) ===")
print_table(overall, "=== OVERALL (FIXED) ===")

# ---------- Compare to original ----------
orig_path = Path("results/lookahead_analysis.json")
if orig_path.exists():
    with open(orig_path) as f:
        orig = json.load(f)
    orig_overall = orig.get("Overall", {})
    print("\n=== COMPARISON: Original vs Fixed (Overall) ===")
    print(f"{'i':<5}{'orig_rate':<12}{'fixed_rate':<12}{'delta':<10}{'new_succ':<10}")
    print("-" * 50)
    for i in sorted(overall.keys()):
        fd = overall[i]
        fr = 100 * fd["success"] / fd["tested"] if fd["tested"] else 0
        od = orig_overall.get(str(i))
        if od:
            or_ = 100 * od["success"] / od["tested"] if od["tested"] else 0
            delta = fr - or_
            new = fd["success"] - od["success"]
            print(f"{i:<5}{or_:<12.1f}{fr:<12.1f}{delta:+<10.1f}{new:<10}")
        else:
            print(f"{i:<5}{'N/A':<12}{fr:<12.1f}{'':<10}{fd['success']:<10}")

# ---------- Layer distribution ----------
print("\n=== LAYER DISTRIBUTION AT EACH i (Overall, successes only) ===")
print(f"{'i':<5}{'success':<10}{'early%':<10}{'middle%':<10}{'late%':<10}")
print("-" * 50)
for i in sorted(overall.keys()):
    d = overall[i]
    s = d["success"]
    if s == 0: continue
    e = 100 * d["early"] / s
    m = 100 * d["middle"] / s
    l = 100 * d["late"] / s
    print(f"{i:<5}{s:<10}{e:<10.1f}{m:<10.1f}{l:<10.1f}")

# ---------- Per-category success rates ----------
print("\n=== SUCCESS RATE BY CATEGORY ===")
print(f"{'i':<5}{'building':<15}{'idiom':<15}{'movie':<15}")
print("-" * 50)
cat_agg = {"building": defaultdict(lambda: {"t": 0, "s": 0}),
           "idiom":    defaultdict(lambda: {"t": 0, "s": 0}),
           "movie":    defaultdict(lambda: {"t": 0, "s": 0})}
for e in overall_entries:
    for i, d in e["per_i"].items():
        cat_agg[e["category"]][i]["t"] += 1
        if d["success"]: cat_agg[e["category"]][i]["s"] += 1
for i in sorted(overall.keys()):
    row = f"{i:<5}"
    for c in ["building", "idiom", "movie"]:
        x = cat_agg[c][i]
        r = 100 * x["s"] / x["t"] if x["t"] else 0
        row += f"{r:.1f}% ({x['s']}/{x['t']})   "
    print(row)

# ---------- Length correlation ----------
print("\n=== LENGTH CORRELATION ===")
print(f"{'i':<5}{'avg_tested':<15}{'avg_success':<15}{'delta':<10}")
print("-" * 50)
for i in sorted(overall.keys()):
    tested = [e["n_tokens"] for e in overall_entries if i in e["per_i"]]
    succ = [e["n_tokens"] for e in overall_entries if i in e["per_i"] and e["per_i"][i]["success"]]
    if not tested: continue
    at = sum(tested) / len(tested)
    as_ = sum(succ) / len(succ) if succ else 0
    print(f"{i:<5}{at:<15.2f}{as_:<15.2f}{as_ - at:+<10.2f}")

# ---------- Elite phrases per i ----------
print("\n=== ELITE PHRASES (succeeded at each i) ===")
for target_i in [3, 4, 5, 6, 7, 8, 9, 10]:
    elite = [e["phrase"] for e in overall_entries
             if target_i in e["per_i"] and e["per_i"][target_i]["success"]]
    with open(f"results/phrases_success_i{target_i}_fixed.txt", "w") as f:
        for p in elite: f.write(p + "\n")
    print(f"  i={target_i}: {len(elite)} phrases -> results/phrases_success_i{target_i}_fixed.txt")

# ---------- Change summary ----------
if orig_path.exists():
    print("\n=== CHANGE SUMMARY ===")
    total_new = 0
    for i in sorted(overall.keys()):
        if i <= 5: continue
        s = overall[i]["success"]
        if s: total_new += s
    print(f"Total NEW successes at i>=6: {total_new}")
    max_i_with_success = max((i for i in overall if overall[i]["success"] > 0), default=0)
    print(f"New effective ceiling: i={max_i_with_success}")

print("\nDone.")
