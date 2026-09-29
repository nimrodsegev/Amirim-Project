"""Recompute every number the paper reports from results/raw/.

Writes results/processed/paper_numbers.json. Numbers that cannot be
recomputed here (probe metrics, which need the hidden-state feature
arrays produced on the cluster) are recorded under "external" with the
script that produced them.
"""
import json, collections, math
from pathlib import Path

RAW = Path("results/raw")
OUT = Path("results/processed/paper_numbers.json")
POOLED_I = ("2", "3", "4", "5")


def load(name):
    with open(RAW / name) as f:
        return json.load(f)


def agg_overall(d):
    """{i: (success, tested, rate)} from an {overall: {i: {...}}} file."""
    o = d["overall"] if "overall" in d else d
    return {int(i): (v["success"], v["tested"], round(100 * v["success"] / v["tested"], 1))
            for i, v in o.items() if v["tested"]}


def agg_per_dataset(d):
    tot, suc = collections.Counter(), collections.Counter()
    for ds in d["per_dataset"].values():
        for i, v in ds.items():
            tot[int(i)] += v["tested"]
            suc[int(i)] += v["success"]
    return {i: (suc[i], tot[i], round(100 * suc[i] / tot[i], 1)) for i in sorted(tot) if tot[i]}


def agg_instances(recs):
    tot, suc = collections.Counter(), collections.Counter()
    for r in recs:
        for i, v in r["per_i"].items():
            tot[int(i)] += 1
            suc[int(i)] += bool(v["success"])
    return {i: (suc[i], tot[i], round(100 * suc[i] / tot[i], 1)) for i in sorted(tot) if tot[i]}


out = {}

# ---- Datasets -------------------------------------------------------------
ctx = load("lookahead_with_context_full.json")
gen = load("generation_truth_context.json")["per_phrase"]
per_phrase_counts = collections.Counter(r["phrase"] for r in ctx)
cat_phrases = collections.defaultdict(set)
for r in ctx:
    cat_phrases[r["category"]].add(r["phrase"])
out["dataset"] = {
    "phrases_total": 1164,
    "phrases_by_category_source": {"building": 127, "idiom": 852, "movie": 185},
    "context_instances": len(ctx),
    "phrases_with_at_least_one_context": len(per_phrase_counts),
    "phrases_with_10_contexts": sum(1 for c in per_phrase_counts.values() if c == 10),
    "context_instances_by_category": dict(collections.Counter(r["category"] for r in ctx)),
    "phrases_with_context_by_category": {k: len(v) for k, v in cat_phrases.items()},
}

# ---- Isolation (no context) ----------------------------------------------
out["isolated"] = {
    "patchscopes_olmo2_32L": agg_per_dataset(load("lookahead_analysis_fixed.json")),
    "patchscopes_olmo2_32L_5slot_prompt": agg_overall(load("lookahead_5x_fixed.json")),
    "patchscopes_olmo3_32L": agg_overall(load("lookahead_olmo3_fixed.json")),
    "generation_olmo2": agg_overall(load("generation_truth_naked.json")),
}

# ---- With natural context -------------------------------------------------
out["in_context"] = {
    "patchscopes_olmo2_32L": agg_instances(ctx),
    "patchscopes_olmo3_32L": agg_instances(load("lookahead_olmo3_with_context.json")),
    "generation_olmo2": agg_instances(gen),
}

# ---- Matched-instance agreement (patchscopes 32L vs generation) -----------
assert len(ctx) == len(gen)
assert all(a["phrase"] == b["phrase"] for a, b in zip(ctx, gen))
cont = collections.defaultdict(collections.Counter)
for a, b in zip(ctx, gen):
    for i in sorted(set(a["per_i"]) & set(b["per_i"]), key=int):
        cont[int(i)][(bool(a["per_i"][i]["success"]), bool(b["per_i"][i]["success"]))] += 1
agree = {}
for i in sorted(cont):
    c = cont[i]
    n = sum(c.values())
    agree[i] = {
        "n": n,
        "both": c[(True, True)], "patchscopes_only": c[(True, False)],
        "generation_only": c[(False, True)], "neither": c[(False, False)],
        "agreement_pct": round(100 * (c[(True, True)] + c[(False, False)]) / n, 1),
    }
out["agreement_32L"] = agree

pooled = collections.Counter()
for i in (2, 3, 4, 5):
    for k, v in cont[i].items():
        pooled[k] += v
n = sum(pooled.values())
out["agreement_32L_pooled_i2_5"] = {
    "n": n,
    "both": pooled[(True, True)], "patchscopes_only": pooled[(True, False)],
    "generation_only": pooled[(False, True)], "neither": pooled[(False, False)],
    "agreement_pct": round(100 * (pooled[(True, True)] + pooled[(False, False)]) / n, 1),
    "patchscopes_pct": round(100 * (pooled[(True, True)] + pooled[(True, False)]) / n, 1),
    "generation_pct": round(100 * (pooled[(True, True)] + pooled[(False, True)]) / n, 1),
}

# ---- Context duplication -------------------------------------------------
ctx_by_phrase = collections.defaultdict(list)
for r in ctx:
    ctx_by_phrase[r["phrase"]].append(r["context_before"])
n_inst = sum(len(v) for v in ctx_by_phrase.values())
n_uniq = sum(len(set(v)) for v in ctx_by_phrase.values())
out["context_duplication"] = {
    "instances": n_inst,
    "unique_phrase_context_pairs": n_uniq,
    "duplicate_instances": n_inst - n_uniq,
    "duplicate_pct": round(100 * (n_inst - n_uniq) / n_inst, 1),
    "phrases_with_duplicates": sum(1 for v in ctx_by_phrase.values() if len(set(v)) != len(v)),
    "unique_context_count_hist_for_10_slot_phrases": dict(sorted(collections.Counter(
        len(set(v)) for v in ctx_by_phrase.values() if len(v) == 10).items())),
    "note": ("The context collector resumed from an earlier five-context run and "
             "re-added the same contexts instead of topping up with new ones, so "
             "most phrases hold five distinct contexts recorded twice. Aggregate "
             "instance rates are essentially unchanged (see "
             "per_phrase_consistency_i3), but the effective sample size is about "
             "half the instance count, and any per-phrase count over raw "
             "instances is quantised to even values."),
}


# ---- Per-phrase consistency at i=3, on DISTINCT contexts ------------------
def consistency(recs, min_ctx=5):
    """recs is aligned with ctx; dedupe by (phrase, context_before)."""
    by = collections.defaultdict(dict)
    for base, r in zip(ctx, recs):
        if "3" in r["per_i"]:
            by[base["phrase"]][base["context_before"]] = bool(r["per_i"]["3"]["success"])
    flat = [v for d in by.values() for v in d.values()]
    sel = {p: list(d.values()) for p, d in by.items() if len(d) >= min_ctx}
    n = len(sel)
    p_pool = sum(sum(v) for v in sel.values()) / sum(len(v) for v in sel.values())
    fr = [sum(v) / len(v) for v in sel.values()]
    exp = collections.Counter()
    for v in sel.values():
        m = len(v)
        pk = [math.comb(m, k) * p_pool ** k * (1 - p_pool) ** (m - k) for k in range(m + 1)]
        exp["hi"] += sum(pk[k] for k in range(m + 1) if k / m >= 0.9)
        exp["lo"] += sum(pk[k] for k in range(m + 1) if k / m <= 0.2)
        exp["mid"] += sum(pk[k] for k in range(m + 1) if 0.4 <= k / m <= 0.6)
    return {
        "phrases_tested": len(by),
        "distinct_instances": len(flat),
        "instance_rate_distinct_pct": round(100 * sum(flat) / len(flat), 1),
        "phrases_with_at_least_5_distinct_contexts": n,
        "pooled_rate": round(p_pool, 3),
        "observed_pct": {
            "at_least_90": round(100 * sum(1 for f in fr if f >= 0.9) / n, 1),
            "at_most_20": round(100 * sum(1 for f in fr if f <= 0.2) / n, 1),
            "between_40_and_60": round(100 * sum(1 for f in fr if 0.4 <= f <= 0.6) / n, 1),
        },
        "binomial_null_pct": {
            "at_least_90": round(100 * exp["hi"] / n, 1),
            "at_most_20": round(100 * exp["lo"] / n, 1),
            "between_40_and_60": round(100 * exp["mid"] / n, 1),
        },
        "per_phrase_rate_histogram_deciles": dict(collections.Counter(
            min(int(f * 10), 9) for f in fr)),
    }


out["per_phrase_consistency_i3"] = {
    "patchscopes_32L": consistency(ctx),
    "generation": consistency(gen),
}


def consistency_bins(recs, min_ctx=5):
    """The six bins used in the figure, observed and under the binomial null."""
    by = collections.defaultdict(dict)
    for base, r in zip(ctx, recs):
        if "3" in r["per_i"]:
            by[base["phrase"]][base["context_before"]] = bool(r["per_i"]["3"]["success"])
    sel = {p: list(d.values()) for p, d in by.items() if len(d) >= min_ctx}
    n = len(sel)
    pool = sum(sum(v) for v in sel.values()) / sum(len(v) for v in sel.values())

    def which(f):
        if f <= 0: return 0
        if f >= 1: return 5
        return 1 if f <= .25 else 2 if f <= .50 else 3 if f <= .75 else 4

    obs = [0] * 6
    exp = [0.0] * 6
    for v in sel.values():
        m = len(v)
        obs[which(sum(v) / m)] += 1
        for k in range(m + 1):
            exp[which(k / m)] += math.comb(m, k) * pool ** k * (1 - pool) ** (m - k)
    labels = ["none", "1-25", "25-50", "50-75", "75-99", "all"]
    return {
        "n_phrases": n,
        "labels": labels,
        "observed_pct": {l: round(100 * o / n, 1) for l, o in zip(labels, obs)},
        "null_pct": {l: round(100 * e / n, 1) for l, e in zip(labels, exp)},
        "central_two_bins_observed_pct": round(100 * (obs[2] + obs[3]) / n, 1),
        "central_two_bins_null_pct": round(100 * (exp[2] + exp[3]) / n, 1),
    }


out["per_phrase_consistency_bins_i3"] = {
    "generation": consistency_bins(gen),
    "patchscopes_32L": consistency_bins(ctx),
}

# ---- Category breakdown, pooled i=2-5 -------------------------------------
cats = {}
for label, recs in (("patchscopes_32L", ctx), ("generation", gen)):
    tot, suc = collections.Counter(), collections.Counter()
    for r in recs:
        for i in POOLED_I:
            if i in r["per_i"]:
                tot[r["category"]] += 1
                suc[r["category"]] += bool(r["per_i"][i]["success"])
    cats[label] = {c: {"n": tot[c], "success_pct": round(100 * suc[c] / tot[c], 1)} for c in sorted(tot)}
out["category_pooled_i2_5"] = cats

# ---- Depth of first successful layer --------------------------------------
fl = collections.Counter()
for r in ctx:
    for v in r["per_i"].values():
        if v["success"] and v.get("first_layer") is not None:
            fl[v["first_layer"]] += 1
tot_fl = sum(fl.values())
flat = sorted(L for L, c in fl.items() for _ in range(c))
out["first_successful_layer"] = {
    "n_successes": tot_fl,
    "median": flat[tot_fl // 2],
    "pct_at_or_before_L10": round(100 * sum(c for L, c in fl.items() if L <= 10) / tot_fl, 1),
    "histogram": {str(L): fl[L] for L in sorted(fl)},
}

# ---- Per-i correlation between probe confidence and phrase success --------
out["probe_phrase_correlation"] = {
    "source": "results/raw/correlation_per_i.json (balanced MLP, patchscopes label)",
    "by_layer_i3": {L: round(v["3"][0], 3) for L, v in load("correlation_per_i.json").items()},
    "n_phrases_i3": load("correlation_per_i.json")["10"]["3"][1],
}

# ---- Numbers that need the cluster feature arrays -------------------------
out["external"] = {
    "note": ("Probe metrics require data/{generation_,}probing_features.npz "
             "(hidden states, produced on the cluster and not stored in this repo). "
             "Values below are transcribed from the September 2026 "
             "deduplicated rerun logs."),
    "layers_probed": [5, 7, 10, 13, 15, 20, 25, 30],
    "probe_config": {
        "architecture": "MLP, one hidden layer of 32 units",
        "split": "GroupShuffleSplit by phrase, 80/20",
        "class_balance": "majority class downsampled in the training set only",
        "pooled_i": [2, 3, 4, 5],
        "seed": 42,
        "analysis_layer_for_falsepos_and_confidence": 25,
        "feature_files": {
            "probing_features.npz": "1.28 GB", 
            "generation_probing_features.npz": "3.02 GB",
            "status": "present on the compute cluster; too large to distribute, but the runs are reproducible from them",
        },
    },
    "patchscopes_label_probe": {
        "pos_rate_pct":   {"5": 6.4, "7": 14.1, "10": 18.2, "13": 16.4, "15": 17.8, "20": 35.8, "25": 46.4, "30": 39.1},
        "accuracy_pct":   {"5": 87.6, "7": 82.3, "10": 79.8, "13": 80.0, "15": 80.8, "20": 71.0, "25": 68.8, "30": 68.7},
        "majority_baseline_pct": {"5": 93.6, "7": 85.9, "10": 81.8, "13": 83.6, "15": 82.2, "20": 64.2, "25": 53.6, "30": 60.9},
        "balanced_accuracy_pct": {"5": 78.5, "7": 75.7, "10": 77.6, "13": 76.5, "15": 77.5, "20": 69.7, "25": 68.6, "30": 68.0},
        "precision_pct":  {"5": 29.5, "7": 41.8, "10": 46.5, "13": 43.4, "15": 47.4, "20": 58.4, "25": 66.2, "30": 59.1},
        "auc_roc":        {"5": 0.906, "7": 0.838, "10": 0.855, "13": 0.855, "15": 0.870, "20": 0.768, "25": 0.741, "30": 0.738},
    },
    "generation_label_probe": {
        "pos_rate_pct": 69.1,
        "accuracy_pct":   {"5": 72.1, "7": 73.6, "10": 73.2, "13": 74.6, "15": 76.0, "20": 74.1, "25": 74.8, "30": 76.7},
        "balanced_accuracy_pct": {"5": 68.6, "7": 70.2, "10": 70.1, "13": 71.5, "15": 73.0, "20": 71.4, "25": 72.5, "30": 74.1},
        "precision_pct":  {"5": 81.1, "7": 82.0, "10": 82.1, "13": 83.0, "15": 83.8, "20": 83.1, "25": 83.9, "30": 84.6},
        "auc_roc":        {"5": 0.751, "7": 0.772, "10": 0.769, "13": 0.792, "15": 0.799, "20": 0.792, "25": 0.806, "30": 0.820},
    },
    "patchscopes_per_layer_success_pct": {
        "2":  {"5": 14.2, "7": 23.4, "10": 28.7, "13": 26.0, "15": 27.4, "20": 49.5, "25": 60.7, "30": 58.1},
        "3":  {"5": 7.2,  "7": 13.7, "10": 17.4, "13": 16.2, "15": 17.8, "20": 37.3, "25": 47.0, "30": 38.2},
        "4":  {"5": 3.9,  "7": 8.6,  "10": 11.1, "13": 10.0, "15": 11.9, "20": 23.4, "25": 33.2, "30": 24.7},
        "5":  {"5": 2.4,  "7": 5.6,  "10": 8.9,  "13": 8.3,  "15": 8.3,  "20": 21.1, "25": 27.5, "30": 20.3},
        "L0_L1_L2_pooled_i2_5": {"0": 0.6, "1": 0.8, "2": 1.5, "any_of_3": 1.8},
    },
    "patchscopes_8L_union_in_context_pct": {"2": 71.9, "3": 56.6, "4": 42.3, "5": 37.3},
    "agreement_8L_pooled_i2_5": {
        "n": 13613, "agreement_pct": 77.8,
        "generation_only_pct": 16.6, "patchscopes_only_pct": 5.6,
        "both_pct": 52.0, "neither_pct": 25.8,
        "patchscopes_pct": 57.6, "generation_pct": 68.5,
        "note": ("Computed on the deduplicated set by matching the two feature "
                 "files on (phrase, instance, i); all four cells are measured, "
                 "not derived from rounded percentages."),
    },
    "category_8L_pooled_i2_5_pct": {"building": 46.4, "idiom": 58.0, "movie": 59.9},
    "generation_confidence": {
        "definition": ("Geometric mean of the probability the model assigned to "
                       "each token it actually generated, over the steps needed "
                       "to complete the phrase."),
        "mean_when_success": 0.877, "mean_when_failure": 0.450,
        "median_when_success": 0.934, "median_when_failure": 0.419,
        "note": ("The breakdown by probe outcome (TP/FN/FP/TN) is not reported: "
                 "it was not recomputed on the deduplicated set."),
    },
    "false_positive_llm_judgement": {
        "n": 212, "valid_alternate_pct": 59.9, "partially_right_pct": 22.2, "unrelated_pct": 17.9,
        "counts": {"valid_alternate": 127, "partially_right": 47, "unrelated": 38},
        "judge": ("ChatGPT, GPT-5.6 Sol, run on the deduplicated set in a single "
                  "batch. The prompt is reproduced verbatim in the appendix."),
    },
    "false_positive_by_model_confidence": {
        "n": 212,
        "mean": 0.636, "median": 0.681,
        "model_confident_gt_0_7_pct": 46.2,
        "model_unsure_lt_0_3_pct": 7.1,
        "middle_pct": 46.7,
        "note": "Independent of the LLM adjudication; splits the same 212 cases by the model's own confidence.",
    },
    "qwen25_14b_generation_pct": {
        "isolated": {"1": 82.1, "2": 54.3, "3": 28.4, "4": 18.8, "5": 17.4, "6": 11.8, "7": 13.9, "8": 15.4},
        "note": ("Isolated only. The in-context replication was run on the "
                 "pre-deduplication contexts and is not reported."),
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {OUT}")


# ---- Additional externally recorded results -------------------------------
NUM = json.load(open(OUT))
NUM["external"]["clp_pilot"] = {
    "note": ("Separate reproduction of CLP (Xie and Zhou, 2026) on Qwen3.5-2B "
             "with the adaptive-mtp toolkit. Not a result of this paper; cited "
             "in the Conclusion only. Source: "
             "sources/notes/clp_reproduction_summary.pdf, which reports the "
             "speed-ups. The test-set size of ten prompts is from the author's "
             "run and is not stated in that summary."),
    "test_prompts": 10,
    "speedup_vs_autoregressive": {
        "plain_autoregressive": 1.00, "entropy": 1.30, "max_probability": 1.37,
        "margin": 1.35, "history": 1.42, "clp": 1.34,
        "fixed_k4": 1.72, "fixed_k2": 1.57,
    },
    "tokens_per_second": {
        "plain_autoregressive": 45.6, "entropy": 59.1, "max_probability": 62.4,
        "margin": 61.7, "history": 64.9, "clp": 61.0,
        "fixed_k4": 78.2, "fixed_k2": 71.5,
    },
    "mismatched_prompts_out_of_10": {"clp": 2, "fixed_k": 6},
    "clp_paper_reported_speedup_range": [1.14, 1.29],
    "correctness_caveat": ("Divergences from autoregressive output appeared for "
                           "every drafting policy tested, including fixed-length "
                           "ones with no adaptive logic, so the cause appears to "
                           "be the model's interaction with the inference engine "
                           "rather than any policy. Frequency scaled with how "
                           "aggressively a policy drafted."),
}

with open(OUT, "w") as f:
    json.dump(NUM, f, indent=2)
print("appended external results")
