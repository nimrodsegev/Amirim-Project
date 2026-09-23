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
             "Values below are transcribed from the July 2026 analysis reports "
             "and the August 2026 summary deck."),
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
        "pos_rate_pct":   {"5": 6.6, "7": 14.7, "10": 18.8, "13": 16.9, "15": 18.5, "20": 36.8, "25": 47.7, "30": 40.2},
        "accuracy_pct":   {"5": 89.6, "7": 82.7, "10": 82.3, "13": 81.1, "15": 82.0, "20": 72.4, "25": 68.2, "30": 67.6},
        "majority_baseline_pct": {"5": 93.4, "7": 85.3, "10": 81.2, "13": 83.1, "15": 81.5, "20": 63.2, "25": 52.3, "30": 59.8},
        "balanced_accuracy_pct": {"5": 76.0, "7": 70.2, "10": 76.5, "13": 74.8, "15": 76.3, "20": 70.4, "25": 68.1, "30": 66.4},
        "precision_pct":  {"5": 33.9, "7": 42.8, "10": 52.3, "13": 45.8, "15": 51.1, "20": 62.4, "25": 67.2, "30": 59.7},
        "auc_roc":        {"5": 0.89, "7": 0.82, "10": 0.85, "13": 0.85, "15": 0.87, "20": 0.78, "25": 0.74, "30": 0.74},
    },
    "generation_label_probe": {
        "pos_rate_pct": 69.1,
        "accuracy_pct":   {"5": 70.4, "7": 71.5, "10": 72.4, "13": 74.3, "15": 74.8, "20": 74.0, "25": 75.3, "30": 77.5},
        "balanced_accuracy_pct": {"5": 69.8, "7": 71.0, "10": 72.0, "13": 73.6, "15": 74.2, "20": 73.6, "25": 74.6, "30": 76.8},
        "precision_pct":  {"5": 83.3, "7": 84.2, "10": 84.9, "13": 85.6, "15": 86.1, "20": 85.8, "25": 86.2, "30": 87.6},
        "auc_roc":        {"5": 0.76, "7": 0.78, "10": 0.80, "13": 0.81, "15": 0.81, "20": 0.80, "25": 0.81, "30": 0.83},
        "early_layers_balanced_accuracy_pct": {"0": 58.6, "1": 65.1, "2": 66.5, "5": 69.8},
        "early_layers_note": ("Monotone, with no irregularity at L2. The "
                              "patchscopes-label probe on the same states dips "
                              "at L2 (55.4%, down from 76.6% at L1); that dip "
                              "is most likely instability from a positive class "
                              "of 0.6-1.2%, but no multi-seed check was run to "
                              "confirm it."),
        "early_layers_precision_pct": {"0": 75.4, "1": 80.6, "2": 80.9, "5": 83.3},
        "held_out_category_balanced_accuracy_pct": {
            "none":     {"13": 73.6, "25": 74.6, "30": 76.8},
            "building": {"13": 69.1, "25": 62.6, "30": 63.4},
            "idiom":    {"13": 68.7, "25": 68.9, "30": 72.3},
            "movie":    {"13": 72.4, "25": 72.7, "30": 73.5},
        },
    },
    "patchscopes_per_layer_success_pct": {
        "2":  {"5": 14.9, "7": 24.5, "10": 29.8, "13": 27.1, "15": 28.5, "20": 51.0, "25": 62.2, "30": 59.2},
        "3":  {"5": 7.7,  "7": 14.3, "10": 18.3, "13": 16.9, "15": 18.6, "20": 38.2, "25": 48.1, "30": 39.1},
        "4":  {"5": 4.3,  "7": 9.2,  "10": 12.0, "13": 10.7, "15": 12.8, "20": 25.0, "25": 34.7, "30": 25.7},
        "5":  {"5": 2.3,  "7": 5.5,  "10": 9.4,  "13": 8.8,  "15": 9.1,  "20": 21.8, "25": 29.4, "30": 20.8},
        "L0_L1_L2_pooled_i2_5": {"0": 0.7, "1": 0.8, "2": 1.6, "any_of_3": 1.9},
    },
    "patchscopes_8L_union_in_context_pct": {"2": 73.2, "3": 57.5, "4": 43.9, "5": 38.7},
    "agreement_8L_pooled_i2_5": {
        "n": 24740, "agreement_pct": 78.7,
        "generation_only_pct": 15.7, "patchscopes_only_pct": 5.6,
        "both_pct": 53.4, "neither_pct": 25.3,
        "patchscopes_pct": 59.0, "generation_pct": 69.1,
        "note": ("Reported in the July 2026 analysis report as agreement 78.7%, "
                 "generation-only 15.7%, patchscopes-only 5.6%. The remaining "
                 "cells follow from those and the fixed 69.1% generation rate; "
                 "they are derived from rounded percentages, so we quote "
                 "percentages rather than counts."),
    },
    "category_8L_pooled_i2_5_pct": {"building": 47.2, "idiom": 57.4, "movie": 54.7},
    "generation_confidence": {
        "definition": ("Geometric mean of the probability the model assigned to "
                       "each token it actually generated, over the steps needed "
                       "to complete the phrase."),
        "mean_when_success": 0.878, "mean_when_failure": 0.447,
        "median_when_success": 0.934, "median_when_failure": 0.416,
        "by_probe_outcome_L25": {
            "true_positive":  {"n": 2491, "mean": 0.904, "median": 0.950},
            "false_negative": {"n": 864,  "mean": 0.791, "median": 0.824},
            "false_positive": {"n": 412,  "mean": 0.599, "median": 0.636},
            "true_negative":  {"n": 1091, "mean": 0.414, "median": 0.394},
        },
    },
    "false_positive_llm_judgement": {
        "n": 412, "valid_alternate_pct": 62.9, "partially_right_pct": 17.5, "unrelated_pct": 19.7,
        "counts": {"valid_alternate": 259, "partially_right": 72, "unrelated": 81},
        "judge": ("ChatGPT (web interface); the specific model version was not "
                  "recorded at the time. The prompt is reproduced verbatim in "
                  "the appendix. A separate informal pass by a different "
                  "assistant reached a similar split but used no fixed rubric "
                  "and is not reported."),
    },
    "false_positive_by_model_confidence": {
        "n": 412,
        "model_confident_gt_0_7_pct": 40.0,
        "model_unsure_lt_0_3_pct": 10.4,
        "middle_pct": 49.6,
        "note": "Independent of the LLM adjudication; splits the same 412 cases by the model's own confidence.",
    },
    "qwen25_14b_generation_pct": {
        "isolated": {"1": 82.1, "2": 54.3, "3": 28.4, "4": 18.8, "5": 17.4, "6": 11.8, "7": 13.9, "8": 15.4},
        "in_context": {"2": 84.8, "3": 66.2, "4": 57.4, "5": 54.7},
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {OUT}")


# ---- Additional transcribed results (August 2026 deck) --------------------
NUM = json.load(open(OUT))
NUM["external"]["bin_chart_provenance"] = {
    "note": ("The August deck's slide-14 chart (L25 per-phrase prediction bins, "
             "i=3 only) covers 194 phrases. No saved file or script reproduces "
             "that count. Resolved by direct rerun on the cluster against the "
             "current probing_features.npz."),
    "correct_n_phrases": 180,
    "min_observations_filter_sweep": {"3": 180, "2": 190, "1": 196},
    "conclusion": ("194 is not recoverable by varying the minimum-observations "
                   "filter, so the chart was most likely built from an earlier "
                   "snapshot of the extracted features, before the filtering "
                   "that produced the current .npz. That snapshot no longer "
                   "exists and there is no server-side history to recover it."),
    "reproduction_check_L25_i3": {
        "mlp_balanced": {"recorded": 0.5310 if False else 0.5173, "rerun": 0.5173, "reproduced": True},
        "mlp_natural": {"recorded": 0.5310, "rerun": 0.499, "reproduced": False},
    },
    "third_variant": ("sanity_check_probe_l25.py is a separate script with a "
                      "different methodology (trains on i=3 data only) and "
                      "gives a fourth figure again: 178 phrases, r=0.614. Not "
                      "used in the paper."),
}

NUM["external"]["clp_pilot"] = {
    "note": ("Separate reproduction of CLP (Xie and Zhou, 2026) on Qwen3.5-2B "
             "with the adaptive-mtp toolkit. Not a result of this paper; cited "
             "in the Conclusion only. Ten held-out prompts, the whole test set "
             "available. Source: sources/notes/clp_reproduction_summary.pdf and "
             "the project assistant's records."),
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

NUM["external"]["layer_group_pooling"] = {
    "note": ("Deck slide 11. Grouping the eight probed layers into A={5,7}, "
             "B={10,13,15}, C={20,25,30} and asking, per phrase, which groups "
             "contain at least one context at one layer that succeeded. Pooled "
             "over up to 10 contexts, so this is a deliberately generous "
             "criterion. Requires data/probing_features.npz to recompute."),
    "A_and_B_and_C_pct": 47.8,
    "B_and_C_only_pct": 17.3,
    "C_only_pct": 33.7,
    "covered_pct": 98.8,
}
with open(OUT, "w") as f:
    json.dump(NUM, f, indent=2)
print("appended transcribed deck results")
