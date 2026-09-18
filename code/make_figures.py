"""Generate the paper's figures from results/raw/ into figures/."""
import json, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, FIG = Path("results/raw"), Path("figures")
FIG.mkdir(exist_ok=True)
NUM = json.load(open("results/processed/paper_numbers.json"))

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"ps": "#3a6ea5", "gen": "#c1671a", "alt": "#4a7c59", "grey": "#7a7a7a"}
IS = [2, 3, 4, 5]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("  wrote figures/" + name + ".pdf")


# Fig 1: context effect (generation) + Patchscopes layer-budget sensitivity
fig, ax = plt.subplots(figsize=(3.2, 2.1))
x = list(range(len(IS)))
iso_g = [NUM["isolated"]["generation_olmo2"][str(i)][2] for i in IS]
ctx_g = [NUM["in_context"]["generation_olmo2"][str(i)][2] for i in IS]
ctx_p8 = [NUM["external"]["patchscopes_8L_union_in_context_pct"][str(i)] for i in IS]
ctx_p32 = [NUM["in_context"]["patchscopes_olmo2_32L"][str(i)][2] for i in IS]
ax.plot(x, ctx_g, "o-", color=C["gen"], label="generation, in context")
ax.plot(x, iso_g, "o--", color=C["gen"], alpha=.5, label="generation, isolated")
ax.plot(x, ctx_p8, "s-", color=C["ps"], label="patchscopes (8L), in context")
ax.plot(x, ctx_p32, "s:", color=C["ps"], alpha=.6, label="patchscopes (32L), in context")
ax.set_xticks(x); ax.set_xticklabels([f"$i={i}$" for i in IS])
ax.set_ylabel("recovery rate (%)"); ax.set_ylim(0, 100)
ax.set_xlabel("lookahead distance")
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", fontsize=6)
save(fig, "context_vs_isolation")

# Fig 2: per-phrase consistency on distinct contexts, vs binomial null
import collections, math
ctx = json.load(open(RAW / "lookahead_with_context_full.json"))
gen = json.load(open(RAW / "generation_truth_context.json"))["per_phrase"]
by = collections.defaultdict(dict)
for base, r in zip(ctx, gen):
    if "3" in r["per_i"]:
        by[base["phrase"]][base["context_before"]] = bool(r["per_i"]["3"]["success"])
sel = {p_: list(d.values()) for p_, d in by.items() if len(d) >= 5}
n = len(sel)
pool = sum(sum(v) for v in sel.values()) / sum(len(v) for v in sel.values())
edges = [i / 10 for i in range(11)]
obs = [0.0] * 10
exp = [0.0] * 10
for v in sel.values():
    m = len(v)
    obs[min(int(sum(v) / m * 10), 9)] += 1
    for k in range(m + 1):
        pk = math.comb(m, k) * pool ** k * (1 - pool) ** (m - k)
        exp[min(int(k / m * 10), 9)] += pk
fig, ax = plt.subplots(figsize=(3.2, 2.0))
xs = [i + .5 for i in range(10)]
ax.bar(xs, [100 * o / n for o in obs], width=.85, color=C["gen"], label="observed")
ax.step([i for i in range(11)], [100 * e / n for e in exp] + [100 * exp[-1] / n],
        where="post", color=C["grey"], lw=1.2, ls="--",
        label=f"binomial null ($p={pool:.2f}$)")
ax.set_xticks(range(0, 11, 2))
ax.set_xticklabels([f"{10*i}%" for i in range(0, 11, 2)])
ax.set_xlabel("share of a phrase's contexts recovered")
ax.set_ylabel("% of phrases")
ax.set_ylim(0, 39)
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
save(fig, "per_phrase_consistency")

# Fig 3: per-layer patchscopes success
fig, ax = plt.subplots(figsize=(3.2, 2.0))
L = NUM["external"]["layers_probed"]
per = NUM["external"]["patchscopes_per_layer_success_pct"]
for i, style in zip(["2", "3", "4", "5"], ["o-", "s-", "^-", "v-"]):
    ax.plot(L, [per[i][str(l)] for l in L], style, ms=3,
            label=f"$i={i}$", alpha=.9)
ax.set_xlabel("layer"); ax.set_ylabel("patchscopes success (%)")
ax.set_xticks(L); ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False)
save(fig, "patchscopes_by_layer")

# Fig 4: the two probes, balanced accuracy by layer
fig, ax = plt.subplots(figsize=(3.2, 2.0))
ps = NUM["external"]["patchscopes_label_probe"]["balanced_accuracy_pct"]
ge = NUM["external"]["generation_label_probe"]["balanced_accuracy_pct"]
ax.plot(L, [ps[str(l)] for l in L], "s-", ms=3, color=C["ps"], label="patchscopes label")
ax.plot(L, [ge[str(l)] for l in L], "o-", ms=3, color=C["gen"], label="generation label")
ax.axhline(50, ls=":", lw=.8, color=C["grey"])
ax.text(30, 51, "chance", ha="right", fontsize=6, color=C["grey"])
ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy (%)")
ax.set_xticks(L); ax.set_ylim(45, 85)
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
save(fig, "probe_balanced_accuracy")

# Fig 5: first successful layer distribution
fig, ax = plt.subplots(figsize=(3.2, 1.8))
hist = NUM["first_successful_layer"]["histogram"]
ls = sorted(int(k) for k in hist)
tot = NUM["first_successful_layer"]["n_successes"]
ax.bar(ls, [100 * hist[str(l)] / tot for l in ls], color=C["alt"], width=.8)
ax.set_xlabel("first layer at which the patchscope succeeds")
ax.set_ylabel("% of successes")
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
save(fig, "first_successful_layer")

print("done")


# Fig 6: per-phrase correlation between probe confidence and true difficulty
corr = json.load(open(RAW / "correlation_per_i.json"))
Ls = sorted(corr, key=int)
fig, ax = plt.subplots(figsize=(3.2, 1.9))
for key, lab, st in (("3", "$i=3$", "o-"), ("all", "pooled $i=2\\ldots5$", "s--")):
    ax.plot([int(l) for l in Ls], [corr[l][key][0] for l in Ls], st, ms=3,
            label=lab, color=C["alt"] if key == "3" else C["grey"],
            alpha=1 if key == "3" else .75)
ax.set_xlabel("layer"); ax.set_ylabel("Pearson $r$")
ax.set_xticks([int(l) for l in Ls]); ax.set_ylim(0, .75)
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center")
save(fig, "probe_phrase_correlation")


# Fig 7: model confidence by probe outcome, and false-positive adjudication
conf = NUM["external"]["generation_confidence"]["by_probe_outcome_L25"]
fp = NUM["external"]["false_positive_llm_judgement"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.95), gridspec_kw={"width_ratios": [1.15, 1]})
a = axes[0]
order = [("true_positive", "TP"), ("false_negative", "FN"),
         ("false_positive", "FP"), ("true_negative", "TN")]
cols = ["#2e8b57", "#7fb89a", C["gen"], C["grey"]]
a.bar(range(4), [conf[k]["mean"] for k, _ in order], color=cols, width=.66)
for j, (k, lab) in enumerate(order):
    a.text(j, conf[k]["mean"] + .02, f'{conf[k]["mean"]:.3f}', ha="center", fontsize=6.5)
a.set_xticks(range(4))
a.set_xticklabels([f'{lab}\n$n$={conf[k]["n"]:,}' for k, lab in order], fontsize=6.5)
a.set_ylabel("mean model confidence"); a.set_ylim(0, 1.05)
a.axvline(1.5, ls=":", lw=.8, color="#c9c9c9")
a.text(0.5, 1.0, "model succeeded", ha="center", fontsize=6.3, color=C["grey"], style="italic")
a.text(2.5, 1.0, "model failed", ha="center", fontsize=6.3, color=C["grey"], style="italic")
a.grid(axis="y", lw=.4, alpha=.3); a.set_axisbelow(True)

b = axes[1]
segs = [("valid_alternate_pct", "valid alternate", "#2e8b57"),
        ("partially_right_pct", "partially right", "#c9a227"),
        ("unrelated_pct", "unrelated", C["grey"])]
left = 0
for k, lab, col in segs:
    v = fp[k]
    b.barh(0, v, left=left, color=col, height=.5, label=lab)
    b.text(left + v / 2, 0, f"{v:.1f}%", ha="center", va="center",
           fontsize=6.8, color="white", fontweight="bold")
    left += v
b.set_xlim(0, 100); b.set_ylim(-.75, .95); b.set_yticks([])
b.legend(frameon=False, fontsize=6.3, ncol=3, loc="upper center",
         handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
b.set_xlabel(f'all {fp["n"]} probe false positives, adjudicated')
for sp in ("left", "right", "top"):
    b.spines[sp].set_visible(False)
save(fig, "confidence_and_falsepos")


# Fig 8: probe at the earliest layers (learned signal, not base rate)
ge = NUM["external"]["generation_label_probe"]
early = ge["early_layers_balanced_accuracy_pct"]
eprec = ge["early_layers_precision_pct"]
keys = ["0", "1", "2", "5"]
fig, ax = plt.subplots(figsize=(3.2, 1.9))
xs = range(len(keys) + 1)
bacc = [early[k] for k in keys] + [ge["balanced_accuracy_pct"]["30"]]
prec = [eprec[k] for k in keys] + [ge["precision_pct"]["30"]]
w = .38
ax.bar([x - w / 2 for x in xs], bacc, width=w, color=C["gen"], label="balanced acc.")
ax.bar([x + w / 2 for x in xs], prec, width=w, color=C["ps"], label="precision")
ax.axhline(50, ls=":", lw=.8, color=C["grey"])
ax.text(-0.42, 51.5, "chance", ha="left", fontsize=6, color=C["grey"])
ax.set_xticks(list(xs))
ax.set_xticklabels([f"L{k}" for k in keys] + ["L30\n(best)"], fontsize=6.8)
ax.set_ylabel("%"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=6.3, ncol=2, loc="upper center")
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
save(fig, "probe_early_layers")
