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
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print("  wrote figures/" + name + ".pdf")


# Fig 1: what natural context adds, for both readouts, over i = 2..5
iso_p = NUM["isolated"]["patchscopes_olmo2_32L"]
iso_g = NUM["isolated"]["generation_olmo2"]
ctx_g = NUM["in_context"]["generation_olmo2"]
ctx_p32 = NUM["in_context"]["patchscopes_olmo2_32L"]

fig, ax = plt.subplots(figsize=(3.3, 2.15))
xs = range(len(IS))
ax.plot(xs, [ctx_g[str(i)][2] for i in IS], "o-", ms=3.4, color=C["gen"],
        label="Generation, in context")
ax.plot(xs, [ctx_p32[str(i)][2] for i in IS], "s-", ms=3.4, color=C["ps"],
        label="Patchscopes, in context")
ax.plot(xs, [iso_g[str(i)][2] for i in IS], "o--", ms=3.4, color=C["gen"],
        alpha=.5, label="Generation, alone")
ax.plot(xs, [iso_p[str(i)][2] for i in IS], "s--", ms=3.4, color=C["ps"],
        alpha=.5, label="Patchscopes, alone")
ax.set_xticks(list(xs)); ax.set_xticklabels([f"${i}$" for i in IS])
ax.set_xlabel("Lookahead distance $i$")
ax.set_ylabel("Recovery rate (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.1, loc="upper right")
save(fig, "context_vs_isolation")

# Fig 2: per-phrase consistency on distinct contexts, vs binomial null.
# Bins separate the two extremes explicitly: most phrases have exactly five
# distinct contexts, so a uniform decile grid would alias badly.
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

LABELS = ["None", "1-25%", "25-50%", "50-75%", "75-99%", "All"]


def which(frac):
    if frac <= 0: return 0
    if frac >= 1: return 5
    if frac <= .25: return 1
    if frac <= .50: return 2
    if frac <= .75: return 3
    return 4


obs = [0.0] * 6
exp = [0.0] * 6
for v in sel.values():
    m = len(v)
    obs[which(sum(v) / m)] += 1
    for k in range(m + 1):
        exp[which(k / m)] += math.comb(m, k) * pool ** k * (1 - pool) ** (m - k)

fig, ax = plt.subplots(figsize=(3.2, 2.0))
xs = range(6)
w = .38
ax.bar([x - w / 2 for x in xs], [100 * o / n for o in obs], width=w,
       color=C["gen"], label="Observed")
ax.bar([x + w / 2 for x in xs], [100 * e / n for e in exp], width=w,
       color="#b9b9b9", label=f"Binomial null ($p={pool:.2f}$)")
ax.set_xticks(list(xs)); ax.set_xticklabels(LABELS, fontsize=6.4)
ax.set_xlabel("Share of a phrase's contexts recovered")
ax.set_ylabel("Phrases (%)")
ax.legend(frameon=False, fontsize=6.4)
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
save(fig, "per_phrase_consistency")

# Fig 3: per-layer patchscopes success
fig, ax = plt.subplots(figsize=(3.2, 2.0))
L = NUM["external"]["layers_probed"]
per = NUM["external"]["patchscopes_per_layer_success_pct"]
for i, style in zip(["2", "3", "4", "5"], ["o-", "s-", "^-", "v-"]):
    ax.plot(L, [per[i][str(l)] for l in L], style, ms=3,
            label=f"$i={i}$", alpha=.9)
ax.set_xlabel("Layer"); ax.set_ylabel("Patchscopes success (%)")
ax.set_xticks(L); ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False)
save(fig, "patchscopes_by_layer")

# Fig 4: the two probes, balanced accuracy by layer
fig, ax = plt.subplots(figsize=(3.2, 2.25))
ps = NUM["external"]["patchscopes_label_probe"]["balanced_accuracy_pct"]
ge = NUM["external"]["generation_label_probe"]["balanced_accuracy_pct"]
base = NUM["external"]["patchscopes_label_probe"]["majority_baseline_pct"]
ax.plot(L, [ps[str(l)] for l in L], "s-", ms=3, color=C["ps"], label="Patchscopes label")
ax.plot(L, [ge[str(l)] for l in L], "o-", ms=3, color=C["gen"], label="Generation label")
ax.plot(L, [base[str(l)] for l in L], "^--", ms=2.6, color=C["grey"], alpha=.75,
        label="Majority-class baseline (raw acc.)")
ax.axhline(50, ls=":", lw=.8, color=C["grey"])
ax.text(5.4, 52.2, "Chance", ha="left", fontsize=6, color=C["grey"])
ax.set_xlabel("Layer"); ax.set_ylabel("%")
ax.set_xticks(L); ax.set_ylim(40, 100)
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, -0.26), handlelength=1.6, columnspacing=1.4)
save(fig, "probe_balanced_accuracy")

# Fig 5: first successful layer distribution
fig, ax = plt.subplots(figsize=(3.2, 1.8))
hist = NUM["first_successful_layer"]["histogram"]
ls = sorted(int(k) for k in hist)
tot = NUM["first_successful_layer"]["n_successes"]
ax.bar(ls, [100 * hist[str(l)] / tot for l in ls], color=C["alt"], width=.8)
ax.set_xlabel("First layer at which the patchscope succeeds")
ax.set_ylabel("Successes (%)")
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
save(fig, "first_successful_layer")

print("done")


# Fig 6: per-phrase correlation between probe confidence and true difficulty
corr = json.load(open(RAW / "correlation_per_i.json"))
Ls = sorted(corr, key=int)
fig, ax = plt.subplots(figsize=(3.2, 1.9))
for key, lab, st in (("3", "$i=3$", "o-"), ("all", "Pooled $i=2\\ldots5$", "s--")):
    ax.plot([int(l) for l in Ls], [corr[l][key][0] for l in Ls], st, ms=3,
            label=lab, color=C["alt"] if key == "3" else C["grey"],
            alpha=1 if key == "3" else .75)
ax.set_xlabel("Layer"); ax.set_ylabel("Pearson $r$")
ax.set_xticks([int(l) for l in Ls]); ax.set_ylim(0, .75)
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center")
save(fig, "probe_phrase_correlation")


# Fig 7: model confidence by probe outcome, and false-positive adjudication
conf = NUM["external"]["generation_confidence"]
fpc = NUM["external"]["false_positive_by_model_confidence"]
fp = NUM["external"]["false_positive_llm_judgement"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 1.95), gridspec_kw={"width_ratios": [1.15, 1]})
a = axes[0]
order = [("mean_when_success", "Recovered"), ("mean_when_failure", "Not recovered"),
         (None, "Probe false\npositives")]
vals = [conf["mean_when_success"], conf["mean_when_failure"], fpc["mean"]]
cols = ["#2e8b57", C["grey"], C["gen"]]
a.bar(range(3), vals, color=cols, width=.6)
for j, v in enumerate(vals):
    a.text(j, v + .02, f"{v:.3f}", ha="center", fontsize=6.5)
a.set_xticks(range(3))
a.set_xticklabels([lab for _, lab in order], fontsize=6.5)
a.set_ylabel("Mean model confidence"); a.set_ylim(0, 1.05)
a.grid(axis="y", lw=.4, alpha=.3); a.set_axisbelow(True)

b = axes[1]
segs = [("valid_alternate_pct", "Valid alternate", "#2e8b57"),
        ("partially_right_pct", "Partially right", "#c9a227"),
        ("unrelated_pct", "Unrelated", C["grey"])]
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
b.set_xlabel(f'All {fp["n"]} probe false positives, adjudicated')
for sp in ("left", "right", "top"):
    b.spines[sp].set_visible(False)
save(fig, "confidence_and_falsepos")


# Fig 8 (probe at the earliest layers) is not produced: the generation-label
# early-layer probe was not rerun on the deduplicated set.

