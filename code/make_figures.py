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


# Fig 1: isolation vs context, both readouts
fig, ax = plt.subplots(figsize=(3.2, 2.1))
x = range(len(IS))
iso_g = [NUM["isolated"]["generation_olmo2"][str(i)][2] for i in IS]
ctx_g = [NUM["in_context"]["generation_olmo2"][str(i)][2] for i in IS]
iso_p = [NUM["isolated"]["patchscopes_olmo2_32L"][str(i)][2] for i in IS]
ctx_p = [NUM["in_context"]["patchscopes_olmo2_32L"][str(i)][2] for i in IS]
ax.plot(x, ctx_g, "o-", color=C["gen"], label="generation, in context")
ax.plot(x, iso_g, "o--", color=C["gen"], alpha=.55, label="generation, isolated")
ax.plot(x, ctx_p, "s-", color=C["ps"], label="patchscopes, in context")
ax.plot(x, iso_p, "s--", color=C["ps"], alpha=.55, label="patchscopes, isolated")
ax.set_xticks(list(x)); ax.set_xticklabels([f"$i={i}$" for i in IS])
ax.set_ylabel("recovery rate (%)"); ax.set_ylim(0, 100)
ax.set_xlabel("lookahead distance")
ax.grid(axis="y", lw=.4, alpha=.3); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
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
