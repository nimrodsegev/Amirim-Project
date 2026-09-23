"""Protocol schematic: lookahead distance and the two readouts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 7.4,
                     "figure.dpi": 300, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02})
PS, GEN, GREY, CUT = "#3a6ea5", "#c1671a", "#8a8a8a", "#b03a3a"
INK = "#1c1c1c"

fig, ax = plt.subplots(figsize=(7.0, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=GREY, fs=7.4, weight="normal",
        tc=INK, style="round,pad=0.1,rounding_size=0.6", lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight)


def arrow(x1, y1, x2, y2, color=GREY, style="-|>", lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=8, color=color, lw=lw,
                                 linestyle=ls, shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---- input sequence -------------------------------------------------------
ax.text(1, 30.6, "context from FineWeb-Edu", fontsize=6.8, color=GREY, style="italic")
box(1, 24.5, 20, 5, "...films such as", fc="#f2f2f2", ec="#d8d8d8")

toks = ["The", "Lord", "of", "the", "Rings"]
x0, w, gap = 23, 8.4, 1.0
read_n = 2   # "The Lord" has been read; cut after it
for k, t in enumerate(toks):
    x = x0 + k * (w + gap)
    is_read = k < read_n
    box(x, 24.5, w, 5, t,
        fc="#e8eef6" if is_read else "white",
        ec=PS if is_read else "#d8d8d8",
        tc=INK if is_read else GREY,
        weight="bold" if is_read else "normal")

cut_x = x0 + read_n * (w + gap) - gap / 2
ax.plot([cut_x, cut_x], [23.2, 31.2], color=CUT, lw=1.3, ls=(0, (3, 2)))
ax.text(cut_x - 0.8, 31.6, "cut", fontsize=6.8, color=CUT, ha="right", weight="bold")

# brace over the unread tokens
unread_l = x0 + read_n * (w + gap)
unread_r = x0 + len(toks) * (w + gap) - gap
ax.annotate("", xy=(unread_l, 23.0), xytext=(unread_r, 23.0),
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.8,
                            connectionstyle="bar,fraction=0.18"))
ax.text((unread_l + unread_r) / 2, 19.6,
        r"target: $i=3$ tokens still unread", fontsize=7.2, ha="center", color=INK)

# the state we read
ax.plot([cut_x - w / 2 - gap / 2, cut_x - w / 2 - gap / 2], [24.3, 17.6],
        color=PS, lw=0.9)
ax.text(cut_x - w / 2 - gap / 2 - 1.2, 21.0, r"$h^{\ell}_p$", fontsize=8,
        color=PS, ha="right", va="center")

# ---- the two readouts -----------------------------------------------------
box(3, 7.4, 42, 7.2, "", fc="#fbfbfb", ec="#e2e2e2")
ax.text(4.8, 12.9, "Readout 1: Patchscopes", fontsize=7.4, color=PS, weight="bold")
box(4.8, 8.2, 21, 3.6, 'Repeat this: [$h^{\\ell}_p$]', fc="white", ec=PS)
arrow(26.4, 10.0, 30.0, 10.0, color=PS)
box(30.2, 8.2, 13.6, 3.6, '"of the Rings"', fc="white", ec="#d8d8d8")

box(52, 7.4, 45, 7.2, "", fc="#fbfbfb", ec="#e2e2e2")
ax.text(53.8, 12.9, "Readout 2: generation", fontsize=7.4, color=GEN, weight="bold")
box(53.8, 8.2, 24, 3.6, "...films such as The Lord", fc="white", ec=GEN)
arrow(78.4, 10.0, 82.0, 10.0, color=GEN)
box(82.2, 8.2, 13.6, 3.6, '"of the Rings"', fc="white", ec="#d8d8d8")

jx = cut_x - w / 2 - gap / 2
ax.plot([jx, jx], [17.6, 16.3], color=PS, lw=0.9)
ax.plot([20, jx], [17.6, 17.6], color=PS, lw=0.9)
ax.plot([jx, 70], [17.6, 17.6], color=GEN, lw=0.9)
arrow(20, 17.6, 20, 14.9, color=PS)
arrow(70, 17.6, 70, 14.9, color=GEN)

# ---- verdict --------------------------------------------------------------
ax.text(50, 4.4, "success $=$ the target string appears in the output "
                 "(case-insensitive, budget $i+3$ tokens)",
        fontsize=7.0, ha="center", color=INK)
ax.text(50, 1.4, "Patchscopes asks what the vector contains; generation asks "
                 "what the model does with it.",
        fontsize=6.9, ha="center", color=GREY, style="italic")

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/protocol.pdf", metadata={"CreationDate": None})
print("wrote figures/protocol.pdf")


# ---------------------------------------------------------------- motivation
# Slide 3: the cost argument. Full-depth decoding of a five-token phrase
# against a hypothetical read-ahead that skips most of it.
fig, ax = plt.subplots(figsize=(3.3, 2.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
TOKS = ["The", "Lord", "of", "the", "Rings"]
BW, GAP, X0, BARH = 14.5, 2.2, 6.0, 12.0


def unit(x, base, tok, n_layers, color):
    """Token box with a depth bar beneath it."""
    h = BARH * (n_layers / 32.0)
    ax.add_patch(FancyBboxPatch((x, base + BARH + 1.0), BW, 5.4,
                                boxstyle="round,pad=0.1,rounding_size=0.5",
                                facecolor="white", edgecolor="#d8d8d8", linewidth=0.7))
    ax.text(x + BW / 2, base + BARH + 3.7, tok, ha="center", va="center",
            fontsize=6.6, color=INK)
    ax.add_patch(FancyBboxPatch((x, base), BW, h,
                                boxstyle="round,pad=0.04,rounding_size=0.3",
                                facecolor=color, edgecolor="none", alpha=.88))
    ax.text(x + BW / 2, base + h - 2.4, str(n_layers), ha="center", va="center",
            fontsize=5.8, color="white", fontweight="bold")


# --- row 1: every token pays full depth
B1 = 29
ax.text(0.5, B1 + BARH + 8.6, "one token at a time", fontsize=6.9,
        color=INK, weight="bold")
ax.text(99, B1 + BARH + 8.6, "160 layer-passes", fontsize=7.0, color=INK,
        ha="right", weight="bold")
for k, t in enumerate(TOKS):
    unit(X0 + k * (BW + GAP), B1, t, 32, PS)

# --- row 2: read the rest out of the state at 'Lord'
B2 = 3
ax.text(0.5, B2 + BARH + 8.6, "if the rest is already encoded", fontsize=6.9,
        color=GEN, weight="bold")
ax.text(99, B2 + BARH + 8.6, "45 layer-passes", fontsize=7.0, color=GEN,
        ha="right", weight="bold")
for k, (t, L) in enumerate(zip(TOKS[:2], [32, 13])):
    unit(X0 + k * (BW + GAP), B2, t, L, GEN)
rx = X0 + 2 * (BW + GAP)
rw = BW * 3 + GAP * 2
ax.add_patch(FancyBboxPatch((rx, B2 + 1.4), rw, BARH + 4.0,
                            boxstyle="round,pad=0.1,rounding_size=0.6",
                            facecolor="#fbf4ee", edgecolor=GEN,
                            linewidth=0.8, linestyle=(0, (2.5, 1.8))))
ax.text(rx + rw / 2, B2 + 10.6, "``of the Rings\'\'", ha="center", va="center",
        fontsize=6.8, color=INK)
ax.text(rx + rw / 2, B2 + 6.0, "read from the hidden state", ha="center",
        va="center", fontsize=6.0, color=GEN, style="italic")

fig.savefig("figures/motivation.pdf", metadata={"CreationDate": None})
plt.close(fig)
print("wrote figures/motivation.pdf")
