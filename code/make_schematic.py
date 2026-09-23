"""Protocol and motivation schematics."""
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
Path("figures").mkdir(exist_ok=True)

# ----------------------------------------------------------------- protocol
fig, ax = plt.subplots(figsize=(7.0, 2.1))
ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec="#d5d5d5", fs=7.2, weight="normal",
        tc=INK, lw=0.75, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.1,rounding_size=0.55",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight)


def arrow(x1, y1, x2, y2, color=GREY, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7.5, color=color, lw=lw,
                                 shrinkA=1.5, shrinkB=1.5))


# --- the input strip -------------------------------------------------------
ax.text(1, 27.6, "context from FineWeb-Edu", fontsize=6.3, color=GREY,
        style="italic")
box(1, 21.4, 17.5, 4.6, "\u2026films such as", fc="#f4f4f4", ec="#e2e2e2",
    tc=GREY, fs=7.0)

toks, TW, TG, TX = ["The", "Lord", "of", "the", "Rings"], 7.4, 0.9, 20.2
for k, t in enumerate(toks):
    x = TX + k * (TW + TG)
    read = k < 2
    box(x, 21.4, TW, 4.6, t,
        fc="#e7eef7" if read else "white",
        ec=PS if read else "#dcdcdc",
        tc=INK if read else GREY,
        weight="bold" if read else "normal")

cut_x = TX + 2 * (TW + TG) - TG / 2
ax.plot([cut_x, cut_x], [20.0, 27.4], color=CUT, lw=1.2, ls=(0, (2.6, 1.8)))
ax.text(cut_x - 0.9, 27.9, "cut", fontsize=6.3, color=CUT, ha="right",
        weight="bold")

ur = TX + 5 * (TW + TG) - TG
ax.plot([cut_x + 0.6, ur], [19.8] * 2, color="#bdbdbd", lw=0.75)
for xe in (cut_x + 0.6, ur):
    ax.plot([xe, xe], [19.8, 18.9], color="#bdbdbd", lw=0.75)
ax.text((cut_x + ur) / 2, 17.2, "target: the $i=3$ unread tokens",
        fontsize=6.5, ha="center", va="center", color=INK)

# the state we lift, and where it goes
hx = cut_x - TW / 2 - TG / 2
ax.plot([hx, hx], [21.2, 13.2], color=PS, lw=0.9)
ax.text(hx - 1.3, 17.6, "$h^{\\ell}_p$", fontsize=8.2, color=PS, ha="right",
        va="center")
ax.plot([16, 62], [13.2, 13.2], color="#c6c6c6", lw=0.8)
arrow(16, 13.2, 16, 10.4, color=PS)
arrow(62, 13.2, 62, 10.4, color=GEN)

# --- the two readouts ------------------------------------------------------
ax.text(1, 9.0, "1", fontsize=6.4, color="white", ha="center", va="center",
        weight="bold",
        bbox=dict(boxstyle="circle,pad=0.22", fc=PS, ec="none"))
ax.text(4.0, 9.0, "Patchscopes", fontsize=7.3, color=PS, weight="bold",
        va="center")
box(1, 2.4, 25.5, 4.6, "Repeat this: [$h^{\\ell}_p$]", ec=PS)
arrow(27.2, 4.7, 31.0, 4.7, color=PS)
box(31.4, 2.4, 15.0, 4.6, "\u201cof the Rings\u201d", ec="#dcdcdc")

ax.text(52, 9.0, "2", fontsize=6.4, color="white", ha="center", va="center",
        weight="bold",
        bbox=dict(boxstyle="circle,pad=0.22", fc=GEN, ec="none"))
ax.text(55.0, 9.0, "generation", fontsize=7.3, color=GEN, weight="bold",
        va="center")
box(52, 2.4, 28.0, 4.6, "\u2026films such as The Lord", ec=GEN)
arrow(80.7, 4.7, 84.5, 4.7, color=GEN)
box(84.9, 2.4, 15.0, 4.6, "\u201cof the Rings\u201d", ec="#dcdcdc")

ax.text(49.2, 5.8, "", ha="center")
ax.plot([49.2, 49.2], [1.2, 10.4], color="#e4e4e4", lw=0.7)

ax.text(1, 0.0, "success $=$ the target string appears in the output, "
                "case-insensitively, within $i+3$ generated tokens",
        fontsize=6.4, color=GREY, ha="left")

fig.savefig("figures/protocol.pdf", metadata={"CreationDate": None})
plt.close(fig)
print("wrote figures/protocol.pdf")

# ---------------------------------------------------------------- motivation
# Slide 3: the cost argument, as two stacked rows of layer columns.
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")
TOKS = ["The", "Lord", "of", "the", "Rings"]
CW, GAP, X0, COLH = 10.4, 2.6, 2.0, 14.0
LIGHT_PS, LIGHT_GEN = "#dce6f2", "#f7e6d6"


def column(x, base, tok, n_layers, edge, fill):
    h = COLH * (n_layers / 32.0)
    ax.text(x + CW / 2, base + COLH + 3.2, tok, ha="center", va="center",
            fontsize=6.7, color=INK)
    ax.add_patch(FancyBboxPatch((x, base), CW, h,
                                boxstyle="round,pad=0.05,rounding_size=0.4",
                                facecolor=fill, edgecolor=edge, linewidth=0.7))
    ax.plot([x + 0.6, x + CW - 0.6], [base + h - 0.8] * 2, color=edge, lw=1.1)
    ax.text(x + CW / 2, base - 3.2, str(n_layers), ha="center", va="center",
            fontsize=5.8, color=GREY)


SPAN = 5 * CW + 4 * GAP           # width of a full five-token row
TOTX = X0 + SPAN + 5.0            # left edge of the running-total label

# --- row 1: every token pays full depth
B1 = 38
ax.text(X0, B1 + COLH + 9.0, "one token at a time", ha="left", va="center",
        fontsize=6.9, color=INK, weight="bold")
for k, t in enumerate(TOKS):
    column(X0 + k * (CW + GAP), B1, t, 32, PS, LIGHT_PS)
ax.text(TOTX, B1 + COLH / 2, "160\nlayer-passes", ha="left", va="center",
        fontsize=7.0, color=INK, weight="bold", linespacing=1.3)

# --- row 2: the rest read out of the state at 'Lord'
B2 = 7
ax.text(X0, B2 + COLH + 9.0, "if the rest is already encoded", ha="left",
        va="center", fontsize=6.9, color=GEN, weight="bold")
for k, (t, L) in enumerate(zip(TOKS[:2], [32, 13])):
    column(X0 + k * (CW + GAP), B2, t, L, GEN, LIGHT_GEN)
rx = X0 + 2 * (CW + GAP)
rw = 3 * CW + 2 * GAP
ax.add_patch(FancyBboxPatch((rx, B2), rw, COLH,
                            boxstyle="round,pad=0.05,rounding_size=0.5",
                            facecolor="none", edgecolor=GEN,
                            linewidth=0.75, linestyle=(0, (2.6, 2.0))))
ax.text(rx + rw / 2, B2 + COLH * 0.62, "\u201cof the Rings\u201d",
        ha="center", va="center", fontsize=6.6, color=INK)
ax.text(rx + rw / 2, B2 + COLH * 0.26, "read from the state",
        ha="center", va="center", fontsize=5.6, color=GEN, style="italic")
ax.text(TOTX, B2 + COLH / 2, "45\nlayer-passes", ha="left", va="center",
        fontsize=7.0, color=GEN, weight="bold", linespacing=1.3)

fig.savefig("figures/motivation.pdf", metadata={"CreationDate": None})
plt.close(fig)
print("wrote figures/motivation.pdf")
