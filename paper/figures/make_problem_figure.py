# -*- coding: utf-8 -*-
"""Input -> Method -> Output explainer figure for the Introduction, in the
same visual language as AlphaMissense's own explanatory figure (Cheng et al.
2023, Fig. 1): a missense variant goes in, our pipeline's three information
sources are shown, a pathogenicity score comes out. Portrait, single-column
width, built with plain matplotlib so every element is placed by hand and
checked against its axis limits before being called finished."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np
import math

COL_TEXT = "#1B222B"
COL_MUTED = "#57646D"
COL_MUT = "#E0703C"
COL_MUT_EDGE = "#B44F22"
COL_NODE = "#93A8B3"
COL_NODE_EDGE = "#5B7A88"
COL_HERO = "#2E6F8E"
COL_PATHO = "#C0392B"
COL_BENIGN = "#2E6F8E"
COL_ARROW = "#57646D"
COL_A = "#2E6F8E"   # nucleotide colors
COL_C = "#C0392B"
COL_G = "#2E8B57"
COL_T = "#8B6D3F"
NUC_COLOR = {"A": COL_A, "C": COL_C, "G": COL_G, "T": COL_T}

fig = plt.figure(figsize=(4.3, 7.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def txt(x, y, s, size=9, color=COL_TEXT, ha="left", va="center", weight="normal",
        style="normal", family="sans-serif", zorder=3):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight,
            style=style, family=family, zorder=zorder)


def mono_seq(x0, y, seq, size=9.5, highlight_idx=None, highlight_color=COL_MUT, dx=0.028):
    for i, ch in enumerate(seq):
        col = highlight_color if (highlight_idx is not None and i == highlight_idx) else COL_TEXT
        w = "bold" if (highlight_idx is not None and i == highlight_idx) else "normal"
        txt(x0 + i * dx, y, ch, size=size, color=col, ha="center", family="monospace", weight=w)
    return x0 + (len(seq) - 1) * dx


def codon(x0, y, seq, size=12, highlight_idx=None, highlight_color=COL_MUT, dx=0.05):
    for i, ch in enumerate(seq):
        col = highlight_color if (highlight_idx is not None and i == highlight_idx) else NUC_COLOR[ch]
        txt(x0 + i * dx, y, ch, size=size, color=col, ha="center", family="monospace",
            weight="bold", style="italic")
    return x0 + (len(seq) - 1) * dx


def dashed_line(x0, x1, y):
    ax.plot([x0, x1], [y, y], color="#B9C4C8", linewidth=0.9, linestyle=(0, (3, 2)))


def down_arrow(x, y_top, y_bot, color=COL_ARROW, lw=1.3, scale=9):
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot), arrowstyle="-|>",
                                  mutation_scale=scale, linewidth=lw, color=color, zorder=5))


# ============================================================ INPUT ========
Y_SEC_INPUT = 0.975
txt(0.02, Y_SEC_INPUT, "Input", size=13, weight="bold")

Y_REF_LBL = 0.935
txt(0.02, Y_REF_LBL, "Reference:", size=9, color=COL_MUTED, style="italic")

Y_REF_DNA = 0.905
codon_end = codon(0.30, Y_REF_DNA, "GGT", size=12, dx=0.045)
dashed_line(codon_end + 0.06, 0.90, Y_REF_DNA)
txt(0.93, Y_REF_DNA, "DNA", size=8.5, color=COL_MUTED, ha="center")

Y_REF_PROT = 0.868
ref_seq = "MDVVAMVNGTVATMIS"
seq_end = mono_seq(0.05, Y_REF_PROT, ref_seq, highlight_idx=8, highlight_color=COL_TEXT, dx=0.0475)
dashed_line(seq_end + 0.05, 0.90, Y_REF_PROT)
txt(0.93, Y_REF_PROT, "Protein", size=8.5, color=COL_MUTED, ha="center")

down_arrow(0.5, 0.845, 0.805)

Y_VAR_LBL = 0.775
txt(0.02, Y_VAR_LBL, "Missense variant:", size=9, color=COL_MUTED, style="italic")

Y_VAR_DNA = 0.745
codon_end2 = codon(0.30, Y_VAR_DNA, "GTT", size=12, highlight_idx=1, dx=0.045)
dashed_line(codon_end2 + 0.06, 0.90, Y_VAR_DNA)
txt(0.93, Y_VAR_DNA, "DNA", size=8.5, color=COL_MUTED, ha="center")

Y_VAR_PROT = 0.708
var_seq = "MDVVAMVNVTVATMIS"
seq_end2 = mono_seq(0.05, Y_VAR_PROT, var_seq, highlight_idx=8, dx=0.0475)
dashed_line(seq_end2 + 0.05, 0.90, Y_VAR_PROT)
txt(0.93, Y_VAR_PROT, "Protein", size=8.5, color=COL_MUTED, ha="center")

down_arrow(0.5, 0.675, 0.638)

# ============================================================ METHOD =======
Y_SEC_METHOD = 0.615
txt(0.02, Y_SEC_METHOD, "This paper's approach", size=13, weight="bold")

Y_CIRC = 0.578
CX = [0.19, 0.5, 0.81]
LABELS = ["Structural contacts\n+ PAE confidence", "Protein language\nmodel (ESM-2)",
          "Graph attention\nnetwork (GATv2)"]
for i, (cx, lbl) in enumerate(zip(CX, LABELS), start=1):
    ax.add_patch(Circle((cx, Y_CIRC), 0.017, facecolor="white", edgecolor=COL_TEXT,
                         linewidth=1.1, zorder=6))
    txt(cx, Y_CIRC, str(i), size=8.5, ha="center", va="center", weight="bold", zorder=7)
    txt(cx, Y_CIRC - 0.032, lbl, size=6.9, ha="center", va="top", weight="bold")

Y_ICON_TOP = 0.50
Y_ICON_CY = 0.435
Y_ICON_BOT = 0.365

# --- icon 1: ring graph with PAE-weighted (varying) edges, mutated node at top ---
def ring_positions(cx, cy, r, n=7):
    pos = {}
    for i in range(n):
        a = math.pi / 2 - i * (2 * math.pi / n)
        pos[i] = (cx + r * math.cos(a), cy - r * math.sin(a) * 0.85)
    return pos

cx1 = CX[0]
r1 = 0.075
pos1 = ring_positions(cx1, Y_ICON_CY, r1, n=7)
ring_edges = [(i, (i + 1) % 7) for i in range(7)]
chord_edges = [(0, 3), (1, 5)]
edge_w = {(0, 1): 0.85, (1, 2): 0.55, (2, 3): 0.35, (3, 4): 0.9, (4, 5): 0.6, (5, 6): 0.4, (6, 0): 0.8,
          (0, 3): 0.2, (1, 5): 0.25}
for e in ring_edges + chord_edges:
    (x1e, y1e), (x2e, y2e) = pos1[e[0]], pos1[e[1]]
    w = edge_w[e]
    t = w
    rC = int(0xB9 + t * (0x2E - 0xB9)); gC = int(0xC4 + t * (0x6F - 0xC4)); bC = int(0xC8 + t * (0x8E - 0xC8))
    ax.plot([x1e, x2e], [y1e, y2e], color=(rC / 255, gC / 255, bC / 255),
            linewidth=0.5 + 2.2 * w, alpha=0.5 + 0.5 * w, zorder=2)
for i, (x, y) in pos1.items():
    if i == 0:
        ax.add_patch(Circle((x, y), 0.014, facecolor=COL_MUT, edgecolor=COL_MUT_EDGE, linewidth=1.0, zorder=4))
    else:
        ax.add_patch(Circle((x, y), 0.011, facecolor=COL_NODE, edgecolor=COL_NODE_EDGE, linewidth=0.8, zorder=3))

# --- icon 2: ESM-2 embedding heatmap grid ---
rng = np.random.RandomState(3)
grid = rng.rand(9, 6)
gx0, gx1 = CX[1] - 0.09, CX[1] + 0.09
gy0, gy1 = Y_ICON_CY - 0.075, Y_ICON_CY + 0.075
extent = [gx0, gx1, gy0, gy1]
ax.imshow(grid, extent=extent, aspect="auto", cmap="cividis", zorder=2,
          interpolation="nearest")
rect = mpatches.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                           edgecolor=COL_NODE_EDGE, linewidth=0.8, zorder=3)
ax.add_patch(rect)

# --- icon 3: small directed attention graph feeding a single readout ---
cx3 = CX[2]
star_pos = {
    "c": (cx3, Y_ICON_CY),
    0: (cx3 - 0.075, Y_ICON_CY + 0.06),
    1: (cx3 + 0.005, Y_ICON_CY + 0.078),
    2: (cx3 + 0.078, Y_ICON_CY + 0.03),
    3: (cx3 + 0.06, Y_ICON_CY - 0.055),
    4: (cx3 - 0.06, Y_ICON_CY - 0.06),
}
att_w = {0: 0.85, 1: 0.3, 2: 0.55, 3: 0.75, 4: 0.4}
for k, w in att_w.items():
    xk, yk = star_pos[k]
    xc, yc = star_pos["c"]
    ax.add_patch(FancyArrowPatch((xk, yk), (xc, yc), arrowstyle="-|>", mutation_scale=5.5,
                                  linewidth=0.5 + 2.0 * w, color=COL_HERO, alpha=0.4 + 0.5 * w, zorder=2))
for k, (xk, yk) in star_pos.items():
    if k == "c":
        ax.add_patch(Circle((xk, yk), 0.016, facecolor=COL_MUT, edgecolor=COL_MUT_EDGE, linewidth=1.1, zorder=4))
    else:
        ax.add_patch(Circle((xk, yk), 0.012, facecolor=COL_NODE, edgecolor=COL_NODE_EDGE, linewidth=0.8, zorder=3))

txt(CX[0], Y_ICON_BOT - 0.028, "contacts weighted by\nAlphaFold confidence", size=6.4,
    color=COL_MUTED, ha="center", style="italic")
txt(CX[1], Y_ICON_BOT - 0.028, "evolutionary +\nbiochemical context", size=6.4,
    color=COL_MUTED, ha="center", style="italic")
txt(CX[2], Y_ICON_BOT - 0.028, "attention reads out only\nthe mutated residue", size=6.4,
    color=COL_MUTED, ha="center", style="italic")

down_arrow(0.5, 0.325, 0.288)

# ============================================================ OUTPUT =======
Y_SEC_OUTPUT = 0.265
txt(0.02, Y_SEC_OUTPUT, "Output", size=13, weight="bold")

txt(0.02, 0.225, "Predicted\npathogenicity:", size=8.7, color=COL_TEXT, weight="bold")

bar_x0, bar_x1 = 0.30, 0.42
bar_y0, bar_y1 = 0.145, 0.245
grad = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(grad, extent=[bar_x0, bar_x1, bar_y0, bar_y1], aspect="auto",
          cmap="coolwarm", origin="lower", zorder=2)
rect2 = mpatches.Rectangle((bar_x0, bar_y0), bar_x1 - bar_x0, bar_y1 - bar_y0, fill=False,
                            edgecolor=COL_TEXT, linewidth=0.9, zorder=3)
ax.add_patch(rect2)
txt(bar_x1 + 0.025, bar_y1 - 0.006, "1  Pathogenic", size=7.6, color=COL_PATHO, weight="bold")
txt(bar_x1 + 0.025, bar_y0 + 0.006, "0  Benign", size=7.6, color=COL_BENIGN, weight="bold")
mid_y = (bar_y0 + bar_y1) / 2
ax.plot([bar_x0 - 0.012, bar_x1 + 0.012], [mid_y, mid_y], color=COL_TEXT, linewidth=1.0, zorder=4)
txt(bar_x1 + 0.025, mid_y, "0.5  decision threshold", size=6.6, color=COL_MUTED, style="italic")

# right-side pie: dataset composition (real, verified numbers)
pie_ax = fig.add_axes([0.60, 0.02, 0.35, 0.20])
sizes = [40.9, 59.1]
colors = [COL_PATHO, COL_BENIGN]
pie_ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 1.2})
pie_ax.set_aspect("equal")

txt(0.02, 0.075, "Training data:\n49,303 ClinVar missense\nvariants, 6,597 proteins\n40.9% pathogenic / 59.1% benign",
    size=6.8, color=COL_MUTED)

fig.savefig("problem_figure.png", dpi=300, facecolor="white", bbox_inches=None)
print("saved")
