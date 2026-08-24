"""Figure 1: the performance-chasing loop and the two classes of memory kernel.

Drawn in matplotlib rather than TikZ so that it can be supplied as a standalone
image file at the resolution the journal requires.

Run:  python sim/fig_loop_schematic.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, panel

BOX_FC = "#F2F6F5"


def box(ax, xy, w, h, title, sub):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.1, edgecolor=INK, facecolor=BOX_FC))
    ax.text(x, y + 0.055, title, ha="center", va="center", fontsize=9.5,
            color=INK, fontweight="bold")
    ax.text(x, y - 0.075, sub, ha="center", va="center", fontsize=9, color=INK)


def arrow(ax, p0, p1, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.3, color=INK, shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(7.4, 3.5))
gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0], hspace=0.55, wspace=0.22)
axL = fig.add_subplot(gs[:, 0])
axT = fig.add_subplot(gs[0, 1])
axB = fig.add_subplot(gs[1, 1])

# ---------------------------------------------------------------- the loop
axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
W, H = 0.40, 0.24
box(axL, (0.24, 0.78), W, H, "capital", "$C(t)$")
box(axL, (0.76, 0.78), W, H, "realised edge", "$p(C)$")
box(axL, (0.76, 0.22), W, H, "track record", r"$\hat{p} = w * p$")
box(axL, (0.24, 0.22), W, H, "flows", r"$\dot C = \kappa C(\hat{p} - r)$")

arrow(axL, (0.44, 0.78), (0.56, 0.78))
arrow(axL, (0.76, 0.66), (0.76, 0.34))
arrow(axL, (0.56, 0.22), (0.44, 0.22))
arrow(axL, (0.24, 0.34), (0.24, 0.66))

axL.text(0.50, 0.985, "price impact,  $p'(C) < 0$", ha="center", va="top",
         fontsize=8, color=INK)
axL.text(0.795, 0.50, "memory kernel $w(u)$", ha="left", fontsize=8, color=INK)
axL.text(0.795, 0.455, r"timescale $\tau$", ha="left", fontsize=8, color=INK)
axL.text(0.50, 0.025, "chasing, rate $\\kappa$", ha="center", va="bottom",
         fontsize=8, color=INK)
axL.text(0.50, 0.52, "loop gain", ha="center", fontsize=8.5, color=ACCENT,
         fontweight="bold")
axL.text(0.50, 0.465, r"$g_{\mathrm{eff}} = \kappa C^{*}|p'(C^{*})|\,\tau$",
         ha="center", fontsize=9, color=ACCENT)
panel(axL, "A")

# ------------------------------------------------------------ kernel shapes
u = np.linspace(0, 3.4, 400)

axT.plot(u, np.exp(-1.35 * u), color=ACCENT, lw=1.9)
axT.fill_between(u, np.exp(-1.35 * u), color=ACCENT, alpha=0.13)
axT.set_title("monotone: cannot cycle", fontsize=9, color=INK, pad=9)
axT.set_xlim(0, 3.4); axT.set_ylim(0, 1.25)
axT.set_yticks([])
axT.set_xticks([])
axT.set_ylabel("$w(u)$", fontsize=8.5)
style(axT)
panel(axT, "B")

wbox = np.where((u >= 0.7) & (u <= 2.6), 0.85, 0.0)
axB.plot(u, wbox, color=ACCENT, lw=1.9)
axB.fill_between(u, wbox, color=ACCENT, alpha=0.13)
axB.annotate("", xy=(0.7, 1.03), xytext=(0.0, 1.03),
             arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
axB.text(0.35, 1.09, "lag", ha="center", fontsize=7.5, color=INK)
axB.annotate("", xy=(2.6, 1.03), xytext=(0.7, 1.03),
             arrowprops=dict(arrowstyle="<->", color=GREY, lw=0.9))
axB.text(1.65, 1.09, "window", ha="center", fontsize=7.5, color=INK)
axB.set_title("humped: can cycle", fontsize=9, color=INK, pad=9)
axB.set_xlim(0, 3.4); axB.set_ylim(0, 1.35)
axB.set_yticks([])
axB.set_xticks([])
axB.set_xlabel("lag $u$ (age of the observation)", fontsize=8.5)
axB.set_ylabel("$w(u)$", fontsize=8.5)
style(axB)
panel(axB, "C")

fig.tight_layout()
os.makedirs(FIGDIR, exist_ok=True)
fig.savefig(os.path.join(FIGDIR, "fig_loop_schematic.png"), dpi=600)
print("figure: fig/fig_loop_schematic.png")
