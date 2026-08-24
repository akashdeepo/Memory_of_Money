"""Ensemble averaging (Section 6.1, Figure 7): does averaging over desynchronized boom-bust strategies
reproduce smooth McLean–Pontiff-style decay?

Ensemble: N strategies, ALL with hard-delay memory in the cyclic/near-cyclic
regime. Heterogeneous (tau_i, g_i, r_i/a_i). Discovery aligned at event time
t = 0 (like publication alignment in McLean–Pontiff), seed capital 1% of C*.
We average the normalized realized edge p_i(t)/a_i across the ensemble.

Analytic side-check: for the delayed logistic limit cycle, the time average of
d(ln C)/dt over a period is zero  =>  <C> = C* exactly  =>  <p> = r even when
the equilibrium is unstable. Grossman–Stiglitz should hold in TIME-AVERAGE
though the point equilibrium is never occupied. We verify the late-time plateau
sits at E[r/a].
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import integrate_dde, ACCENT, GREY, INK, FIGDIR, style, panel

rng = np.random.default_rng(1)

N = 400
T_EVENT = 30.0                      # years of event time after discovery
tau = rng.uniform(1.0, 5.0, N)      # lookback windows, years
g = rng.lognormal(np.log(2.0), 0.35, N)   # gains; mostly above pi/2, some below
rr = rng.uniform(0.30, 0.50, N)     # r/a: Grossman–Stiglitz floor as fraction of raw edge

t_common = np.linspace(0.0, T_EVENT, 2400)
P = np.empty((N, t_common.size))

n_cyclic = 0
for i in range(N):
    s_max = T_EVENT / tau[i]                       # horizon in units of tau_i
    s, C = integrate_dde(None, g[i], s_max, h=1.0 / 512, C0=0.01)
    if g[i] > np.pi / 2:
        n_cyclic += 1
    p_norm = 1.0 - (1.0 - rr[i]) * C               # p/a = 1 - (1 - r/a) * C/C*
    P[i] = np.interp(t_common, s * tau[i], p_norm)

mean_p = P.mean(axis=0)

# ---- diagnostics ----
late = t_common > 0.7 * T_EVENT
plateau = mean_p[late].mean()
gs_level = rr.mean()
i_min = np.argmin(mean_p)
print(f"strategies in cyclic regime (g > pi/2): {n_cyclic}/{N}")
print(f"late-time plateau of mean p/a: {plateau:.4f}   vs   E[r/a] = {gs_level:.4f}")
print(f"deepest point of the mean: {mean_p[i_min]:.4f} at t = {t_common[i_min]:.2f} y "
      f"(undershoot below plateau: {plateau - mean_p[i_min]:.4f})")
print(f"late-time residual ripple of the mean (max-min, last 30%): "
      f"{mean_p[late].max() - mean_p[late].min():.4f}")

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

# left: individuals are violent
for i in range(0, 30):
    ax1.plot(t_common, P[i], color=GREY, lw=0.7, alpha=0.35)
ax1.plot(t_common, P[3], color=INK, lw=1.4, alpha=0.9)   # one highlighted individual
ax1.axhline(gs_level, color=GREY, lw=1, ls="--")
ax1.set_title("individual strategies: perpetual boom-bust", fontsize=10)
ax1.set_xlabel("event time since discovery (years)")
ax1.set_ylabel("realized edge  p / a")
style(ax1)
panel(ax1, "A")

# right: the average is smooth
ax2.plot(t_common, mean_p, color=ACCENT, lw=2.2,
         label="ensemble mean (N = 400)")
ax2.axhline(1.0, color=GREY, lw=1, ls=":")
ax2.axhline(gs_level, color=GREY, lw=1, ls="--")
ax2.set_ylim(-0.1, 1.18)
ax2.annotate("raw edge (pre-discovery)", (T_EVENT * 0.99, 1.0), ha="right",
             xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK)
ax2.annotate("Grossman–Stiglitz floor  E[r/a]", (T_EVENT * 0.99, gs_level), ha="right",
             xytext=(0, -11), textcoords="offset points", fontsize=8, color=INK)
ax2.set_title("their average: smooth McLean–Pontiff-style decay", fontsize=10)
ax2.set_xlabel("event time since discovery (years)")
ax2.set_ylabel("realized edge  p / a")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
style(ax2)
panel(ax2, "B")

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "e001b_averaging.png"), dpi=600)
print("figure written: fig/e001b_averaging.png")
