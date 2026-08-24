"""The trailing-window kernel institutions actually use, and what it implies (Section 6.1, Figure 8).
A 'trailing 3-year track record' is a BOX average, not a pure delay (which is a
reporting lag). Box threshold pi^2/2 = 4.93 vs delay pi/2 = 1.57. Calibration
gave g ~ 2-3 -> ABOVE delay threshold, BELOW box threshold.
Q1: at g = 2-3 with a box kernel, is the response damped (no sustained cycle)?
Q2: does the one-undershoot ensemble signature of e001b survive in the DAMPED
    regime (synchronized start -> ringing -> one visible dip)?
"""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, integrate_dde
WARM = "#B5541C"
rng = np.random.default_rng(101)

print("== Q1: box kernel at calibrated gains (threshold pi^2/2 = 4.935) ==")
for g in (2.0, 3.0, 4.0, 4.9, 5.5):
    t, C = integrate_dde(None, g, 400.0, box=True, C0=0.5)
    late = t > 300
    amp = (C[late].max() - C[late].min())/2
    mid = (C[(t>150)&(t<250)].max() - C[(t>150)&(t<250)].min())/2
    print(f"  g={g:4.1f}: late amplitude {amp:.2e}  (mid {mid:.2e})  -> "
          f"{'SUSTAINED' if amp > 1e-4 and amp > 0.9*mid else 'damped'}")

print("\n== Q2: ensemble under BOX kernel, gains BELOW threshold (damped regime) ==")
def box_scaled(g, rr, s_max, h=1/256, x0=0.01):
    """dx/ds = A x (1 - M), M = box-average of x over last 1; A = g/(1-rr)."""
    A = g/(1.0-rr)
    Nd = round(1/h); n = round(s_max/h)
    x = np.empty(n+1); M = np.empty(n+1)
    x[0] = x0; M[0] = x0
    def xd(i):
        j = i - Nd
        if j <= 0: return x0
        j0 = int(np.floor(j)); fr = j - j0
        return x[j0]*(1-fr) + x[min(j0+1, n)]*fr
    for i in range(n):
        c, m = x[i], M[i]
        d0, d1, d2 = xd(i), xd(i+0.5), xd(i+1.0)
        f = lambda cc, mm: (A*cc*(1-mm), cc)
        k1 = (A*c*(1-m), c - d0)
        c2, m2 = c+0.5*h*k1[0], m+0.5*h*k1[1]
        k2 = (A*c2*(1-m2), c2 - d1)
        c3, m3 = c+0.5*h*k2[0], m+0.5*h*k2[1]
        k3 = (A*c3*(1-m3), c3 - d1)
        c4, m4 = c+h*k3[0], m+h*k3[1]
        k4 = (A*c4*(1-m4), c4 - d2)
        x[i+1] = max(c + h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]), 1e-12)
        M[i+1] = m + h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    return np.linspace(0, s_max, n+1), x

N, T_EV = 400, 30.0
tau = rng.uniform(1.0, 5.0, N)
gs = rng.lognormal(np.log(2.5), 0.35, N)         # centred at 2.5: BELOW box threshold
rr = rng.uniform(0.30, 0.50, N)
tc = np.linspace(0, T_EV, 2400); P = np.empty((N, tc.size))
for i in range(N):
    s, x = box_scaled(gs[i], rr[i], T_EV/tau[i])
    P[i] = np.interp(tc, s*tau[i], 1.0 - (1.0-rr[i])*x)      # p/a
mean_p = 100*P.mean(0)
above = int((gs > np.pi**2/2).sum())
late = tc > 0.7*T_EV; plateau = mean_p[late].mean(); i_min = np.argmin(mean_p)
print(f"  gains above box threshold: {above}/{N}  (i.e. essentially all DAMPED)")
print(f"  ensemble mean: trough {mean_p[i_min]:.1f} at t={tc[i_min]:.1f}y; plateau {plateau:.1f} "
      f"(E[r/a]={100*rr.mean():.1f}); undershoot depth = {plateau-mean_p[i_min]:.1f}")
print(f"  late ripple (max-min over last 30%): {mean_p[late].max()-mean_p[late].min():.2f}")

fig, ax = plt.subplots(figsize=(7.4, 3.8))
ax.plot(tc, mean_p, color=ACCENT, lw=2.2, label="ensemble mean, trailing-window kernel, gains below threshold")
ax.axhline(100, color=GREY, lw=1, ls=":"); ax.axhline(100*rr.mean(), color=GREY, lw=1, ls="--")
ax.axhline(0, color=INK, lw=0.9)
ax.annotate(f"single undershoot: {mean_p[i_min]:.0f}", (tc[i_min], mean_p[i_min]),
            xytext=(tc[i_min]+3.5, mean_p[i_min]-8), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax.set_xlabel("event years since discovery"); ax.set_ylabel("realized edge p/a (×100)")
ax.set_title("damped regime: a single undershoot survives in the mean", fontsize=10)
ax.legend(frameon=False, fontsize=8.5); style(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "e001d_audit_kernel.png"), dpi=600)
print("  figure: fig/e001d_audit_kernel.png")
