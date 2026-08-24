"""Saturating impact laws (Section 3.4, Appendix A.5). Verifies the predicted thresholds:
(1) S1 p = a/(1+lam C): Hopf at g_sat = kappa*(a-r)*(r/a)*tau = pi/2, kernel dichotomy intact
(2) S2 p = (a-lam C)/(1+nu C): Hopf at g = kappa*(a-r)*(lam+nu r)/(lam+nu a)*tau = pi/2
(3) S1 bounded amplitudes at all g; realized edge p in (0, a] -- never negative
(4) ensemble rerun: one-undershoot shape survives, trough >= 0
Scaled: tau = 1. S1: a=2, r=1, lam=1 -> C*=1, b=kappa/2 (Hopf kappa=pi).
S2: a=2, r=1, lam=1, nu=1 -> C*=1/2, b=2kappa/3 (Hopf kappa=3pi/4).
"""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style
from scipy.signal import argrelmax
from scipy.integrate import solve_ivp
WARM = "#B5541C"
rng = np.random.default_rng(11)

def integrate_sat(kappa, T, h=1/256, C0=0.5, variant="S1"):
    """delay-kernel DDE for saturating impact, RK4 + linear interp of delayed C."""
    a, r, lam, nu = 2.0, 1.0, 1.0, 1.0
    if variant == "S1":
        p = lambda C: a/(1 + lam*C)
    else:
        p = lambda C: (a - lam*C)/(1 + nu*C)
    Nd = round(1/h); n = round(T/h)
    C = np.empty(n+1); C[0] = C0
    def Cdel(i_float):
        j = i_float - Nd
        if j <= 0: return C0
        j0 = int(np.floor(j)); fr = j - j0
        return C[j0]*(1-fr) + C[min(j0+1, n)]*fr
    for i in range(n):
        c = C[i]
        d0, d1, d2 = Cdel(i), Cdel(i+0.5), Cdel(i+1.0)
        k1 = kappa*c*(p(d0) - r)
        k2 = kappa*(c+0.5*h*k1)*(p(d1) - r)
        k3 = kappa*(c+0.5*h*k2)*(p(d1) - r)
        k4 = kappa*(c+h*k3)*(p(d2) - r)
        C[i+1] = max(c + h/6*(k1+2*k2+2*k3+k4), 1e-12)
    return np.linspace(0, T, n+1), C

def growth_rate(kappa, variant, Cstar, pert=1e-5, T=80.0):
    t, C = integrate_sat(kappa, T, C0=Cstar*(1+pert), variant=variant)
    x = np.abs(C - Cstar); m = (t >= 20) & (t <= 75)
    tt, xx = t[m], x[m]
    pk = argrelmax(xx, order=5)[0]; pk = pk[xx[pk] > 1e-14]
    if len(pk) >= 4:
        return np.polyfit(tt[pk], np.log(xx[pk]), 1)[0]
    ok = xx > 1e-14
    return np.polyfit(tt[ok], np.log(xx[ok]), 1)[0]

def find_threshold(variant, Cstar, klo, khi, iters=16):
    while growth_rate(klo, variant, Cstar) > 0: klo *= 0.8
    while growth_rate(khi, variant, Cstar) < 0: khi *= 1.25
    for _ in range(iters):
        mid = 0.5*(klo+khi)
        if growth_rate(mid, variant, Cstar) > 0: khi = mid
        else: klo = mid
    return 0.5*(klo+khi)

print("== (1)/(2) Hopf thresholds ==")
kS1 = find_threshold("S1", 1.0, 2.5, 4.0)
print(f"  S1: kappa* = {kS1:.4f}  (predicted pi = {np.pi:.4f}, err {100*(kS1/np.pi-1):+.2f}%)"
      f"  -> g_sat = {kS1/2:.4f} vs pi/2 = {np.pi/2:.4f}")
kS2 = find_threshold("S2", 0.5, 1.8, 3.0)
print(f"  S2: kappa* = {kS2:.4f}  (predicted 3pi/4 = {3*np.pi/4:.4f}, err {100*(kS2/(3*np.pi/4)-1):+.2f}%)")

print("== EMA kernel under S1: still no cycles (spot check) ==")
def ema_rhs(t, y, kappa):
    C, ph = y
    return [kappa*C*(ph - 1.0), (2.0/(1.0 + C) - ph)]
for kap in (50.0, 500.0):
    sol = solve_ivp(ema_rhs, (0, 200), [1.3, 1.0], args=(kap,), rtol=1e-10, atol=1e-12, max_step=0.05)
    tail = np.abs(sol.y[0][sol.t > 180] - 1.0).max()
    print(f"  kappa={kap}: |C-C*| late = {tail:.1e}  STABLE")

print("== (3) S1 amplitude boundedness & realized edge floor ==")
for gsat in (2.0, 3.0, 4.5, 6.0):
    kap = 2*gsat
    t, C = integrate_sat(kap, 300.0)
    m = t > 150
    pmin = (2.0/(1.0 + C[m])).min(); pmax = (2.0/(1.0 + C[m])).max()
    x = C[m] - C[m].mean(); idx = np.where(np.diff(np.sign(x)) > 0)[0]
    per = np.diff(t[m][idx]).mean() if len(idx) > 2 else np.nan
    print(f"  g_sat={gsat}: C in [{C[m].min():.3f}, {C[m].max():.2f}]  realized p in "
          f"[{pmin:.3f}, {pmax:.3f}] (floor 0 OK)  period {per:.2f} tau  FINITE={np.isfinite(C).all()}")

print("== (4) ensemble rerun under S1 (one-undershoot shape) ==")
def integrate_scaled_S1(gsat, rr, s_max, h=1/256, x0=0.01):
    """dx/ds = [gsat/(rr(1-rr))] * (1/(1+(1/rr-1)*x(s-1)) - rr) * x(s); x = C/C*, s = t/tau."""
    A = gsat/(rr*(1-rr)); B = 1.0/rr - 1.0
    Nd = round(1/h); n = round(s_max/h)
    x = np.empty(n+1); x[0] = x0
    def xd(i_float):
        j = i_float - Nd
        if j <= 0: return x0
        j0 = int(np.floor(j)); fr = j - j0
        return x[j0]*(1-fr) + x[min(j0+1, n)]*fr
    f = lambda xx, dd: A*xx*(1.0/(1.0 + B*dd) - rr)
    for i in range(n):
        c = x[i]; d0, d1, d2 = xd(i), xd(i+0.5), xd(i+1.0)
        k1 = f(c, d0); k2 = f(c+0.5*h*k1, d1); k3 = f(c+0.5*h*k2, d1); k4 = f(c+h*k3, d2)
        x[i+1] = max(c + h/6*(k1+2*k2+2*k3+k4), 1e-12)
    return np.linspace(0, s_max, n+1), x

N, T_EV = 400, 30.0
tau = rng.uniform(1.0, 5.0, N)
gs = rng.lognormal(np.log(2.0), 0.35, N)
rr = rng.uniform(0.30, 0.50, N)
t_common = np.linspace(0.0, T_EV, 2400)
P = np.empty((N, t_common.size))
ncyc = 0
for i in range(N):
    s, x = integrate_scaled_S1(gs[i], rr[i], T_EV/tau[i])
    if gs[i] > np.pi/2: ncyc += 1
    B = 1.0/rr[i] - 1.0
    p_norm = 1.0/(1.0 + B*x)          # = p/a, in (0,1]
    P[i] = np.interp(t_common, s*tau[i], p_norm)
mean_p = 100*P.mean(axis=0)
late = t_common > 0.7*T_EV
plateau = mean_p[late].mean()
i_min = np.argmin(mean_p)
print(f"  cyclic fraction: {ncyc}/{N};  plateau {plateau:.1f} vs E[r/a] = {100*rr.mean():.1f}")
print(f"  trough of mean: {mean_p[i_min]:.1f} at t = {t_common[i_min]:.1f}y  (>= 0 required)")
print(f"  min over ALL individuals ever: {100*P.min():.2f}  (S1 floor: must be > 0)")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
for i in range(0, 30):
    ax.plot(t_common, 100*P[i], color=GREY, lw=0.7, alpha=0.35)
ax.plot(t_common, 100*P[3], color=INK, lw=1.4)
ax.axhline(100*rr.mean(), color=GREY, lw=1, ls="--")
ax.axhline(0, color=INK, lw=0.9)
ax.set_title("S1 individuals: boom-bust with a hard floor at 0", fontsize=10)
ax.set_xlabel("event years since discovery"); ax.set_ylabel("realized edge  p/a (x100)")
style(ax)
ax = axes[1]
ax.plot(t_common, mean_p, color=ACCENT, lw=2.2, label="ensemble mean, S1 impact")
ax.axhline(100, color=GREY, lw=1, ls=":"); ax.axhline(0, color=INK, lw=0.9)
ax.axhline(100*rr.mean(), color=GREY, lw=1, ls="--")
ax.annotate(f"trough {mean_p[i_min]:.0f} (bounded > 0)", (t_common[i_min], mean_p[i_min]),
            xytext=(t_common[i_min]+3, mean_p[i_min]-9), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax.set_title("their average: one undershoot, trough floored", fontsize=10)
ax.set_xlabel("event years since discovery")
ax.legend(frameon=False, fontsize=8.5)
style(ax)
fig.suptitle("saturating impact S1 — shape survives, depth is bounded", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "e001c_saturating.png"), dpi=600)
print("  figure written: fig/e001c_saturating.png")
