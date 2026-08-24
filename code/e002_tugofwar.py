"""Two-species model (Section 3.5): numerical verification of the Hayes-equation thresholds.

Scaled units: tau = 1, N* = 1 (a = 2, r = 1, lambda = 1).  kappa+ = g, kappa- = rho*g.  mu = 0.
Hard-delay chaser memory:
    dC+/dt = g  C+ (1 - N(t-1))
    dC-/dt = rho g C- [ theta (N(t) - 1) + (1 - theta)(N(t-1) - 1) ]
    N      = C+ - C-
Linear theory (Hayes equation for n = N - 1):   dn/dt = -D n(t) - K n(t-1)
    K = g Khat,  Khat = (1+beta) + (1-theta) rho beta       (lagged gain)
    D = g Dhat,  Dhat = theta rho beta                       (instantaneous damping)
    beta = C-*/N*  (contrarian capitalisation, a state of the ecology)
    threshold  g* = arccos(-Dhat/Khat) / sqrt(Khat^2 - Dhat^2),   = inf if Khat <= Dhat
    frequency at onset  omega*tau = arccos(-Dhat/Khat)  in [pi/2, pi)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq
from scipy.signal import argrelmax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, panel
WARM = "#B5541C"      # second series colour (contrarians)
LIGHT = "#C9D3CE"

# ------------------------------------------------------------------ theory
def g_star(beta, rho, theta):
    Kh = (1 + beta) + (1 - theta) * rho * beta
    Dh = theta * rho * beta
    if Kh <= Dh:
        return np.inf
    return np.arccos(-Dh / Kh) / np.sqrt(Kh**2 - Dh**2)

def omega_onset(beta, rho, theta):
    Kh = (1 + beta) + (1 - theta) * rho * beta
    Dh = theta * rho * beta
    return np.arccos(-Dh / Kh)

# ------------------------------------------------------------------ DDE integrator
def integrate(g, rho, theta, beta, T, h=1/256, Cp0=None, Cm0=None):
    """RK4 with linear interpolation of the delayed net positioning."""
    Nd = round(1 / h); n = round(T / h)
    Cp = np.empty(n + 1); Cm = np.empty(n + 1); Nn = np.empty(n + 1)
    Cp[0] = (1 + beta) if Cp0 is None else Cp0
    Cm[0] = beta if Cm0 is None else Cm0
    Nn[0] = Cp[0] - Cm[0]
    N0 = Nn[0]                                   # constant history

    def Ndel(i_float):
        j = i_float - Nd
        if j <= 0:
            return N0
        j0 = int(np.floor(j)); fr = j - j0
        return Nn[j0] * (1 - fr) + Nn[min(j0 + 1, n)] * fr

    def f(cp, cm, Nnow, Ndl):
        return (g * cp * (1 - Ndl),
                rho * g * cm * (theta * (Nnow - 1) + (1 - theta) * (Ndl - 1)))

    for i in range(n):
        cp, cm = Cp[i], Cm[i]
        d0, d1, d2 = Ndel(i), Ndel(i + 0.5), Ndel(i + 1.0)
        k1p, k1m = f(cp, cm, cp - cm, d0)
        c2p, c2m = cp + 0.5*h*k1p, cm + 0.5*h*k1m
        k2p, k2m = f(c2p, c2m, c2p - c2m, d1)
        c3p, c3m = cp + 0.5*h*k2p, cm + 0.5*h*k2m
        k3p, k3m = f(c3p, c3m, c3p - c3m, d1)
        c4p, c4m = cp + h*k3p, cm + h*k3m
        k4p, k4m = f(c4p, c4m, c4p - c4m, d2)
        Cp[i+1] = cp + h/6*(k1p + 2*k2p + 2*k3p + k4p)
        Cm[i+1] = cm + h/6*(k1m + 2*k2m + 2*k3m + k4m)
        Nn[i+1] = Cp[i+1] - Cm[i+1]
    return np.linspace(0, T, n + 1), Cp, Cm

# ------------------------------------------------------------------ linear growth rate
def growth_rate(g, rho, theta, beta, T=60.0, pert=1e-3, win=(15.0, 55.0)):
    t, Cp, Cm = integrate(g, rho, theta, beta, T, Cp0=(1 + beta) * (1 + pert), Cm0=beta)
    nn = Cp - Cm - 1.0
    m = (t >= win[0]) & (t <= win[1])
    tt, x = t[m], np.abs(nn[m])
    pk = argrelmax(x, order=5)[0]
    pk = pk[x[pk] > 1e-13]
    if len(pk) >= 4:
        return np.polyfit(tt[pk], np.log(x[pk]), 1)[0]
    ok = x > 1e-13
    return np.polyfit(tt[ok], np.log(x[ok]), 1)[0]

def measure_threshold(rho, theta, beta, iters=14):
    gs = g_star(beta, rho, theta)
    # small perturbation: the perturbation itself shifts beta via the conservation law (Q's boundary
    # term), and g*(beta) is steep near the outright boundary -- pert=1e-3 biases those cases by ~1-3%
    kw = dict(pert=1e-5, T=140.0, win=(40.0, 130.0)) if gs > 3 else dict(pert=1e-5)
    gr = lambda g: growth_rate(g, rho, theta, beta, **kw)
    lo, hi = 0.7 * gs, 1.4 * gs
    while gr(lo) > 0: lo *= 0.7
    while gr(hi) < 0: hi *= 1.4
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if gr(mid) > 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def measure_period(rho, theta, beta, factor=1.08, T=160.0):
    g = factor * g_star(beta, rho, theta)
    t, Cp, Cm = integrate(g, rho, theta, beta, T, Cp0=(1 + beta) * 1.10, Cm0=beta)
    nn = Cp - Cm - 1.0
    m = t >= 0.6 * T
    tt, x = t[m], nn[m] - nn[m].mean()
    idx = np.where(np.diff(np.sign(x)) > 0)[0]
    return np.diff(tt[idx]).mean() if len(idx) > 2 else np.nan

# ------------------------------------------------------------------ EMA (3-ODE) check
def ema_rhs(t, y, g, rho, theta):
    Cp, Cm, ph = y
    N = Cp - Cm
    return [g * Cp * (ph - 1),
            rho * g * Cm * (theta * (N - 1) - (1 - theta) * (ph - 1)),
            (2 - N) - ph]

def ema_check():
    print("== EMA kernel: always stable (tr J = -D - 1/tau < 0) ==")
    for theta in (0.0, 1.0):
        for beta in (0.5, 2.0):
            g, rho = 50.0, 1.0
            sol = solve_ivp(ema_rhs, (0, 120), [(1+beta)*1.3, beta, 1.0], args=(g, rho, theta),
                            rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.02)
            y = sol.sol(np.linspace(100, 120, 2000))
            dev = np.abs(y[0] - y[1] - 1).max()
            print(f"  theta={theta:.0f} beta={beta}: g=50 -> |N-N*| late = {dev:.1e}  STABLE")

# ------------------------------------------------------------------ main checks
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    ema_check()

    print("== Delay kernel: Hopf thresholds  (measured vs Hayes prediction) ==")
    cases = [  # (theta, rho, beta)
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.25), (0.0, 1.0, 0.5), (0.0, 1.0, 1.0),
        (0.5, 1.0, 0.5),
        (1.0, 1.0, 0.25), (1.0, 1.0, 0.5), (1.0, 1.0, 1.0), (1.0, 1.0, 3.0),
        (1.0, 2.0, 0.5), (1.0, 2.0, 0.9),
        (1.0, 3.0, 0.25), (1.0, 3.0, 0.4),
    ]
    rows = []
    for th, rh, be in cases:
        pred = g_star(be, rh, th)
        meas = measure_threshold(rh, th, be)
        rows.append((th, rh, be, pred, meas))
        print(f"  theta={th:.1f} rho={rh:.0f} beta={be:<5}: g* = {meas:.4f}   (pred {pred:.4f}, "
              f"err {100*(meas/pred-1):+.2f}%)")

    print("== Unconditional stability  (Khat <= Dhat  <=>  beta(rho-1) >= 1) ==")
    for g in (2.0, 10.0, 50.0):
        r = growth_rate(g, 3.0, 1.0, 0.6, T=80.0)
        print(f"  theta=1 rho=3 beta=0.6: g={g:>4} -> growth rate {r:+.4f}  (must stay negative)")

    print("== Period at onset, theta=1, rho=1 (see-saw mode: 4tau -> 2tau) ==")
    betas_p = [0.0, 0.5, 1.0, 3.0, 8.0]
    per_meas = []
    for be in betas_p:
        P = measure_period(1.0, 1.0, be)
        per_meas.append(P)
        print(f"  beta={be:<4}: period {P:.3f} tau   (linear pred {2*np.pi/omega_onset(be,1,1):.3f} tau)")

    print("== Conservation law  Q = ln(C- C+^rho) - rho g theta int_{t-1}^t n ds ==")
    for th in (0.0, 0.5, 1.0):
        rho, be = 1.0, 0.5
        g = 1.06 * g_star(be, rho, th)              # just above threshold: bounded cycle
        t, Cp, Cm = integrate(g, rho, th, be, 80.0, Cp0=(1+be)*0.6, Cm0=be)
        nn = Cp - Cm - 1
        I = cumulative_trapezoid(nn, t, initial=0.0)
        Nd = round(1/(t[1]-t[0]))
        Ilag = np.concatenate([np.zeros(Nd), I[:-Nd]])
        n0 = nn[0]
        boundary = np.where(np.arange(len(t)) < Nd, I + n0 * (1 - t), I - Ilag)
        Q = np.log(Cm) + rho*np.log(Cp) - rho*g*th*boundary
        amp = np.abs(nn[t > 40]).max()
        print(f"  theta={th}: g={g:.3f}: max|Q - Q0| = {np.abs(Q-Q[0]).max():.2e}   (cycle amplitude {amp:.2f})")

    print("== Seeded contrarians are crushed by the boom (conservation-law corollary) ==")
    t, Cp, Cm = integrate(1.2, 1.0, 1.0, 0.0, 60.0, Cp0=0.01, Cm0=0.01)
    print(f"  seeds C+=C-=0.01 -> final C+ = {Cp[-1]:.4f}, C- = {Cm[-1]:.2e}, "
          f"beta = {Cm[-1]/(Cp[-1]-Cm[-1]):.2e}   (pred C- ~ 1e-4)")

    figures(rows, betas_p, per_meas)

# ------------------------------------------------------------------ figures
def figures(rows, betas_p, per_meas):
    # ---- Fig 1: threshold vs beta, three theta, two rho
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    bb = np.linspace(0, 2, 400)
    for ax, rho in zip(axes, (1.0, 3.0)):
        for th, col in ((0.0, WARM), (0.5, GREY), (1.0, ACCENT)):
            gs = np.array([g_star(b, rho, th) for b in bb])
            ax.plot(bb, gs, color=col, lw=2, label=f"θ = {th:g}")
        ax.axhline(np.pi/2, color=INK, lw=1, ls="--")
        ax.text(1.98, np.pi/2, "no contrarians (π/2)", ha="right", va="bottom", fontsize=8, color=INK)
        if rho > 1:
            bu = 1/(rho-1)
            ax.axvspan(bu, 2, color=LIGHT, alpha=0.5, lw=0)
            ax.text(bu + 0.05, 5.4, "θ=1: stable at\nany gain", fontsize=8, color=INK, va="top")
        for th, rh, be, pred, meas in rows:
            if rh == rho and be <= 2:
                col = {0.0: WARM, 0.5: GREY, 1.0: ACCENT}[th]
                ax.plot(be, meas, "o", ms=5, color=col, mec="white", mew=0.8)
        ax.set_ylim(0, 5.8)
        ax.set_xlabel("β  (contrarian capital / net positioning)")
        ax.set_title(("contrarians as aggressive as chasers  (ρ = 1)" if rho == 1
                      else f"contrarians {rho:g}× more aggressive  (ρ = {rho:g})"), fontsize=10)
        style(ax)
        panel(ax, "A" if rho == 1 else "B")
    axes[0].set_ylabel("oscillation threshold  g*")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", title="information quality θ", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e002_thresholds.png"), dpi=600)

    # ---- Fig 2: (beta, rho) map at theta = 1
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bb = np.linspace(0.02, 3, 300)
    rho_unc = 1 + 1/bb
    def neutral(b):
        f = lambda r: g_star(b, r, 1.0) - np.pi/2
        hi = 1 + 1/b - 1e-6
        return brentq(f, 1e-6, hi) if f(hi) > 0 else np.nan
    rho_neu = np.array([neutral(b) for b in bb])
    ax.fill_between(bb, 0, rho_neu, color=WARM, alpha=0.18, lw=0)
    ax.fill_between(bb, rho_neu, np.minimum(rho_unc, 6), color=GREY, alpha=0.18, lw=0)
    ax.fill_between(bb, np.minimum(rho_unc, 6), 6, color=ACCENT, alpha=0.18, lw=0)
    ax.plot(bb, rho_neu, color=INK, lw=1.6)
    ax.plot(bb, rho_unc, color=ACCENT, lw=1.6)
    ax.axhline(np.pi/2, color=INK, lw=0.8, ls=":")
    ax.text(0.04, np.pi/2 + 0.08, "ρ = π/2", fontsize=8, color=INK)
    ax.text(1.6, 0.45, "informed contrarians make it WORSE\n(threshold below π/2)", fontsize=8.5, color=INK, ha="center")
    ax.text(1.75, 1.35, "damp, conditionally", fontsize=8.5, color=INK, ha="center")
    ax.text(1.4, 4.3, "contrarians win outright:\nstable at any gain  (β(ρ−1) ≥ 1)", fontsize=8.5, color=INK, ha="center")
    ax.set_xlim(0, 3); ax.set_ylim(0, 6)
    ax.set_xlabel("β  (contrarian capital / net positioning)")
    ax.set_ylabel("ρ  (contrarian / chaser aggressiveness)")
    ax.set_title("Fully informed contrarians (θ = 1): three regimes", fontsize=10.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e002_regime_map.png"), dpi=600)

    # ---- Fig 3: same gain, four ecologies
    g = 1.45
    panels = [
        (0.0, 1.0, 0.0, "no contrarians  (v1)", "decays: g < π/2"),
        (0.0, 1.0, 0.15, "stale contrarians  θ=0, β=0.15", f"cycles: g* = {g_star(0.15,1,0):.2f}"),
        (1.0, 1.0, 0.5, "informed, equally aggressive  θ=1, ρ=1", f"cycles: g* = {g_star(0.5,1,1):.2f}"),
        (1.0, 3.0, 0.6, "informed, 3× aggressive  θ=1, ρ=3", "stable at any gain"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for ax, (th, rh, be, title, note) in zip(axes.flat, panels):
        t, Cp, Cm = integrate(g, rh, th, be, 70.0, Cp0=(1+be)*0.4, Cm0=be)
        ax.plot(t, Cp - Cm, color=ACCENT, lw=1.8, label="net positioning N")
        if be > 0:
            ax.plot(t, Cm, color=WARM, lw=1.1, alpha=0.9, label="contrarian book C₋")
        ax.axhline(1, color=GREY, lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.text(0.98, 0.94, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
        style(ax)
    for ax in axes[1]: ax.set_xlabel("t / τ")
    for ax in axes[:, 0]: ax.set_ylabel("capital / N*")
    axes[1, 1].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.suptitle(f"Same market (g = {g}), four ecologies", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e002_four_ecologies.png"), dpi=600)

    # ---- Fig 4: see-saw mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.7))
    bb = np.linspace(0, 8, 300)
    ax1.plot(bb, [2*np.pi/omega_onset(b, 1, 1) for b in bb], color=ACCENT, lw=2, label="linear theory")
    ax1.plot(betas_p, per_meas, "o", color=ACCENT, mec="white", mew=0.8, ms=6, label="simulation (8% above g*)")
    ax1.axhline(4, color=GREY, lw=0.8, ls=":"); ax1.axhline(2, color=GREY, lw=0.8, ls=":")
    ax1.set_xlabel("β  (θ = 1, ρ = 1)"); ax1.set_ylabel("cycle period / τ")
    ax1.set_ylim(1.5, 4.5)
    ax1.set_title("period at onset: 4τ (v1) → 2τ (see-saw)", fontsize=10)
    ax1.legend(frameon=False, fontsize=8.5)
    style(ax1)
    be = 5.0; gg = 1.10 * g_star(be, 1, 1)
    t, Cp, Cm = integrate(gg, 1.0, 1.0, be, 60.0, Cp0=(1+be)*1.1, Cm0=be)
    m = t >= 30
    ax2.plot(t[m], Cp[m], color=GREY, lw=1.2, label="chaser book C₊")
    ax2.plot(t[m], Cm[m], color=WARM, lw=1.2, label="contrarian book C₋")
    ax2.plot(t[m], Cp[m] - Cm[m], color=ACCENT, lw=2, label="net N")
    ax2.set_xlabel("t / τ"); ax2.set_ylabel("capital / N*")
    ax2.set_title(f"β = 5: two huge books, net exposure see-saws (g = {gg:.2f})", fontsize=10)
    ax2.legend(frameon=False, fontsize=8.5, loc="center right")
    style(ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e002_seesaw.png"), dpi=600)
    print("figures written: fig/e002_thresholds.png, e002_regime_map.png, e002_four_ecologies.png, e002_seesaw.png")

if __name__ == "__main__":
    main()
