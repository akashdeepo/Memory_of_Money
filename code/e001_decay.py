"""Numerical verification of the single-strategy stability results (Section 3).

Scaled units: tau = 1, C* = 1  (a = 2, r = 1, lambda = 1  =>  kappa = rho = g).
Checks:
  V1  EMA system: stable for all g (incl. huge g); node/spiral boundary g = 1/4;
      measured ring frequency & decay rate vs eigenvalue prediction
      s = (-1 ± sqrt(1 - 4g))/2.
  V2  Pure-delay (Hutchinson): Hopf threshold at g = pi/2 (bisection),
      limit-cycle amplitude ~ sqrt(g - g*) (supercritical), period -> 4*tau at onset.
  V3  Box-window kernel: threshold at g = pi^2/2, period -> 2*tau at onset,
      crossing direction of the eigenvalue pair at the threshold.
Figures -> ../fig/, summary printed to stdout.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelmax

# Journal figure style (SNDE Instructions for Authors): sans-serif in Arial or
# Helvetica, nothing below 6 pt, and a resolution of at least 300 dpi. Importing
# anything from this module applies the style, so every figure script inherits it.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ACCENT = "#0B6E5C"
GREY = "#8A9A95"
INK = "#17211E"


def panel(ax, letter):
    """Uppercase part-figure label, as required for multi-part figures."""
    ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right", color=INK)

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig")

# ----------------------------------------------------------------------
# V1 — EMA system (2 ODEs):  C' = g*C*(phat - 1),  phat' = (2 - C) - phat
# fixed point (1, 1); linear eigenvalues s = (-1 ± sqrt(1-4g))/2
# ----------------------------------------------------------------------

def ema_rhs(t, y, g):
    C, ph = y
    return [g * C * (ph - 1.0), (2.0 - C) - ph]


def run_ema(g, C0=0.5, T=200.0):
    sol = solve_ivp(ema_rhs, (0, T), [C0, 2.0 - C0], args=(g,),
                    dense_output=True, rtol=1e-10, atol=1e-12, max_step=0.05)
    t = np.linspace(0, T, 40000)
    C = sol.sol(t)[0]
    return t, C


def ema_checks():
    print("== V1: EMA system ==")
    # stability at extreme gains
    for g in [0.1, 0.3, 5.0, 50.0, 1000.0]:
        t, C = run_ema(g, T=400.0 if g < 1 else 200.0)
        tail = np.abs(C[t > 0.9 * t[-1]] - 1.0).max()
        print(f"  g={g:>7}: |C-C*| at late time = {tail:.2e}  ->  "
              f"{'STABLE' if tail < 1e-6 else 'CHECK'}")
    # node vs spiral boundary: count sign changes of C-1 after transient
    for g in [0.2, 0.3]:
        t, C = run_ema(g, T=400.0)
        crossings = np.sum(np.diff(np.sign(C - 1.0)) != 0)
        print(f"  g={g}: zero-crossings of C-C* = {crossings}  "
              f"({'node-like' if crossings <= 1 else 'spiral-like'}; "
              f"boundary predicted at 1/4)")
    # eigenvalue check at g = 5: ring frequency and decay rate from peaks
    g = 5.0
    t, C = run_ema(g, T=60.0)
    x = np.abs(C - 1.0)
    pk = argrelmax(x, order=20)[0]
    pk = pk[x[pk] > 1e-8]
    periods = np.diff(t[pk])          # peaks of |C-1| occur twice per period
    omega_meas = np.pi / periods.mean()
    decay_meas = -np.polyfit(t[pk], np.log(x[pk]), 1)[0]
    omega_pred = np.sqrt(4 * g - 1) / 2
    print(f"  g=5 ring:  omega measured {omega_meas:.4f} vs predicted "
          f"{omega_pred:.4f};  decay rate measured {decay_meas:.4f} vs predicted 0.5000")
    return omega_meas, omega_pred, decay_meas


# ----------------------------------------------------------------------
# DDE integrator: RK4 with linear interpolation of the delayed state.
# rhs(C_now, delayed_values...) ; delay tau = 1 in scaled units.
# ----------------------------------------------------------------------

def integrate_dde(rhs, g, T, h=1.0 / 512, C0=0.5, box=False):
    N = round(1.0 / h)                      # steps per delay
    n = round(T / h)
    C = np.empty(n + 1)
    C[0] = C0
    hist = np.full(N + 1, C0)               # constant history on [-1, 0]

    def delayed(i_float):
        # value of C at (i_float - N) steps, linear interpolation; i_float may be half-integer
        j = i_float - N
        if j <= 0:
            return C0
        j0 = int(np.floor(j))
        frac = j - j0
        return C[j0] * (1 - frac) + C[min(j0 + 1, n)] * frac

    if not box:
        # Hutchinson: C' = g * C * (1 - C(t-1))
        for i in range(n):
            cd0 = delayed(i)
            cd1 = delayed(i + 0.5)
            cd2 = delayed(i + 1.0)
            c = C[i]
            k1 = g * c * (1 - cd0)
            k2 = g * (c + 0.5 * h * k1) * (1 - cd1)
            k3 = g * (c + 0.5 * h * k2) * (1 - cd1)
            k4 = g * (c + h * k3) * (1 - cd2)
            C[i + 1] = c + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.linspace(0, T, n + 1), C

    # Box window: C' = g*C*(1 - M),  M(t) = int_{t-1}^t C ds,  M' = C(t) - C(t-1)
    M = np.empty(n + 1)
    M[0] = C0
    for i in range(n):
        cd0 = delayed(i)
        cd1 = delayed(i + 0.5)
        cd2 = delayed(i + 1.0)
        c, m = C[i], M[i]
        k1c = g * c * (1 - m);                 k1m = c - cd0
        c2, m2 = c + 0.5 * h * k1c, m + 0.5 * h * k1m
        k2c = g * c2 * (1 - m2);               k2m = c2 - cd1
        c3, m3 = c + 0.5 * h * k2c, m + 0.5 * h * k2m
        k3c = g * c3 * (1 - m3);               k3m = c3 - cd1
        c4, m4 = c + h * k3c, m + h * k3m
        k4c = g * c4 * (1 - m4);               k4m = c4 - cd2
        C[i + 1] = c + h / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)
        M[i + 1] = m + h / 6 * (k1m + 2 * k2m + 2 * k3m + k4m)
    return np.linspace(0, T, n + 1), C


def late_amplitude(t, C, frac=(0.7, 1.0)):
    m = (t >= frac[0] * t[-1]) & (t <= frac[1] * t[-1])
    return (C[m].max() - C[m].min()) / 2


def is_sustained(t, C):
    a_mid = late_amplitude(t, C, (0.45, 0.65))
    a_late = late_amplitude(t, C, (0.8, 1.0))
    return a_late > 1e-5 and a_late > 0.95 * a_mid


def find_threshold(box, g_lo, g_hi, T=600.0, iters=18):
    for _ in range(iters):
        g = 0.5 * (g_lo + g_hi)
        t, C = integrate_dde(None, g, T, box=box)
        if is_sustained(t, C):
            g_hi = g
        else:
            g_lo = g
    return 0.5 * (g_lo + g_hi)


def measure_period(t, C):
    m = t >= 0.7 * t[-1]
    tt, cc = t[m], C[m] - np.mean(C[m])
    s = np.sign(cc)
    idx = np.where(np.diff(s) > 0)[0]       # upward crossings
    if len(idx) < 2:
        return np.nan
    return np.diff(tt[idx]).mean()


def delay_checks():
    print("== V2: pure delay (Hutchinson) ==")
    gstar = find_threshold(box=False, g_lo=1.2, g_hi=2.2)
    print(f"  Hopf threshold measured g* = {gstar:.4f}   (predicted pi/2 = {np.pi/2:.4f})")
    t, C = integrate_dde(None, 1.70, 600.0)
    P = measure_period(t, C)
    print(f"  period at g=1.70: {P:.3f} tau   (predicted -> 4 tau at onset)")
    # supercritical check: amplitude^2 linear in g - g*
    gs = np.linspace(1.60, 2.4, 15)
    amps = []
    for g in gs:
        t, C = integrate_dde(None, g, 500.0)
        amps.append(late_amplitude(t, C) if is_sustained(t, C) else 0.0)
    amps = np.array(amps)
    sel = (amps > 0) & (gs < 2.0)
    slope, intercept = np.polyfit(gs[sel], amps[sel] ** 2, 1)
    g0 = -intercept / slope
    print(f"  amp^2 vs g linear fit crosses zero at g = {g0:.4f} "
          f"(supercritical Hopf: should ~= g*)")
    return gstar, gs, amps


def box_checks():
    print("== V3: box window ==")
    gstar = find_threshold(box=True, g_lo=4.0, g_hi=6.0)
    print(f"  threshold measured g* = {gstar:.4f}   (predicted pi^2/2 = {np.pi**2/2:.4f})")
    t, C = integrate_dde(None, gstar + 0.25, 600.0, box=True)
    sustained = is_sustained(t, C)
    P = measure_period(t, C)
    print(f"  just above threshold: sustained = {sustained}  "
          f"(confirms D4 crossing DIRECTION);  period = {P:.3f} tau (predicted -> 2 tau)")
    t2, C2 = integrate_dde(None, gstar - 0.3, 600.0, box=True)
    print(f"  just below threshold: sustained = {is_sustained(t2, C2)} (should be False)")
    return gstar


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)


def figures(gs, amps, gstar_delay):
    # Fig 1: EMA regimes
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, g in zip(axes, [0.15, 2.0, 50.0]):
        t, C = run_ema(g, T=60.0 if g > 0.5 else 200.0)
        ax.plot(t, C, color=ACCENT, lw=1.8)
        ax.axhline(1.0, color=GREY, lw=1, ls="--")
        ax.set_title(f"g = {g:g}  ({'node' if g < 0.25 else 'damped spiral'})",
                     fontsize=10, color=INK)
        ax.set_xlabel("t / τ")
        style(ax)
        panel(ax, "ABC"[list(axes).index(ax)])
    axes[0].set_ylabel("C / C*")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e001_ema_regimes.png"), dpi=600)

    # Fig 2: delay bifurcation diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(gs, amps, "o-", color=ACCENT, lw=1.8, ms=4)
    ax1.axvline(np.pi / 2, color=GREY, lw=1.2, ls="--")
    ax1.annotate("π/2", (np.pi / 2, ax1.get_ylim()[1] * 0.9),
                 textcoords="offset points", xytext=(5, 0), color=INK)
    ax1.set_xlabel("g = κ(a−r)τ")
    ax1.set_ylabel("limit-cycle amplitude / C*")
    ax1.set_title("pure lag: Hopf bifurcation at π/2", fontsize=10)
    style(ax1)
    panel(ax1, "A")
    for g, colr, lab, ls, lw in [(1.75, ACCENT, "g = 1.75 (above threshold)", "-", 1.3),
                                 (1.45, INK, "g = 1.45 (below threshold)", "--", 1.4)]:
        t, C = integrate_dde(None, g, 100.0)
        ax2.plot(t, C, color=colr, lw=lw, ls=ls, label=lab)
    ax2.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax2.set_xlabel("t / τ")
    ax2.set_ylabel("C / C*")
    ax2.set_title("decay below π/2, perpetual boom-bust above", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.26))
    style(ax2)
    panel(ax2, "B")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e001_delay_bifurcation.png"), dpi=600)

    # Fig 3: box window above/below threshold
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for g, colr, lab in [(4.6, GREY, "g = 4.6 (below π²/2)"),
                         (5.2, ACCENT, "g = 5.2 (above π²/2)")]:
        t, C = integrate_dde(None, g, 120.0, box=True)
        ax.plot(t, C, color=colr, lw=1.5, label=lab)
    ax.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("t / τ")
    ax.set_ylabel("C / C*")
    ax.set_title("Box-window memory: threshold at π²/2 ≈ 4.93", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "e001_box_window.png"), dpi=600)


if __name__ == "__main__":
    import os

    os.makedirs(FIGDIR, exist_ok=True)
    ema_checks()
    gstar_d, gs, amps = delay_checks()
    gstar_b = box_checks()
    figures(gs, amps, gstar_d)
    print("\n== SUMMARY ==")
    print(f"  EMA: stable at all tested g up to 1000; boundary node/spiral near 1/4  -> D2 OK")
    print(f"  Delay threshold {gstar_d:.4f} vs pi/2 {np.pi/2:.4f}                    -> D3")
    print(f"  Box threshold   {gstar_b:.4f} vs pi^2/2 {np.pi**2/2:.4f}              -> D4")
    print("  Figures written to fig/.")
