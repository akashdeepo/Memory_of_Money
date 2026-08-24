"""Compounding-bankroll dynamics (Appendix B.3). Numerics.

Scaled: tau = 1, lambda = 1, a - r+ = 1  =>  N* = 1,  p = 1 + r+ - N.
  C+' = g C+ [1 - N(t-1)]                        chasers (allocator flow, delay kernel)
  S   = l x B ;  N = C+ - S                       contrarian short
  B'  = B [ -l x p - r- ]                          bankroll compounds realized P&L, fixed burn r-
  x'  = ( sigma(-s/w) - x ) / eps                  exposure relaxes to target on a short time eps
  s   = theta p(t) + (1-theta) p(t-1)              contrarian signal (information quality theta)
"""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, integrate_dde as hutch
WARM = "#B5541C"; LIGHT = "#C9D3CE"

def sig(z):
    return 1.0/(1.0 + np.exp(-np.clip(z, -60, 60)))

# ------------------------------------------------------------------ v1 cycle statistics (Pi, period, stale skill)
def v1_cycle_stats(g, rplus, T=600.0, h=1/256):
    t, C = hutch(None, g, T, h=h, C0=0.5)
    m = t >= 0.5*T
    N = C[m]; tt = t[m]
    p = 1 + rplus - N
    Nd = round(1/h)
    p_lag = 1 + rplus - C[m.nonzero()[0] - Nd]        # p(t-1)
    Pi_inf = np.mean(np.maximum(-p, 0.0))               # informed sharp switch: eats (-p)+
    Pi_stale = np.mean((-p) * (p_lag < 0))              # stale sharp switch: short when p(t-1)<0
    Pi_pass = np.mean(-p)                               # passive: should be -r+
    x = N - N.mean(); idx = np.where(np.diff(np.sign(x)) > 0)[0]
    period = np.diff(tt[idx]).mean() if len(idx) > 2 else np.nan
    amp_up = N.max() - 1.0
    return dict(Pi=Pi_inf, Pi_stale=Pi_stale, Pi_pass=Pi_pass, period=period, amp_up=amp_up)

# ------------------------------------------------------------------ v2.3 integrator
def integrate(g, l, rplus, rminus, w, theta, T, h=1/128, eps=0.05, C0=0.5, B0=0.05, x0=0.0):
    Nd = round(1/h); n = round(T/h)
    Cp = np.empty(n+1); B = np.empty(n+1); X = np.empty(n+1); Nn = np.empty(n+1)
    Cp[0], B[0], X[0] = C0, B0, x0
    Nn[0] = C0 - l*x0*B0
    N0 = Nn[0]
    def Ndel(i_float):
        j = i_float - Nd
        if j <= 0: return N0
        j0 = int(np.floor(j)); fr = j - j0
        return Nn[j0]*(1-fr) + Nn[min(j0+1, n)]*fr
    def f(cp, b, x, Ndl):
        Nnow = cp - l*x*b
        p = 1 + rplus - Nnow
        pd = 1 + rplus - Ndl
        s = theta*p + (1-theta)*pd
        return (g*cp*(1 - Ndl),
                b*(-l*x*p - rminus),
                (sig(-s/w) - x)/eps)
    for i in range(n):
        cp, b, x = Cp[i], B[i], X[i]
        d0, d1, d2 = Ndel(i), Ndel(i+0.5), Ndel(i+1.0)
        k1 = f(cp, b, x, d0)
        k2 = f(cp+0.5*h*k1[0], b+0.5*h*k1[1], x+0.5*h*k1[2], d1)
        k3 = f(cp+0.5*h*k2[0], b+0.5*h*k2[1], x+0.5*h*k2[2], d1)
        k4 = f(cp+h*k3[0], b+h*k3[1], x+h*k3[2], d2)
        Cp[i+1] = cp + h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        B[i+1]  = max(b + h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]), 1e-300)
        X[i+1]  = x + h/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        Nn[i+1] = Cp[i+1] - l*X[i+1]*B[i+1]
    t = np.linspace(0, T, n+1)
    return t, Cp, B, X, Nn

# ------------------------------------------------------------------ classification
def envelope(t, N, win=8.0):
    """max |N-1| over consecutive windows of length win (2 cycles)."""
    h = t[1]-t[0]; k = round(win/h)
    n = (len(t)//k)*k
    e = np.abs(N[:n]-1.0).reshape(-1, k).max(axis=1)
    te = t[:n].reshape(-1, k).mean(axis=1)
    return te, e

def classify(t, B, N, B0):
    T = t[-1]
    late = t >= 0.5*T
    if B[late].max() < 1e-4*B0:
        return "E", dict()
    te, e = envelope(t, N)
    m1 = (te >= 0.5*T) & (te < 0.75*T); m2 = te >= 0.75*T
    cv1 = e[m1].std()/e[m1].mean(); cv2 = e[m2].std()/e[m2].mean()
    Bl = B[late]; cvB = Bl.std()/Bl.mean()
    # slow oscillation count in B over the late half
    x = Bl - Bl.mean(); idx = np.where(np.diff(np.sign(x)) > 0)[0]
    nosc = len(idx)
    info = dict(cv_env_1=cv1, cv_env_2=cv2, cv_B=cvB, n_slow_osc=nosc,
                B_late_mean=Bl.mean(), env_late_mean=e[m2].mean())
    if cv2 < 0.03 and cvB < 0.03:
        return "P", info
    if cv2 > 0.10 and cv2 > 0.7*cv1:
        return "M", info
    if cv2 < 0.7*cv1:
        return "D", info
    return "?", info

# ------------------------------------------------------------------ main
RPLUS, RMINUS, W = 0.5, 0.05, 0.05
GS = [1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]

def stage1():
    print("== Stage 1: v1 cycle statistics (r+ = %.2f) ==" % RPLUS)
    print("   g    period  amp_up   Pi(informed)  Pi(stale)  stale/informed  Pi(passive)  l*=r-/Pi")
    out = {}
    for g in GS:
        st = v1_cycle_stats(g, RPLUS)
        ratio = st['Pi_stale']/st['Pi'] if st['Pi'] > 1e-9 else np.nan
        lstar = RMINUS/st['Pi'] if st['Pi'] > 1e-9 else np.inf
        out[g] = dict(st, ratio=ratio, lstar=lstar)
        print(f"  {g:4.2f}  {st['period']:6.2f}  {st['amp_up']:6.3f}   {st['Pi']:8.4f}     {st['Pi_stale']:8.4f}   {ratio:8.3f}       {st['Pi_pass']:8.4f}   {lstar:7.2f}")
    return out

def stage2(v1):
    print("== Stage 2: v2.3 sweeps (theta=1 informed; then theta=0 stale) ==")
    results = []
    T = 1200.0
    for theta in (1.0, 0.0):
        for g in (1.8, 2.5):
            lstar = v1[g]['lstar'] if theta == 1 else (RMINUS/v1[g]['Pi_stale'] if v1[g]['Pi_stale'] > 1e-9 else np.inf)
            print(f"  theta={theta:.0f} g={g}: predicted survival bar l* = {lstar:.2f}")
            for l in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
                t, Cp, B, X, N = integrate(g, l, RPLUS, RMINUS, W, theta, T)
                cls, info = classify(t, B, N, 0.05)
                late = t >= 0.5*T
                p = 1 + RPLUS - N[late]
                meanp = p.mean(); eat = np.mean(X[late]*(-p))
                results.append(dict(theta=theta, g=g, l=l, cls=cls, info=info, meanp=meanp, eat=eat, t=t, B=B, N=N, X=X))
                extra = "" if cls == "E" else (f"  <p>={meanp:.3f} (T2: r+={RPLUS})  l<x(-p)>={l*eat:.4f} (T3: r-={RMINUS})  "
                        f"cv_env late={info['cv_env_2']:.3f} (earlier {info['cv_env_1']:.3f}) cv_B={info['cv_B']:.3f} slow-osc={info['n_slow_osc']}")
                print(f"     l={l:<5}: {cls}{extra}")
    return results

def figures(v1, results):
    # ---- Fig 1: v1 statistics
    gs = np.array(GS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(gs, [v1[g]['period'] for g in gs], "o-", color=ACCENT)
    axes[0].axhline(4, color=GREY, ls=":", lw=0.8); axes[0].set_ylabel("cycle period / τ"); axes[0].set_title("period grows with gain", fontsize=10)
    axes[1].plot(gs, [v1[g]['Pi'] for g in gs], "o-", color=ACCENT, label="informed  ⟨(−p)₊⟩")
    axes[1].plot(gs, [v1[g]['Pi_stale'] for g in gs], "s-", color=WARM, label="stale  ⟨(−p)·1[p(t−τ)<0]⟩")
    axes[1].axhline(RMINUS, color=INK, ls="--", lw=1); axes[1].text(gs[-1], RMINUS, "burn r₋ (ℓ=1)", ha="right", va="bottom", fontsize=8)
    axes[1].set_ylabel("edge available to a contrarian"); axes[1].set_title(f"what the overshoot feeds  (r₊ = {RPLUS})", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(gs, [v1[g]['ratio'] for g in gs], "o-", color=WARM)
    axes[2].axhline(0, color=GREY, lw=0.8); axes[2].set_ylim(-0.3, 1.05)
    axes[2].set_ylabel("stale skill / informed skill"); axes[2].set_title("the quadrature question", fontsize=10)
    for ax in axes: ax.set_xlabel("g"); style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "e003_v1_stats.png"), dpi=600)

    # ---- Fig 2: sweep traces theta=1, g=1.8 : B(t) and envelope for several l
    for theta, g, tag in ((1.0, 1.8, "informed_g1p8"), (1.0, 2.5, "informed_g2p5"), (0.0, 2.5, "stale_g2p5")):
        rs = [r for r in results if r['theta'] == theta and r['g'] == g]
        fig, axes = plt.subplots(len(rs), 2, figsize=(12, 1.55*len(rs)), sharex=True)
        for ax_row, r in zip(axes, rs):
            t, B, N = r['t'], r['B'], r['N']
            te, e = envelope(t, N)
            ax_row[0].plot(t, N, color=ACCENT, lw=0.5)
            ax_row[0].plot(te, 1+e, color=INK, lw=1.2)
            ax_row[0].set_ylabel(f"ℓ={r['l']:g}\n{r['cls']}", rotation=0, ha="right", va="center", fontsize=9)
            ax_row[1].semilogy(t, B, color=WARM, lw=1.2)
            for ax in ax_row: style(ax)
        axes[0][0].set_title("net positioning N (envelope in black)", fontsize=10)
        axes[0][1].set_title("contrarian bankroll B  (log)", fontsize=10)
        axes[-1][0].set_xlabel("t / τ"); axes[-1][1].set_xlabel("t / τ")
        fig.suptitle(f"θ = {theta:g}, g = {g}, r₊ = {RPLUS}, r₋ = {RMINUS}: leverage sweep", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGDIR + rf"\e003_sweep_{tag}.png", dpi=600)
    print("figures written")

if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    v1 = stage1()
    res = stage2(v1)
    figures(v1, res)
    import pickle
    pickle.dump(dict(v1=v1, res=[{k: v for k, v in r.items() if k not in ('t','B','N','X')} for r in res]),
                open(os.path.join(os.path.dirname(FIGDIR), "sim", "e003_results.pkl"), "wb"))
