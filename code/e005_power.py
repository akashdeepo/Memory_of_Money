"""Power analysis for the pooled spectral resonance test (Section 7, Table 5).

Model (H1): monthly factor return = slow damped-oscillator component (noise-driven,
period P, damping ratio zeta) + white noise.  Null (H0): slow AR(1) red-noise
component with the SAME slow variance and the same correlation time + white noise.
Test: Whittle likelihood ratio on the pooled (averaged over N series) periodogram,
H1 spectrum = white + Lorentzian peak at f0 ; H0 spectrum = white + AR(1).
Null distribution of the LR by parametric bootstrap; power = P(LR_H1 > 95th pct of LR_H0).
Calibration: white std 3.0%/mo (HML-like); slow-component std 0.4%/mo.
"""
import numpy as np, warnings, sys, time
from scipy.optimize import minimize
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)

SIG_W, SIG_S = 3.0, 0.4          # %/month

def sim_osc(N, T, P, zeta):
    w0 = 2*np.pi/P; wd = w0*np.sqrt(max(1-zeta**2, 1e-6)); gam = zeta*w0
    phi1 = 2*np.exp(-gam)*np.cos(wd); phi2 = -np.exp(-2*gam)
    # stationary variance of AR(2) driven by unit noise -> scale to SIG_S
    x = np.zeros((N, T+600)); e = rng.standard_normal((N, T+600))
    for t in range(2, T+600):
        x[:, t] = phi1*x[:, t-1] + phi2*x[:, t-2] + e[:, t]
    x = x[:, 600:]; x *= SIG_S/np.sqrt(x.var(axis=1, keepdims=True).mean())
    return x + SIG_W*rng.standard_normal((N, T))

def sim_ar1(N, T, P, zeta):
    gam = zeta*2*np.pi/P; rho = np.exp(-gam)      # same correlation time
    x = np.zeros((N, T+600)); e = rng.standard_normal((N, T+600))
    for t in range(1, T+600):
        x[:, t] = rho*x[:, t-1] + e[:, t]
    x = x[:, 600:]; x *= SIG_S/np.sqrt(x.var(axis=1, keepdims=True).mean())
    return x + SIG_W*rng.standard_normal((N, T))

def pooled_periodogram(X):
    N, T = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    F = np.fft.rfft(Xc, axis=1)
    I = (np.abs(F)**2)/T
    f = np.fft.rfftfreq(T, d=1.0)
    return f[1:-1], I[:, 1:-1].mean(axis=0), N

def S_h0(f, th):      # white + AR(1)-type Lorentzian at zero
    w, A, fc = np.exp(th)
    return w + A/(1 + (f/fc)**2)
def S_h1(f, th):      # white + damped-oscillator response
    w, A, f0, z = np.exp(th[0]), np.exp(th[1]), np.exp(th[2]), 1/(1+np.exp(-th[3]))
    return w + A/((1 - (f/f0)**2)**2 + (2*z*f/f0)**2)

def whittle(S, I, N):   # averaged periodogram over N series ~ Gamma(N, S/N)
    return -N*np.sum(np.log(S) + I/S)

def fit(f, I, N, model, inits):
    best = None
    for th0 in inits:
        r = minimize(lambda th: -whittle(model(f, th), I, N), th0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        if best is None or r.fun < best.fun: best = r
    return -best.fun, best.x

def LR(X, P_guess):
    f, I, N = pooled_periodogram(X)
    m = f <= 1/24            # low-frequency band: periods >= 2 years
    f, I = f[m], I[m]
    lw = np.log(SIG_W**2)
    l0, _ = fit(f, I, N, S_h0, [np.array([lw, np.log(0.5), np.log(0.01)]),
                                np.array([lw, np.log(2.0), np.log(0.003)])])
    l1, th1 = fit(f, I, N, S_h1, [np.array([lw, np.log(0.05), np.log(1/P_guess), 0.0]),
                                  np.array([lw, np.log(0.2), np.log(1/P_guess), -1.0]),
                                  np.array([lw, np.log(0.05), np.log(1/(1.5*P_guess)), 0.0])])
    return 2*(l1 - l0), 1/np.exp(th1[2])

def power(N, T, P, zeta, R=60):
    lr0 = np.array([LR(sim_ar1(N, T, P, zeta), P)[0] for _ in range(R)])
    crit = np.percentile(lr0, 95)
    out = [LR(sim_osc(N, T, P, zeta), P) for _ in range(R)]
    lr1 = np.array([o[0] for o in out]); per = np.array([o[1] for o in out])
    return np.mean(lr1 > crit), np.median(per), np.percentile(per, [10, 90])

if __name__ == "__main__":
    t0 = time.time()
    print("power = P(reject red-noise null at 5%) | median estimated period [10-90%]")
    print(f"  calibration: white {SIG_W}%/mo, slow {SIG_S}%/mo; R=60 sims each")
    grid = [(1, 1176, 96, 0.3), (5, 1176, 96, 0.3), (20, 1176, 96, 0.3), (60, 1176, 96, 0.3),
            (20, 1176, 96, 0.15), (20, 1176, 96, 0.5),
            (20, 1176, 144, 0.3), (20, 1176, 192, 0.3),
            (20, 360, 96, 0.3), (60, 360, 96, 0.3)]
    for N, T, P, z in grid:
        pw, med, (lo, hi) = power(N, T, P, z)
        print(f"  N={N:3d} T={T:4d}mo ({T//12:3d}y) P={P//12:2d}y zeta={z:.2f}:  power={pw:.2f}   "
              f"period est {med/12:5.1f}y [{lo/12:4.1f},{hi/12:4.1f}]   ({time.time()-t0:.0f}s)")
        sys.stdout.flush()
