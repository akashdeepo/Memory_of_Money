"""Verification of the conservation law (Appendix B.2), three ways.
(1) symbolic (sympy): dQ/dt == 0 identically for delay and EMA kernels, general theta;
(2) numerical convergence: max|Q - Q0| -> 0 as the step h -> 0 (so the sim's 1e-5 was quadrature error);
(3) exact prediction of the 'seeded contrarians crushed' number from Q's boundary term.
"""
import numpy as np, sympy as sp, warnings
warnings.filterwarnings("ignore")
from e002_tugofwar import integrate

# ---------- (1) symbolic ----------
t, tau, kp, rho, lam, th = sp.symbols('t tau kappa_p rho lambda theta', positive=True)
n = sp.Function('n')          # crowding N - N*
Cp = sp.Function('Cp'); Cm = sp.Function('Cm')
# delay kernel: phat - r = -lam*n(t-tau);  p - r = -lam*n(t)
dlnCp = kp * (-lam * n(t - tau))
dlnCm = -rho*kp * (th * (-lam*n(t)) + (1 - th) * (-lam*n(t - tau)))
I = sp.Integral(n(s := sp.Symbol('s')), (s, t - tau, t))
Q_delay = sp.log(Cm(t)) + rho*sp.log(Cp(t)) - rho*kp*lam*th*I
# dQ/dt with d ln C = given rates, d/dt of the integral = n(t) - n(t - tau)
dQ = dlnCm + rho*dlnCp - rho*kp*lam*th*(n(t) - n(t - tau))
print("(1a) delay kernel:  dQ/dt simplifies to", sp.simplify(sp.expand(dQ)))
# EMA kernel: q = phat - r,  q' = (-lam n - q)/tau  =>  lam n = -q - tau q'
q = sp.Function('q')
lam_n = -q(t) - tau*sp.diff(q(t), t)
dlnCp_e = kp*q(t)
dlnCm_e = -rho*kp*(th*(-lam_n) + (1 - th)*q(t))
dQ_e = dlnCm_e + rho*dlnCp_e + rho*kp*th*tau*sp.diff(q(t), t)   # Q_EMA = ln(Cm Cp^rho) + rho kp th tau q
print("(1b) EMA kernel:    dQ/dt simplifies to", sp.simplify(sp.expand(dQ_e)))

# ---------- (2) numerical convergence ----------
def Q_series(t, Cp_, Cm_, g, rho, th):
    nn = Cp_ - Cm_ - 1.0
    h = t[1] - t[0]; Nd = round(1/h)
    # exact-history-aware running integral of n over [t-1, t] via cumulative Simpson (even spacing)
    from scipy.integrate import cumulative_simpson
    I = cumulative_simpson(nn, dx=h, initial=0.0)
    Ilag = np.concatenate([np.zeros(Nd), I[:-Nd]])
    n0 = nn[0]
    idx = np.arange(len(t))
    window = np.where(idx < Nd, I + n0*(1 - t), I - Ilag)     # constant history n0 on [-1,0]
    return np.log(Cm_) + rho*np.log(Cp_) - rho*g*th*window

print("(2) convergence of max|Q-Q0| with step size (theta=1, rho=1, beta=0.5, g=1.06 g*, amplitude ~1.4):")
from e002_tugofwar import g_star
g = 1.06*g_star(0.5, 1.0, 1.0)
for h in (1/128, 1/256, 1/512, 1/1024):
    tt, Cp_, Cm_ = integrate(g, 1.0, 1.0, 0.5, 60.0, h=h, Cp0=1.5*0.6, Cm0=0.5)
    Q = Q_series(tt, Cp_, Cm_, g, 1.0, 1.0)
    print(f"    h = 1/{round(1/h):<5d}: max|Q - Q0| = {np.abs(Q - Q[0]).max():.2e}")

# ---------- (3) boundary term predicts the crushed-seed number exactly ----------
g, rho, th = 1.2, 1.0, 1.0
eps = 0.01
tt, Cp_, Cm_ = integrate(g, rho, th, 0.0, 60.0, Cp0=eps, Cm0=eps)
n0 = eps - eps - 1.0                       # constant history: N = 0 => n = -1 on [-1,0]
Q0 = np.log(eps) + rho*np.log(eps) - rho*g*th*(n0*1.0)
Cm_pred = np.exp(Q0) / Cp_[-1]**rho        # at the end n -> 0, boundary term -> 0
print(f"(3) seeded contrarians: predicted C- = exp(Q0)/C+^rho = {Cm_pred:.4e}   simulated C- = {Cm_[-1]:.4e}")
print(f"    (naive estimate ignoring the boundary term: {eps**(1+rho):.1e}; the factor e^{{rho g theta}} = {np.exp(rho*g*th):.3f} is the boundary term)")
