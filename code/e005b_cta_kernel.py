"""Managed futures: the flow-performance lag kernel and the loop gain g_eff on the
one strategy with observable capital (BarclayHedge CTA assets). Produces the point
estimates of Table 3 and Figure 6; the intervals come from e005c_cta_bootstrap.py.
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, panel
WARM = "#B5541C"
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cta")

# ---------- AUM ----------
ind = pd.read_excel(os.path.join(D, "MUM_CTA_Industry.xls"), header=None).iloc[2:]
ind.columns = ["year", "q1", "q2", "q3", "q4"]
ind = ind.apply(pd.to_numeric, errors="coerce").dropna(subset=["year"])
aum_ann = ind.set_index("year")["q4"].dropna()                       # year-end, 1980-2025
aum_q = ind.melt(id_vars="year", var_name="q", value_name="aum").dropna()
aum_q["t"] = aum_q["year"] + aum_q["q"].str[1].astype(int)/4.0       # 2000.25 = end Q1
aum_q = aum_q.sort_values("t").set_index("t")["aum"]
aum_q = aum_q[aum_q.index >= 2000.25]
sysq = pd.read_excel(os.path.join(D, "quarterly_MUM_Sys.xls"), header=None).iloc[2:, :5]
sysq.columns = ["year", "q1", "q2", "q3", "q4"]
sysq = sysq.apply(pd.to_numeric, errors="coerce").dropna(subset=["year"])
sysq = sysq.melt(id_vars="year", var_name="q", value_name="aum").dropna()
sysq["t"] = sysq["year"] + sysq["q"].str[1].astype(int)/4.0
sysq = sysq.sort_values("t").set_index("t")["aum"]; sysq = sysq[sysq.index >= 2000.0]

# ---------- performance ----------
b = pd.read_excel(os.path.join(D, "btop50_monthly.xls"), header=None)
rows = []
for _, r in b.iterrows():
    try: y = int(r[0])
    except: continue
    vals = pd.to_numeric(r[1:13], errors="coerce").values
    if 1900 < y < 2100 and np.nanmax(np.abs(vals)) < 1.0:
        for m, v in enumerate(vals, 1):
            if not np.isnan(v): rows.append((pd.Timestamp(y, m, 1), v))
btop = pd.Series(dict(rows)).sort_index()                            # monthly ROR, 1987-01+
a = pd.read_excel(os.path.join(D, "aqr_tsmom_monthly.xlsx"), sheet_name="TSMOM Factors", header=None)
a = a[pd.to_datetime(a[0], errors="coerce").notna()]
tsmom = pd.Series(pd.to_numeric(a[1], errors="coerce").values,
                  index=pd.to_datetime(a[0]).dt.to_period("M").dt.to_timestamp()).dropna()
print(f"AUM annual {int(aum_ann.index.min())}-{int(aum_ann.index.max())} (n={len(aum_ann)}); "
      f"industry quarterly n={len(aum_q)}; systematic quarterly n={len(sysq)}")
print(f"BTOP50 {btop.index.min():%Y-%m}..{btop.index.max():%Y-%m}; TSMOM {tsmom.index.min():%Y-%m}..{tsmom.index.max():%Y-%m}")

def comp(s, freq):  # compound monthly returns to annual ('Y') or quarterly ('Q') totals
    g = (1 + s).groupby(s.index.to_period(freq)).prod() - 1
    return g

# ---------- ANNUAL flow-performance kernel, 1988-2025 ----------
R = comp(btop, "Y"); R.index = R.index.year
A = aum_ann
df = pd.DataFrame({"A": A, "R": R}).dropna()
df["flow"] = (df["A"] - df["A"].shift(1) * (1 + df["R"])) / df["A"].shift(1)   # performance-adjusted flow rate
for k in range(0, 6): df[f"R{k}"] = df["R"].shift(k)
dfa = df.dropna()
X = sm.add_constant(dfa[[f"R{k}" for k in range(0, 6)]])
m_ann = sm.OLS(dfa["flow"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
print(f"\n== ANNUAL kernel (industry AUM, BTOP50), n={int(m_ann.nobs)}, R2={m_ann.rsquared:.2f} ==")
print("  lag  coef    se     t")
for k in range(0, 6):
    print(f"  {k:3d}  {m_ann.params[f'R{k}']:+.3f}  {m_ann.bse[f'R{k}']:.3f}  {m_ann.tvalues[f'R{k}']:+.2f}")
w_ann = m_ann.params[[f"R{k}" for k in range(1, 6)]].values
print(f"  sum(lags 1-5) = {w_ann.sum():+.3f}  centroid = {np.sum(np.arange(1,6)*w_ann)/w_ann.sum():.2f} y  "
      f"peak lag = {1+int(np.argmax(w_ann))}")

# ---------- QUARTERLY Almon kernel, systematic AUM 2000-2026 ----------
Rq = comp(btop, "Q"); Rq.index = Rq.index.year + Rq.index.quarter/4.0
dq = pd.DataFrame({"A": sysq, "R": Rq}).dropna()
dq["flow"] = (dq["A"] - dq["A"].shift(1) * (1 + dq["R"])) / dq["A"].shift(1)
L, deg = 16, 3
for k in range(1, L+1): dq[f"R{k}"] = dq["R"].shift(k)
dq["R0"] = dq["R"]
dqq = dq.dropna()
lags = np.arange(1, L+1)
M = np.vstack([lags**j for j in range(deg+1)]).T                    # L x (deg+1): w = M a
Z = dqq[[f"R{k}" for k in lags]].values @ M                          # Almon regressors
Xq = sm.add_constant(np.column_stack([dqq["R0"].values, Z]))
m_q = sm.OLS(dqq["flow"].values, Xq).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
a_hat = m_q.params[2:]; Va = m_q.cov_params()[2:, 2:]
w_q = M @ a_hat; se_w = np.sqrt(np.diag(M @ Va @ M.T))
print(f"\n== QUARTERLY Almon(deg {deg}) kernel (systematic AUM, BTOP50), n={int(m_q.nobs)}, R2={m_q.rsquared:.2f} ==")
print("  lag(q)  w      se      | lag(y)")
for k, w, s in zip(lags, w_q, se_w):
    print(f"  {k:5d}  {w:+.3f}  {s:.3f}   | {k/4:.2f}")
cent_q = np.sum(lags*w_q)/w_q.sum()/4
print(f"  sum(lags 1-16q) = {w_q.sum():+.3f}  centroid = {cent_q:.2f} y  peak lag = {lags[np.argmax(w_q)]/4:.2f} y")

# ---------- capacity / impact slope ----------
Rt = comp(tsmom, "Y"); Rt.index = Rt.index.year
dc = pd.DataFrame({"logA": np.log(aum_ann), "R_next": Rt.shift(-1), "B_next": R.shift(-1)}).dropna()
for col, nm in (("R_next", "TSMOM (excess)"), ("B_next", "BTOP50 (total)")):
    mc = sm.OLS(dc[col], sm.add_constant(dc[["logA"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    print(f"\n== impact: next-year {nm} on log AUM, n={int(mc.nobs)} ==  slope {mc.params['logA']:+.4f} "
          f"(se {mc.bse['logA']:.4f}, t {mc.tvalues['logA']:+.2f})  => dp/dlogC = {mc.params['logA']:+.3f}/yr")
    if col == "R_next": slope_x = mc.params['logA']; se_slope = mc.bse['logA']

# ---------- assemble g_eff ----------
kappa_a = max(w_ann.sum(), 0); kappa_q = max(w_q.sum(), 0)
tau_a = np.sum(np.arange(1,6)*w_ann)/w_ann.sum() if w_ann.sum() > 0 else np.nan
print("\n== loop gain g_eff = kappa * |dp/dlogC| * tau ==")
for nm, kap, tau in (("annual kernel", kappa_a, tau_a), ("quarterly Almon", kappa_q, cent_q)):
    g = kap*abs(slope_x)*tau if not np.isnan(tau) else np.nan
    print(f"  {nm:16s}: kappa={kap:.2f}  |dp/dlogC|={abs(slope_x):.3f}  tau={tau:.2f}y  ->  g_eff = {g:.2f}   "
          f"(delay thr 1.57, box thr 4.93)")

# ---------- figure ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
ax.bar(np.arange(1,6), w_ann, color=ACCENT, width=0.7)
ax.errorbar(np.arange(1,6), w_ann, yerr=1.645*m_ann.bse[[f"R{k}" for k in range(1,6)]].values, fmt="none", ecolor=INK, lw=1)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("lag (years)"); ax.set_ylabel("flow rate per unit trailing return")
ax.set_title(f"annual kernel, industry assets 1988–2025 (n={int(m_ann.nobs)})", fontsize=10)
style(ax); panel(ax, "A")
ax = axes[1]
ax.plot(lags/4, w_q, color=ACCENT, lw=2)
ax.fill_between(lags/4, w_q-1.645*se_w, w_q+1.645*se_w, color=ACCENT, alpha=0.15, lw=0)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("lag (years)"); ax.set_ylabel("Almon kernel weight")
ax.set_title(f"quarterly Almon kernel, systematic assets 2000–2026 (n={int(m_q.nobs)})", fontsize=10)
style(ax); panel(ax, "B")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "e005b_cta_kernel.png"), dpi=600)
print("\nfigure: fig/e005b_cta_kernel.png")
