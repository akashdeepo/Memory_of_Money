"""Predictors against placebos after calendar-month fixed effects (Section 6.2, Figure 10).
Within-group calendar-month demeaning (each signal's return minus what ITS OWN
peer group earned that month) == calendar-month fixed effects, identified from
dispersion of event-time within a month. Then:
  (a) event-year-3 gap for predictors and placebos separately;
  (b) difference-in-differences (pred gap - plac gap) with paired bootstrap.
Crowding story predicts: predictor gap < 0, placebo gap ~ 0, DiD < 0.
Calendar-artifact story predicts: both gaps ~ 0 after demeaning, DiD ~ 0.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, panel
WARM = "#B5541C"
rng = np.random.default_rng(33)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
doc_all = pd.read_csv(os.path.join(BASE, "data", "SignalDoc.csv"))

def load(kind):
    f = "PredictorPortsFull.parquet" if kind == "Predictor" else "PlaceboPortsFull.parquet"
    p = pd.read_parquet(os.path.join(BASE, "data", f))
    ls = p[p['port'] == 'LS'][['signalname', 'date', 'ret']].copy()
    ls['date'] = pd.to_datetime(ls['date'])
    d = doc_all[doc_all['Cat.Signal'] == kind][['Acronym', 'Year', 'SampleStartYear', 'SampleEndYear']]
    ls = ls.merge(d, left_on='signalname', right_on='Acronym', how='inner')
    ins = ls[(ls['date'].dt.year >= ls['SampleStartYear']) & (ls['date'].dt.year <= ls['SampleEndYear'])]
    mu = ins.groupby('signalname')['ret'].mean().rename('mu')
    n = ins.groupby('signalname')['ret'].size().rename('n_ins')
    ls = ls.merge(mu, on='signalname').merge(n, on='signalname')
    ls = ls[ls['n_ins'] >= 60].copy()
    ls['y'] = ls['ret'] * np.sign(ls['mu'])
    ls['y_dm'] = ls['y'] - ls.groupby('date')['y'].transform('mean')   # WITHIN-GROUP demeaning
    em = (ls['date'].dt.year - ls['Year']) * 12 + ls['date'].dt.month
    ls['ey'] = np.where(em > 12, np.ceil((em - 12) / 12.0), np.floor((em - 1) / 12.0)).astype(int)
    return ls[(ls.ey >= -6) & (ls.ey <= 15)]

def cells(d, col):
    c = d.groupby(['signalname', 'ey'])[col].agg(['mean', 'size']).reset_index()
    pm = c.pivot(index='signalname', columns='ey', values='mean')
    pn = c.pivot(index='signalname', columns='ey', values='size').fillna(0)
    return np.where(np.isnan(pm.values), 0.0, pm.values), pn.values, pm.columns.values

def gap_from(M, N, yrs, idx=None):
    if idx is not None: M, N = M[idx], N[idx]
    tot = (M * N).sum(0); cnt = N.sum(0)
    m = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=yrs)
    nb = np.mean([m.get(y, np.nan) for y in (1, 2, 4, 5)])
    return m.get(3, np.nan) - nb, m

P = load("Predictor"); Q = load("Placebo")
MP, NP, YP = cells(P, "y_dm"); MQ, NQ, YQ = cells(Q, "y_dm")
gp, mp = gap_from(MP, NP, YP); gq, mq = gap_from(MQ, NQ, YQ)

print("== within-group calendar-demeaned event-year means (%/month) ==")
print("  ey  :" + "".join(f"{y:>7d}" for y in range(0, 8)))
print("  pred:" + "".join(f"{mp.get(y, np.nan):7.2f}" for y in range(0, 8)))
print("  plac:" + "".join(f"{mq.get(y, np.nan):7.2f}" for y in range(0, 8)))

B = 5000
bp = np.empty(B); bq = np.empty(B)
for b in range(B):
    bp[b] = gap_from(MP, NP, YP, rng.integers(0, MP.shape[0], MP.shape[0]))[0]
    bq[b] = gap_from(MQ, NQ, YQ, rng.integers(0, MQ.shape[0], MQ.shape[0]))[0]
did = bp - bq
print(f"\n== year-3 gap, within-group calendar FE (signal-cluster bootstrap, B={B}) ==")
print(f"  predictors: {gp:+.3f}  90% CI [{np.percentile(bp,5):+.3f}, {np.percentile(bp,95):+.3f}]  "
      f"P(<0) = {np.mean(bp<0)*100:.1f}%")
print(f"  placebos  : {gq:+.3f}  90% CI [{np.percentile(bq,5):+.3f}, {np.percentile(bq,95):+.3f}]  "
      f"P(<0) = {np.mean(bq<0)*100:.1f}%")
print(f"  DiD (pred-plac): {gp-gq:+.3f}  90% CI [{np.percentile(did,5):+.3f}, {np.percentile(did,95):+.3f}]  "
      f"P(DiD<0) = {np.mean(did<0)*100:.1f}%")

# figure
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
ax = axes[0]
yrs = np.arange(-2, 11)
ax.axhline(0, color=INK, lw=0.9)
ax.plot(yrs, [mp.get(y, np.nan) for y in yrs], "o-", color=ACCENT, lw=2, ms=4.5, label=f"predictors (n={MP.shape[0]})")
ax.plot(yrs, [mq.get(y, np.nan) for y in yrs], "s--", color=WARM, lw=1.6, ms=4, label=f"placebos (n={MQ.shape[0]})")
ax.axvline(3, color=GREY, lw=6, alpha=0.25)
ax.set_xlabel("event years since publication"); ax.set_ylabel("abnormal L/S return (%/mo)")
ax.set_title("after calendar-month fixed effects (year 3 shaded)", fontsize=10)
ax.legend(frameon=False, fontsize=8.5); style(ax); panel(ax, "A")
ax = axes[1]
ax.hist(did, bins=60, color=GREY, alpha=0.55, lw=0)
ax.axvline(0, color=INK, lw=1.2)
ax.axvline(gp-gq, color=ACCENT, lw=2)
ax.text(0.97, 0.93, f"DiD = {gp-gq:+.2f}", color=INK, fontsize=9, ha="right", transform=ax.transAxes)
ax.set_xlabel("difference-in-differences of the year-3 gap (%/mo)")
ax.set_ylabel("bootstrap draws")
ax.set_title(f"signal-cluster bootstrap: P(DiD < 0) = {np.mean(did<0)*100:.0f}%", fontsize=10)
style(ax); panel(ax, "B")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "e004d_did.png"), dpi=600)
print("\nfigure: fig/e004d_did.png")
