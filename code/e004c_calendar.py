"""Calendar-month fixed-effects check (Section 6.2): does the year-3 dip survive?
If the dip is the GFC landing at event-year 3, removing each calendar month's
cross-signal mean kills it. If it is publication-synchronized crowding, it
survives (identified from signals with DIFFERENT publication years in the SAME month).
Predictors vs placebos, raw %/month, sign-aligned.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style
WARM = "#B5541C"
rng = np.random.default_rng(21)
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
    ls['kind'] = kind
    return ls[['signalname', 'date', 'y', 'Year', 'kind']]

pred, plac = load("Predictor"), load("Placebo")
print("publication-year distribution (quartiles):")
for nm, d in (("predictors", pred), ("placebos", plac)):
    q = d.groupby('signalname')['Year'].first().quantile([.25, .5, .75]).values
    print(f"  {nm:11s} n={d.signalname.nunique():3d}  Q1/med/Q3 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}")

both = pd.concat([pred, plac])
# calendar-month demeaning using ALL signals present that month (the common factor)
cal = both.groupby('date')['y'].transform('mean')
both['y_dm'] = both['y'] - cal

def curve(d, col, lo=-6, hi=15):
    em = (d['date'].dt.year - d['Year']) * 12 + d['date'].dt.month
    ey = np.where(em > 12, np.ceil((em - 12) / 12.0), np.floor((em - 1) / 12.0)).astype(int)
    x = d.assign(ey=ey)
    x = x[(x.ey >= lo) & (x.ey <= hi)]
    return x, x.groupby('ey')[col].mean()

print("\n== event-year means, RAW (y) vs CALENDAR-DEMEANED (y_dm), %/month ==")
print("  ey   :" + "".join(f"{y:>7d}" for y in range(0, 8)))
rows = {}
for kind in ("Predictor", "Placebo"):
    d = both[both.kind == kind]
    for col, tag in (("y", "raw "), ("y_dm", "demn")):
        _, m = curve(d, col)
        rows[(kind, col)] = m
        print(f"  {kind[:4]:4s} {tag}:" + "".join(f"{m.get(y, np.nan):7.2f}" for y in range(0, 8)))

def yr3_gap(m):    # year-3 vs mean of neighbours 1,2,4,5
    nb = np.mean([m.get(y, np.nan) for y in (1, 2, 4, 5)])
    return m.get(3, np.nan) - nb

print("\n== year-3 gap (year3 minus mean of years 1,2,4,5), %/month ==")
for kind in ("Predictor", "Placebo"):
    for col, tag in (("y", "raw"), ("y_dm", "calendar-demeaned")):
        print(f"  {kind:9s} {tag:18s}: {yr3_gap(rows[(kind, col)]):+.3f}")

# bootstrap the demeaned year-3 gap by signal cluster
def boot_gap(d, col, B=3000):
    x, _ = curve(d, col)
    cell = x.groupby(['signalname', 'ey'])[col].agg(['mean', 'size']).reset_index()
    pm = cell.pivot(index='signalname', columns='ey', values='mean')
    pn = cell.pivot(index='signalname', columns='ey', values='size').fillna(0)
    M = np.where(np.isnan(pm.values), 0.0, pm.values); N = pn.values
    yrs = pm.columns.values; nu = M.shape[0]
    out = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, nu, nu)
        w = N[idx]; v = M[idx]
        tot = (v * w).sum(0); cnt = w.sum(0)
        m = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=yrs)
        out[b] = yr3_gap(m)
    return out

print("\n== signal-cluster bootstrap of the calendar-demeaned year-3 gap ==")
for kind in ("Predictor", "Placebo"):
    g = boot_gap(both[both.kind == kind], "y_dm")
    print(f"  {kind:9s}: mean {g.mean():+.3f}  90% CI [{np.percentile(g,5):+.3f}, {np.percentile(g,95):+.3f}]  "
          f"P(gap<0) = {np.mean(g < 0)*100:.1f}%")
