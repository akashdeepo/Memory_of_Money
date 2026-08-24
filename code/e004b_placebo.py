"""Placebo-portfolio check (Section 6.2).
Identical pipeline on CZ's 114 'Placebo' signals vs the 204 predictors.
Model demand: no real edge -> no crowding -> NO publication-synchronized dip.
Run in both normalizations: in-sample=100 scaling, and raw %/month.
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
rng = np.random.default_rng(7)
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
    n_ins = ins.groupby('signalname')['ret'].size().rename('n_ins')
    return ls.merge(mu, on='signalname').merge(n_ins, on='signalname')

def prep(ls, mode):
    if mode == "norm":
        d = ls[(ls['mu'] > 0) & (ls['n_ins'] >= 60)].copy()
        d['y'] = 100.0 * d['ret'] / d['mu']
    else:                       # raw %/month, sign-aligned by in-sample sign
        d = ls[ls['n_ins'] >= 60].copy()
        d['y'] = d['ret'] * np.sign(d['mu'])
    return d

def event_curve(df, align='Year', lo=-10, hi=15):
    em = (df['date'].dt.year - df[align]) * 12 + df['date'].dt.month
    ey = np.where(em > 12, np.ceil((em - 12) / 12.0), np.floor((em - 1) / 12.0)).astype(int)
    d = df.assign(ey=ey)
    d = d[(d.ey >= lo) & (d.ey <= hi)]
    return d, d.groupby('ey')['y'].mean()

def stats(m):
    dip = m.loc[[y for y in range(2, 6) if y in m.index]].mean()
    plateau = m.loc[[y for y in range(8, 16) if y in m.index]].mean()
    tr = m.loc[[y for y in range(1, 7) if y in m.index]]
    return dict(dip=dip, plateau=plateau, R1=plateau - dip, trough=tr.min(), tyear=tr.idxmin())

def shuffle_p(df, obs_R1, B=1000):
    sig_year = df.groupby('signalname')['Year'].first()
    base = df[['signalname', 'date', 'y']]
    sigs = sig_year.index.values
    hits = 0
    for b in range(B):
        perm = pd.Series(rng.permutation(sig_year.values), index=sigs)
        _, m = event_curve(base.assign(Year=base['signalname'].map(perm)))
        if stats(m)['R1'] >= obs_R1: hits += 1
    return hits / B

results = {}
for kind in ("Predictor", "Placebo"):
    ls = load(kind)
    for mode in ("norm", "raw"):
        d0 = prep(ls, mode)
        d, m = event_curve(d0)
        S = stats(m)
        p = shuffle_p(d0, S['R1'])
        results[(kind, mode)] = (m, S, p, d0['signalname'].nunique())
        unit = "(in-sample=100)" if mode == "norm" else "(%/month, sign-aligned)"
        print(f"{kind:9s} {mode:4s} {unit:26s} n={d0['signalname'].nunique():3d}  "
              f"dip(2-5)={S['dip']:7.2f}  plateau={S['plateau']:7.2f}  R1={S['R1']:7.2f}  "
              f"trough={S['trough']:7.2f} (yr {S['tyear']})  shuffle-p={p:.3f}")

# year-3 detail, raw returns: the decisive comparison
print("\n== event-year means, RAW %/month (the like-for-like comparison) ==")
mp = results[("Predictor", "raw")][0]; mq = results[("Placebo", "raw")][0]
print("  ey :  " + "".join(f"{y:>7d}" for y in range(0, 9)))
print("  pred: " + "".join(f"{mp.get(y, np.nan):7.2f}" for y in range(0, 9)))
print("  plac: " + "".join(f"{mq.get(y, np.nan):7.2f}" for y in range(0, 9)))
