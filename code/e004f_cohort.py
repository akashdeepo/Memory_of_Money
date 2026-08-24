"""Publication-vintage check (Section 6.2, Table 4): is the 'event-year-3 dip' an event-time effect or a
publication-VINTAGE x crisis effect? For each publication cohort, event-year 3
falls in a different calendar window. If the dip is event-time, every cohort
shows it. If it is the 2004-06 vintage being hit by 2008-09, only that cohort shows it.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e004e_crowdability import load   # reuse loader (prints its own header)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(55)

both = pd.concat([load("Predictor"), load("Placebo")])
sub = both[both.ey.isin([1, 2, 3, 4, 5])]
piv = sub.pivot_table(index=['signalname'], columns='ey', values='y_dm', aggfunc='mean')
gap = (piv[3] - piv[[1, 2, 4, 5]].mean(axis=1)).rename('gap')
meta = both.groupby('signalname').agg(Year=('Year', 'first'), kind=('kind', 'first'))
X = meta.join(gap, how='inner').dropna()

def boot_mean(v, B=4000):
    v = np.asarray(v)
    d = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
    return v.mean(), np.percentile(d, 5), np.percentile(d, 95), np.mean(d < 0)

print("\n== year-3 gap by publication cohort (calendar-demeaned, %/mo) ==")
print("  cohort        yr3 falls in    n    mean gap   90% CI            P(<0)")
bins = [(1970, 1994), (1995, 1999), (2000, 2003), (2004, 2008), (2009, 2016)]
for lo, hi in bins:
    v = X[(X.Year >= lo) & (X.Year <= hi)]['gap']
    if len(v) < 8: continue
    m, l, u, p = boot_mean(v)
    print(f"  {lo}-{hi}     {lo+3}-{hi+3}    {len(v):3d}   {m:+.3f}    [{l:+.3f},{u:+.3f}]   {p*100:.0f}%")

print("\n== excluding the 2004-2008 vintage entirely ==")
v = X[~((X.Year >= 2004) & (X.Year <= 2008))]['gap']
m, l, u, p = boot_mean(v)
print(f"  n={len(v)}  mean gap {m:+.3f}  90% CI [{l:+.3f},{u:+.3f}]  P(<0) = {p*100:.0f}%")
v2 = X[(X.Year >= 2004) & (X.Year <= 2008)]['gap']
m2, l2, u2, p2 = boot_mean(v2)
print(f"  2004-08 vintage only: n={len(v2)}  mean gap {m2:+.3f}  90% CI [{l2:+.3f},{u2:+.3f}]  P(<0) = {p2*100:.0f}%")

print("\n== placebo-style check: same statistic at event-year 2 and 4 (should be ~0) ==")
for k in (2, 4):
    g2 = (piv[k] - piv[[y for y in (1,2,3,4,5) if y != k]].mean(axis=1)).dropna()
    m, l, u, p = boot_mean(g2.values)
    print(f"  event-year {k}: mean {m:+.3f}  90% CI [{l:+.3f},{u:+.3f}]  P(<0) = {p*100:.0f}%")
