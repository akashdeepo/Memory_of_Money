"""Publication-aligned event study on the Chen-Zimmermann Open Source Asset Pricing
panel (Section 6.2, Figure 9). Design:
- 212 published predictors, original-paper long-short returns (percent/month).
- Normalize each predictor by its own in-sample mean (in-sample = 100).
- Align event clocks at publication (t0 = Dec of publication year); average, NO heavy smoothing.
- T1: mean(event-years 2-5) < mean(8-15)?   T2: trough <= 0?
- T3: rebound R2 = plateau(8-15) - min single year in (1-6); monotone-decay null => R2 <= 0.
- Inference: predictor cluster bootstrap AND calendar-month bootstrap; placebo = shuffled
  publication years. Control C1: align at SampleEndYear instead (CZ's choice).
"""
import numpy as np, pandas as pd, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e001_decay import ACCENT, GREY, INK, FIGDIR, style, panel
WARM = "#B5541C"
rng = np.random.default_rng(4)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- load & prepare ----------------
port = pd.read_parquet(os.path.join(BASE, "data", "PredictorPortsFull.parquet"))
ls = port[port['port'] == 'LS'][['signalname', 'date', 'ret']].copy()
ls['date'] = pd.to_datetime(ls['date'])
doc = pd.read_csv(os.path.join(BASE, "data", "SignalDoc.csv"))
doc = doc[doc['Cat.Signal'] == 'Predictor'][['Acronym', 'Year', 'SampleStartYear', 'SampleEndYear']]
ls = ls.merge(doc, left_on='signalname', right_on='Acronym', how='inner')

ins = ls[(ls['date'].dt.year >= ls['SampleStartYear']) & (ls['date'].dt.year <= ls['SampleEndYear'])]
mu = ins.groupby('signalname')['ret'].mean().rename('mu')
n_ins = ins.groupby('signalname')['ret'].size().rename('n_ins')
ls = ls.merge(mu, on='signalname').merge(n_ins, on='signalname')
keep = (ls['mu'] > 0) & (ls['n_ins'] >= 60)
dropped = ls.loc[~keep, 'signalname'].nunique()
ls = ls[keep]
print(f"predictors kept: {ls['signalname'].nunique()} (dropped {dropped}: nonpositive in-sample mean or <60 in-sample months)")
ls['norm'] = 100.0 * ls['ret'] / ls['mu']

def event_curve(df, align_col, lo=-10, hi=15):
    """event-year k: k>=1 -> months (12(k-1),12k] after Dec of align year;
       k<=0 -> calendar year alignyear+k."""
    em = (df['date'].dt.year - df[align_col]) * 12 + df['date'].dt.month
    ey = np.where(em > 12, np.ceil((em - 12) / 12.0), np.floor((em - 1) / 12.0)).astype(int)
    d = df.assign(ey=ey)
    d = d[(d.ey >= lo) & (d.ey <= hi)]
    g = d.groupby('ey')['norm']
    return d, g.mean(), g.size()

d_pub, m_pub, n_pub = event_curve(ls, 'Year')
d_end, m_end, n_end = event_curve(ls, 'SampleEndYear')

def stats(m):
    dip = m.loc[[y for y in range(2, 6) if y in m.index]].mean()
    plateau = m.loc[[y for y in range(8, 16) if y in m.index]].mean()
    tr_years = [y for y in range(1, 7) if y in m.index]
    trough_year = m.loc[tr_years].idxmin(); trough = m.loc[tr_years].min()
    return dict(dip25=dip, plateau=plateau, R1=plateau - dip, trough=trough,
                trough_year=trough_year, R2=plateau - trough)

S = stats(m_pub)
print("\n== Publication-aligned event-year means (in-sample = 100) ==")
print(m_pub.round(1).to_string())
print(f"\ndip(y2-5) = {S['dip25']:.1f}   plateau(y8-15) = {S['plateau']:.1f}   "
      f"trough = {S['trough']:.1f} (year {S['trough_year']})   R1 = {S['R1']:.1f}   R2 = {S['R2']:.1f}")

# ---------------- bootstraps (cluster-resample precomputed cells) ----------------
def boot_fast(d, kind, B=4000):
    if kind == 'predictor':
        cell = d.groupby(['signalname', 'ey'])['norm'].agg(['mean', 'size']).reset_index()
        key = 'signalname'
    else:
        cell = d.groupby(['date', 'ey'])['norm'].agg(['mean', 'size']).reset_index()
        key = 'date'
    piv_m = cell.pivot(index=key, columns='ey', values='mean')
    piv_n = cell.pivot(index=key, columns='ey', values='size').fillna(0)
    Mv = np.where(np.isnan(piv_m.values), 0.0, piv_m.values)
    Nv = piv_n.values
    years = piv_m.columns.values
    nunit = Mv.shape[0]
    R1s = np.empty(B); R2s = np.empty(B); troughs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, nunit, nunit)
        w = Nv[idx]; x = Mv[idx]
        tot = (x * w).sum(0); cnt = w.sum(0)
        m = pd.Series(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), index=years).dropna()
        st = stats(m)
        R1s[b] = st['R1']; R2s[b] = st['R2']; troughs[b] = st['trough']
    return R1s, R2s, troughs

print("\n== Inference ==")
for kind in ('predictor', 'calendar'):
    R1s, R2s, troughs = boot_fast(d_pub, kind)
    print(f"{kind}-bootstrap:  P(R1>0) = {np.mean(R1s > 0)*100:.1f}%  "
          f"(R1 90% CI [{np.percentile(R1s,5):.1f},{np.percentile(R1s,95):.1f}])   "
          f"P(R2>0) = {np.mean(R2s > 0)*100:.1f}%   "
          f"P(trough<=0) = {np.mean(troughs <= 0)*100:.1f}%  "
          f"(trough 90% CI [{np.percentile(troughs,5):.1f},{np.percentile(troughs,95):.1f}])")

# ---------------- placebo: shuffle publication years across predictors ----------------
print("\n== Placebo: publication years shuffled across predictors (1000x) ==")
sig_year = doc.set_index('Acronym')['Year']
base = ls[['signalname', 'date', 'norm']].copy()
sigs = base['signalname'].unique()
pl_R1 = np.empty(1000)
for b in range(1000):
    perm = pd.Series(rng.permutation(sig_year.loc[sigs].values), index=sigs)
    df = base.assign(Year=base['signalname'].map(perm))
    _, m, _ = event_curve(df, 'Year')
    pl_R1[b] = stats(m)['R1']
p_placebo = np.mean(pl_R1 >= S['R1'])
print(f"observed R1 = {S['R1']:.1f}; placebo mean {pl_R1.mean():.1f}; "
      f"P(placebo >= observed) = {p_placebo:.3f}")

# ---------------- C1: sample-end alignment ----------------
S_end = stats(m_end)
print(f"\n== C1: sample-end alignment ==  dip(2-5) = {S_end['dip25']:.1f}  plateau = {S_end['plateau']:.1f}  "
      f"R1 = {S_end['R1']:.1f}  trough = {S_end['trough']:.1f} (yr {S_end['trough_year']})  R2 = {S_end['R2']:.1f}")
print(f"publication-alignment sharpening: R1_pub - R1_end = {S['R1'] - S_end['R1']:.1f}")

# ---------------- figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax = axes[0]
yrs = m_pub.index.values
ax.axhline(100, color=GREY, lw=0.9, ls=":"); ax.axhline(0, color=INK, lw=0.9)
ax.axhspan(S['plateau']-2, S['plateau']+2, color=GREY, alpha=0.25, lw=0)
ax.axvspan(1.5, 5.5, color=WARM, alpha=0.10, lw=0)
ax.bar(yrs, m_pub.values, color=[WARM if 2 <= y <= 5 else ACCENT for y in yrs], width=0.8)
ax.set_ylim(-32, 152)
ax.annotate(f"trough: year {S['trough_year']}, {S['trough']:.0f}", (S['trough_year'], S['trough']),
            xytext=(S['trough_year']+2.5, S['trough']-16), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
ax.text(11.0, S['plateau']+5, f"plateau (years 8–15) = {S['plateau']:.0f}", fontsize=8.5, color=INK, ha="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
ax.text(15.4, 103, "in-sample mean = 100", fontsize=8, color=INK, ha="right")
ax.set_xlabel("event years since publication"); ax.set_ylabel("normalized L/S return")
ax.set_title("aligned at publication, unsmoothed (204 predictors)", fontsize=10)
style(ax); panel(ax, "A")
ax = axes[1]
ax.axhline(100, color=GREY, lw=0.9, ls=":"); ax.axhline(0, color=INK, lw=0.9)
ax.plot(m_pub.index, m_pub.values, "o-", color=ACCENT, lw=1.8, ms=4, label="aligned at publication")
ax.plot(m_end.index, m_end.values, "s--", color=GREY, lw=1.4, ms=4, label="aligned at sample end (panel convention)")
ax.set_xlabel("event years"); ax.legend(frameon=False, fontsize=8.5)
ax.set_title("alignment comparison", fontsize=10)
style(ax); panel(ax, "B")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "e004_undershoot.png"), dpi=600)
print("\nfigure written: fig/e004_undershoot.png")
np.save(os.path.join(BASE, "data", "e004_placebo_R1.npy"), pl_R1)
m_pub.to_csv(os.path.join(BASE, "data", "e004_curve_pub.csv"))
m_end.to_csv(os.path.join(BASE, "data", "e004_curve_end.csv"))
