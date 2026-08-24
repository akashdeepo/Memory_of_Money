"""Cross-sectional check (Section 6.2): does the publication-year-3 dip scale with
ex-ante CROWDABILITY? Per-signal calendar-demeaned year-3 gap regressed on
attention (citations), capacity (breadth, VW flag), in-sample strength, controls.
Crowding => attention negative, capacity positive. Selection => t-stat loads, capacity not.
"""
import os, sys, warnings
import numpy as np, pandas as pd, statsmodels.api as sm
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
doc = pd.read_csv(os.path.join(BASE, "data", "SignalDoc.csv"))

def load(kind):
    f = "PredictorPortsFull.parquet" if kind == "Predictor" else "PlaceboPortsFull.parquet"
    p = pd.read_parquet(os.path.join(BASE, "data", f))
    ls = p[p['port'] == 'LS'][['signalname', 'date', 'ret', 'Nlong', 'Nshort']].copy()
    ls['date'] = pd.to_datetime(ls['date'])
    d = doc[doc['Cat.Signal'] == kind][['Acronym', 'Year', 'SampleStartYear', 'SampleEndYear',
                                        'GScholarCites202509', 'Stock Weight']]
    ls = ls.merge(d, left_on='signalname', right_on='Acronym', how='inner')
    ins = ls[(ls['date'].dt.year >= ls['SampleStartYear']) & (ls['date'].dt.year <= ls['SampleEndYear'])]
    g = ins.groupby('signalname')['ret']
    stat = pd.DataFrame({'mu': g.mean(), 'sd': g.std(), 'n_ins': g.size()})
    stat['tstat'] = stat['mu'] / (stat['sd'] / np.sqrt(stat['n_ins']))
    ls = ls.merge(stat, on='signalname')
    ls = ls[ls['n_ins'] >= 60].copy()
    ls['y'] = ls['ret'] * np.sign(ls['mu'])
    ls['y_dm'] = ls['y'] - ls.groupby('date')['y'].transform('mean')
    em = (ls['date'].dt.year - ls['Year']) * 12 + ls['date'].dt.month
    ls['ey'] = np.where(em > 12, np.ceil((em - 12) / 12.0), np.floor((em - 1) / 12.0)).astype(int)
    ls['kind'] = kind
    return ls

def main():
    both = pd.concat([load("Predictor"), load("Placebo")])
    # per-signal year-3 gap = mean(ey==3) - mean(ey in {1,2,4,5})
    sub = both[both.ey.isin([1, 2, 3, 4, 5])]
    piv = sub.pivot_table(index='signalname', columns='ey', values='y_dm', aggfunc='mean')
    gap = (piv[3] - piv[[1, 2, 4, 5]].mean(axis=1)).rename('gap')

    meta = both.groupby('signalname').agg(
        kind=('kind', 'first'), Year=('Year', 'first'),
        cites=('GScholarCites202509', 'first'), sw=('Stock Weight', 'first'),
        tstat=('tstat', 'first'),
        breadth=('Nlong', lambda x: x.mean()), nshort=('Nshort', 'mean'))
    X = meta.join(gap, how='inner').dropna(subset=['gap', 'cites'])
    X['nstocks'] = X['breadth'] + X['nshort']
    X['log_cites'] = np.log(X['cites'])
    X['cites_per_yr'] = np.log(X['cites'] / np.maximum(2026 - X['Year'], 1))
    X['log_cap'] = np.log(np.maximum(X['nstocks'], 1))
    X['vw'] = (X['sw'] == 'VW').astype(float)
    X['is_pred'] = (X['kind'] == 'Predictor').astype(float)
    X['tstat'] = X['tstat'].clip(-15, 15)
    X['cohort'] = pd.cut(X['Year'], [1960, 1995, 2003, 2008, 2020], labels=False)
    print(f"n signals = {len(X)}  (predictors {int(X.is_pred.sum())}, placebos {int((1-X.is_pred).sum())})")
    print(f"mean gap = {X.gap.mean():+.3f} %/mo   median {X.gap.median():+.3f}")

    def run(cols, label, data=X):
        d = data.dropna(subset=cols + ['gap'])
        A = sm.add_constant(pd.get_dummies(d[cols + ['cohort']], columns=['cohort'], drop_first=True).astype(float))
        res = sm.OLS(d['gap'], A).fit(cov_type='HC3')
        print(f"\n-- {label}  (n={int(res.nobs)}, R2={res.rsquared:.3f}) --")
        for c in cols:
            print(f"   {c:14s} beta={res.params[c]:+.4f}  se={res.bse[c]:.4f}  t={res.tvalues[c]:+.2f}  p={res.pvalues[c]:.3f}")
        return res

    run(['log_cites'], "attention only")
    run(['log_cites', 'log_cap', 'vw'], "attention + capacity")
    run(['log_cites', 'log_cap', 'vw', 'tstat', 'is_pred'], "full model")
    run(['cites_per_yr', 'log_cap', 'vw', 'tstat', 'is_pred'], "attention as cites/yr")
    run(['log_cites', 'log_cap', 'vw', 'tstat'], "predictors only", X[X.is_pred == 1])
    run(['log_cites', 'log_cap', 'vw', 'tstat'], "placebos only", X[X.is_pred == 0])

    print("\n-- quintile sort on attention (log citations) --")
    X['q'] = pd.qcut(X['log_cites'], 5, labels=False)
    print(X.groupby('q')['gap'].agg(['mean', 'median', 'size']).round(3).to_string())
    print("\n-- quintile sort on capacity (log #stocks) --")
    X['qc'] = pd.qcut(X['log_cap'], 5, labels=False)
    print(X.groupby('qc')['gap'].agg(['mean', 'median', 'size']).round(3).to_string())
    X.to_csv(os.path.join(BASE, "data", "e004_crowdability.csv"))


if __name__ == '__main__':
    main()
