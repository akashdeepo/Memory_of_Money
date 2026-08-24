"""Shared loading and model fitting for the managed-futures analysis.

Used by e005b_cta_kernel.py (point estimates and figure) and
e005c_cta_bootstrap.py (block-bootstrap intervals), so that both work from
exactly the same series and specification.

Data files live in data/cta/ and are described in the replication README.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "cta")
L_QUARTERS, ALMON_DEG = 16, 3


# ----------------------------------------------------------------- raw series
def load_series():
    """Return (aum_ann, sysq, btop_monthly, tsmom_monthly)."""
    ind = pd.read_excel(os.path.join(DATA, "MUM_CTA_Industry.xls"), header=None).iloc[2:]
    ind.columns = ["year", "q1", "q2", "q3", "q4"]
    ind = ind.apply(pd.to_numeric, errors="coerce").dropna(subset=["year"])
    aum_ann = ind.set_index("year")["q4"].dropna()

    sysq = pd.read_excel(os.path.join(DATA, "quarterly_MUM_Sys.xls"),
                         header=None).iloc[2:, :5]
    sysq.columns = ["year", "q1", "q2", "q3", "q4"]
    sysq = sysq.apply(pd.to_numeric, errors="coerce").dropna(subset=["year"])
    sysq = sysq.melt(id_vars="year", var_name="q", value_name="aum").dropna()
    sysq["t"] = sysq["year"] + sysq["q"].str[1].astype(int) / 4.0
    sysq = sysq.sort_values("t").set_index("t")["aum"]
    sysq = sysq[sysq.index >= 2000.0]

    b = pd.read_excel(os.path.join(DATA, "btop50_monthly.xls"), header=None)
    rows = []
    for _, r in b.iterrows():
        try:
            y = int(r[0])
        except (TypeError, ValueError):
            continue
        vals = pd.to_numeric(r[1:13], errors="coerce").values
        if 1900 < y < 2100 and np.nanmax(np.abs(vals)) < 1.0:
            for m, v in enumerate(vals, 1):
                if not np.isnan(v):
                    rows.append((pd.Timestamp(y, m, 1), v))
    btop = pd.Series(dict(rows)).sort_index()

    a = pd.read_excel(os.path.join(DATA, "aqr_tsmom_monthly.xlsx"),
                      sheet_name="TSMOM Factors", header=None)
    a = a[pd.to_datetime(a[0], errors="coerce").notna()]
    tsmom = pd.Series(pd.to_numeric(a[1], errors="coerce").values,
                      index=pd.to_datetime(a[0]).dt.to_period("M").dt.to_timestamp()).dropna()
    return aum_ann, sysq, btop, tsmom


def compound(s, freq):
    """Compound monthly returns to annual ('Y') or quarterly ('Q') totals."""
    return (1 + s).groupby(s.index.to_period(freq)).prod() - 1


# ----------------------------------------------------- design matrices (once)
def quarterly_design(sysq, btop):
    """Rows of the quarterly Almon regression: (y, R0, R1..RL) as a DataFrame."""
    Rq = compound(btop, "Q")
    Rq.index = Rq.index.year + Rq.index.quarter / 4.0
    dq = pd.DataFrame({"A": sysq, "R": Rq}).dropna()
    dq["flow"] = (dq["A"] - dq["A"].shift(1) * (1 + dq["R"])) / dq["A"].shift(1)
    for k in range(1, L_QUARTERS + 1):
        dq[f"R{k}"] = dq["R"].shift(k)
    dq["R0"] = dq["R"]
    return dq.dropna()


def annual_capacity_design(aum_ann, tsmom, btop):
    """Rows of the capacity regression: next-year TSMOM excess return on log assets.

    The BTOP50 next-year total return is carried as a column and included in the
    listwise deletion, so that the TSMOM and BTOP50 capacity slopes reported in
    Section 5.2 are estimated on a common sample of years.
    """
    Rt = compound(tsmom, "Y")
    Rt.index = Rt.index.year
    Rb = compound(btop, "Y")
    Rb.index = Rb.index.year
    return pd.DataFrame({"logA": np.log(aum_ann),
                         "R_next": Rt.shift(-1),
                         "B_next": Rb.shift(-1)}).dropna()


# ------------------------------------------------------------------- fitting
_ALMON_M = np.vstack([np.arange(1, L_QUARTERS + 1) ** j
                      for j in range(ALMON_DEG + 1)]).T          # L x (deg+1)


def fit_quarterly(dqq, hac=4):
    """Fit the Almon kernel. Returns (weights, se_weights, kappa, centroid_years, peak_years)."""
    lags = np.arange(1, L_QUARTERS + 1)
    Z = dqq[[f"R{k}" for k in lags]].values @ _ALMON_M
    X = sm.add_constant(np.column_stack([dqq["R0"].values, Z]))
    kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": hac}} if hac else {}
    m = sm.OLS(dqq["flow"].values, X).fit(**kw)
    a_hat = m.params[2:]
    w = _ALMON_M @ a_hat
    se_w = np.sqrt(np.diag(_ALMON_M @ m.cov_params()[2:, 2:] @ _ALMON_M.T)) if hac else None
    kappa = w.sum()
    centroid = np.sum(lags * w) / kappa / 4 if kappa != 0 else np.nan
    peak = lags[np.argmax(w)] / 4
    return w, se_w, kappa, centroid, peak


def fit_capacity(dc, hac=2):
    """Fit next-year excess return on log assets. Returns (slope, se)."""
    kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": hac}} if hac else {}
    m = sm.OLS(dc["R_next"], sm.add_constant(dc[["logA"]])).fit(**kw)
    return m.params["logA"], m.bse["logA"]
