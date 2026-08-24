"""Preregistered spectral test, stage one: build the panel and run the power gate.

Implements steps 1-3 of the registered procedure (OSF rhke5, DOI 10.17605/OSF.IO/RHKE5)
and STOPS. No test statistic is computed on real data by this script: the registration
requires the effective-N power gate to be evaluated before any test statistic, and its
verdict reported either way.

Registered classification (fixed in the filing, not revisited here):
  heavily chased : value, size, momentum, low risk, quality, profitability
  lightly chased : accruals, debt issuance, investment, low leverage, profit growth,
                   seasonality, short-term reversal
  macro control  : the market factor
Series whose theme is in neither list (e.g. carry, composite "All" columns) are assigned
to no pool; they are counted and reported.

Implementation choices documented here because the registration does not fix them:
  - "slow component (periods of two years and above)": FFT low-pass per series,
    frequencies strictly above 1/24 cycles per month zeroed, on the series' longest
    contiguous non-missing run, demeaned first.
  - pairwise correlations use months where both series' slow components exist, and a
    pair enters rbar only with at least 120 overlapping months.
  - the power re-run uses N = round(effective N) independent simulated series and
    T = the median included-series length in the heavily chased pool, with the
    registered gate point P = 96 months, zeta = 0.3, at the 5% level, R = 60.

Run:  python sim/e005d_gate.py
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FDIR = os.path.join(BASE, "data", "factors")
MIN_MONTHS = 360          # registered inclusion rule
MIN_OVERLAP = 120         # documented implementation choice
CUTOFF = 1.0 / 24.0       # cycles per month: periods >= 2 years

HEAVY = {"value", "size", "momentum", "low_risk", "quality", "profitability"}
LIGHT = {"accruals", "debt_issuance", "investment", "low_leverage",
         "profit_growth", "seasonality", "short_term_reversal"}

# JKP theme spellings observed in the file -> registered theme names
JKP_MAP = {
    "value": "value", "size": "size", "momentum": "momentum",
    "low_risk": "low_risk", "quality": "quality", "profitability": "profitability",
    "accruals": "accruals", "debt_issuance": "debt_issuance",
    "investment": "investment", "low_leverage": "low_leverage",
    "profit_growth": "profit_growth", "seasonality": "seasonality",
    "short_term_reversal": "short_term_reversal", "skewness": None,
}


def month_index(dates):
    return pd.to_datetime(dates).dt.to_period("M")


def add_series(store, source, theme, label, s):
    """s: pd.Series indexed by Period[M]."""
    s = s.dropna().astype(float)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    if len(s) == 0:
        return
    store.append(dict(source=source, theme=theme, label=label, s=s))


def load_jkp(store):
    for zname, kind in (("jkp_themes.zip", "theme"), ("jkp_mkt.zip", "mkt")):
        zz = zipfile.ZipFile(os.path.join(FDIR, zname))
        with zz.open(zz.namelist()[0]) as f:
            df = pd.read_csv(f)
        df["m"] = month_index(df["date"])
        for (name, loc), g in df.groupby(["name", "location"]):
            theme = "market" if kind == "mkt" else JKP_MAP.get(name, name)
            if theme is None:
                theme = "unclassified"
            add_series(store, "JKP", theme, f"jkp:{name}:{loc}",
                       g.set_index("m")["ret"])


def load_devil(store):
    sheets = {"MKT": "market", "SMB": "size", "HML FF": "value", "UMD": "momentum"}
    xls = os.path.join(FDIR, "aqr_devil_monthly.xlsx")
    for sheet, theme in sheets.items():
        df = pd.read_excel(xls, sheet_name=sheet, header=18)
        df = df.rename(columns={df.columns[0]: "DATE"}).dropna(subset=["DATE"])
        df["m"] = month_index(df["DATE"])
        for col in df.columns:
            if col in ("DATE", "m") or str(col).startswith("Unnamed"):
                continue
            add_series(store, "AQR-Devil", theme, f"devil:{sheet}:{col}",
                       pd.to_numeric(df.set_index("m")[col], errors="coerce"))


def century_theme(colname):
    c = str(colname).lower()
    if "value" in c:
        return "value"
    if "momentum" in c:
        return "momentum"
    if "defensive" in c:
        return "low_risk"
    if "carry" in c:
        return "unclassified"       # not in either registered list
    return "unclassified"           # composites such as "All ..." columns


def load_century(store):
    xls = os.path.join(FDIR, "aqr_century_monthly.xlsx")
    df = pd.read_excel(xls, sheet_name="Century of Factor Premia", header=18)
    df = df.rename(columns={df.columns[0]: "DATE"}).dropna(subset=["DATE"])
    df["m"] = month_index(df["DATE"])
    for col in df.columns:
        if col in ("DATE", "m") or str(col).startswith("Unnamed"):
            continue
        add_series(store, "AQR-Century", century_theme(col),
                   f"century:{col}", pd.to_numeric(df.set_index("m")[col], errors="coerce"))


def read_french_zip(zname):
    zz = zipfile.ZipFile(os.path.join(FDIR, zname))
    with zz.open(zz.namelist()[0]) as f:
        lines = f.read().decode("utf8", errors="replace").splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.strip().startswith(",") and len(l.split(",")) > 1)
    rows = []
    cols = [c.strip() for c in lines[hdr].split(",")][1:]
    for l in lines[hdr + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0][:6].isdigit() or len(parts[0]) != 6:
            break                                    # end of the monthly block
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["m"] = pd.PeriodIndex(pd.to_datetime(df["date"], format="%Y%m"), freq="M")
    out = df.set_index("m")[cols].apply(pd.to_numeric, errors="coerce") / 100.0
    return out.where(out > -0.99)                    # -99.99 marks missing


def load_french(store):
    regions = [("F-F_Research_Data_Factors_CSV.zip", "F-F_Momentum_Factor_CSV.zip", "US"),
               ("Developed_3_Factors_CSV.zip", "Developed_Mom_Factor_CSV.zip", "DEV"),
               ("Europe_3_Factors_CSV.zip", "Europe_Mom_Factor_CSV.zip", "EU"),
               ("Japan_3_Factors_CSV.zip", "Japan_Mom_Factor_CSV.zip", "JP"),
               ("Asia_Pacific_ex_Japan_3_Factors_CSV.zip",
                "Asia_Pacific_ex_Japan_Mom_Factor_CSV.zip", "APxJ"),
               ("North_America_3_Factors_CSV.zip", "North_America_Mom_Factor_CSV.zip", "NA")]
    for f3, fm, reg in regions:
        d3 = read_french_zip(f3)
        for col, theme in (("Mkt-RF", "market"), ("SMB", "size"), ("HML", "value")):
            if col in d3:
                add_series(store, "French", theme, f"french:{col}:{reg}", d3[col])
        dm = read_french_zip(fm)
        mom = next((c for c in dm.columns if "mom" in c.lower()), None)
        if mom:
            add_series(store, "French", "momentum", f"french:Mom:{reg}", dm[mom])


def longest_run(s):
    """Longest contiguous monthly run of a Period-indexed series."""
    idx = s.index.astype("int64")
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(idx) - 1]])
    k = np.argmax(ends - starts)
    return s.iloc[starts[k]:ends[k] + 1]


def slow_component(s):
    x = s.values - s.values.mean()
    F = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1.0)
    F[f > CUTOFF] = 0.0
    return pd.Series(np.fft.irfft(F, n=len(x)), index=s.index)


def main():
    store = []
    load_jkp(store)
    load_devil(store)
    load_century(store)
    load_french(store)
    print(f"series loaded: {len(store)}")

    rows = []
    for d in store:
        run = longest_run(d["s"])
        keep = len(run) >= MIN_MONTHS
        pool = ("heavy" if d["theme"] in HEAVY else
                "light" if d["theme"] in LIGHT else
                "market" if d["theme"] == "market" else "none")
        rows.append(dict(label=d["label"], source=d["source"], theme=d["theme"],
                         pool=pool, months=len(run), included=keep))
        d["run"] = run
    cat = pd.DataFrame(rows)
    cat.to_csv(os.path.join(FDIR, "panel_catalog.csv"), index=False)

    print("\nby source (loaded / >=360m):")
    for src, g in cat.groupby("source"):
        print(f"  {src:12s} {len(g):4d} / {int(g.included.sum()):4d}")
    print("\nincluded series by pool:")
    inc = cat[cat.included]
    for pool, g in inc.groupby("pool"):
        print(f"  {pool:8s} {len(g):4d}   themes: "
              f"{dict(g.theme.value_counts())}")
    print(f"  excluded-by-length: {int((~cat.included).sum())}; "
          f"unpooled (unlisted themes): {len(inc[inc.pool=='none'])}")

    # slow components for the heavily chased pool
    heavy = [d for d in store if d["theme"] in HEAVY and len(d["run"]) >= MIN_MONTHS]
    slows = {d["label"]: slow_component(d["run"]) for d in heavy}
    labels = list(slows)
    N = len(labels)

    cors, skipped = [], 0
    for i in range(N):
        for j in range(i + 1, N):
            a, b = slows[labels[i]], slows[labels[j]]
            both = a.index.intersection(b.index)
            if len(both) < MIN_OVERLAP:
                skipped += 1
                continue
            c = np.corrcoef(a.loc[both].values, b.loc[both].values)[0, 1]
            if np.isfinite(c):
                cors.append(c)
    cors = np.array(cors)
    rbar = cors.mean()
    neff = N / (1.0 + (N - 1.0) * rbar)
    T_typ = int(np.median([len(d["run"]) for d in heavy]))

    print(f"\nheavily chased pool: N = {N}")
    print(f"pairs used = {len(cors)} (skipped for overlap < {MIN_OVERLAP}: {skipped})")
    print(f"rbar = {rbar:.4f}   (pairwise correlation of slow components; "
          f"10-90% [{np.percentile(cors,10):.3f}, {np.percentile(cors,90):.3f}])")
    print(f"effective N = {neff:.2f}    typical length T = {T_typ} months "
          f"({T_typ/12:.1f} years)")

    # registered gate: power at effective N, typical length, P = 8y, zeta = 0.3
    from e005_power import power
    N_sim = max(1, int(round(neff)))
    print(f"\nrunning the registered power gate: power(N={N_sim}, T={T_typ}, "
          f"P=96, zeta=0.3), R=60 ...")
    pw, med, (lo, hi) = power(N_sim, T_typ, 96, 0.3, R=60)
    print(f"power = {pw:.2f}   period estimate median {med/12:.1f}y "
          f"[{lo/12:.1f}, {hi/12:.1f}]")

    verdict = "PASS" if pw >= 0.8 else "FAIL"
    print(f"\nGATE {verdict}: power {pw:.2f} vs registered threshold 0.80")
    if verdict == "FAIL":
        print("Per the registration, the test is NOT run; this outcome is reported "
              "with the measured rbar and effective N.")
    else:
        print("Per the registration, the test proceeds (steps 4-7).")

    json.dump(dict(N=N, pairs=int(len(cors)), rbar=float(rbar),
                   neff=float(neff), N_sim=N_sim, T_typ=T_typ,
                   power=float(pw), gate=verdict),
              open(os.path.join(FDIR, "gate_result.json"), "w"), indent=1)
    print(f"\nwritten: data/factors/panel_catalog.csv, data/factors/gate_result.json")


if __name__ == "__main__":
    main()
