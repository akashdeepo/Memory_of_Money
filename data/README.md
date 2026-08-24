# Data

Only freely redistributable data is included here. The three vendor series used in the
managed-futures analysis are **not** redistributed, because their providers do not grant
redistribution rights; the scripts read them from this directory once you have obtained
them yourself. This is the usual arrangement for vendor data in finance and does not
affect reproducibility: every step from the raw files onward is in `code/`.

## Included

- `SignalDoc.csv` — signal documentation (publication years, sample windows) from the
  Open Source Asset Pricing project of Chen and Zimmermann, release 2025.10, which is
  published openly at <https://www.openassetpricing.com>. Please cite their paper.

## To obtain: Open Source Asset Pricing portfolio returns

Needed by `e004*.py`. Roughly 30 MB.

```python
pip install openassetpricing
import openassetpricing as oap
op = oap.OpenAP(2025.10)
op.dl_port('PredictorPortsFull', 'pandas').to_parquet('data/PredictorPortsFull.parquet')
```

The 114 placebo portfolios (`PlaceboPortsFull`) are not exposed by the package; download
them from the release's public Google Drive folder linked at openassetpricing.com and
save as `data/PlaceboPortsFull.parquet`.

## To obtain: managed-futures series

Needed by `e005b_cta_kernel.py` and `e005c_cta_bootstrap.py`. Save all three into
`data/cta/`.

| File | Source | Notes |
|---|---|---|
| `MUM_CTA_Industry.xls` | BarclayHedge, CTA industry assets under management | year-end 1980 onward, quarterly from 2000; the sheet is a year x quarter grid with two header rows |
| `quarterly_MUM_Sys.xls` | BarclayHedge, systematic-trader assets under management | quarterly from 1999Q4; same grid layout |
| `btop50_monthly.xls` | BarclayHedge, BTOP50 index monthly rates of return | monthly from 1987; year in column 0, twelve monthly returns as decimals |
| `aqr_tsmom_monthly.xlsx` | AQR Data Library, "Time Series Momentum Factors" | sheet `TSMOM Factors`; monthly from 1985 |

`code/cta_data.py` documents exactly how each file is parsed, so you can check that a
newer vintage still matches the expected layout before running the analysis.
