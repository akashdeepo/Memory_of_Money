# The Memory of Money: Track-Record Windows and Whether Crowding Cycles

Replication package for the paper of that name (Akash Deep). Every number, table and
figure in the paper is produced by the scripts in `code/`; nothing is hand-computed.

> **Status.** The manuscript is under review. This repository is archived on Zenodo;
> cite the DOI in `CITATION.cff` rather than the moving `main` branch.


---

## 1. Environment

Python 3.13.3 was used for the results in the paper. Any Python ≥ 3.11 should work.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Exact package versions used for the published results are pinned in
`requirements-frozen.txt`. The looser `requirements.txt` is sufficient to reproduce the
results to plotting precision.

## 2. Data

One of the five data sources is freely redistributable and is included in `data/`. The
rest must be obtained from their providers before the corresponding scripts will run;
their licences do not permit redistribution here. See `data/README.md` for the exact
retrieval steps and the expected file layouts.

| Source | Used by | Included? | How to obtain |
|---|---|---|---|
| Open Source Asset Pricing, release 2025.10: `SignalDoc` | `e004*` | yes | included |
| Open Source Asset Pricing: `PredictorPortsFull`, `PlaceboPortsFull` | `e004*` | no | `pip install openassetpricing`; see `data/README.md` |
| BarclayHedge CTA industry and systematic-trader assets under management | `e005b`, `e005c` | no | barclayhedge.com |
| BarclayHedge BTOP50 index, monthly | `e005b`, `e005c` | no | barclayhedge.com |
| AQR Time Series Momentum Factors, monthly | `e005b`, `e005c` | no | AQR Data Library |
| Global Factor Data; Kenneth French library | Section 7 test (not yet run) | no | see paper, Appendix C |

The theory scripts (`e001*`, `e002*`, `e003*`, `e005_power`, `fig_loop_schematic`) need
no data at all and reproduce Table 1, Table 5 and Figures 1--5, 7 and 8 out of the box.

## 3. What produces what

Run any script from the repository root, e.g. `python code/e001_decay.py`. Scripts are
independent of one another except where a dependency is noted.

### Theory (Sections 3, and Appendices A and B)

| Paper object | Script | Notes |
|---|---|---|
| Table 1, rows 1–3 (exponential, pure lag, trailing window) | `e001_decay.py` | thresholds located by bisection on the measured linear growth rate |
| Table 1, rows 4–7 (window + decision lag) | `e001e_boxlag_thresholds.py` | closed form and numerical root-finding on the unwrapped phase, agreeing to 2e-16 |
| Figure 2 (`fig:e001`), exponential and pure-lag regimes | `e001_decay.py` | also writes `e001_box_window.png`, not used in the paper |
| Section 3.4 verification numbers (ringing frequency 2.1815, decay 0.5008, thresholds 1.5787 and 4.959) | `e001_decay.py` | printed to stdout |
| Section 3.4 saturating-impact thresholds (κ\* = π, 3π/4); Corollary 1 | `e001c_saturating.py` | reproduces Proposition 1's impact-law independence |
| Figure 3 (`fig:e002`), threshold curves and regime map | `e002_tugofwar.py` | slow: 13 thresholds by bisection, each an integrated DDE |
| Propositions 4–6, thirteen thresholds to <0.05% | `e002_tugofwar.py` | printed to stdout |
| Appendix B.2, conservation law residual scaling | `e002_verify_D2.py` | symbolic check plus O(h²) residual scaling |
| Appendix B.3, bankroll coexistence identities | `e003_bankroll.py` | writes `e003_results.pkl` |

### Aggregation and the anomaly panel (Section 6)

| Paper object | Script | Notes |
|---|---|---|
| Figure 5 top (`fig:e001b`), 400-strategy ensemble under a pure lag | `e001b_averaging.py` | |
| Figure 5 bottom, same ensemble under a trailing window below threshold | `e001d_audit_kernel.py` | source of the 3.3-point undershoot and the 0.013%/month figure |
| Figure 6 (`fig:e004`), publication-aligned event study | `e004_undershoot.py` | needs OSAP; 4000-draw cluster and calendar bootstraps, plus a 1000-draw shuffled-publication placebo |
| Section 6.2 placebo-portfolio results | `e004b_placebo.py` | needs OSAP placebo portfolios |
| Section 6.2 calendar-month fixed effects | `e004c_calendar.py` | needs OSAP |
| Figure 7 (`fig:did`), predictors against placebos, difference-in-differences | `e004d_did.py` | needs OSAP |
| Section 6.2 cross-sectional regression | `e004e_crowdability.py` | needs OSAP; writes `data/e004_crowdability.csv` |
| Table 4 (`tab:cohort`), year-three gap by publication cohort | `e004f_cohort.py` | needs OSAP; this is the diagnostic that identifies the artifact |

### Measurement and power (Sections 5 and 7)

| Paper object | Script | Notes |
|---|---|---|
| Table 3 (`tab:cta`) point estimates, and Figure 4 (`fig:cta`) | `e005b_cta_kernel.py` | Almon kernel, capacity regression, HAC standard errors |
| Table 3 (`tab:cta`) 90% intervals and tail probabilities | `e005c_cta_bootstrap.py` | circular block bootstrap, blocks of 8 quarters and 3 years, 2000 draws, seed 5; prints a row-by-row comparison against the values in the paper |
| series loading and model specification shared by the two above | `cta_data.py` | not run directly |
| Table 5 (`tab:power`), power of the pooled spectral test | `e005_power.py` | slow: 10 cells × 60 replications, each with a parametric bootstrap null |

Figure 1 (`fig:loop`) is drawn in TikZ inside the manuscript and has no script.
Table 2 (`tab:kernellit`) is a synthesis of published estimates; its sources are the
citations in the table itself.

### Regenerating every figure as vector PDF

```bash
python code/_regen_pdf.py e001_decay.py e001b_averaging.py e001d_audit_kernel.py \
    e002_tugofwar.py e004_undershoot.py e004d_did.py e005b_cta_kernel.py
```

This patches `Figure.savefig` so each PNG is written with a PDF sibling, and writes
into `fig/`.

## 4. Random seeds and reproducibility

Every stochastic script seeds its generator explicitly (`np.random.default_rng(...)`),
so bootstrap intervals and simulated ensembles reproduce exactly on the same package
versions. Threshold-finding is deterministic. `e005c_cta_bootstrap.py` prints its
recomputed values next to the ones printed in Table 3, so any divergence is visible
without reading the paper. Reported thresholds carry a small upward
bias from finite integration time, the signature of critical slowing down near a
bifurcation; this is discussed in Section 3.4 and is not a numerical error.

## 5. Known gaps

The spectral test of Section 7 has not been run: `e005_power.py` establishes its power,
but the test itself awaits the factor panels listed in Appendix C. When it is run, the
code will be added here and released as a new version.

## 6. License

Code is released under the MIT License (see `LICENSE`). Data files retain the licenses
of their original providers.
