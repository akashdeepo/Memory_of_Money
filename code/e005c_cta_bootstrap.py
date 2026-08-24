"""Circular block bootstrap for the managed-futures kernel, capacity and loop gain.

Produces the 90% intervals and tail probabilities of Table 3 (tab:cta).

Design, as described in Section 5.1 of the paper:
  - quarterly Almon regression (systematic assets, n = 90 quarters): circular block
    bootstrap with blocks of eight quarters, giving kappa, the kernel centroid tau and
    the peak lag in each draw;
  - annual capacity regression (industry assets, TSMOM excess return): circular block
    bootstrap with blocks of three years, giving dp/dlogC;
  - the assembled quantities b = kappa |dp/dlogC| and g_eff = b tau are formed
    draw-by-draw by pairing the two resamples, which are independent.
  - 2000 draws, seeded.

In draws where the estimated total flow response kappa is not positive the kernel
centroid is not a weighted mean of lags and the loop does not exist; those draws are
excluded from the intervals for tau, b and g_eff (7% of draws) and their frequency is
reported. Kappa's own interval, and the peak lag, use every finite draw.

Run:  python sim/e005c_cta_bootstrap.py
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cta_data import (load_series, quarterly_design, annual_capacity_design,
                      fit_quarterly, fit_capacity)

N_DRAWS = 2000
BLOCK_Q = 8          # quarters
BLOCK_A = 3          # years
SEED = 5

rng = np.random.default_rng(SEED)


def circular_block_index(n, block, rng):
    """Indices of a circular block bootstrap resample of length n."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    idx = np.concatenate([(s + np.arange(block)) % n for s in starts])
    return idx[:n]


def main():
    aum_ann, sysq, btop, tsmom = load_series()
    dqq = quarterly_design(sysq, btop)
    dc = annual_capacity_design(aum_ann, tsmom, btop)
    print(f"quarterly design n = {len(dqq)};  annual capacity design n = {len(dc)}")

    # ---- point estimates (HAC), for reference ----
    w0, se0, kappa0, tau0, peak0 = fit_quarterly(dqq)
    slope0, se_slope0 = fit_capacity(dc)
    b0 = max(kappa0, 0.0) * abs(slope0)
    g0 = b0 * tau0
    print(f"\npoint estimates: peak = {peak0:.2f} y, tau = {tau0:.2f} y, "
          f"kappa = {kappa0:.2f}, dp/dlogC = {slope0:+.3f} (se {se_slope0:.3f}), "
          f"b = {b0:.3f}/y, g_eff = {g0:.2f}")

    # ---- bootstrap ----
    nq, na = len(dqq), len(dc)
    peaks = np.full(N_DRAWS, np.nan)
    taus = np.full(N_DRAWS, np.nan)
    kappas = np.full(N_DRAWS, np.nan)
    slopes = np.full(N_DRAWS, np.nan)
    for d in range(N_DRAWS):
        qi = circular_block_index(nq, BLOCK_Q, rng)
        try:
            _, _, kap, tau, peak = fit_quarterly(dqq.iloc[qi], hac=None)
            kappas[d], taus[d], peaks[d] = kap, tau, peak
        except Exception:
            pass
        ai = circular_block_index(na, BLOCK_A, rng)
        try:
            slopes[d], _ = fit_capacity(dc.iloc[ai], hac=None)
        except Exception:
            pass

    finite = np.isfinite(kappas) & np.isfinite(taus) & np.isfinite(slopes)
    # The kernel centroid tau = sum(k w_k)/sum(w_k) is not defined as a lag when the
    # total flow response is not positive: as kappa -> 0 it diverges, and for kappa < 0
    # it is not a weighted mean at all. Draws in which the estimated loop does not
    # exist are therefore excluded from the intervals for tau, b and g_eff, and the
    # frequency of such draws is reported instead.
    ok = finite & (kappas > 0)
    print(f"usable draws: {finite.sum()} / {N_DRAWS} finite; "
          f"{ok.sum()} with kappa > 0 ({100*np.mean(kappas[finite] <= 0):.0f}% discarded)")
    kap, tau, slope, peak = kappas[ok], taus[ok], slopes[ok], peaks[ok]
    peak_all = peaks[finite]          # the peak lag is defined whatever the sign of kappa
    b = kap * np.abs(slope)
    g = b * tau
    kap_all = kappas[finite]          # kappa's own interval uses every finite draw

    def ci(x, lo=5, hi=95):
        return np.percentile(x, lo), np.percentile(x, hi)

    print("\n== Table 3: point estimate and 90% interval ==")
    print(f"  kernel peak lag        {peak0:6.2f} y   "
          f"P(peak >= 2y) = {100*np.mean(peak_all >= 2.0):.0f}%")
    print(f"  kernel centroid tau    {tau0:6.2f} y   [{ci(tau)[0]:.1f}, {ci(tau)[1]:.1f}]")
    print(f"  flow sensitivity kappa {kappa0:6.2f}     [{ci(kap_all)[0]:.2f}, {ci(kap_all)[1]:.2f}]   "
          f"P(kappa < 0) = {100*np.mean(kap_all < 0):.0f}%")
    print(f"  impact slope dp/dlogC  {slope0:+6.3f}    [{ci(slope)[0]:+.3f}, {ci(slope)[1]:+.3f}]   "
          f"P(<0) = {100*np.mean(slope < 0):.0f}%")
    print(f"  loop rate b            {b0:6.3f}/y   [{ci(b)[0]:.2f}, {ci(b)[1]:.2f}]")
    print(f"  loop gain g_eff        {g0:6.2f}     [{ci(g)[0]:.2f}, {ci(g)[1]:.2f}]   "
          f"P(g > pi/2) = {100*np.mean(g > np.pi/2):.1f}%   "
          f"P(g < 1/4) = {100*np.mean(g < 0.25):.0f}%")

    # ---- comparison with the values printed in the paper ----
    print("\n== agreement with Table 3 as published ==")
    published = [
        ("peak >= 2y",      100*np.mean(peak_all >= 2.0), 72.0, "%"),
        ("tau lo",          ci(tau)[0],                 1.7,  "y"),
        ("tau hi",          ci(tau)[1],                 5.8,  "y"),
        ("kappa lo",        ci(kap_all)[0],           -0.16,  ""),
        ("kappa hi",        ci(kap_all)[1],            3.49,  ""),
        ("slope lo",        ci(slope)[0],            -0.071,  ""),
        ("slope hi",        ci(slope)[1],            -0.024,  ""),
        ("b lo",            ci(b)[0],                  0.02,  "/y"),
        ("b hi",            ci(b)[1],                  0.16,  "/y"),
        ("g lo",            ci(g)[0],                  0.05,  ""),
        ("g hi",            ci(g)[1],                  0.40,  ""),
        ("P(g > pi/2)",     100*np.mean(g > np.pi/2),   0.0,  "%"),
        ("P(g < 1/4)",      100*np.mean(g < 0.25),     73.0,  "%"),
    ]
    for name, got, want, unit in published:
        flag = "" if abs(got - want) <= max(0.05 * abs(want), 0.02) else "   <-- DIFFERS"
        print(f"  {name:<14} recomputed {got:+8.3f}{unit:<3}  published {want:+8.3f}{unit:<3}{flag}")


if __name__ == "__main__":
    main()
