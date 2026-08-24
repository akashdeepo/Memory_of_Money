"""Stability thresholds for a trailing window followed by a decision lag.

Produces the box-plus-lag rows of Table 1 (tab:thresholds) in the paper, by two
independent routes:

  (a) closed form.  For a window of width tw preceded by nothing and followed by a
      decision lag D, the transfer function is

          W(s) = e^{-sD} (1 - e^{-s tw}) / (s tw),

      and at s = i*omega this factorises as

          W(i w) = e^{-i w (D + tw/2)} * sinc(w tw / 2),      sinc(x) = sin(x)/x,

      so for w tw < 2*pi the phase is exactly -w (D + tw/2).  A Hopf bifurcation
      needs phase = -pi/2, giving

          w*   = pi / (2D + tw)
          b*   = w* / sinc(w* tw / 2)
          T*   = 2 pi / w*

  (b) numerical root-finding on the unwrapped phase of W(i w) evaluated directly
      from the exponentials, with b* = w*/|W(i w*)|.

Route (b) makes no use of the sinc factorisation, so agreement between the two is a
check on the algebra of Appendix A.4.  The pure-window case (D = 0) must reproduce
b* tw = pi^2/2 and T* = 2 tw, and the pure-lag case is checked separately in
e001_decay.py.

Run:  python sim/e001e_boxlag_thresholds.py
"""
import numpy as np
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# transfer function of a trailing window of width tw followed by a lag D
# ----------------------------------------------------------------------


def W(s, tw, D):
    """W(s) = e^{-sD} (1 - e^{-s tw}) / (s tw), with the removable singularity at s=0."""
    if abs(s) < 1e-12:
        return complex(1.0, 0.0)
    return np.exp(-s * D) * (1.0 - np.exp(-s * tw)) / (s * tw)


def phase_excess(w, tw, D):
    """arg W(i w) + pi/2, unwrapped.  Zero at the marginal frequency.

    The phase is continuous and decreasing on 0 < w tw < 2 pi, so we track it by
    accumulating the principal argument along a fine grid rather than trusting
    np.angle's branch at a single point.
    """
    grid = np.linspace(1e-9, w, 4000)
    ang = np.unwrap([np.angle(W(1j * x, tw, D)) for x in grid])
    return ang[-1] + np.pi / 2


def threshold_numeric(tw, D):
    """Marginal (w*, b*, period) by root-finding on the unwrapped phase."""
    hi = 2 * np.pi / tw * 0.999          # stay inside the first sinc lobe
    w_star = brentq(phase_excess, 1e-6, hi, args=(tw, D), xtol=1e-14, rtol=1e-15)
    b_star = w_star / abs(W(1j * w_star, tw, D))
    return w_star, b_star, 2 * np.pi / w_star


def threshold_closed(tw, D):
    """Marginal (w*, b*, period) in closed form."""
    w_star = np.pi / (2 * D + tw)
    x = w_star * tw / 2
    b_star = w_star / (np.sin(x) / x)
    return w_star, b_star, 2 * np.pi / w_star


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # rows of Table 1, plus the pure-window row as a control
    rows = [
        ("3y window (no lag)", 3.0, 0.0, 1.645, 6.0),
        ("3y window + 6-month lag", 3.0, 0.5, 1.002, 8.0),
        ("3y window + 1-year lag", 3.0, 1.0, 0.732, 10.0),
        ("3y window + 18-month lag", 3.0, 1.5, 0.582, 12.0),
        ("5y window + 1-year lag", 5.0, 1.0, 0.559, 14.0),
    ]

    print("Table 1, box-plus-lag rows: b* (per year) and period at onset\n")
    hdr = f"{'kernel':<26}{'b* closed':>10}{'b* numeric':>12}{'paper':>8}" \
          f"{'period':>9}{'paper':>8}"
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for name, tw, D, b_paper, T_paper in rows:
        _, b_c, T_c = threshold_closed(tw, D)
        _, b_n, T_n = threshold_numeric(tw, D)
        worst = max(worst, abs(b_c - b_n) / b_c, abs(T_c - T_n) / T_c)
        print(f"{name:<26}{b_c:>10.4f}{b_n:>12.4f}{b_paper:>8.3f}"
              f"{T_c:>9.2f}{T_paper:>8.1f}")
        assert abs(b_c - b_paper) < 5e-3, f"{name}: b* disagrees with the paper"
        assert abs(T_c - T_paper) < 5e-2, f"{name}: period disagrees with the paper"

    print(f"\nclosed form vs numerical, worst relative difference: {worst:.2e}")

    # control: the pure window must give g* = b* tw = pi^2/2 and T* = 2 tw
    _, b_box, T_box = threshold_closed(3.0, 0.0)
    g_box = b_box * 3.0
    print(f"pure 3y window: g* = b* tw = {g_box:.6f}  (pi^2/2 = {np.pi**2 / 2:.6f}); "
          f"period = {T_box:.4f} y  (2 tw = 6 y)")
    assert abs(g_box - np.pi**2 / 2) < 1e-9
    assert abs(T_box - 6.0) < 1e-9

    # the decision lag is what does the damage: report the ratio
    b_ref = threshold_closed(3.0, 0.0)[1]
    print("\nthreshold reduction caused by the decision lag alone:")
    for name, tw, D, _, _ in rows[1:]:
        if tw == 3.0:
            print(f"  {name:<26} b*/b*(no lag) = {threshold_closed(tw, D)[1] / b_ref:.3f}")

    print("\nAll Table 1 box-plus-lag entries reproduced.")
