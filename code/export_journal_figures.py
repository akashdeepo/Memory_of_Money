"""Copy the paper's figures into submission/ named Figure 1 ... Figure 10.

The journal requires each figure as its own file, named according to its order in the
manuscript, at a minimum of 300 dpi. Run after regenerating the figures.

Run:  python sim/export_journal_figures.py
"""
import os
import shutil
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "fig")
OUT = os.path.join(BASE, "submission", "figures")

# document order -> source file (see paper/sections/*.tex)
ORDER = [
    (1, "fig_loop_schematic", "the performance-chasing loop and the two kernel classes"),
    (2, "e001_ema_regimes", "exponential memory relaxes at every gain"),
    (3, "e001_delay_bifurcation", "a pure lag cycles above pi/2"),
    (4, "e002_thresholds", "cycling threshold with opposing capital"),
    (5, "e002_regime_map", "three regimes with informed contrarians"),
    (6, "e005b_cta_kernel", "managed-futures flow kernel"),
    (7, "e001b_averaging", "aggregation hides the regime"),
    (8, "e001d_audit_kernel", "damped regime, single undershoot"),
    (9, "e004_undershoot", "publication-aligned event study"),
    (10, "e004d_did", "predictors against placebos"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        Image = None
    missing, rows = [], []
    for n, stem, what in ORDER:
        src = os.path.join(FIG, stem + ".png")
        if not os.path.exists(src):
            missing.append(stem)
            continue
        dst = os.path.join(OUT, f"Figure {n}.png")
        shutil.copyfile(src, dst)
        size = os.path.getsize(dst) / 1024
        if Image is not None:
            with Image.open(dst) as im:
                w, h = im.size
                dpi = im.info.get("dpi", (0, 0))[0]
            rows.append((n, stem, f"{w}x{h}", f"{dpi:.0f}", f"{size:.0f} KB", what))
        else:
            rows.append((n, stem, "?", "?", f"{size:.0f} KB", what))

    print(f"wrote {len(rows)} figures to {OUT}\n")
    print(f"{'#':>2}  {'source':<24}{'pixels':<12}{'dpi':<6}{'size':<10}what")
    print("-" * 96)
    bad = []
    for n, stem, px, dpi, size, what in rows:
        flag = ""
        if dpi not in ("?",) and float(dpi) < 300:
            flag = "  <-- BELOW 300 dpi"
            bad.append(n)
        print(f"{n:>2}  {stem:<24}{px:<12}{dpi:<6}{size:<10}{what}{flag}")
    if missing:
        print("\nMISSING (regenerate these first):", ", ".join(missing))
    if bad:
        print("\nFigures below the journal's 300 dpi minimum:", bad)
    if not missing and not bad:
        print("\nAll figures present and at or above 300 dpi.")


if __name__ == "__main__":
    main()
