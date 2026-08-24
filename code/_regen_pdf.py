"""Re-run the figure scripts with savefig patched to also emit a vector PDF sibling."""
import runpy, sys, os, matplotlib
matplotlib.use("Agg")
import matplotlib.figure as mf
_orig = mf.Figure.savefig
def savefig(self, fname, *a, **k):
    _orig(self, fname, *a, **k)
    if isinstance(fname, str) and fname.lower().endswith(".png"):
        k2 = dict(k); k2.pop("dpi", None)
        _orig(self, fname[:-4] + ".pdf", *a, **k2)
mf.Figure.savefig = savefig
here = os.path.dirname(os.path.abspath(__file__))
for script in sys.argv[1:]:
    print("=== running", script, flush=True)
    sys.argv = [script]
    runpy.run_path(os.path.join(here, script), run_name="__main__")
