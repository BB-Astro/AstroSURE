"""Row-bias destriping of FLC chips with BB's StripeField engine.

Runs BB's destripe_astro (the StripeField backend, single source of
truth imported from ModulePixinsightByBB) on every SCI chip, BEFORE
AstroSURE. In detector coordinates the ACS striping is exactly
horizontal, so the angle is forced to 0 and the greedy scan is skipped.

Why upstream: AstroSURE partially erases the striping in a
content-dependent way, so on the denoised drizzle the residual is no
longer constant along a line and StripeField's own source mask excludes
the bright zones from the estimate (Wiener shrinks the unmeasured
profile to zero there: no correction precisely where BB sees the
residue). On the raw FLC the stripe is at full amplitude, the row
profile is measured on the row's background pixels and the correction
applies to the WHOLE row, bright zones included.

One pass per chip: the engine's standard high-frequency pass
(detrend 25, row-to-row striping). A low-frequency pass was tried on
10 Aug and REMOVED: every LF row-profile estimator tested (median per
row, column-block floor) measured mostly SKY, not banding (the band
profiles of the two exposures were 91-96% correlated at the dither lag
on the galaxy chip), and with two exposures the symmetric part of an
LF banding is indistinguishable from sky along the row-transverse
axis. The LF mottling seen in the denoised products is in the DATA
(the MAST DRC carries the same 0.0009 e-/s RMS at 25-400 px scales,
consistent with the ~1% ACS L-flat residual); it is a field
correction problem, not a destriping one. See PLAN_ENTRAINEMENT.md.

Meant to run AFTER deepcr_flc.py (CR-free rows give cleaner medians)
and BEFORE asure2_flc.py. Writes <stem>_dsf.fits siblings, SCI
replaced, ERR/DQ/WCS untouched.

Usage:
    python pipeline/destripe_flc.py exp1_flc_dcr.fits exp2_flc_dcr.fits
        [--win 1024] [--suffix _dsf]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

# The StripeField engine (destripe_astro.py) is vendored in this directory.
# Set STRIPEFIELD_DIR to use an external copy instead (e.g. the PixInsight
# distribution from github.com/BB-Astro/Pixinsight_Scripts).
STRIPEFIELD_DIR = Path(os.environ.get("STRIPEFIELD_DIR",
                                      Path(__file__).resolve().parent))
if not (STRIPEFIELD_DIR / "destripe_astro.py").exists():
    sys.exit(f"destripe_astro.py not found in {STRIPEFIELD_DIR}")
sys.path.insert(0, str(STRIPEFIELD_DIR))
from destripe_astro import destripe_greedy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="StripeField destriping on FLC chips")
    ap.add_argument("flc", nargs="+", help="FLC FITS files (ideally deepCR-cleaned)")
    ap.add_argument("--win", type=int, default=1024,
                    help="along-row window of the StripeField engine "
                         "(0 = constant profile along the row)")
    ap.add_argument("--suffix", default="_dsf")
    args = ap.parse_args()

    win = args.win if args.win > 0 else None
    for f in args.flc:
        path = Path(f).resolve()
        hdul = fits.open(path)
        n_sci = sum(1 for h in hdul if h.name == "SCI")
        for ver in range(1, n_sci + 1):
            sci = hdul["SCI", ver]
            res = destripe_greedy(sci.data.astype(np.float64),
                                  angles=[0.0], win=win, verbose=False)
            rms = float(np.std(res["stripes"]))
            sci.data = res["corrected"].astype(np.float32)
            sci.header["HISTORY"] = (f"StripeField destripe (destripe_astro, "
                                     f"theta=0, win={args.win})")
            try:
                hdul["ERR", ver].header["ERRSTALE"] = (
                    True, "SCI destriped; ERR still original FLC noise")
            except KeyError:
                pass
            print(f"  {path.name} SCI,{ver}: stripe field RMS {rms:.3f} e-",
                  flush=True)
        out = path.with_name(path.stem + args.suffix + ".fits")
        hdul.writeto(out, overwrite=True)
        hdul.close()
        print(f"ecrit: {out}", flush=True)


if __name__ == "__main__":
    main()
