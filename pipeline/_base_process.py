"""Historical FLC experiment: AstroSURE run-5 denoise then ACS destriping.

This is the first end-to-end chip experiment (Arp 70 era), kept because it
is a compact, readable template: for each exposure and each chip
(SCI,1 / SCI,2):
  1. extract the chip as a temporary mono FITS (EXPTIME propagated)
  2. AstroSURE run-5 inference (native FLC domain: denoise + cosmic rays)
  3. BB-Astro StripeField destriping, fixed horizontal angle (ACS rows)
then reassemble a full *_flc_asure.fits with the original structure
(ERR/DQ/WCS untouched) so AstroDrizzle/PixInsight can consume it.
Note: ERR arrays are NOT rescaled to the denoised data.

The current production chain is pipeline/batch_arp.py (deepCR ->
StripeField -> AstroSURE 2-channel -> drizzle -> blend).

Usage:
    python pipeline/_base_process.py exp1_flc.fits exp2_flc.fits
        [--model training/checkpoints_run5/best.pth]
        [--destripe pipeline/destripe_astro.py] [--workdir work]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]


def stripe_metric(img):
    """Std of the row-median profile after source masking, in e-."""
    med = np.median(img)
    sig = 1.4826 * np.median(np.abs(img - med))
    masked = np.where(img < med + 3 * sig, img, np.nan)
    rows = np.nanmedian(masked, axis=1)
    return float(np.nanstd(rows - np.nanmedian(rows)))


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")
    return r


def main():
    ap = argparse.ArgumentParser(
        description="AstroSURE run-5 denoise + StripeField destripe, per chip")
    ap.add_argument("flc", nargs="+", help="FLC FITS files to process")
    ap.add_argument("--model",
                    default=str(REPO / "training/checkpoints_run5/best.pth"),
                    help="1-channel AstroSURE checkpoint")
    ap.add_argument("--destripe",
                    default=str(Path(__file__).resolve().parent
                                / "destripe_astro.py"),
                    help="StripeField engine (vendored copy by default)")
    ap.add_argument("--workdir", default="work",
                    help="scratch directory for per-chip files")
    args = ap.parse_args()

    py = sys.executable
    infer = REPO / "training" / "infer_simple.py"
    work = Path(args.workdir)
    work.mkdir(exist_ok=True)

    for f in args.flc:
        flc = Path(f).resolve()
        stem = flc.stem.replace("_flc", "")
        out_path = flc.parent / f"{stem}_flc_asure.fits"
        hdul = fits.open(flc)
        exptime = float(hdul[0].header["EXPTIME"])
        for ver in (1, 2):
            sci = hdul["SCI", ver]
            chip_in = work / f"{stem}_chip{ver}.fits"
            hdr = fits.Header()
            hdr["EXPTIME"] = exptime
            fits.writeto(chip_in, sci.data.astype(np.float32), hdr,
                         overwrite=True)

            den = work / f"{stem}_chip{ver}_asure.fits"
            run([py, str(infer), str(chip_in), args.model, str(den)])

            dsdir = work / f"ds_{stem}_chip{ver}"
            run([py, args.destripe, str(den), "-o", str(dsdir),
                 "--angles", "0.0"])
            final = dsdir / f"{den.stem}_destriped.fits"
            result = fits.getdata(final).astype(np.float32)

            s0 = stripe_metric(sci.data.astype(np.float32))
            s1 = stripe_metric(fits.getdata(den).astype(np.float32))
            s2 = stripe_metric(result)
            print(f"{stem} chip{ver}: stripes raw {s0:.3f} e- -> after "
                  f"AstroSURE {s1:.3f} -> after destriping {s2:.3f}",
                  flush=True)

            sci.data = result
            sci.header["HISTORY"] = \
                "AstroSURE run5 N2N denoise + BB StripeField destripe"
        hdul.writeto(out_path, overwrite=True)
        hdul.close()
        print(f"written: {out_path}", flush=True)

    print("PIPELINE DONE")


if __name__ == "__main__":
    main()
