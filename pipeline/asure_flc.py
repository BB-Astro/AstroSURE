"""AstroSURE production pipeline, FLC stage.

Canonical order (PLAN_ENTRAINEMENT.md): FLC -> AstroSURE -> (StripeField)
-> drizzle last. This tool covers the AstroSURE stage on full FLC files:
each SCI chip is denoised (noise + cosmic rays + most of the ACS striping)
and written back into a structural copy (ERR/DQ/WCS untouched, ERR not
rescaled), ready for any drizzle.

Usage:
    python pipeline/asure_flc.py exp1_flc.fits exp2_flc.fits ... [--suffix _asure]
Outputs land next to each input as <stem>_asure.fits.
Validated end to end on Arp 70 (Gaia stars 100.2 %, CR 99.9 % removed,
striping 0.64 -> 0.11 e-).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

PY = sys.executable
TRAINING = Path(__file__).resolve().parent.parent / "training"
MODEL = TRAINING / "checkpoints_run5" / "best.pth"


def process(flc_path: Path, suffix: str, work: Path) -> Path:
    out_path = flc_path.with_name(flc_path.stem + suffix + ".fits")
    hdul = fits.open(flc_path)
    exptime = float(hdul[0].header["EXPTIME"])
    n_sci = sum(1 for h in hdul if h.name == "SCI")
    for ver in range(1, n_sci + 1):
        sci = hdul["SCI", ver]
        chip_in = work / f"{flc_path.stem}_c{ver}.fits"
        hdr = fits.Header()
        hdr["EXPTIME"] = exptime
        fits.writeto(chip_in, sci.data.astype(np.float32), hdr, overwrite=True)
        chip_out = work / f"{flc_path.stem}_c{ver}_asure.fits"
        r = subprocess.run([PY, str(TRAINING / "infer_simple.py"), str(chip_in),
                            str(MODEL), str(chip_out)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
            raise RuntimeError(f"inference failed on {chip_in.name}")
        sci.data = fits.getdata(chip_out).astype(np.float32)
        sci.header["HISTORY"] = "AstroSURE run5 N2N denoise (asure_flc.py)"
        print(f"  {flc_path.name} SCI,{ver} ok", flush=True)
    hdul.writeto(out_path, overwrite=True)
    hdul.close()
    return out_path


def main():
    ap = argparse.ArgumentParser(description="AstroSURE on full FLC files")
    ap.add_argument("flc", nargs="+", help="FLC FITS files")
    ap.add_argument("--suffix", default="_asure")
    args = ap.parse_args()
    work = Path(args.flc[0]).resolve().parent / "asure_work"
    work.mkdir(exist_ok=True)
    for f in args.flc:
        out = process(Path(f).resolve(), args.suffix, work)
        print(f"ecrit: {out}", flush=True)


if __name__ == "__main__":
    main()
