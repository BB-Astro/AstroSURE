"""Drizzle AstroSURE-processed FLC exposures onto a reference DRC grid.

Real STScI drizzle kernel (pip package `drizzle`), full SIP + lookup-table
distortion from the FLC headers, per-exposure sky subtraction, square
kernel. The output is sky-subtracted, in e-/s, on the reference grid, so
it compares pixel to pixel with the MAST DRC.

Usage:
    python pipeline/drizzle_asure.py ref_drc.fits out.fits exp1_asure.fits exp2_asure.fits ...
"""

import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from drizzle.resample import Drizzle

# DQ bits accepted in the final product: 16 hot pixel, 64 warm pixel,
# 256 full-well saturation, 2048 A-to-D saturation (PSF cores must stay).
# Everything else (bad detector pixel, bad column, bleed trails, and the
# SVM cosmic-ray flags 4096/8192) is weighted to zero, like AstroDrizzle.
# The mask is NOT dilated: the SVM routinely misflags undersampled PSF
# cores as CRs in BOTH exposures of a pair, and a grown mask then guts
# the star (measured on Arp 70: one Gaia star at 65% instead of 90%).
DQ_GOOD = 16 | 64 | 256 | 2048


def clipped_sky(data):
    """Sigma-clipped sky median: a plain median over a chip hosting a
    bright galaxy is inflated and over-subtracts, leaving a visible step
    between the chip footprints (the AstroDrizzle equivalent is skymatch)."""
    v = data[::4, ::4].ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("clipped_sky: aucune valeur finie dans l'entree")
    for _ in range(5):
        med = np.median(v)
        sig = 1.4826 * np.median(np.abs(v - med))
        if sig == 0.0:
            # Degenerate (constant) input: the clip would empty the array
            # and the next median would be NaN, which then contaminates
            # the whole image through the background re-anchor.
            return float(med)
        keep = np.abs(v - med) < 3 * sig
        if keep.all():
            break
        v = v[keep]
    return float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_drc", help="reference drizzled product (grid + WCS)")
    ap.add_argument("output", help="output FITS")
    ap.add_argument("flc", nargs="+", help="processed FLC files")
    ap.add_argument("--pixfrac", type=float, default=0.8)
    args = ap.parse_args()

    ref = fits.open(args.ref_drc)
    out_wcs = WCS(ref["SCI"].header)
    out_shape = ref["SCI"].data.shape
    driz = Drizzle(kernel="square", out_shape=out_shape, fillval=np.nan)

    for fname in args.flc:
        hdul = fits.open(fname)
        exptime = float(hdul[0].header["EXPTIME"])
        n_sci = sum(1 for h in hdul if h.name == "SCI")
        # ONE sky per exposure, pooled over its chips: independent per-chip
        # skies leave a visible step between the chip footprints whenever a
        # big galaxy biases one chip's estimate (Arp 130). The chips share
        # the same sky physically (same visit, same minutes).
        pool = np.concatenate([
            (hdul["SCI", v].data.astype(np.float32) / exptime)[::8, ::8].ravel()
            for v in range(1, n_sci + 1)])
        sky = clipped_sky(pool.reshape(1, -1))
        for ver in range(1, n_sci + 1):
            sci = hdul["SCI", ver]
            wcs_in = WCS(sci.header, hdul)
            data = sci.data.astype(np.float32) / exptime
            data -= sky
            h, w = data.shape
            yy, xx = np.mgrid[0:h, 0:w]
            world = wcs_in.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
            ox, oy = out_wcs.all_world2pix(world, 0).T
            pixmap = np.dstack([ox.reshape(h, w), oy.reshape(h, w)]).astype(np.float64)
            wht = np.isfinite(data).astype(np.float32)
            try:
                dq = hdul["DQ", ver].data
                bad = (dq & ~DQ_GOOD) != 0
                wht[bad] = 0.0
                print(f"  {fname.split('/')[-1]} SCI,{ver}: {bad.mean()*100:.2f}% "
                      f"DQ-masked", flush=True)
            except KeyError:
                pass
            driz.add_image(np.nan_to_num(data), exptime, pixmap, weight_map=wht,
                           pixfrac=args.pixfrac, in_units="cps")
            print(f"  {fname.split('/')[-1]} SCI,{ver} drizzle ok", flush=True)

    hdr = fits.Header(out_wcs.to_header())
    hdr["HISTORY"] = "AstroSURE chain drizzle (drizzle_asure.py)"
    fits.writeto(args.output, driz.out_img.astype(np.float32), hdr, overwrite=True)
    print(f"ecrit: {args.output}")


if __name__ == "__main__":
    main()
