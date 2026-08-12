"""Pre-clean cosmic rays on FLC exposures with deepCR (ACS-WFC model),
with cross-exposure FLUX veto of misflagged star cores.

deepCR flags undersampled bright PSF cores as CRs (measured on Arp 70:
core probability 0.84-0.90 on a G=17.3 star; the default 0.5 threshold
then costs 12 pt of stellar flux through the full chain). The veto is
the project's partner-confirmation doctrine, applied per pixel: a real
cosmic ray has NO counterpart flux at the same sky position in the
partner exposure, a star always does. A flagged pixel whose partner
flux exceeds 0.3x the local flux (and the partner noise floor) is a
static source and is left alone. Vetoing on the partner's deepCR
PROBABILITY instead does not work: a core flagged in only one pose
(subpixel centering makes one pose CR-like and the other not) gets no
veto and loses ~15 pt (measured, G=18.69 on Arp 70).

CRs sitting on bright structure can be vetoed too (partner halo
confirms them); that is fine, they are exactly the ones the downstream
2-channel N2N model erases (spike in channel 1, absent from channel 2).
This stage only needs to catch what the pair logic cannot: single-pose
CRs on faint background, the inter-chip gap and the dither margins.

Repairs use the DQ-validated masked median fill (asure2_flc), not the
learned inpaint: CR footprints are small and the masked median is
unbiased there (the learned inpaint would be another model to QC).

Writes a sibling <stem>_dcr.fits per input, SCI replaced, ERR/DQ/WCS
untouched. Meant to run BEFORE asure2_flc.py.

Usage:
    python pipeline/deepcr_flc.py exp1_flc.fits exp2_flc.fits
        [--threshold 0.5] [--veto-ratio 0.3] [--suffix _dcr]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from deepCR import deepCR
from scipy.ndimage import label, map_coordinates, sum_labels

from asure2_flc import masked_median_fill


def main():
    ap = argparse.ArgumentParser(description="deepCR CR pre-clean on FLC files")
    ap.add_argument("flc", nargs="+", help="FLC FITS files of the SAME pointing")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="CR probability threshold (higher = fewer detections)")
    ap.add_argument("--veto-ratio", type=float, default=0.3,
                    help="partner/local flux ratio above which a detection "
                         "is vetoed as a static source")
    ap.add_argument("--suffix", default="_dcr")
    args = ap.parse_args()
    if len(args.flc) < 2:
        sys.exit("deepcr_flc: il faut au moins 2 expositions du même champ, "
                 "toute la protection des étoiles repose sur le veto par flux "
                 "partenaire (avec 1 pose, sig=NaN et le veto ne se déclenche "
                 "jamais : les cœurs de PSF flaggés seraient tous écrasés)")

    mdl = deepCR(mask="ACS-WFC", device="CPU")
    exposures = []
    for f in args.flc:
        path = Path(f).resolve()
        hdul = fits.open(path)
        exptime = float(hdul[0].header["EXPTIME"])
        chips = {}
        n_sci = sum(1 for h in hdul if h.name == "SCI")
        for ver in range(1, n_sci + 1):
            sci = hdul["SCI", ver]
            prob = mdl.clean(sci.data.astype(np.float32), binary=False,
                             inpaint=False, segment=True, patch=1024)
            rate = sci.data.astype(np.float32) / exptime
            sky = float(np.median(rate[::8, ::8]))
            chips[ver] = (prob.astype(np.float32), rate - sky,
                          WCS(sci.header, hdul))
        exposures.append((path, hdul, chips))
        print(f"  {path.name}: cartes de probabilite calculees", flush=True)

    for i, (path, hdul, chips) in enumerate(exposures):
        for ver, (prob, rate, wcs) in chips.items():
            sci = hdul["SCI", ver]
            h, w = prob.shape
            yy, xx = np.mgrid[0:h, 0:w]
            world = wcs.all_pix2world(
                np.column_stack([xx.ravel(), yy.ravel()]), 0)
            acc = np.zeros((h, w), dtype=np.float32)
            nval = np.zeros((h, w), dtype=np.int32)
            for j, (_, _, chips_j) in enumerate(exposures):
                if j == i or ver not in chips_j:
                    continue
                _, p_rate, p_wcs = chips_j[ver]
                px, py = p_wcs.all_world2pix(world, 0).T
                py = py.reshape(h, w)
                px = px.reshape(h, w)
                mapped = map_coordinates(p_rate, [py, px], order=1,
                                         mode="constant", cval=0.0)
                inside = ((py >= 0) & (py <= p_rate.shape[0] - 1) &
                          (px >= 0) & (px <= p_rate.shape[1] - 1))
                acc += np.where(inside, mapped, 0.0)
                nval += inside
            partner = acc / np.maximum(nval, 1)
            if not (nval > 0).any():
                sys.exit(f"deepcr_flc: aucune couverture partenaire pour "
                         f"{path.name} SCI,{ver} (mauvais appariement des "
                         f"expositions ?), le veto serait inopérant")
            v = partner[nval > 0][::37]
            sig = 1.4826 * float(np.median(np.abs(v - np.median(v))))
            # Clump-level veto: per-pixel comparison lets positive sky
            # noise "confirm" the faint fringe pixels of a CR footprint
            # (measured: 80-90% of flagged pixels wrongly vetoed). Summed
            # over the whole connected component, a star core is matched
            # by the partner while a CR sums against plain sky.
            mask = prob > args.threshold
            veto = np.zeros((h, w), dtype=bool)
            lab, nlab = label(mask)
            if nlab:
                idx = np.arange(1, nlab + 1)
                flux_ref = sum_labels(rate, lab, idx)
                flux_par = sum_labels(partner, lab, idx)
                npix = sum_labels(np.ones((h, w), dtype=np.float32), lab, idx)
                veto_lab = ((flux_par > args.veto_ratio * flux_ref) &
                            (flux_par > 3 * sig * np.sqrt(npix)))
                veto[mask] = veto_lab[lab[mask] - 1]
            kept = mask & ~veto
            sci.data = masked_median_fill(sci.data.astype(np.float32), kept)
            sci.header["HISTORY"] = (f"deepCR ACS-WFC mask thr={args.threshold} "
                                     f"partner flux veto ratio="
                                     f"{args.veto_ratio}, masked median fill")
            try:
                hdul["ERR", ver].header["ERRSTALE"] = (
                    True, "SCI CR-cleaned; ERR still original FLC noise")
            except KeyError:
                pass
            print(f"  {path.name} SCI,{ver}: {mask.mean()*100:.2f}% px flagges, "
                  f"{(mask & veto).mean()*100:.3f}% vetoes (flux partenaire), "
                  f"{kept.mean()*100:.2f}% repares", flush=True)
        out = path.with_name(path.stem + args.suffix + ".fits")
        hdul.writeto(out, overwrite=True)
        hdul.close()
        print(f"ecrit: {out}", flush=True)


if __name__ == "__main__":
    main()
