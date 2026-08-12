"""Shared helpers for the independent QC of the 116 new chips (read-only)."""

import json
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path("/Volumes/MacStudio_SideCar/TheGrandDenoise/AstroSURE")
CAT_RAW = ROOT / "training_data" / "pairs_catalog_new.json"
CAT_CLEAN = ROOT / "training_data_crclean" / "pairs_catalog_new_crclean.json"
CAT_FULL = ROOT / "training_data_crclean" / "pairs_catalog_full_crclean.json"


def load_catalog(path):
    with open(path) as f:
        return json.load(f)


def group_label(g):
    return f"{g['galaxy']}/{g['filter']}/chip{g['chip']}"


def field_of(g):
    return g["galaxy"]


def clean_wcs_header(header):
    hdr = header.copy()
    for kw in list(hdr.keys()):
        if "D2IM" in kw or "CPDIS" in kw or "DP" in kw:
            del hdr[kw]
    return hdr


def open_chip(path, want_data=True):
    """Return dict with data (float32, e-/s not applied), sci header, primary header, wcs."""
    with fits.open(path, memmap=False) as hdul:
        sci = hdul["SCI"]
        hdr = sci.header
        prim = hdul[0].header
        data = sci.data.astype(np.float32) if want_data else None
        wcs = WCS(clean_wcs_header(hdr))
    return {
        "data": data,
        "hdr": hdr,
        "prim": prim,
        "wcs": wcs,
        "path": str(path),
        "name": os.path.basename(path),
    }


def rootname_of(path):
    """hst_9810_01_acs_wfc_f814w_j8r001s0_chip1.fits -> j8r001s0 (exposure rootname)."""
    base = os.path.basename(path)
    parts = base.replace(".fits", "").split("_")
    # ... _<rootname>_chip<N>
    return parts[-2]


def cd_angle_deg(hdr):
    return float(np.degrees(np.arctan2(hdr["CD1_2"], hdr["CD1_1"])))


def sky_and_sigma(data, step=4):
    """Robust background level and sigma from a decimated view."""
    v = data[::step, ::step].ravel()
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, 1.4826 * mad


def local_maxima(img, thresh, box=3, edge=12):
    """Coordinates of pixels that are the strict max of their (2*box+1)^2 neighbourhood
    and above `thresh`. Returns (ys, xs)."""
    from scipy.ndimage import maximum_filter

    mx = maximum_filter(img, size=2 * box + 1, mode="nearest")
    cand = (img >= mx) & (img > thresh)
    cand[:edge, :] = False
    cand[-edge:, :] = False
    cand[:, :edge] = False
    cand[:, -edge:] = False
    ys, xs = np.nonzero(cand)
    return ys, xs


def psf_like(img, ys, xs, sky, sigma, min_ratio=0.15, max_ratio=0.95):
    """Keep maxima whose 4-neighbour mean over peak ratio is PSF-like.

    A cosmic ray hit on 1-2 pixels has almost no flux in its neighbours; a real
    (undersampled) ACS/WFC star still puts >~15% of its peak in the ring.
    This is the project's own criterion, used only as a pre-filter here: the
    ground truth is the partner-exposure test.
    """
    peak = img[ys, xs] - sky
    ring = (
        img[ys - 1, xs] + img[ys + 1, xs] + img[ys, xs - 1] + img[ys, xs + 1]
    ) / 4.0 - sky
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ring / np.maximum(peak, 1e-6)
    return (r > min_ratio) & (r < max_ratio), r


def aper_flux(img, ys, xs, sky, r=3):
    """Sum of sky-subtracted pixels in a (2r+1)^2 box around each (y, x)."""
    out = np.empty(len(ys), dtype=np.float64)
    n = (2 * r + 1) ** 2
    for i, (y, x) in enumerate(zip(ys, xs)):
        out[i] = float(np.sum(img[y - r:y + r + 1, x - r:x + r + 1])) - n * sky
    return out


def centroid(img, y, x, half=3):
    """Flux-weighted centroid in a (2*half+1)^2 window; returns (yc, xc) in image coords."""
    sub = img[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
    sub = sub - np.median(sub)
    sub[sub < 0] = 0.0
    tot = sub.sum()
    if tot <= 0:
        return None
    yy, xx = np.mgrid[y - half:y + half + 1, x - half:x + half + 1]
    return float((sub * yy).sum() / tot), float((sub * xx).sum() / tot)
