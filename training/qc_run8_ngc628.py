"""Run-8 QC on the reference NGC 628 chip (F814W, visit j96r21, chip 1).

Protocol per PLAN_ENTRAINEMENT.md (corrected 2026-08-05): reference stars
are selected on the deepCR-cleaned chip (never on the raw FLC), photometry
is measured at the same positions on the raw original and on each denoised
version, per brightness tier. Background noise from a decimated MAD.

This group has only 2 exposures, so it was EXCLUDED from run-8 training
(units need >= 3 files): the run-8 numbers below are out-of-sample, while
run 5 saw this chip during training.

Adds a fine-detail score: correlation of the high-pass denoised image with
the high-pass PARTNER exposure (an independent measurement of the same
sky) inside the galaxy mask. Noise is independent between the two, so a
higher correlation means more real structure kept, not more noise.

Usage: python qc_run8_ngc628.py [model.pth]  (from training/, env dip)
The optional argument overrides the 2-channel checkpoint under test
(default: checkpoints_run8/best.pth); the run-5 column stays the anchor.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CMP = HERE / "compare_NGC628"
DATA = ROOT / "training_data" / "NGC_628" / "F814W"
PARTNER = DATA / "hst_10402_21_acs_wfc_f814w_j96r21jg_chip1.fits"
MODEL8 = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "checkpoints_run8" / "best.pth"

sys.path.insert(0, str(ROOT / "pipeline"))
from asure2_flc import measure_shift  # noqa: E402


def clean_wcs_header(header):
    hdr = header.copy()
    for kw in list(hdr.keys()):
        if "D2IM" in kw or "CPDIS" in kw or "DP" in kw:
            del hdr[kw]
    return hdr


def sky_and_sigma(data, step=4):
    v = data[::step, ::step].ravel()
    v = v[np.isfinite(v)]
    med = float(np.median(v))
    return med, 1.4826 * float(np.median(np.abs(v - med)))


def local_maxima(img, thresh, box=3, edge=12):
    mx = maximum_filter(img, size=2 * box + 1, mode="nearest")
    cand = (img >= mx) & (img > thresh)
    cand[:edge, :] = False
    cand[-edge:, :] = False
    cand[:, :edge] = False
    cand[:, -edge:] = False
    return np.nonzero(cand)


def psf_like(img, ys, xs, sky, min_ratio=0.15, max_ratio=0.95):
    peak = img[ys, xs] - sky
    ring = (img[ys - 1, xs] + img[ys + 1, xs]
            + img[ys, xs - 1] + img[ys, xs + 1]) / 4.0 - sky
    with np.errstate(invalid="ignore", divide="ignore"):
        r = ring / np.maximum(peak, 1e-6)
    return (r > min_ratio) & (r < max_ratio)


def aper_flux(img, ys, xs, sky, r=3):
    out = np.empty(len(ys))
    n = (2 * r + 1) ** 2
    for i, (y, x) in enumerate(zip(ys, xs)):
        out[i] = float(np.sum(img[y - r:y + r + 1, x - r:x + r + 1])) - n * sky
    return out


def build_partner_channel():
    """Partner exposure resampled onto the reference grid, e-/s, sky-sub."""
    with fits.open(CMP / "original.fits") as h:
        ref_exp = float(h[0].header["EXPTIME"])
        ref_rate = h["SCI"].data.astype(np.float32) / ref_exp
        ref_rate -= float(np.median(ref_rate[::8, ::8]))
        ref_wcs = WCS(clean_wcs_header(h["SCI"].header))
    with fits.open(PARTNER) as h:
        par_exp = float(h[0].header["EXPTIME"])
        par = h["SCI"].data.astype(np.float32) / par_exp
        par -= float(np.median(par[::8, ::8]))
        par_wcs = WCS(clean_wcs_header(h["SCI"].header))

    hgt, wid = ref_rate.shape
    yy, xx = np.mgrid[0:hgt, 0:wid]
    world = ref_wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    px, py = par_wcs.all_world2pix(world, 0).T
    py = py.reshape(hgt, wid)
    px = px.reshape(hgt, wid)
    aligned = map_coordinates(np.nan_to_num(par), [py, px], order=1, mode="nearest")

    cy, cx = hgt // 2, wid // 2
    sy, sx = measure_shift(ref_rate[cy - 512:cy + 512, cx - 512:cx + 512],
                           aligned[cy - 512:cy + 512, cx - 512:cx + 512])
    mag = (sy ** 2 + sx ** 2) ** 0.5
    print(f"partner residual shift: ({sy:+.3f}, {sx:+.3f}) px", flush=True)
    if 0.05 < mag < 2.0:
        aligned = map_coordinates(np.nan_to_num(par), [py + sy, px + sx],
                                  order=1, mode="nearest")
        print("  -> refinement applied")

    out = CMP / "partner_aligned.fits"
    hdr = fits.Header()
    hdr["EXPTIME"] = 1.0
    fits.writeto(out, aligned.astype(np.float32), hdr, overwrite=True)
    return out


def detail_score(den, partner_al, gal_mask, clip):
    """Correlation of high-pass structures with the independent partner
    exposure inside the galaxy mask. The high-pass is clipped at +-clip
    so bright knots and stars do not dominate the sum: what remains is
    the faint texture BB cares about."""
    hp_d = np.clip(den - gaussian_filter(den, 4), -clip, clip)
    hp_p = np.clip(partner_al - gaussian_filter(partner_al, 4), -clip, clip)
    a = hp_d[gal_mask]
    b = hp_p[gal_mask]
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    partner_path = build_partner_channel()

    den8_path = CMP / "denoised_run8.fits"
    r = subprocess.run([sys.executable, str(HERE / "infer2.py"),
                        str(CMP / "original.fits"), str(partner_path),
                        str(MODEL8), str(den8_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        raise RuntimeError("run-8 inference failed")
    print(f"denoised_run8.fits written", flush=True)

    orig = fits.getdata(CMP / "original.fits", extname="SCI").astype(np.float32)
    clean = fits.getdata(CMP / "original_crclean.fits").astype(np.float32)
    if clean.ndim == 3:
        clean = clean[0]
    with fits.open(CMP / "denoised_run5.fits") as h:
        den5 = next(x.data for x in h if x.data is not None).astype(np.float32)
    with fits.open(den8_path) as h:
        den8 = next(x.data for x in h if x.data is not None).astype(np.float32)
    partner_al = fits.getdata(partner_path).astype(np.float32)
    ref_exp = float(fits.getheader(CMP / "original.fits")["EXPTIME"])

    sky_o, sig_o = sky_and_sigma(orig)
    sky_c, sig_c = sky_and_sigma(clean)

    # Star selection on the CLEANED chip only (QC rule)
    ys, xs = local_maxima(clean, sky_c + 10 * sig_c)
    keep = psf_like(clean, ys, xs, sky_c)
    ys, xs = ys[keep], xs[keep]
    amp = (clean[ys, xs] - sky_c) / sig_c
    tiers = [("10-30s", (amp >= 10) & (amp < 30)),
             ("30-100s", (amp >= 30) & (amp < 100)),
             ("100-500s", (amp >= 100) & (amp < 500))]

    print("\n=== QC NGC 628 F814W j96r21je chip1 (selection on deepCR-clean) ===")
    results = {}
    for name, img in [("run5", den5), ("run8", den8)]:
        sky_d, sig_d = sky_and_sigma(img)
        f_o = aper_flux(orig, ys, xs, sky_o)
        f_d = aper_flux(img, ys, xs, sky_d)
        row = {"fond": sig_o / sig_d, "offset_e": sky_d - sky_o}
        for tname, m in tiers:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratios = f_d[m] / f_o[m]
            row[tname] = float(np.median(ratios[np.isfinite(ratios)])) * 100
            row[f"n_{tname}"] = int(m.sum())
        results[name] = row

    # Galaxy mask from smooth surface brightness on the clean chip
    smooth = gaussian_filter(clean - sky_c, 8)
    gal_mask = smooth > 1.5 * sig_c
    print(f"galaxy mask: {gal_mask.mean() * 100:.1f}% of chip")
    clip = 10 * sig_o / ref_exp
    for name, img in [("orig", orig), ("run5", den5), ("run8", den8)]:
        sc = detail_score(img / ref_exp, partner_al, gal_mask, clip)
        results.setdefault(name, {})["detail"] = sc

    print(f"\n{'metric':<14}{'run5':>10}{'run8':>10}")
    for key in ["fond", "offset_e", "10-30s", "30-100s", "100-500s", "detail"]:
        v5 = results["run5"].get(key, float('nan'))
        v8 = results["run8"].get(key, float('nan'))
        print(f"{key:<14}{v5:>10.3f}{v8:>10.3f}")
    print(f"(stars per tier: " +
          ", ".join(f"{t}={results['run5'][f'n_{t}']}" for t, _ in tiers) + ")")
    print(f"detail score original: {results['orig']['detail']:.3f} "
          f"(upper bound includes its own noise floor)")


if __name__ == "__main__":
    main()
