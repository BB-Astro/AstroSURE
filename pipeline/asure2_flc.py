"""AstroSURE production pipeline, FLC stage (2-channel model).

Production model: run 9 epoch 20 (QC-selected champion), see
DEFAULT_MODEL. Same contract as asure_flc.py but for the 2-channel
model: each SCI chip of each exposure is denoised using the OTHER
exposure(s) as the second input channel (mean of the others if more
than two), resampled onto the reference grid through the full
distortion WCS, with a phase-correlation residual-shift check. Outputs
land next to each input as <stem>_asure9e20b.fits (DQ/WCS untouched;
ERR kept as-is and flagged ERRSTALE, it no longer describes the SCI
noise).

Usage:
    python pipeline/asure2_flc.py exp1_flc.fits exp2_flc.fits ... [--model ...]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates

from drizzle_asure import clipped_sky

# Same acceptance as drizzle_asure.DQ_GOOD: hot/warm pixels and saturation
# stay, everything else is detector garbage. It must be cleaned BEFORE
# inference: a DQ-flagged clump of deeply negative pixels (measured -79 e-
# on Arp 141, DQ=4096+16) is smeared by the U-Net into a dark streak wider
# than the DQ footprint, which then survives the drizzle's DQ masking.
DQ_GOOD = 16 | 64 | 256 | 2048

PY = sys.executable
TRAINING = Path(__file__).resolve().parent.parent / "training"
# Production champion: run-9 epoch 20, selected by QC (detail 0.890, Gaia
# 100.4% on Arp 70), NOT by val loss (which prefers the smoothest epoch).
DEFAULT_MODEL = TRAINING / "checkpoints_run9" / "best_qc.pth"
# 1-channel champion (run 5, via the checkpoints symlink): used to fill
# the inter-chip gap band that no partner exposure covers. There the
# 2-channel model sees channel2 = channel1, treats every source as
# confirmed and keeps the cosmic rays (the speckled band BB spotted in
# PixInsight on Arp 130); the 1-channel model removes CRs without a
# partner (99.9% validated on Arp 70).
MODEL_1CH = TRAINING / "checkpoints" / "best.pth"

MAX_REFINE_PX = 2.0   # beyond this a measured shift means something is wrong
MIN_REFINE_PX = 0.05  # below this the WCS alignment is already good enough


def masked_median_fill(elec, bad, size=5):
    """Replace DQ-bad pixels with the median of the GOOD pixels of their
    size x size neighbourhood.

    A plain median_filter is biased wherever the window itself is
    contaminated: on the dead column pair of Arp 141 (SCI2 x=1456-1457,
    -45 e-, 10 of the 25 window samples bad) the repaired column lands
    ~2 e- low (p37 of the good pixels instead of p50), and the U-Net
    widens that groove into a ~20 px dark trail that survives the
    drizzle DQ mask. Bad regions larger than the window (big CR blobs)
    are filled iteratively from the outside in."""
    out = elec.copy()
    todo = bad.copy()
    pad = size // 2
    while todo.any():
        data = np.pad(out, pad, mode="reflect")
        good = np.pad(~todo, pad, mode="reflect")
        ys, xs = np.nonzero(todo)
        vals = np.empty((size * size, len(ys)), dtype=np.float32)
        k = 0
        for dy in range(size):
            for dx in range(size):
                vals[k] = np.where(good[ys + dy, xs + dx],
                                   data[ys + dy, xs + dx], np.nan)
                k += 1
        fillable = np.isfinite(vals).sum(axis=0) > 0
        if not fillable.any():
            break
        med = np.nanmedian(vals[:, fillable], axis=0)
        out[ys[fillable], xs[fillable]] = med
        todo[ys[fillable], xs[fillable]] = False
    return out


def measure_shift(a, b, upsample=50, search_px=3.0):
    """Phase correlation restricted to +-search_px around zero: returns
    (sy, sx) such that b ~= a shifted by (sy, sx), i.e. b(y, x) = a(y - sy,
    x - sx). Subpixel by local DFT upsampling of the cross-power spectrum
    (Guizar-Sicairos style).

    The search is LOCAL by design: this measures a residual on top of an
    alignment that is already good to ~1 px. A global argmax instead locks
    onto detector-fixed content (ACS striping, hot pixels, warm columns)
    common to both exposures, which after sky alignment produces a strong
    false peak at exactly minus the dither (measured on Arp 70: |s| = 58-62
    px = the dither, on a correctly aligned pair)."""
    def prep(x):
        # Row-median subtraction removes most of the ACS striping.
        x = x - np.median(x, axis=1, keepdims=True)
        x = x - np.median(x)
        lim = np.percentile(np.abs(x), 99.5) + 1e-12
        return np.clip(x, -lim, lim)

    a = prep(a)
    b = prep(b)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * win)
    fb = np.fft.fft2(b * win)
    r = np.conj(fa) * fb
    r /= np.abs(r) + 1e-12

    # Low frequencies only: bilinear resampling flattens the phase slope at
    # high frequencies (and noise dominates there), which biases the
    # fractional part toward integers by ~0.1 px if kept.
    n, m = r.shape
    fy = np.fft.fftfreq(n)
    fx = np.fft.fftfreq(m)
    r = r * np.outer(np.abs(fy) < 0.15, np.abs(fx) < 0.15)

    # Evaluate the correlation on a 1/upsample grid within +-search_px of
    # zero: c(y, x) = Re sum r[k] exp(2i pi (fy y + fx x))
    ys = np.arange(-search_px * upsample, search_px * upsample + 1) / upsample
    xs = ys
    ey = np.exp(2j * np.pi * np.outer(ys, fy))
    ex = np.exp(2j * np.pi * np.outer(fx, xs))
    cc = (ey @ r @ ex).real
    iy, ix = np.unravel_index(np.argmax(cc), cc.shape)
    return float(ys[iy]), float(xs[ix])


def resample_partner(ref_wcs, ref_shape, part_data, part_wcs, ref_rate,
                     refine=False):
    """Resample a partner chip (e-/s, sky-subtracted) onto the reference
    grid: full distortion WCS mapping, bilinear. The phase-correlation
    residual is always MEASURED and reported as a diagnostic; it is only
    APPLIED with refine=True. On sparse CR-riddled fields (Arp 70: 2x390 s,
    ~14000 CR/chip) the correlation has too little clean common signal and
    reports ~1-3 px residuals on a pair whose pure-WCS alignment is
    Gaia-validated to <0.5 px: default is therefore to trust the WCS."""
    h, w = ref_shape
    yy, xx = np.mgrid[0:h, 0:w]
    world = ref_wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    px, py = part_wcs.all_world2pix(world, 0).T
    py = py.reshape(h, w)
    px = px.reshape(h, w)
    part_data = np.nan_to_num(part_data)
    aligned = map_coordinates(part_data, [py, px], order=1, mode="nearest")
    ph, pw = part_data.shape
    # Pixels of the reference grid that the partner never observed (the
    # dither band along the edges): edge-replicated values there look like
    # "source in channel 1, absent from channel 2", the exact signature of
    # a cosmic ray, and the model erases real stars (measured on Arp 70:
    # 3 Gaia stars at 0-4% flux, all within ~60 px = the dither, of a chip
    # edge covered by a single exposure).
    footprint = (py >= 0) & (py <= ph - 1) & (px >= 0) & (px <= pw - 1)

    # Residual check on a central crop (the WCS should already be <0.1 px
    # for same-visit dithers; inter-visit zero points can be off by ~1 px,
    # cf. wcs_offsets.json in training).
    cy, cx = h // 2, w // 2
    half = min(512, cy, cx)
    ref_c = ref_rate[cy - half:cy + half, cx - half:cx + half]
    ali_c = aligned[cy - half:cy + half, cx - half:cx + half]
    sy, sx = measure_shift(ref_c, ali_c)
    mag = (sy ** 2 + sx ** 2) ** 0.5
    if not refine or mag <= MIN_REFINE_PX:
        return aligned, footprint, (sy, sx, False)
    if mag > MAX_REFINE_PX:
        print(f"    WARNING residual shift {mag:.2f} px > {MAX_REFINE_PX}, "
              f"not applied (check the WCS)")
        return aligned, footprint, (sy, sx, False)
    # aligned(y,x) = ref(y-sy, x-sx): sample the partner at +s to align
    aligned = map_coordinates(part_data, [py + sy, px + sx], order=1,
                              mode="nearest")
    return aligned, footprint, (sy, sx, True)


def load_exposure(path):
    hdul = fits.open(path)
    exptime = float(hdul[0].header["EXPTIME"])
    n_sci = sum(1 for h in hdul if h.name == "SCI")
    chips = {}
    for ver in range(1, n_sci + 1):
        sci = hdul["SCI", ver]
        elec = sci.data.astype(np.float32)
        try:
            dq = hdul["DQ", ver].data
            bad = (dq & ~DQ_GOOD) != 0
            elec = masked_median_fill(elec, bad)
        except KeyError:
            bad = None
        rate = elec / exptime
        rate -= float(np.median(rate[::8, ::8]))
        chips[ver] = (rate, WCS(sci.header, hdul), elec)
    return hdul, exptime, chips


def main():
    ap = argparse.ArgumentParser(description="AstroSURE 2-channel N2N denoise on FLC files (run 9 e20 champion)")
    ap.add_argument("flc", nargs="+", help="FLC FITS files of the SAME pointing (>=2)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--suffix", default="_asure9e20b")
    ap.add_argument("--refine", action="store_true",
                    help="apply the measured residual shift (default: "
                         "measure and report only, trust the WCS)")
    ap.add_argument("--blend", type=float, default=1.0,
                    help="denoising strength: output = blend*denoised + "
                         "(1-blend)*input. The kept noise is about "
                         "(1-blend) times the input noise; star flux moves "
                         "TOWARD 100%% (the (1-blend) term is raw signal). "
                         "1.0 = full denoise (default)")
    args = ap.parse_args()
    if len(args.flc) < 2:
        sys.exit("need at least 2 exposures (the model input is a pair)")

    paths = [Path(f).resolve() for f in args.flc]
    work = paths[0].parent / "asure_work"
    work.mkdir(exist_ok=True)

    exposures = [load_exposure(p) for p in paths]

    for i, path in enumerate(paths):
        hdul, exptime, chips = exposures[i]
        out_path = path.with_name(path.stem + args.suffix + ".fits")
        for ver, (ref_rate, ref_wcs, ref_elec) in chips.items():
            sci = hdul["SCI", ver]
            acc = np.zeros_like(ref_rate)
            nvalid = np.zeros(ref_rate.shape, dtype=np.int32)
            for j, (_, _, chips_j) in enumerate(exposures):
                if j == i or ver not in chips_j:
                    continue
                part_rate, part_wcs, _ = chips_j[ver]
                aligned, footprint, (sy, sx, applied) = resample_partner(
                    ref_wcs, ref_rate.shape, part_rate, part_wcs, ref_rate,
                    refine=args.refine)
                tag = "applied" if applied else "kept WCS"
                print(f"  {path.name} SCI,{ver} <- {paths[j].name}: "
                      f"residual ({sy:+.3f}, {sx:+.3f}) px, {tag}", flush=True)
                acc += np.where(footprint, aligned, 0.0)
                nvalid += footprint
            # Where no partner ever observed (dither band along the edges),
            # fall back to the reference itself: the model then sees a
            # "confirmed" source and keeps it instead of erasing it as a
            # cosmic ray. Honest physics: with single-exposure coverage,
            # star and CR are indistinguishable, so that band keeps CRs,
            # exactly like the classic MAST chain there.
            covered = nvalid > 0
            partner = np.where(covered, acc / np.maximum(nvalid, 1), ref_rate)
            pct = 100.0 * (~covered).mean()
            if pct > 0:
                print(f"    {pct:.1f}% of the chip has no partner coverage "
                      f"(channel 2 = channel 1 there)", flush=True)

            chip_ref = work / f"{path.stem}_c{ver}_ref.fits"
            hdr = fits.Header()
            hdr["EXPTIME"] = exptime
            fits.writeto(chip_ref, ref_elec.astype(np.float32), hdr, overwrite=True)
            chip_par = work / f"{path.stem}_c{ver}_par.fits"
            hdr = fits.Header()
            hdr["EXPTIME"] = 1.0
            fits.writeto(chip_par, partner.astype(np.float32), hdr, overwrite=True)

            chip_out = work / f"{path.stem}_c{ver}_asure.fits"
            r = subprocess.run([PY, str(TRAINING / "infer2.py"), str(chip_ref),
                                str(chip_par), args.model, str(chip_out)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-1500:])
                print(r.stderr[-1500:])
                raise RuntimeError(f"inference failed on {chip_ref.name}")
            den = fits.getdata(chip_out).astype(np.float32)
            if not covered.all():
                chip_1c = work / f"{path.stem}_c{ver}_asure1c.fits"
                r = subprocess.run([PY, str(TRAINING / "infer_simple.py"),
                                    str(chip_ref), str(MODEL_1CH),
                                    str(chip_1c), "--tile-scale"],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-1500:])
                    print(r.stderr[-1500:])
                    raise RuntimeError(f"1-channel inference failed on {chip_ref.name}")
                den1 = fits.getdata(chip_1c).astype(np.float32)
                w = gaussian_filter(covered.astype(np.float32), 8)
                den = w * den + (1.0 - w) * den1
                print(f"    gap band ({(~covered).mean()*100:.1f}% of chip) "
                      f"filled with the 1-channel model", flush=True)
            # Re-anchor the denoised background on the input background:
            # the model shifts the sky level by a small content-dependent
            # offset (+1.4 e- measured on the Arp 130 galaxy chip vs +0.15
            # on the empty chip), invisible on a noisy chip but glaring as
            # a chip-to-chip step once the background is smooth. Measured
            # on the DIFFERENCE image: the galaxy halo cancels pixel by
            # pixel there, and the sigma-clip rejects the CR spikes.
            # (Comparing clipped_sky(raw) - clipped_sky(den) does NOT work:
            # the clip keeps the halo on the noisy raw but rejects it on
            # the smooth denoised chip, so the halo pollutes the shift.)
            shift = clipped_sky(ref_elec - den)
            den += shift
            print(f"    background re-anchored ({shift:+.3f} e-)", flush=True)
            if args.blend < 1.0:
                # Strength knob: re-inject a fraction of the input (the
                # DQ/deepCR pre-cleaned chip, so no CR comes back with the
                # noise). Convex blend of two estimators of the same
                # scene, so photometry is preserved by construction.
                den = args.blend * den + (1.0 - args.blend) * ref_elec
                print(f"    blend {args.blend:.2f} (garde ~"
                      f"{(1-args.blend)*100:.0f}% du bruit)", flush=True)
            sci.data = den
            sci.header["HISTORY"] = (f"AstroSURE 2-channel N2N denoise "
                                     f"({Path(args.model).name}, asure2_flc.py)")
            try:
                hdul["ERR", ver].header["ERRSTALE"] = (
                    True, "SCI denoised; ERR still original FLC noise")
            except KeyError:
                pass
            print(f"  {path.name} SCI,{ver} ok", flush=True)
        hdul.writeto(out_path, overwrite=True)
        print(f"ecrit: {out_path}", flush=True)

    for hdul, _, _ in exposures:
        hdul.close()


if __name__ == "__main__":
    main()
