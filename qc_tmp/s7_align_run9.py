"""Section 7: WCS zero-point QC of the run-9 galaxy groups.

Same methodology as s3_align.py / s3b_align_clean.py:
  1. PSF-like maxima in the group's reference exposure (files[0]),
  2. projected into every other exposure through the full WCS (SIP included),
  3. both positions refined by a 7x7 flux-weighted centroid,
  4. the sigma-clipped median of (centroid - projection) is that exposure's
     WCS zero-point offset, in the exact sign convention consumed by
     dataset_n2n._extract_pair (tx = xy_b - xy_a + (off_b - off_a)).

Two extra passes that s3b did not have inline:
  - VERIFICATION: the match is redone with the measured offset applied; the
    leftover systematic is the residual after correction (target < 0.1 px).
  - GRADIENT TEST: the residual is also fitted per image quadrant. A pure
    translation shows the same offset everywhere; a relative roll or a
    distortion-solution mismatch shows a gradient across the field, which
    no single translation can fix (this is what disqualified NGC 4258
    chip2 in the previous extension).

Read-only on the FITS data; writes qc_tmp/s7_align_run9.json.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (ROOT, load_catalog, group_label, open_chip,  # noqa: E402
                       sky_and_sigma, local_maxima, psf_like, centroid,
                       cd_angle_deg)
from s3b_align_clean import clipped_median  # noqa: E402

CAT_RUN9 = ROOT / "training_data" / "pairs_catalog_run9_new.json"
OUT_JSON = Path(__file__).resolve().parent / "s7_align_run9.json"

MIN_MATCH = 12          # minimum matched stars for a usable measurement
MIN_GROUP = 4           # a unit below this is not worth keeping (run-8 needs >= 3)
CLOSURE_PAIRS = 4       # direct pair measurements per group for the closure test
# A whole-group unit is only accepted if the closure - the disagreement
# between a directly measured pair offset and the difference of the two
# stored offsets - stays under this. It is the outcome the training actually
# sees, so it catches units whose star sample was too poor for the plane fit
# to return a gradient verdict (NGC 4639/F606W/chip1: no exposure flagged,
# closure 0.166 px; split by visit it drops to 0.013 px).
CLOSURE_MAX_PX = 0.12
MIN_GRAD_STARS = 25     # below this the plane fit cannot decide on a gradient
MIN_SPAN_FRAC = 0.35    # stars must cover this fraction of the chip in x AND y

# Quadrant-to-quadrant peak-to-peak of the residual above which the exposure
# cannot be registered by a translation. Calibrated on the previous extension:
# the healthy inter-visit exposures sit at 0.06-0.35 px on both chips, while
# the single defective one (NGC 4258 visit 02, j8r002yl) sits at 1.06 px on
# chip1 and 1.11 px on chip2 -- it is the EXPOSURE that is broken, not the
# group, so run 9 drops the exposure instead of the whole chip group.
GRADIENT_MAX_PX = 0.50
GRADIENT_ERR_MAX = 0.15  # above this the plane fit cannot decide (no verdict)

# Backstop on the relative roll read straight from the CD matrices, expressed
# as the differential displacement it causes at the chip corner (radius 2290
# px). It still works on the star-poor fields where the plane fit returns no
# verdict, and it treats chip1 and chip2 of one exposure alike.
#
# Calibration: the trainer recomputes the translation PER PATCH, at the patch
# centre, through each exposure's own WCS (dataset_n2n._extract_pair), so a
# roll that the CD matrix models is absorbed except within the patch itself.
# The leftover is r_patch x theta with r_patch = 181 px (256x256 half
# diagonal), i.e. 0.079 x this number. 0.80 px at the chip corner is therefore
# 0.063 px inside a patch, just under the 0.1 px target, and it still catches
# the genuinely broken exposure of the previous extension (NGC 4258 visit 02,
# 1.04 px from the group median roll, residual gradient 1.5-1.7 px).
ROLL_MAX_PX = 0.80


def outer_ring_ratio(d, ys, xs, sky):
    """Mean of the 8 pixels at Chebyshev radius 2, over the peak.

    The inner-ring test of psf_like already rejects single-pixel hits, but a
    2-3 pixel cosmic ray passes it. The ACS/WFC PSF still puts a few percent
    of its peak at radius 2, a cosmic ray puts essentially nothing there.
    This matters here because the run-9 chips are RAW (no CR cleaning) and
    the halo fields are star-poor: without it the matched sample is diluted
    by CR/CR coincidences and every derived statistic gets noisy.
    """
    peak = d[ys, xs] - sky
    acc = np.zeros(len(ys), dtype=np.float64)
    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2),
                   (-2, -2), (-2, 2), (2, -2), (2, 2)):
        acc += d[ys + dy, xs + dx] - sky
    with np.errstate(invalid="ignore", divide="ignore"):
        return acc / 8.0 / np.maximum(peak, 1e-6)


def star_candidates(d, sky, sig, nsig, maxn=800):
    ys, xs = local_maxima(d, sky + nsig * sig, box=3, edge=32)
    if len(ys) == 0:
        return ys, xs
    keep, _ = psf_like(d, ys, xs, sky, sig, 0.25, 0.90)
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0:
        return ys, xs
    keep = outer_ring_ratio(d, ys, xs, sky) > 0.03
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0:
        return ys, xs
    pk = d[ys, xs]
    keep = pk < 60000            # saturated cores bleed and bias the centroid
    ys, xs, pk = ys[keep], xs[keep], pk[keep]
    if len(ys) == 0:
        return ys, xs
    o = np.argsort(-pk)[:maxn]
    return ys[o], xs[o]


def adaptive_candidates(d, sky, sig):
    """Galaxy fields at high galactic latitude are star-poor: relax the
    detection threshold until there are enough compact sources."""
    for nsig in (40, 25, 15, 10):
        ys, xs = star_candidates(d, sky, sig, nsig)
        if len(ys) >= 40:
            return ys, xs, nsig
    return ys, xs, 10


def present_in(O, sky_coords, win=3):
    """Is each sky position a significant, PSF-like source in exposure O?"""
    d = O["data"]
    sky, sig = sky_and_sigma(d)
    xy = O["wcs"].all_world2pix(sky_coords, 0)
    h, w = d.shape
    hit = np.zeros(len(sky_coords), bool)
    for i, (xf, yf) in enumerate(zip(xy[:, 0], xy[:, 1])):
        xb, yb = int(round(xf)), int(round(yf))
        if xb < 12 or yb < 12 or xb > w - 13 or yb > h - 13:
            continue
        loc = d[yb - win:yb + win + 1, xb - win:xb + win + 1]
        if loc.max() - sky < 20 * sig:
            continue
        dy, dx = np.unravel_index(np.argmax(loc), loc.shape)
        yb2, xb2 = yb - win + dy, xb - win + dx
        pk = d[yb2, xb2] - sky
        ring = (d[yb2 - 1, xb2] + d[yb2 + 1, xb2]
                + d[yb2, xb2 - 1] + d[yb2, xb2 + 1]) / 4 - sky
        if pk <= 0 or not (0.20 < ring / pk < 0.95):
            continue
        hit[i] = True
    return hit


def vet_candidates(A, others, ys, xs, n_needed=2):
    """Keep only the reference sources that other exposures confirm.

    These are RAW chips: a 1200 s ACS/WFC exposure carries ~10^4 cosmic rays
    per chip, so ~3% of the reference candidates find a CR in the partner
    within the +-2 px search box purely by chance. On a star-poor halo field
    that is most of the matched sample, and it inflates the per-star scatter
    to >1 px (measured on NGC 1316) which in turn fakes field gradients.
    Cosmic rays are uncorrelated between exposures, so requiring the source
    in two OTHER exposures removes them essentially completely.
    """
    if not others:
        return ys, xs
    sky_coords = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
    votes = np.zeros(len(ys), int)
    for O in others:
        votes += present_in(O, sky_coords)
    keep = votes >= min(n_needed, len(others))
    return ys[keep], xs[keep]


def match_xy(A, B, ys, xs, sky_b, sig_b, win, pre=(0.0, 0.0)):
    """Matched-star residuals WITH their positions.

    Returns an (n, 4) array: x_ref, y_ref, dx, dy, where (dx, dy) is
    centroid_B - WCS_projection_of_A_centroid, in B pixels, after `pre`
    has been applied to the projection.
    """
    da, db = A["data"], B["data"]
    sky_coords = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
    xy = B["wcs"].all_world2pix(sky_coords, 0)
    out = []
    hb, wb = db.shape
    for (xa, ya, xbf, ybf) in zip(xs, ys, xy[:, 0] + pre[0], xy[:, 1] + pre[1]):
        xb, yb = int(round(xbf)), int(round(ybf))
        if xb < 12 or yb < 12 or xb > wb - 13 or yb > hb - 13:
            continue
        loc = db[yb - win:yb + win + 1, xb - win:xb + win + 1]
        if loc.max() - sky_b < 20 * sig_b:
            continue
        dy, dx = np.unravel_index(np.argmax(loc), loc.shape)
        yb2, xb2 = yb - win + dy, xb - win + dx
        pk = db[yb2, xb2] - sky_b
        ring = (db[yb2 - 1, xb2] + db[yb2 + 1, xb2]
                + db[yb2, xb2 - 1] + db[yb2, xb2 + 1]) / 4 - sky_b
        if pk <= 0 or not (0.20 < ring / pk < 0.95):
            continue
        ca = centroid(da, int(ya), int(xa), 3)
        cb = centroid(db, yb2, xb2, 3)
        if ca is None or cb is None:
            continue
        s2 = A["wcs"].all_pix2world([[ca[1], ca[0]]], 0)
        p = B["wcs"].all_world2pix(s2, 0)[0]
        out.append((ca[1], ca[0], cb[1] - (p[0] + pre[0]), cb[0] - (p[1] + pre[1])))
    return np.array(out).reshape(-1, 4)


def plane_gradient(P):
    """Field dependence of the residual, from a robust plane fit.

    Splitting the chip into quadrant medians needs ~4x more stars than
    these halo fields provide, and with ~10 stars per quadrant the spread
    is dominated by centroid noise (measured: spurious 2.3 px "gradients"
    on a group whose closure residual is 0.04 px). A single 3-parameter
    plane per axis, fitted on all matched stars with sigma clipping, is
    well constrained by 25 stars and comes with an error bar.

    Returns (ptp, ptp_err, n_used) where ptp is the peak-to-peak of the
    fitted plane across the chip corners, or (None, None, n) when there
    are too few stars to decide.
    """
    n = len(P)
    if n < MIN_GRAD_STARS:
        return None, None, n
    x = P[:, 0] / 4096.0
    y = P[:, 1] / 2048.0
    A = np.column_stack([np.ones_like(x), x, y])
    # Evaluate the fitted plane over the region the stars actually cover, not
    # over the chip corners. In a star-poor halo field the usable sources sit
    # in one part of the frame, and extrapolating a plane to the far corners
    # produced 2 px "gradients" with 0.6 px error bars on NGC 1316 - pure
    # extrapolation variance, not a real field dependence.
    x0, x1 = np.percentile(x, [5, 95])
    y0, y1 = np.percentile(y, [5, 95])
    if (x1 - x0) < MIN_SPAN_FRAC or (y1 - y0) < MIN_SPAN_FRAC:
        return None, None, n
    corners = np.array([[1.0, x0, y0], [1.0, x1, y0],
                        [1.0, x0, y1], [1.0, x1, y1]])
    ptps, errs, nused = [], [], []
    for col in (2, 3):
        v = P[:, col]
        keep = np.ones(n, bool)
        coef = None
        for _ in range(6):
            if keep.sum() < 12:
                break
            coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
            r = v - A @ coef
            m = np.median(r[keep])
            s = 1.4826 * np.median(np.abs(r[keep] - m)) + 1e-6
            new = np.abs(r - m) < 3.0 * s
            if (new == keep).all():
                break
            keep = new
        if coef is None or keep.sum() < 12:
            return None, None, n
        coef, *_ = np.linalg.lstsq(A[keep], v[keep], rcond=None)
        resid = v[keep] - A[keep] @ coef
        dof = max(keep.sum() - 3, 1)
        s2 = float(resid @ resid) / dof
        try:
            cov = s2 * np.linalg.inv(A[keep].T @ A[keep])
        except np.linalg.LinAlgError:
            return None, None, n
        pred = corners @ coef
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", corners, cov, corners), 0))
        i_hi, i_lo = int(np.argmax(pred)), int(np.argmin(pred))
        ptps.append(float(pred[i_hi] - pred[i_lo]))
        errs.append(float(np.hypot(se[i_hi], se[i_lo])))
        nused.append(int(keep.sum()))
    k = int(np.argmax(ptps))
    return ptps[k], errs[k], min(nused)


def measure(A, B, cand=None):
    """Offset of B relative to A, plus verification and gradient diagnostics."""
    da, db = A["data"], B["data"]
    sky_a, sig_a = sky_and_sigma(da)
    sky_b, sig_b = sky_and_sigma(db)
    if cand is None:
        ys, xs, nsig = adaptive_candidates(da, sky_a, sig_a)
    else:
        ys, xs, nsig = cand
    if len(ys) < 10:
        return None

    P = match_xy(A, B, ys, xs, sky_b, sig_b, win=2)
    if len(P) < MIN_MATCH:
        return None
    mx, _, _ = clipped_median(P[:, 2])
    my, _, _ = clipped_median(P[:, 3])

    # fine pass: re-centre the search window on the coarse offset
    P2 = match_xy(A, B, ys, xs, sky_b, sig_b, win=1, pre=(mx, my))
    if len(P2) < MIN_MATCH:
        P2, base = P, (0.0, 0.0)
    else:
        base = (mx, my)
    fx, sx, nx = clipped_median(P2[:, 2])
    fy, sy, ny = clipped_median(P2[:, 3])
    off_x, off_y = base[0] + fx, base[1] + fy

    # VERIFICATION pass: apply the final offset and re-measure
    P3 = match_xy(A, B, ys, xs, sky_b, sig_b, win=1, pre=(off_x, off_y))
    if len(P3) < MIN_MATCH:
        return None
    vx, _, _ = clipped_median(P3[:, 2])
    vy, _, _ = clipped_median(P3[:, 3])
    resid_sys = float(np.hypot(vx, vy))
    resid_scatter = float(np.median(np.hypot(P3[:, 2] - vx, P3[:, 3] - vy)))
    grad, grad_err, grad_n = plane_gradient(P3)

    return {
        "n_cand": int(len(ys)), "nsig": nsig, "n_match": int(len(P3)),
        "n_used": int(min(nx, ny)),
        "off_x": float(off_x), "off_y": float(off_y),
        "off_r": float(np.hypot(off_x, off_y)),
        "resid_sys": resid_sys,          # leftover systematic after correction
        "resid_scatter": resid_scatter,  # per-star scatter (centroid noise floor)
        "gradient_ptp": grad,
        "gradient_err": grad_err,
        "gradient_n": grad_n,
    }


def is_gradient(m):
    """A gradient counts only when the fit is well constrained AND large.

    Requiring a small ABSOLUTE error bar is what separates a real defect
    from a noisy fit. Checked against the header roll angles, which give
    the rotation independently of the images: the exposures this flags are
    exactly those whose CD matrix really is rotated with respect to the
    group reference (NGC 4258 visit 02: 0.028 deg = 1.12 px at the chip
    corner, measured gradient 1.5-1.7 px), while the star-poor halo fields
    where the plane fit is ill-conditioned return err > 0.15 px and get no
    verdict at all rather than a false positive.
    """
    g, e = m["gradient_ptp"], m["gradient_err"]
    if g is None or e is None:
        return False
    if e > GRADIENT_ERR_MAX:
        return False
    return g > GRADIENT_MAX_PX and g > 5.0 * e


def median_roll_index(files):
    """Index of the exposure whose roll is the group median.

    Anchoring on files[0] is arbitrary: if that exposure sits at one end of
    the roll distribution, every other exposure looks rotated and the roll
    backstop fires on half the group (measured: 25 spurious drops out of 110
    across the run-9 fields, 1 with the median anchor). The median also makes
    chip1 and chip2 of a group pick the same reference exposure.
    """
    angles = []
    for f in files:
        with fits.open(f, memmap=False) as hdul:
            angles.append(cd_angle_deg(hdul["SCI"].header))
    a = np.asarray(angles)
    return int(np.argmin(np.abs(a - np.median(a))))


def measure_group(files, ref_idx=0):
    """Measure every exposure of a group against files[ref_idx].

    The reference's candidate list is built and vetted ONCE against three
    partner exposures, then reused for every pair: same stars everywhere,
    so the offsets are strictly comparable and the CR rejection is paid for
    only once.
    """
    A = open_chip(files[ref_idx])
    sky_a, sig_a = sky_and_sigma(A["data"])
    ys, xs, nsig = adaptive_candidates(A["data"], sky_a, sig_a)
    vet_idx = [j for j in range(len(files)) if j != ref_idx][:3]
    others = [open_chip(f) for f in (files[j] for j in vet_idx)]
    ys, xs = vet_candidates(A, others, ys, xs)
    del others
    print(f"    {len(ys)} sources confirmees par >= 2 autres poses "
          f"(seuil {nsig} sigma)", flush=True)
    cand = (ys, xs, nsig)

    rows = {}
    for j in range(len(files)):
        if j == ref_idx:
            continue
        B = open_chip(files[j])
        m = measure(A, B, cand)
        if m is not None:
            droll = cd_angle_deg(B["hdr"]) - cd_angle_deg(A["hdr"])
            m["droll_deg"] = float(droll)
            # differential displacement a relative roll causes at the chip
            # corner, i.e. the part of the misregistration that no single
            # translation can absorb
            m["droll_px"] = float(2290.0 * abs(np.radians(droll)))
        rows[j] = m
        del B
    del A
    return rows


def closure_test(files, offsets_by_idx, rng):
    """True residual after correction.

    The offsets are all measured against the same reference, so a direct
    measurement between two NON-reference exposures is an independent check:
    if the field really is a rigid translation, the direct offset must equal
    the difference of the two stored offsets. What is left is the residual
    the training will actually see.
    """
    idx = sorted(offsets_by_idx)
    if len(idx) < 2:
        return []
    pairs = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:]]
    if len(pairs) > CLOSURE_PAIRS:
        pick = rng.choice(len(pairs), size=CLOSURE_PAIRS, replace=False)
        pairs = [pairs[k] for k in pick]
    out = []
    for a, b in pairs:
        A = open_chip(files[a])
        B = open_chip(files[b])
        m = measure(A, B)
        del A, B
        if m is None:
            continue
        pred = (offsets_by_idx[b][0] - offsets_by_idx[a][0],
                offsets_by_idx[b][1] - offsets_by_idx[a][1])
        out.append({
            "pair": f"{a}-{b}",
            "direct": [m["off_x"], m["off_y"]],
            "predicted": list(pred),
            "closure": float(np.hypot(m["off_x"] - pred[0], m["off_y"] - pred[1])),
        })
    return out


def visit_of(name):
    """hst_14704_04_acs_wfc_f814w_jd8f04g4_chip1.fits -> '04' (HST visit)."""
    m = re.match(r"hst_(\d+)_(\w+?)_acs", name)
    return m.group(2) if m else "?"


def evaluate(files, rng, tag):
    """Measure one candidate unit. Returns (offsets_by_index, rows, closure)."""
    ref_idx = median_roll_index(files)
    rows = measure_group(files, ref_idx)
    if n_bad(rows) > len(rows) / 2 and len(files) > 2:
        alt = 0 if ref_idx != 0 else 1
        print(f"    {tag}: reference suspecte -> re-ancrage sur la pose {alt}")
        ref_idx = alt
        rows = measure_group(files, ref_idx)

    keep = {ref_idx: (0.0, 0.0)}
    bad = []
    for j in sorted(rows):
        m = rows[j]
        if m is None:
            bad.append((j, "mesure WCS impossible"))
        elif is_gradient(m):
            bad.append((j, f"gradient de champ {m['gradient_ptp']:.3f} "
                           f"+/- {m['gradient_err']:.3f} px (> {GRADIENT_MAX_PX})"))
        elif m.get("droll_px", 0.0) > ROLL_MAX_PX:
            bad.append((j, f"roulis relatif {m['droll_px']:.2f} px au coin "
                           f"(> {ROLL_MAX_PX})"))
        else:
            keep[j] = (m["off_x"], m["off_y"])
    closure = closure_test(files, keep, rng) if len(keep) >= 2 else []
    return keep, rows, bad, closure


def n_bad(rows):
    return sum(1 for m in rows.values() if m is None or is_gradient(m))


def main():
    cat_path = Path(sys.argv[1]) if len(sys.argv) > 1 else CAT_RUN9
    cat = load_catalog(cat_path)
    rng = np.random.default_rng(9)
    results, subgroups, excluded, dropped, closures = [], [], [], [], []

    for g in cat:
        gl = group_label(g)
        files = g["files"]
        print(f"\n=== {gl} ({len(files)} poses) ===", flush=True)

        keep, rows, bad, closure = evaluate(files, rng, "entier")
        for j, m in rows.items():
            if m is not None:
                m.update(group=gl, galaxy=g["galaxy"], filter=g["filter"],
                         chip=g["chip"], j=j, file=Path(files[j]).name)
                results.append(m)

        cv_whole = [c["closure"] for c in closure]
        closure_ok = (not cv_whole) or float(np.median(cv_whole)) <= CLOSURE_MAX_PX
        if not bad and closure_ok and len(keep) >= MIN_GROUP:
            # The whole group is one rigid unit: keep it whole, it is the
            # biggest and therefore the quietest stacked target.
            subgroups.append({
                "group": gl, "galaxy": g["galaxy"], "filter": g["filter"],
                "chip": g["chip"], "scope": "groupe entier",
                "files": [files[j] for j in sorted(keep)],
                "offsets": {Path(files[j]).name: list(keep[j]) for j in keep},
                "closure": [c["closure"] for c in closure],
            })
            cv = [c["closure"] for c in closure]
            closures.extend(closure)
            print(f"    -> unite entiere de {len(keep)} poses, fermeture "
                  f"median {np.median(cv) if cv else float('nan'):.3f} px")
            continue

        # Inter-visit field gradients: each HST visit is internally rigid, so
        # split by visit instead of throwing half the group away. Two units of
        # 4 are worth more than one unit of 4 plus 4 discarded exposures.
        by_visit = {}
        for i, f in enumerate(files):
            by_visit.setdefault(visit_of(Path(f).name), []).append(i)
        why_split = (f"{len(bad)} poses en defaut" if bad
                     else f"fermeture {np.median(cv_whole):.3f} px "
                          f"> {CLOSURE_MAX_PX}")
        print(f"    {why_split} -> decoupage par visite "
              f"{ {k: len(v) for k, v in sorted(by_visit.items())} }")

        got = False
        for vis, idxs in sorted(by_visit.items()):
            sub = [files[i] for i in idxs]
            if len(sub) < MIN_GROUP:
                for i in idxs:
                    dropped.append((gl, Path(files[i]).name,
                                    f"visite {vis} : {len(sub)} poses seulement "
                                    f"(< {MIN_GROUP}) apres decoupage"))
                continue
            k2, r2, b2, c2 = evaluate(sub, rng, f"visite {vis}")
            for j, why in b2:
                dropped.append((f"{gl} v{vis}", Path(sub[j]).name, why))
            if len(k2) < MIN_GROUP:
                for j in sorted(k2):
                    dropped.append((f"{gl} v{vis}", Path(sub[j]).name,
                                    f"visite {vis} reduite a {len(k2)} poses "
                                    f"(< {MIN_GROUP})"))
                continue
            cv = [c["closure"] for c in c2]
            closures.extend(c2)
            subgroups.append({
                "group": f"{gl} v{vis}", "galaxy": g["galaxy"],
                "filter": g["filter"], "chip": g["chip"],
                "scope": f"visite {vis}",
                "files": [sub[j] for j in sorted(k2)],
                "offsets": {Path(sub[j]).name: list(k2[j]) for j in k2},
                "closure": cv,
            })
            got = True
            print(f"    -> visite {vis} : unite de {len(k2)} poses, fermeture "
                  f"median {np.median(cv) if cv else float('nan'):.3f} px")
        if not got:
            if not bad and len(keep) >= MIN_GROUP:
                # Rejected only on closure and no visit is big enough: keep
                # the whole unit rather than lose the field, and say so.
                closures.extend(closure)
                subgroups.append({
                    "group": gl, "galaxy": g["galaxy"], "filter": g["filter"],
                    "chip": g["chip"],
                    "scope": f"groupe entier (fermeture {np.median(cv_whole):.3f} px)",
                    "files": [files[j] for j in sorted(keep)],
                    "offsets": {Path(files[j]).name: list(keep[j]) for j in keep},
                    "closure": cv_whole,
                })
                print(f"    -> conserve entier malgre la fermeture "
                      f"{np.median(cv_whole):.3f} px (aucune visite >= {MIN_GROUP})")
            else:
                excluded.append((gl, "aucune visite ne donne >= "
                                     f"{MIN_GROUP} poses alignables"))

    offsets = {}
    for sg in subgroups:
        offsets.update(sg["offsets"])

    json.dump({"measurements": results, "subgroups": subgroups,
               "offsets": offsets, "excluded": excluded,
               "dropped_exposures": dropped, "closure": closures},
              open(OUT_JSON, "w"), indent=1)

    print("\n--- unites retenues ---")
    for sg in sorted(subgroups, key=lambda x: x["group"]):
        cv = sg["closure"]
        print(f"{sg['group']:34s} {len(sg['files']):2d} poses  "
              f"fermeture med={np.median(cv) if cv else float('nan'):.3f} px")

    if results:
        o = np.array([r["off_r"] for r in results])
        print(f"\nTOTAL {len(results)} mesures de pose")
        print(f"  |offset| avant correction : median {np.median(o):.3f} px, "
              f"max {o.max():.3f} px, >0.3 px : {int((o > 0.3).sum())}/{len(o)}")
    if closures:
        cv = np.array([c["closure"] for c in closures])
        print(f"  residu apres correction (fermeture, {len(cv)} paires directes) : "
              f"median {np.median(cv):.3f} px, p90 {np.percentile(cv, 90):.3f} px, "
              f"max {cv.max():.3f} px")
    print(f"\nUnites : {len(subgroups)}  |  poses cataloguees : "
          f"{sum(len(sg['files']) for sg in subgroups)}")
    print(f"Poses ecartees : {len(dropped)}")
    for gl, f, why in dropped:
        print(f"  {gl} {f} : {why}")
    print(f"Groupes exclus : {len(excluded)}")
    for gl, why in excluded:
        print(f"  {gl} : {why}")
    print(f"\nOffsets mesures : {len(offsets)} chips -> {OUT_JSON}")


if __name__ == "__main__":
    main()
