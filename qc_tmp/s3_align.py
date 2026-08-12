"""Section 3: WCS alignment accuracy, measured on common-star centroids.

For a sample of pairs in every group:
  - find PSF-like local maxima in A (well above background, not saturated),
  - project each to B through WCS (all_world2pix / all_pix2world, SIP included),
  - refine both positions by a 7x7 flux-weighted centroid,
  - the residual (centroid_B - projected_B) is the alignment error.

Also reports the WCS translation used by the dataset and the overlap fraction.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_RAW, load_catalog, group_label, open_chip,
                       sky_and_sigma, local_maxima, psf_like, centroid)

PAIRS_PER_GROUP = 3
MAX_STARS = 400


def analyse_pair(A, B, verbose=False):
    da, db = A["data"], B["data"]
    sky_a, sig_a = sky_and_sigma(da)
    sky_b, sig_b = sky_and_sigma(db)

    # overlap: project A corners into B
    h, w = da.shape
    corners = np.array([[10, 10], [w - 10, 10], [w - 10, h - 10], [10, h - 10]], float)
    sky = A["wcs"].all_pix2world(corners, 0)
    xy_b = B["wcs"].all_world2pix(sky, 0)
    x0, x1 = max(0, xy_b[:, 0].min()), min(w, xy_b[:, 0].max())
    y0, y1 = max(0, xy_b[:, 1].min()), min(h, xy_b[:, 1].max())
    overlap = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / (w * h)

    # global translation at image centre
    c = np.array([[w / 2, h / 2]], float)
    s = A["wcs"].all_pix2world(c, 0)
    cb = B["wcs"].all_world2pix(s, 0)
    tx, ty = float(cb[0, 0] - c[0, 0]), float(cb[0, 1] - c[0, 1])

    # candidate stars in A: bright, PSF-like, isolated, not saturated
    thr = sky_a + 40 * sig_a
    ys, xs = local_maxima(da, thr, box=3, edge=30)
    if len(ys) == 0:
        return None
    keep, ratio = psf_like(da, ys, xs, sky_a, sig_a, 0.20, 0.90)
    ys, xs = ys[keep], xs[keep]
    peaks = da[ys, xs]
    keep = peaks < 60000  # avoid saturated / bleeding
    ys, xs, peaks = ys[keep], xs[keep], peaks[keep]
    if len(ys) == 0:
        return None
    order = np.argsort(-peaks)[:MAX_STARS]
    ys, xs = ys[order], xs[order]

    # project to B
    sky_coords = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
    xy = B["wcs"].all_world2pix(sky_coords, 0)

    res_x, res_y = [], []
    for (xa, ya, xb_f, yb_f) in zip(xs, ys, xy[:, 0], xy[:, 1]):
        xb, yb = int(round(xb_f)), int(round(yb_f))
        if xb < 12 or yb < 12 or xb > db.shape[1] - 13 or yb > db.shape[0] - 13:
            continue
        # must also be a significant peak in B (else it is a CR, not a star)
        loc = db[yb - 2:yb + 3, xb - 2:xb + 3]
        if loc.max() - sky_b < 20 * sig_b:
            continue
        # snap to the local max in B (within +-2 px) then centroid
        dy, dx = np.unravel_index(np.argmax(loc), loc.shape)
        yb2, xb2 = yb - 2 + dy, xb - 2 + dx
        ca = centroid(da, int(ya), int(xa), 3)
        cb2 = centroid(db, yb2, xb2, 3)
        if ca is None or cb2 is None:
            continue
        # predicted position of A's centroid in B
        s2 = A["wcs"].all_pix2world([[ca[1], ca[0]]], 0)
        p = B["wcs"].all_world2pix(s2, 0)[0]
        res_x.append(cb2[1] - p[0])
        res_y.append(cb2[0] - p[1])

    if len(res_x) < 10:
        return {"n": len(res_x), "overlap": overlap, "tx": tx, "ty": ty}
    rx, ry = np.array(res_x), np.array(res_y)
    r = np.hypot(rx, ry)
    return {
        "n": len(rx),
        "overlap": overlap,
        "tx": tx, "ty": ty,
        "med_dx": float(np.median(rx)), "med_dy": float(np.median(ry)),
        "med_r": float(np.median(r)),
        "p90_r": float(np.percentile(r, 90)),
        "rms_x": float(np.std(rx)), "rms_y": float(np.std(ry)),
        "sig_a": sig_a, "sig_b": sig_b,
    }


def main():
    cat = load_catalog(CAT_RAW)
    rng = np.random.default_rng(7)
    results = []
    print(f"{'group':28s} {'pair':>7s} {'n*':>5s} {'ovlp':>6s} {'tx':>8s} {'ty':>8s} "
          f"{'med|dr|':>8s} {'p90':>7s} {'med dx':>7s} {'med dy':>7s}")
    for g in cat:
        files = g["files"]
        combos = list(combinations(range(len(files)), 2))
        pick = rng.choice(len(combos), size=min(PAIRS_PER_GROUP, len(combos)), replace=False)
        cache = {}
        for k in pick:
            ia, ib = combos[k]
            for i in (ia, ib):
                if i not in cache:
                    cache[i] = open_chip(files[i])
            r = analyse_pair(cache[ia], cache[ib])
            if r is None:
                print(f"{group_label(g):28s} {ia}-{ib:<5d} no star found")
                continue
            r["group"] = group_label(g)
            r["galaxy"] = g["galaxy"]
            r["pair"] = f"{ia}-{ib}"
            results.append(r)
            if "med_r" in r:
                print(f"{group_label(g):28s} {r['pair']:>7s} {r['n']:5d} {r['overlap']:6.3f} "
                      f"{r['tx']:8.2f} {r['ty']:8.2f} {r['med_r']:8.3f} {r['p90_r']:7.3f} "
                      f"{r['med_dx']:7.3f} {r['med_dy']:7.3f}", flush=True)
            else:
                print(f"{group_label(g):28s} {r['pair']:>7s} {r['n']:5d} {r['overlap']:6.3f} "
                      f"{r['tx']:8.2f} {r['ty']:8.2f}   too few stars", flush=True)
        cache.clear()

    with open("qc_tmp/s3_align.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\n--- par champ ---")
    gals = sorted({r["galaxy"] for r in results})
    for gal in gals:
        rs = [r for r in results if r["galaxy"] == gal and "med_r" in r]
        if not rs:
            print(f"{gal:10s} aucune mesure")
            continue
        med = np.median([r["med_r"] for r in rs])
        worst = max(r["med_r"] for r in rs)
        ov = min(r["overlap"] for r in rs)
        nn = int(np.median([r["n"] for r in rs]))
        print(f"{gal:10s} med|dr|={med:.3f} px  pire paire={worst:.3f} px  "
              f"recouvrement min={ov:.3f}  n*~{nn}")


if __name__ == "__main__":
    main()
