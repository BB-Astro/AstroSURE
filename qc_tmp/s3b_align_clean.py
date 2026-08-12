"""Section 3b: alignment measured on the CR-cleaned chips (= the actual training data).

Two passes per pair:
  1. coarse: match A's PSF-like maxima to B's local max within +-2 px, take the
     sigma-clipped median offset (robust to accidental CR/CR matches);
  2. fine: re-match with the coarse offset applied and a +-1 px window, and
     report the clipped median/scatter of the residual.

Also, per group, every file is measured against file 0 so a single exposure with
a bad WCS zero point shows up as an isolated non-zero offset.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_CLEAN, load_catalog, group_label, open_chip,
                       sky_and_sigma, local_maxima, psf_like, centroid)


def clipped_median(v, nsig=3.0, iters=4):
    v = np.asarray(v, float)
    keep = np.ones(len(v), bool)
    for _ in range(iters):
        if keep.sum() < 5:
            break
        m = np.median(v[keep])
        s = 1.4826 * np.median(np.abs(v[keep] - m)) + 1e-6
        new = np.abs(v - m) < nsig * s
        if (new == keep).all():
            break
        keep = new
    if keep.sum() == 0:
        return float(np.median(v)), float(np.std(v)), 0
    return float(np.median(v[keep])), float(1.4826 * np.median(np.abs(v[keep] - np.median(v[keep])))), int(keep.sum())


def star_candidates(d, sky, sig, nsig=40, maxn=600):
    ys, xs = local_maxima(d, sky + nsig * sig, box=3, edge=32)
    if len(ys) == 0:
        return ys, xs
    keep, _ = psf_like(d, ys, xs, sky, sig, 0.25, 0.90)
    ys, xs = ys[keep], xs[keep]
    if len(ys) == 0:
        return ys, xs
    pk = d[ys, xs]
    keep = pk < 60000
    ys, xs, pk = ys[keep], xs[keep], pk[keep]
    if len(ys) == 0:
        return ys, xs
    o = np.argsort(-pk)[:maxn]
    return ys[o], xs[o]


def match(A, B, ys, xs, sky_b, sig_b, win, pre=(0.0, 0.0)):
    da, db = A["data"], B["data"]
    sky_coords = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
    xy = B["wcs"].all_world2pix(sky_coords, 0)
    rx, ry = [], []
    for (xa, ya, xbf, ybf) in zip(xs, ys, xy[:, 0] + pre[0], xy[:, 1] + pre[1]):
        xb, yb = int(round(xbf)), int(round(ybf))
        if xb < 12 or yb < 12 or xb > db.shape[1] - 13 or yb > db.shape[0] - 13:
            continue
        loc = db[yb - win:yb + win + 1, xb - win:xb + win + 1]
        if loc.max() - sky_b < 20 * sig_b:
            continue
        dy, dx = np.unravel_index(np.argmax(loc), loc.shape)
        yb2, xb2 = yb - win + dy, xb - win + dx
        # the B counterpart must be PSF-like too
        pk = db[yb2, xb2] - sky_b
        ring = (db[yb2 - 1, xb2] + db[yb2 + 1, xb2] + db[yb2, xb2 - 1] + db[yb2, xb2 + 1]) / 4 - sky_b
        if pk <= 0 or not (0.20 < ring / pk < 0.95):
            continue
        ca = centroid(da, int(ya), int(xa), 3)
        cb = centroid(db, yb2, xb2, 3)
        if ca is None or cb is None:
            continue
        s2 = A["wcs"].all_pix2world([[ca[1], ca[0]]], 0)
        p = B["wcs"].all_world2pix(s2, 0)[0]
        rx.append(cb[1] - p[0])
        ry.append(cb[0] - p[1])
    return np.array(rx), np.array(ry)


def measure(A, B):
    da, db = A["data"], B["data"]
    sky_a, sig_a = sky_and_sigma(da)
    sky_b, sig_b = sky_and_sigma(db)
    ys, xs = star_candidates(da, sky_a, sig_a)
    if len(ys) < 10:
        return None
    rx, ry = match(A, B, ys, xs, sky_b, sig_b, win=2)
    if len(rx) < 8:
        return None
    mx, _, _ = clipped_median(rx)
    my, _, _ = clipped_median(ry)
    rx2, ry2 = match(A, B, ys, xs, sky_b, sig_b, win=1, pre=(mx, my))
    if len(rx2) < 8:
        rx2, ry2 = rx, ry
    fx, sx, nx = clipped_median(rx2)
    fy, sy, ny = clipped_median(ry2)
    r = np.hypot(rx2 - 0.0, ry2 - 0.0)
    rr, _, _ = clipped_median(r)
    # residual scatter about the systematic offset
    rres = np.hypot(rx2 - fx, ry2 - fy)
    return {
        "n_cand": int(len(ys)), "n_match": int(len(rx2)), "n_used": int(min(nx, ny)),
        "off_x": fx, "off_y": fy, "off_r": float(np.hypot(fx, fy)),
        "scat_x": sx, "scat_y": sy,
        "med_abs_r": float(np.median(r)),
        "med_scatter": float(np.median(rres)),
    }


def main():
    cat = load_catalog(CAT_CLEAN)
    out = []
    print("Décalage systématique de chaque exposition vs exposition 0 du groupe "
          "(données CR-nettoyées)\n")
    print(f"{'group':26s} {'file':>4s} {'ncand':>6s} {'nmatch':>7s} "
          f"{'off_x':>7s} {'off_y':>7s} {'|off|':>7s} {'scat':>7s}")
    for g in cat:
        files = g["files"]
        A = open_chip(files[0])
        for j in range(1, len(files)):
            B = open_chip(files[j])
            m = measure(A, B)
            if m is None:
                print(f"{group_label(g):26s} {j:4d}   pas assez d'appariements")
                continue
            m["group"] = group_label(g)
            m["galaxy"] = g["galaxy"]
            m["ref"] = Path(files[0]).name
            m["file"] = Path(files[j]).name
            m["j"] = j
            out.append(m)
            flag = "  <<<" if m["off_r"] > 0.3 else ""
            print(f"{group_label(g):26s} {j:4d} {m['n_cand']:6d} {m['n_match']:7d} "
                  f"{m['off_x']:7.3f} {m['off_y']:7.3f} {m['off_r']:7.3f} "
                  f"{m['med_scatter']:7.3f}{flag}", flush=True)
            del B
        del A

    with open("qc_tmp/s3b_align_clean.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n--- synthèse par champ (|offset| systématique vs exposition de référence) ---")
    for gal in sorted({r["galaxy"] for r in out}):
        rs = [r for r in out if r["galaxy"] == gal]
        offs = np.array([r["off_r"] for r in rs])
        sc = np.array([r["med_scatter"] for r in rs])
        print(f"{gal:10s} n={len(rs):3d}  |off| med={np.median(offs):.3f} "
              f"max={offs.max():.3f}  dispersion med={np.median(sc):.3f} px  "
              f"expo >0.3px: {int((offs>0.3).sum())}/{len(rs)}")


if __name__ == "__main__":
    main()
