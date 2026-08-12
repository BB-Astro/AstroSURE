"""Section 3c: turn the per-exposure WCS offsets into a per-PAIR verdict,
and test whether the offset is a pure translation or has a field gradient."""

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import CAT_CLEAN, load_catalog, group_label, open_chip, sky_and_sigma
from s3b_align_clean import star_candidates, match, clipped_median


def visit_of(name):
    m = re.match(r"hst_(\d+)_(\w+?)_acs", name)
    return f"{m.group(1)}_{m.group(2)}" if m else "?"


def main():
    off = json.load(open("qc_tmp/s3b_align_clean.json"))
    cat = load_catalog(CAT_CLEAN)
    by_group = {}
    for r in off:
        by_group.setdefault(r["group"], {})[r["j"]] = (r["off_x"], r["off_y"])

    print("=== Décalage par PAIRE, dérivé des offsets par exposition ===")
    print(f"{'group':26s} {'npairs':>6s} {'<=0.1':>6s} {'0.1-0.3':>8s} "
          f"{'0.3-0.5':>8s} {'0.5-1':>6s} {'>1':>4s} {'median':>7s} {'max':>6s}  visites")
    tot = np.zeros(5, int)
    all_d = []
    per_group_rows = []
    for g in cat:
        gl = group_label(g)
        n = len(g["files"])
        o = np.zeros((n, 2))
        for j in range(1, n):
            o[j] = by_group.get(gl, {}).get(j, (np.nan, np.nan))
        ds = []
        for a, b in combinations(range(n), 2):
            ds.append(float(np.hypot(*(o[b] - o[a]))))
        ds = np.array(ds)
        bins = np.array([
            (ds <= 0.1).sum(), ((ds > 0.1) & (ds <= 0.3)).sum(),
            ((ds > 0.3) & (ds <= 0.5)).sum(), ((ds > 0.5) & (ds <= 1.0)).sum(),
            (ds > 1.0).sum()])
        tot += bins
        all_d.append(ds)
        visits = sorted({visit_of(Path(f).name) for f in g["files"]})
        print(f"{gl:26s} {len(ds):6d} {bins[0]:6d} {bins[1]:8d} {bins[2]:8d} "
              f"{bins[3]:6d} {bins[4]:4d} {np.median(ds):7.3f} {ds.max():6.3f}  "
              f"{len(visits)} ({','.join(v.split('_')[1] for v in visits)})")
        per_group_rows.append((gl, g["galaxy"], len(ds), bins.tolist(),
                               float(np.median(ds)), float(ds.max()), len(visits)))
    ds = np.concatenate(all_d)
    print(f"{'TOTAL':26s} {len(ds):6d} {tot[0]:6d} {tot[1]:8d} {tot[2]:8d} "
          f"{tot[3]:6d} {tot[4]:4d} {np.median(ds):7.3f} {ds.max():6.3f}")
    print(f"\nPaires avec desalignement > 0.3 px : {int(tot[2:].sum())}/{len(ds)} "
          f"({100*tot[2:].sum()/len(ds):.1f} %)")
    print(f"Paires avec desalignement > 0.5 px : {int(tot[3:].sum())}/{len(ds)} "
          f"({100*tot[3:].sum()/len(ds):.1f} %)")

    with open("qc_tmp/s3c_pairwise.json", "w") as f:
        json.dump({"rows": per_group_rows,
                   "hist_total": tot.tolist(),
                   "n_pairs": int(len(ds))}, f, indent=1)

    # ---- field dependence on a few worst pairs ----
    print("\n=== Dependance au champ du residu (quadrants de l'image) ===")
    tests = [
        ("NGC 598/F814W/chip2", 0, 5),
        ("NGC 4258/F814W/chip2", 0, 2),
        ("IC 10/F814W/chip1", 0, 4),
        ("NGC 3031/F606W/chip1", 0, 7),
        ("NGC 1569/F606W/chip1", 0, 8),
    ]
    gmap = {group_label(g): g for g in cat}
    for gl, ia, ib in tests:
        g = gmap[gl]
        A = open_chip(g["files"][ia])
        B = open_chip(g["files"][ib])
        sky_a, sig_a = sky_and_sigma(A["data"])
        sky_b, sig_b = sky_and_sigma(B["data"])
        ys, xs = star_candidates(A["data"], sky_a, sig_a, maxn=2000)
        rx, ry = match(A, B, ys, xs, sky_b, sig_b, win=2)
        # rebuild coordinates of matched stars: rerun with position bookkeeping
        # (match() drops entries; redo inline for quadrant stats)
        from qc_common import centroid
        skyc = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
        xy = B["wcs"].all_world2pix(skyc, 0)
        pts = []
        for (xa, ya, xbf, ybf) in zip(xs, ys, xy[:, 0], xy[:, 1]):
            xb, yb = int(round(xbf)), int(round(ybf))
            if xb < 12 or yb < 12 or xb > B["data"].shape[1] - 13 or yb > B["data"].shape[0] - 13:
                continue
            loc = B["data"][yb - 2:yb + 3, xb - 2:xb + 3]
            if loc.max() - sky_b < 20 * sig_b:
                continue
            dy, dx = np.unravel_index(np.argmax(loc), loc.shape)
            yb2, xb2 = yb - 2 + dy, xb - 2 + dx
            pk = B["data"][yb2, xb2] - sky_b
            ring = (B["data"][yb2 - 1, xb2] + B["data"][yb2 + 1, xb2]
                    + B["data"][yb2, xb2 - 1] + B["data"][yb2, xb2 + 1]) / 4 - sky_b
            if pk <= 0 or not (0.20 < ring / pk < 0.95):
                continue
            ca = centroid(A["data"], int(ya), int(xa), 3)
            cb = centroid(B["data"], yb2, xb2, 3)
            if ca is None or cb is None:
                continue
            s2 = A["wcs"].all_pix2world([[ca[1], ca[0]]], 0)
            p = B["wcs"].all_world2pix(s2, 0)[0]
            pts.append((ca[1], ca[0], cb[1] - p[0], cb[0] - p[1]))
        if len(pts) < 40:
            print(f"{gl} {ia}-{ib}: seulement {len(pts)} etoiles, saute")
            continue
        P = np.array(pts)
        print(f"\n{gl} paire {ia}-{ib}  n={len(P)}")
        for qy, ly in ((0, "bas"), (1, "haut")):
            for qx, lx in ((0, "gauche"), (1, "droite")):
                m = ((P[:, 1] > 1024 * qy) & (P[:, 1] <= 1024 * (qy + 1) + 1024 * qy) &
                     (P[:, 0] > 2048 * qx) & (P[:, 0] <= 2048 * (qx + 1)))
                if m.sum() < 5:
                    continue
                mx, _, _ = clipped_median(P[m, 2])
                my, _, _ = clipped_median(P[m, 3])
                print(f"   {ly:6s} {lx:7s} n={m.sum():4d}  dx={mx:+.3f}  dy={my:+.3f}")
        del A, B


if __name__ == "__main__":
    main()
