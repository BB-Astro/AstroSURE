"""Section 4e: are the 'destroyed stars' actually detector-fixed artifacts?

The partner test cannot tell a real star from a hot pixel / warm column: both
reproduce at the same place in every exposure of a group (the dithers are only
a few pixels, so a detector artifact stays inside the 7x7 aperture).

Built here: a detector-artifact map from exposures of DIFFERENT fields (the sky
is unrelated, so any pixel that peaks in several of them is a detector defect).
The section-4 damage statistics are then recomputed with those pixels removed.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip,
                       sky_and_sigma, local_maxima, psf_like, aper_flux)
from s3b_align_clean import measure

R = 3
# one exposure from each distinct field, per chip
MAP_SRC = {
    "chip1": [("NGC 4258/F814W/chip1", 0), ("NGC 598/F606W/chip1", 0),
              ("NGC 598/F814W/chip1", 0), ("NGC 1569/F606W/chip1", 0),
              ("IC 10/F814W/chip1", 0), ("NGC 3031/F606W/chip1", 0)],
    "chip2": [("NGC 4258/F814W/chip2", 0), ("NGC 598/F606W/chip2", 0),
              ("NGC 598/F814W/chip2", 0), ("NGC 1569/F606W/chip2", 0),
              ("IC 10/F814W/chip2", 0), ("NGC 3031/F606W/chip2", 0)],
}
CASES = [
    ("NGC 4258/F814W/chip1", 0, [1, 4]),
    ("NGC 4258/F814W/chip2", 0, [1, 4]),
    ("NGC 598/F606W/chip1", 0, [1, 2]),
    ("NGC 1569/F606W/chip1", 0, [1, 2]),
    ("NGC 1569/F606W/chip2", 0, [1, 2]),
    ("IC 10/F814W/chip1", 0, [1, 2]),
    ("IC 10/F814W/chip2", 0, [1, 2]),
    ("NGC 3031/F606W/chip1", 0, [1, 2]),
    ("NGC 3031/F606W/chip2", 0, [1, 2]),
]


def build_map(raw, srcs, min_hits=3):
    """Pixels that are a >30 sigma local max in >= min_hits unrelated fields."""
    acc = None
    for gl, ia in srcs:
        C = open_chip(raw[gl]["files"][ia])
        sky, sig = sky_and_sigma(C["data"])
        ys, xs = local_maxima(C["data"], sky + 30 * sig, box=2, edge=5)
        m = np.zeros(C["data"].shape, np.uint8)
        m[ys, xs] = 1
        acc = m if acc is None else acc + m
        del C
    return acc >= min_hits, acc


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}

    maps = {}
    for chip, srcs in MAP_SRC.items():
        bad, acc = build_map(raw, srcs)
        maps[chip] = bad
        print(f"{chip}: {int(bad.sum())} pixels artefact detecteur "
              f"(pic >30sigma dans >=3 champs sans rapport, sur {acc.size} px)")
        cols = np.bincount(np.nonzero(bad)[1], minlength=bad.shape[1])
        top = np.argsort(-cols)[:12]
        print(f"   colonnes les plus touchees: "
              f"{[(int(c), int(cols[c])) for c in top if cols[c] > 5]}")

    print(f"\n{'groupe':24s} {'N* strict':>9s} {'dont artefact':>13s} "
          f"{'surv* hors art.':>15s} {'abime hors art.':>15s} {'abime artefacts':>15s}")
    res = []
    for gl, ia, ibs in CASES:
        chip = "chip1" if gl.endswith("chip1") else "chip2"
        bad = maps[chip]
        A = open_chip(raw[gl]["files"][ia])
        Ac = open_chip(cln[gl]["files"][ia])
        sky_a, sig_a = sky_and_sigma(A["data"])
        sky_c, sig_c = sky_and_sigma(Ac["data"])
        da, dc = A["data"], Ac["data"]
        h, w = da.shape
        ys, xs = local_maxima(da, sky_a + 30 * sig_a, box=3, edge=45)
        pk = (da[ys, xs] - sky_a) / sig_a
        ispsf, ring = psf_like(da, ys, xs, sky_a, sig_a, 0.25, 0.95)
        fa = aper_flux(da, ys, xs, sky_a, R)
        fc = aper_flux(dc, ys, xs, sky_c, R)
        rats = []
        for ib in ibs:
            B = open_chip(raw[gl]["files"][ib])
            m = measure(open_chip(cln[gl]["files"][ia]), open_chip(cln[gl]["files"][ib]))
            off = (m["off_x"], m["off_y"]) if m else (0.0, 0.0)
            sky_b, _ = sky_and_sigma(B["data"])
            sc = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
            xyb = B["wcs"].all_world2pix(sc, 0)
            xf, yf = xyb[:, 0] + off[0], xyb[:, 1] + off[1]
            inb = (xf > R + 2) & (xf < w - R - 3) & (yf > R + 2) & (yf < h - R - 3)
            xb = np.clip(np.round(xf).astype(int), R + 2, w - R - 3)
            yb = np.clip(np.round(yf).astype(int), R + 2, h - R - 3)
            fb = aper_flux(B["data"], yb, xb, sky_b, R)
            with np.errstate(invalid="ignore", divide="ignore"):
                rr = fb / np.maximum(fa, 1e-6)
            rr[~inb] = np.nan
            rats.append(rr)
            del B
        M = np.vstack(rats)
        fin = np.all(np.isfinite(M), axis=0) & (fa > 0)
        star = ispsf & fin & np.all((M > 0.5) & (M < 2.0), axis=0)
        # artifact if the peak, or any pixel of its 3x3 core, is in the map
        art = np.zeros(len(ys), bool)
        for i in np.where(star)[0]:
            art[i] = bad[ys[i] - 1:ys[i] + 2, xs[i] - 1:xs[i] + 2].any()
        with np.errstate(invalid="ignore", divide="ignore"):
            surv = fc / np.maximum(fa, 1e-6)
        diff = da - dc
        clean_star = star & ~art
        art_star = star & art
        row = {"group": gl, "n_star": int(star.sum()), "n_art": int(art_star.sum())}
        if clean_star.sum():
            row["surv"] = float(np.median(surv[clean_star]))
            row["dmg"] = float((diff[ys[clean_star], xs[clean_star]] > 5 * sig_a).mean())
            row["lost"] = float(np.mean(surv[clean_star] < 0.5))
            for lo, hi, nm in ((30, 100, "t30_100"), (100, 1e12, "t100p")):
                m2 = clean_star & (pk >= lo) & (pk < hi)
                row[nm] = {"n": int(m2.sum()),
                           "surv": float(np.median(surv[m2])) if m2.sum() >= 5 else None,
                           "dmg": float((diff[ys[m2], xs[m2]] > 5 * sig_a).mean()) if m2.sum() >= 5 else None}
        if art_star.sum():
            row["dmg_art"] = float((diff[ys[art_star], xs[art_star]] > 5 * sig_a).mean())
        res.append(row)
        print(f"{gl:24s} {row['n_star']:9d} {row['n_art']:13d} "
              f"{100*row.get('surv',float('nan')):14.1f}% "
              f"{100*row.get('dmg',float('nan')):14.1f}% "
              f"{100*row.get('dmg_art',float('nan')):14.1f}%", flush=True)
        del A, Ac
    print(f"\n{'groupe':24s} {'30-100 sigma':>26s} {'>100 sigma':>26s}  (hors artefacts)")
    for r in res:
        def f(k):
            v = r.get(k)
            if not v or v.get("surv") is None:
                return f"n={v['n'] if v else 0}"
            return f"n={v['n']} surv={100*v['surv']:.0f}% abime={100*v['dmg']:.0f}%"
        print(f"{r['group']:24s} {f('t30_100'):>26s} {f('t100p'):>26s}")
    with open("qc_tmp/s4e_artifacts.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
