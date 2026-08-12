"""Section 4g: split the deepCR damage between saturated stars (bleed trails,
which look exactly like a cosmic-ray track) and ordinary unsaturated stars.

A source is called SATURATED if its column contains >= 4 consecutive pixels
above 0.7 x the chip's saturation plateau near the peak, or simply if its peak
exceeds SAT_LEVEL. Damage on unsaturated stars is the metric that matters for
training: it is the one that would re-teach 'bright point = artifact'.
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
CASES = [
    ("NGC 4258/F814W/chip1", 0, [1, 4]),
    ("NGC 4258/F814W/chip2", 0, [1, 4]),
    ("NGC 598/F606W/chip1", 0, [1, 2]),
    ("NGC 598/F814W/chip1", 0, [1, 2]),
    ("NGC 1569/F606W/chip1", 0, [1, 2]),
    ("NGC 1569/F606W/chip2", 0, [1, 2]),
    ("IC 10/F814W/chip1", 0, [1, 2]),
    ("IC 10/F814W/chip2", 0, [1, 2]),
    ("NGC 3031/F606W/chip1", 0, [1, 2]),
    ("NGC 3031/F606W/chip2", 0, [1, 2]),
]


def bleed_length(da, y, x, level):
    """Number of consecutive pixels above `level` along the column through (y,x)."""
    n = 1
    i = y - 1
    while i >= 0 and da[i, x] > level:
        n += 1
        i -= 1
    i = y + 1
    while i < da.shape[0] and da[i, x] > level:
        n += 1
        i += 1
    return n


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    res = []
    print(f"{'groupe':24s} {'plateau':>8s} | {'N* NON SATUREES':>32s} | {'N* SATUREES':>24s}")
    print(f"{'':24s} {'e-':>8s} | {'n':>6s} {'survie':>8s} {'perdues':>8s} {'abimees':>7s} | "
          f"{'n':>6s} {'survie':>8s} {'abimees':>7s}")
    for gl, ia, ibs in CASES:
        A = open_chip(raw[gl]["files"][ia])
        Ac = open_chip(cln[gl]["files"][ia])
        sky, sig = sky_and_sigma(A["data"])
        sky_c, _ = sky_and_sigma(Ac["data"])
        da, dc = A["data"], Ac["data"]
        h, w = da.shape
        # saturation plateau: mode of the very high tail
        hi = da[da > np.percentile(da, 99.999)]
        plateau = float(np.median(hi)) if len(hi) else 65535.0
        sat_level = 0.6 * plateau

        ys, xs = local_maxima(da, sky + 30 * sig, box=3, edge=45)
        pk_e = da[ys, xs]
        pk_sig = (pk_e - sky) / sig
        ispsf, ring = psf_like(da, ys, xs, sky, sig, 0.25, 0.95)
        fa = aper_flux(da, ys, xs, sky, R)
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
        with np.errstate(invalid="ignore", divide="ignore"):
            surv = fc / np.maximum(fa, 1e-6)
        diff = da - dc

        idx = np.where(star)[0]
        sat = np.zeros(len(ys), bool)
        for i in idx:
            if pk_e[i] > sat_level or bleed_length(da, ys[i], xs[i], sat_level) >= 4:
                sat[i] = True
        uns = star & ~sat
        sats = star & sat
        row = {"group": gl, "plateau": plateau, "sat_level": sat_level,
               "n_unsat": int(uns.sum()), "n_sat": int(sats.sum())}
        if uns.sum():
            row["u_surv"] = float(np.median(surv[uns]))
            row["u_lost"] = float(np.mean(surv[uns] < 0.5))
            row["u_dmg"] = float((diff[ys[uns], xs[uns]] > 5 * sig).mean())
            for lo, hi_, nm in ((30, 100, "u30_100"), (100, 300, "u100_300"), (300, 1e12, "u300p")):
                m2 = uns & (pk_sig >= lo) & (pk_sig < hi_)
                row[nm] = {"n": int(m2.sum()),
                           "surv": float(np.median(surv[m2])) if m2.sum() >= 5 else None,
                           "dmg": float((diff[ys[m2], xs[m2]] > 5 * sig).mean()) if m2.sum() >= 5 else None}
        if sats.sum():
            row["s_surv"] = float(np.median(surv[sats]))
            row["s_dmg"] = float((diff[ys[sats], xs[sats]] > 5 * sig).mean())
        res.append(row)
        print(f"{gl:24s} {plateau:8.0f} | {row['n_unsat']:6d} "
              f"{100*row.get('u_surv',float('nan')):7.1f}% {100*row.get('u_lost',float('nan')):7.1f}% "
              f"{100*row.get('u_dmg',float('nan')):6.1f}% | {row['n_sat']:6d} "
              f"{100*row.get('s_surv',float('nan')):7.1f}% {100*row.get('s_dmg',float('nan')):6.1f}%",
              flush=True)
        del A, Ac

    print(f"\nEtoiles NON SATUREES par etage (n / survie / % abimees)")
    print(f"{'groupe':24s} {'30-100 sigma':>24s} {'100-300 sigma':>24s} {'>300 sigma':>24s}")
    for r in res:
        def f(k):
            v = r.get(k)
            if not v or v.get("surv") is None:
                return f"n={v['n'] if v else 0}"
            return f"n={v['n']} {100*v['surv']:.0f}% dmg{100*v['dmg']:.0f}%"
        print(f"{r['group']:24s} {f('u30_100'):>24s} {f('u100_300'):>24s} {f('u300p'):>24s}")
    with open("qc_tmp/s4g_saturation.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
