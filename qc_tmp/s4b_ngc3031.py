"""Section 4b: hard confirmation of the NGC 3031 (M81-DEEP) deepCR failure.

A source is declared a real star only if it is present in TWO independent
partner exposures (a CR/CR/CR triple coincidence is negligible), which removes
any doubt raised by the 5.5 % CR pixel density of this field. Same test applied
to IC 10 as a control.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip,
                       sky_and_sigma, local_maxima, aper_flux)
from s3b_align_clean import measure

R = 3
RATIO = 0.3

CASES = [
    ("NGC 3031/F606W/chip1", 0, [1, 2]),
    ("NGC 3031/F606W/chip1", 1, [0, 3]),
    ("IC 10/F814W/chip1", 0, [1, 2]),
    ("NGC 598/F606W/chip1", 0, [1, 2]),
]


def run(graw, gcln, ia, ibs):
    A = open_chip(graw["files"][ia])
    Ac = open_chip(gcln["files"][ia])
    sky_a, sig_a = sky_and_sigma(A["data"])
    sky_c, sig_c = sky_and_sigma(Ac["data"])
    da, dc = A["data"], Ac["data"]
    h, w = da.shape

    ys, xs = local_maxima(da, sky_a + 30 * sig_a, box=3, edge=45)
    o = np.argsort(-da[ys, xs])[:40000]
    ys, xs = ys[o], xs[o]
    fa = aper_flux(da, ys, xs, sky_a, R)
    fc = aper_flux(dc, ys, xs, sky_c, R)
    peak_sig = (da[ys, xs] - sky_a) / sig_a

    ratios = []
    for ib in ibs:
        B = open_chip(graw["files"][ib])
        m = measure(open_chip(gcln["files"][ia]), open_chip(gcln["files"][ib]))
        off = (m["off_x"], m["off_y"]) if m else (0.0, 0.0)
        sky_b, sig_b = sky_and_sigma(B["data"])
        skyc = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
        xyb = B["wcs"].all_world2pix(skyc, 0)
        xb = np.clip(np.round(xyb[:, 0] + off[0]).astype(int), R + 2, w - R - 3)
        yb = np.clip(np.round(xyb[:, 1] + off[1]).astype(int), R + 2, h - R - 3)
        inb = ((xyb[:, 0] + off[0] > R + 2) & (xyb[:, 0] + off[0] < w - R - 3)
               & (xyb[:, 1] + off[1] > R + 2) & (xyb[:, 1] + off[1] < h - R - 3))
        fb = aper_flux(B["data"], yb, xb, sky_b, R)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = fb / np.maximum(fa, 1e-6)
        rr[~inb] = np.nan
        ratios.append(rr)
        del B

    R2 = np.vstack(ratios)
    both = np.all(R2 > RATIO, axis=0) & np.all(np.isfinite(R2), axis=0) & (fa > 0)
    none = np.all(R2 <= RATIO, axis=0) & np.all(np.isfinite(R2), axis=0) & (fa > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        surv = fc / np.maximum(fa, 1e-6)

    out = {
        "n_peaks": int(len(ys)),
        "n_star2": int(both.sum()),
        "n_cr2": int(none.sum()),
        "star_surv_med": float(np.median(surv[both])) if both.sum() else None,
        "star_surv_p25": float(np.percentile(surv[both], 25)) if both.sum() else None,
        "star_surv_p75": float(np.percentile(surv[both], 75)) if both.sum() else None,
        "star_lost": float(np.mean(surv[both] < 0.5)) if both.sum() else None,
        "cr_surv_med": float(np.median(surv[none])) if none.sum() else None,
    }
    for lo, hi, nm in ((30, 100, "30-100"), (100, 300, "100-300"), (300, 1e9, ">300")):
        m2 = both & (peak_sig >= lo) & (peak_sig < hi)
        out[nm] = {"n": int(m2.sum()),
                   "surv": float(np.median(surv[m2])) if m2.sum() >= 5 else None}
    # brightest confirmed stars, individually
    idx = np.where(both)[0]
    if len(idx):
        top = idx[np.argsort(-fa[idx])][:8]
        out["examples"] = [
            {"x": int(xs[i]), "y": int(ys[i]),
             "peak_raw": float(da[ys[i], xs[i]]), "peak_clean": float(dc[ys[i], xs[i]]),
             "sig": float(peak_sig[i]), "flux_raw": float(fa[i]),
             "flux_clean": float(fc[i]), "surv": float(surv[i]),
             "partner_ratios": [float(r[i]) for r in ratios]}
            for i in top]
    del A, Ac
    return out


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    res = []
    for gl, ia, ibs in CASES:
        print(f"\n### {gl}  chip index {ia}, partenaires {ibs}", flush=True)
        r = run(raw[gl], cln[gl], ia, ibs)
        r["group"] = gl
        r["ia"] = ia
        res.append(r)
        print(f"  pics >30sigma {r['n_peaks']}")
        print(f"  confirmes par 2 partenaires : {r['n_star2']} etoiles / {r['n_cr2']} rayons cosmiques")
        if r["star_surv_med"] is not None:
            print(f"  survie ETOILES (double confirmation) : med={100*r['star_surv_med']:.1f} % "
                  f"[q25={100*r['star_surv_p25']:.1f} %, q75={100*r['star_surv_p75']:.1f} %]  "
                  f"perdues={100*r['star_lost']:.1f} %")
        print(f"  survie rayons cosmiques : med={100*r['cr_surv_med']:.1f} %")
        for nm in ("30-100", "100-300", ">300"):
            v = r[nm]
            print(f"    {nm:8s} n={v['n']:6d} survie="
                  f"{'%.1f %%' % (100*v['surv']) if v['surv'] is not None else 'n/a'}")
        if "examples" in r:
            print("  8 etoiles confirmees les plus brillantes (pic brut -> pic nettoye) :")
            for e in r["examples"]:
                print(f"    ({e['x']:5d},{e['y']:5d}) {e['sig']:7.0f}sig  "
                      f"pic {e['peak_raw']:9.1f} -> {e['peak_clean']:9.1f}   "
                      f"flux {e['flux_raw']:10.1f} -> {e['flux_clean']:10.1f}  "
                      f"({100*e['surv']:6.1f} %)  ratios partenaires "
                      f"{['%.2f' % z for z in e['partner_ratios']]}")
    with open("qc_tmp/s4b_ngc3031.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
