"""Section 4d: hardened star definition, to separate genuine deepCR damage from
contamination of the star sample by CR/CR coincidences.

A source is a STAR only if ALL of:
  * PSF-like in the raw chip: mean of the 4 neighbours over peak in [0.25, 0.95]
    (a cosmic ray deposits its charge on 1-2 pixels and fails this);
  * present in TWO other exposures with a flux ratio in [0.5, 2.0] (a real star
    of the same EXPTIME reproduces its flux; a coincidence rarely does);
  * not saturated in the partner (peak < 60000 e-).
Survival is measured inside the same chip (raw vs cleaned), no alignment.
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
    ("NGC 1569/F606W/chip2", 0, [1, 2]),
    ("NGC 1569/F606W/chip1", 0, [1, 2]),
    ("NGC 3031/F606W/chip1", 0, [1, 2]),
    ("IC 10/F814W/chip2", 0, [1, 2]),
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
    pk_sig = (da[ys, xs] - sky_a) / sig_a
    ispsf, ring = psf_like(da, ys, xs, sky_a, sig_a, 0.25, 0.95)
    fa = aper_flux(da, ys, xs, sky_a, R)
    fc = aper_flux(dc, ys, xs, sky_c, R)

    rats, sat = [], np.zeros(len(ys), bool)
    for ib in ibs:
        B = open_chip(graw["files"][ib])
        m = measure(open_chip(gcln["files"][ia]), open_chip(gcln["files"][ib]))
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
        sat |= B["data"][yb, xb] > 60000
        del B

    M = np.vstack(rats)
    fin = np.all(np.isfinite(M), axis=0) & (fa > 0)
    star = ispsf & fin & ~sat & np.all((M > 0.5) & (M < 2.0), axis=0)
    cr = fin & ~ispsf & np.all(M <= 0.3, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        surv = fc / np.maximum(fa, 1e-6)
    diff = da - dc

    out = {"file": Path(graw["files"][ia]).name,
           "n_peak": int(len(ys)), "n_star": int(star.sum()), "n_cr": int(cr.sum())}
    if star.sum():
        s = surv[star]
        out.update(star_surv_med=float(np.median(s)),
                   star_surv_q25=float(np.percentile(s, 25)),
                   star_surv_q75=float(np.percentile(s, 75)),
                   star_lost=float(np.mean(s < 0.5)),
                   star_damaged=float((diff[ys[star], xs[star]] > 5 * sig_a).mean()),
                   ring_med=float(np.median(ring[star])))
        for lo, hi, nm in ((30, 100, "t30_100"), (100, 300, "t100_300"), (300, 1e12, "t300p")):
            m2 = star & (pk_sig >= lo) & (pk_sig < hi)
            out[nm] = {"n": int(m2.sum()),
                       "surv": float(np.median(surv[m2])) if m2.sum() >= 5 else None,
                       "dmg": float((diff[ys[m2], xs[m2]] > 5 * sig_a).mean()) if m2.sum() >= 5 else None}
        idx = np.where(star)[0]
        top = idx[np.argsort(-fa[idx])][:6]
        out["examples"] = [{"x": int(xs[i]), "y": int(ys[i]), "sig": float(pk_sig[i]),
                            "ring": float(ring[i]),
                            "pk_raw": float(da[ys[i], xs[i]]), "pk_cln": float(dc[ys[i], xs[i]]),
                            "surv": float(surv[i]),
                            "ratios": [float(r[i]) for r in rats]} for i in top]
    if cr.sum():
        out["cr_surv_med"] = float(np.median(surv[cr]))
    del A, Ac
    return out


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    res = []
    print(f"{'groupe':24s} {'Npic':>6s} {'N* strict':>9s} {'NCR':>6s} {'ring':>5s} "
          f"{'surv*':>7s} {'q25':>7s} {'perdu':>6s} {'abime':>6s} {'survCR':>7s}")
    for gl, ia, ibs in CASES:
        r = run(raw[gl], cln[gl], ia, ibs)
        r["group"] = gl
        res.append(r)
        print(f"{gl:24s} {r['n_peak']:6d} {r['n_star']:9d} {r['n_cr']:6d} "
              f"{r.get('ring_med',float('nan')):5.2f} "
              f"{100*r.get('star_surv_med',float('nan')):6.1f}% "
              f"{100*r.get('star_surv_q25',float('nan')):6.1f}% "
              f"{100*r.get('star_lost',float('nan')):5.1f}% "
              f"{100*r.get('star_damaged',float('nan')):5.1f}% "
              f"{100*r.get('cr_surv_med',float('nan')):6.1f}%", flush=True)
    print(f"\n{'groupe':24s} {'30-100s':>22s} {'100-300s':>22s} {'>300s':>22s}   (n, survie, % abime)")
    for r in res:
        def f(k):
            v = r.get(k)
            if not v or v.get("surv") is None:
                return f"n={v['n'] if v else 0}"
            return f"n={v['n']} {100*v['surv']:.0f}% dmg{100*v['dmg']:.0f}%"
        print(f"{r['group']:24s} {f('t30_100'):>22s} {f('t100_300'):>22s} {f('t300p'):>22s}")
    print("\nExemples (6 etoiles strictes les plus brillantes par champ) :")
    for r in res:
        print(f"  {r['group']}")
        for e in r.get("examples", []):
            print(f"    ({e['x']:5d},{e['y']:5d}) {e['sig']:7.0f}sig ring={e['ring']:.2f}  "
                  f"pic {e['pk_raw']:9.1f} -> {e['pk_cln']:9.1f}  survie {100*e['surv']:6.1f} %  "
                  f"ratios {['%.2f' % z for z in e['ratios']]}")
    with open("qc_tmp/s4d_strict.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
