"""Section 4 (final): deepCR quality per field, with two-partner star confirmation.

A source > 30 sigma in raw chip A is declared a REAL STAR only if its aperture
flux is > 0.3x in TWO other exposures of the same group (triple CR coincidence
is negligible even at the 5 % CR density of M81-DEEP), and a COSMIC RAY if it
is absent from both. Survival is then measured inside chip A only (raw vs
cleaned), so it involves no alignment at all.
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


def run(graw, gcln, ia, ibs):
    A = open_chip(graw["files"][ia])
    Ac = open_chip(gcln["files"][ia])
    sky_a, sig_a = sky_and_sigma(A["data"])
    sky_c, sig_c = sky_and_sigma(Ac["data"])
    da, dc = A["data"], Ac["data"]
    h, w = da.shape
    diff = da - dc
    touched = diff != 0

    def classify(thr_lo, thr_hi, maxn):
        ys, xs = local_maxima(da, sky_a + thr_lo * sig_a, box=3, edge=45)
        pk = (da[ys, xs] - sky_a) / sig_a
        m = pk < thr_hi
        ys, xs = ys[m], xs[m]
        if len(ys) > maxn:
            o = np.random.default_rng(0).choice(len(ys), maxn, replace=False)
            ys, xs = ys[o], xs[o]
        fa = aper_flux(da, ys, xs, sky_a, R)
        fc = aper_flux(dc, ys, xs, sky_c, R)
        rats = []
        for ib in ibs:
            B = open_chip(graw["files"][ib])
            m2 = measure(open_chip(gcln["files"][ia]), open_chip(gcln["files"][ib]))
            off = (m2["off_x"], m2["off_y"]) if m2 else (0.0, 0.0)
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
        star = np.all(M > RATIO, axis=0) & fin
        cr = np.all(M <= RATIO, axis=0) & fin
        with np.errstate(invalid="ignore", divide="ignore"):
            surv = fc / np.maximum(fa, 1e-6)
        return ys, xs, pk[:len(ys)] if len(pk) == len(ys) else (da[ys, xs] - sky_a) / sig_a, \
            star, cr, surv

    out = {"sky_raw": sky_a, "sig_raw": sig_a, "sky_clean": sky_c, "sig_clean": sig_c,
           "sig_ratio": sig_c / sig_a, "sky_shift": sky_c - sky_a,
           "ncrpix": int(Ac["hdr"].get("NCRPIX", -1)),
           "ncrpix_pct": 100.0 * Ac["hdr"].get("NCRPIX", 0) / da.size,
           "file": Path(graw["files"][ia]).name}

    # bright > 30 sigma
    ys, xs, pk, star, cr, surv = classify(30, 1e12, 40000)
    out["n_star"] = int(star.sum())
    out["n_cr"] = int(cr.sum())
    out["star_cr"] = float(star.sum() / max(cr.sum(), 1))
    if star.sum():
        s = surv[star]
        out["star_surv_med"] = float(np.median(s))
        out["star_surv_q25"] = float(np.percentile(s, 25))
        out["star_lost50"] = float(np.mean(s < 0.5))
        yy, xx = ys[star], xs[star]
        # real damage: peak pixel lowered by more than 5 sigma
        dmg = (diff[yy, xx] > 5 * sig_a)
        out["star_peak_damaged"] = float(dmg.mean())
        out["star_peak_touched"] = float(touched[yy, xx].mean())
        for lo, hi, nm in ((30, 100, "t30_100"), (100, 1e12, "t100p")):
            m = star & (pk >= lo) & (pk < hi)
            out[nm] = {"n": int(m.sum()),
                       "surv": float(np.median(surv[m])) if m.sum() >= 5 else None}
    if cr.sum():
        out["cr_surv_med"] = float(np.median(surv[cr]))
        out["cr_kill95"] = float(np.mean(surv[cr] < 0.05))
        for lo, hi, nm in ((30, 100, "c30_100"), (100, 1e12, "c100p")):
            m = cr & (pk >= lo) & (pk < hi)
            out[nm] = {"n": int(m.sum()),
                       "surv": float(np.median(surv[m])) if m.sum() >= 5 else None}

    # faint 10-30 sigma
    ys, xs, pk, star, cr, surv = classify(10, 30, 40000)
    out["nf_star"] = int(star.sum())
    out["nf_cr"] = int(cr.sum())
    out["f_star_cr"] = float(star.sum() / max(cr.sum(), 1))
    if star.sum():
        s = surv[star]
        out["faint_surv_med"] = float(np.median(s))
        out["faint_surv_q25"] = float(np.percentile(s, 25))
        out["faint_lost50"] = float(np.mean(s < 0.5))
        out["faint_peak_damaged"] = float((diff[ys[star], xs[star]] > 5 * sig_a).mean())
    if cr.sum():
        out["faint_cr_surv"] = float(np.median(surv[cr]))
    del A, Ac
    return out


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    res = []
    hdr = (f"{'groupe':24s} {'NCR%':>5s} {'sig r/b':>7s} {'d ciel':>7s} "
           f"{'N*':>6s} {'NCR':>6s} {'*/CR':>6s} {'surv*':>7s} {'perdu':>6s} "
           f"{'abime':>6s} {'survCR':>7s} {'surv faible*':>12s} {'perdu f':>7s} {'*/CR f':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for gl, ia, ibs in CASES:
        r = run(raw[gl], cln[gl], ia, ibs)
        r["group"] = gl
        r["galaxy"] = raw[gl]["galaxy"]
        res.append(r)
        print(f"{gl:24s} {r['ncrpix_pct']:5.2f} {r['sig_ratio']:7.3f} "
              f"{r['sky_shift']:7.2f} {r['n_star']:6d} {r['n_cr']:6d} {r['star_cr']:6.2f} "
              f"{100*r.get('star_surv_med',float('nan')):6.1f}% "
              f"{100*r.get('star_lost50',float('nan')):5.1f}% "
              f"{100*r.get('star_peak_damaged',float('nan')):5.1f}% "
              f"{100*r.get('cr_surv_med',float('nan')):6.1f}% "
              f"{100*r.get('faint_surv_med',float('nan')):11.1f}% "
              f"{100*r.get('faint_lost50',float('nan')):6.1f}% "
              f"{r.get('f_star_cr',float('nan')):6.2f}", flush=True)
    print("\nDetail par etage de brillance (etoiles confirmees / rayons cosmiques confirmes)")
    print(f"{'groupe':24s} {'*30-100':>16s} {'*>100':>16s} {'CR30-100':>16s} {'CR>100':>16s}")
    for r in res:
        def fmt(k):
            v = r.get(k)
            if not v or v.get("surv") is None:
                return f"n={v['n'] if v else 0}"
            return f"n={v['n']} {100*v['surv']:.0f}%"
        print(f"{r['group']:24s} {fmt('t30_100'):>16s} {fmt('t100p'):>16s} "
              f"{fmt('c30_100'):>16s} {fmt('c100p'):>16s}")
    with open("qc_tmp/s4c_final.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
