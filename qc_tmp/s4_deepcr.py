"""Section 4: quality of the deepCR cleaning, validated by the partner exposure.

For one chip per field (plus IC 10 and NGC 3031 which are mandatory):
  1. detect local maxima > 30 sigma in the RAW chip, PSF-like pre-filter aside;
  2. classify each with the PARTNER exposure at the same sky position
     (aperture flux ratio B/A > 0.3 => real star; else cosmic ray);
  3. measure the survival of each class in the CLEANED chip;
  4. background sigma before/after;
  5. aperture photometry of faint 10-30 sigma confirmed stars before/after.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip,
                       sky_and_sigma, local_maxima, psf_like, aper_flux)
from s3b_align_clean import measure

R_APER = 3          # 7x7 aperture
RATIO_TRUE = 0.3    # partner flux ratio above which the source is a real star

# one chip per field; IC 10 and NGC 3031 mandatory
TARGETS = [
    ("NGC 4258/F814W/chip1", 0, 1),
    ("NGC 598/F606W/chip1", 0, 1),
    ("NGC 598/F814W/chip1", 0, 1),
    ("NGC 1569/F606W/chip1", 0, 1),
    ("IC 10/F814W/chip1", 0, 1),
    ("IC 10/F814W/chip2", 0, 1),
    ("NGC 3031/F606W/chip1", 0, 1),
    ("NGC 3031/F606W/chip2", 0, 1),
]


def analyse(graw, gclean, ia, ib):
    A = open_chip(graw["files"][ia])
    B = open_chip(graw["files"][ib])
    Ac = open_chip(gclean["files"][ia])

    sky_a, sig_a = sky_and_sigma(A["data"])
    sky_b, sig_b = sky_and_sigma(B["data"])
    sky_ac, sig_ac = sky_and_sigma(Ac["data"])

    # WCS offset A->B, so the partner test looks at the right place
    m = measure(open_chip(gclean["files"][ia]), open_chip(gclean["files"][ib]))
    off = (m["off_x"], m["off_y"]) if m else (0.0, 0.0)

    da, db, dc = A["data"], B["data"], Ac["data"]
    h, w = da.shape

    res = {"file": Path(graw["files"][ia]).name,
           "partner": Path(graw["files"][ib]).name,
           "sky_raw": sky_a, "sigma_raw": sig_a,
           "sky_clean": sky_ac, "sigma_clean": sig_ac,
           "sigma_ratio": sig_ac / sig_a,
           "wcs_off": off,
           "ncrpix": int(Ac["hdr"].get("NCRPIX", -1)),
           "ncrpix_frac": float(Ac["hdr"].get("NCRPIX", 0)) / da.size}

    # ---------- bright peaks (>30 sigma) ----------
    ys, xs = local_maxima(da, sky_a + 30 * sig_a, box=3, edge=40)
    keepnb = db[ys, xs] < 60000
    ys, xs = ys[keepnb], xs[keepnb]
    if len(ys) > 40000:
        o = np.argsort(-da[ys, xs])[:40000]
        ys, xs = ys[o], xs[o]
    _, ratio_ring = psf_like(da, ys, xs, sky_a, sig_a)

    # project to B
    skyc = A["wcs"].all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
    xyb = B["wcs"].all_world2pix(skyc, 0)
    xb = np.round(xyb[:, 0] + off[0]).astype(int)
    yb = np.round(xyb[:, 1] + off[1]).astype(int)
    ok = (xb > R_APER + 2) & (yb > R_APER + 2) & (xb < w - R_APER - 3) & (yb < h - R_APER - 3)
    ys, xs, xb, yb, ratio_ring = ys[ok], xs[ok], xb[ok], yb[ok], ratio_ring[ok]

    fa = aper_flux(da, ys, xs, sky_a, R_APER)
    fb = aper_flux(db, yb, xb, sky_b, R_APER)
    fc = aper_flux(dc, ys, xs, sky_ac, R_APER)
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = fb / np.maximum(fa, 1e-6)          # partner ratio
        surv = fc / np.maximum(fa, 1e-6)        # survival after cleaning

    is_star = (rp > RATIO_TRUE) & (fa > 0)
    is_cr = (rp <= RATIO_TRUE) & (fa > 0)

    peak_sig = (da[ys, xs] - sky_a) / sig_a
    res["n_peaks_30s"] = int(len(ys))
    res["n_star"] = int(is_star.sum())
    res["n_cr"] = int(is_cr.sum())
    res["star_cr_ratio"] = float(is_star.sum() / max(is_cr.sum(), 1))
    if is_star.sum():
        res["star_surv_med"] = float(np.median(surv[is_star]))
        res["star_surv_p10"] = float(np.percentile(surv[is_star], 10))
        res["star_surv_p90"] = float(np.percentile(surv[is_star], 90))
        res["star_surv_lt50"] = float(np.mean(surv[is_star] < 0.5))
        res["star_ring_med"] = float(np.median(ratio_ring[is_star]))
    if is_cr.sum():
        res["cr_surv_med"] = float(np.median(surv[is_cr]))
        res["cr_surv_p90"] = float(np.percentile(surv[is_cr], 90))
        res["cr_surv_gt50"] = float(np.mean(surv[is_cr] > 0.5))
        res["cr_ring_med"] = float(np.median(ratio_ring[is_cr]))

    # by brightness tier (confirmed stars only)
    tiers = {}
    for lo, hi, name in ((30, 100, "30-100s"), (100, 1e9, ">100s")):
        m2 = is_star & (peak_sig >= lo) & (peak_sig < hi)
        if m2.sum() >= 5:
            tiers[name] = {"n": int(m2.sum()),
                           "surv_med": float(np.median(surv[m2])),
                           "surv_p10": float(np.percentile(surv[m2], 10))}
        else:
            tiers[name] = {"n": int(m2.sum())}
    # cosmic rays by tier
    for lo, hi, name in ((30, 100, "cr30-100s"), (100, 1e9, "cr>100s")):
        m2 = is_cr & (peak_sig >= lo) & (peak_sig < hi)
        if m2.sum() >= 5:
            tiers[name] = {"n": int(m2.sum()), "surv_med": float(np.median(surv[m2]))}
        else:
            tiers[name] = {"n": int(m2.sum())}
    res["tiers"] = tiers

    # ---------- faint stars 10-30 sigma ----------
    ysf, xsf = local_maxima(da, sky_a + 10 * sig_a, box=3, edge=40)
    pk = (da[ysf, xsf] - sky_a) / sig_a
    sel = pk < 30
    ysf, xsf = ysf[sel], xsf[sel]
    if len(ysf) > 60000:
        o = np.random.default_rng(1).choice(len(ysf), 60000, replace=False)
        ysf, xsf = ysf[o], xsf[o]
    skyc = A["wcs"].all_pix2world(np.column_stack([xsf, ysf]).astype(float), 0)
    xyb = B["wcs"].all_world2pix(skyc, 0)
    xbf = np.round(xyb[:, 0] + off[0]).astype(int)
    ybf = np.round(xyb[:, 1] + off[1]).astype(int)
    ok = (xbf > R_APER + 2) & (ybf > R_APER + 2) & (xbf < w - R_APER - 3) & (ybf < h - R_APER - 3)
    ysf, xsf, xbf, ybf = ysf[ok], xsf[ok], xbf[ok], ybf[ok]
    faf = aper_flux(da, ysf, xsf, sky_a, R_APER)
    fbf = aper_flux(db, ybf, xbf, sky_b, R_APER)
    fcf = aper_flux(dc, ysf, xsf, sky_ac, R_APER)
    with np.errstate(invalid="ignore", divide="ignore"):
        rpf = fbf / np.maximum(faf, 1e-6)
        survf = fcf / np.maximum(faf, 1e-6)
    starf = (rpf > RATIO_TRUE) & (faf > 0)
    crf = (rpf <= RATIO_TRUE) & (faf > 0)
    res["n_faint"] = int(len(ysf))
    res["n_faint_star"] = int(starf.sum())
    res["n_faint_cr"] = int(crf.sum())
    if starf.sum():
        res["faint_star_surv_med"] = float(np.median(survf[starf]))
        res["faint_star_surv_p10"] = float(np.percentile(survf[starf], 10))
        res["faint_star_surv_lt50"] = float(np.mean(survf[starf] < 0.5))
    if crf.sum():
        res["faint_cr_surv_med"] = float(np.median(survf[crf]))

    # ---------- how many modified pixels sit on a confirmed star ----------
    diff = da - dc
    touched = diff != 0
    res["frac_px_modified"] = float(touched.mean())
    if is_star.sum():
        yy, xx = ys[is_star], xs[is_star]
        res["star_peak_touched"] = float(touched[yy, xx].mean())
        # any modified pixel inside the 7x7 aperture
        cnt = 0
        for y, x in zip(yy, xx):
            if touched[y - R_APER:y + R_APER + 1, x - R_APER:x + R_APER + 1].any():
                cnt += 1
        res["star_aper_touched"] = cnt / len(yy)
    if is_cr.sum():
        yy, xx = ys[is_cr], xs[is_cr]
        res["cr_peak_touched"] = float(touched[yy, xx].mean())
    del A, B, Ac
    return res


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    out = []
    for gl, ia, ib in TARGETS:
        print(f"\n### {gl}", flush=True)
        r = analyse(raw[gl], cln[gl], ia, ib)
        r["group"] = gl
        r["galaxy"] = raw[gl]["galaxy"]
        out.append(r)
        print(f"  fichier            {r['file']}")
        print(f"  NCRPIX             {r['ncrpix']} ({100*r['ncrpix_frac']:.2f} % des pixels)")
        print(f"  px reellement modifies {100*r['frac_px_modified']:.2f} %")
        print(f"  sigma fond  brut={r['sigma_raw']:.2f}  propre={r['sigma_clean']:.2f}  "
              f"ratio={r['sigma_ratio']:.4f}")
        print(f"  fond median brut={r['sky_raw']:.2f}  propre={r['sky_clean']:.2f}")
        print(f"  pics >30sigma      {r['n_peaks_30s']}  -> etoiles {r['n_star']}  "
              f"rayons cosmiques {r['n_cr']}  (ratio {r['star_cr_ratio']:.3f})")
        if "star_surv_med" in r:
            print(f"  survie ETOILES     med={100*r['star_surv_med']:.1f} %  "
                  f"p10={100*r['star_surv_p10']:.1f} %  p90={100*r['star_surv_p90']:.1f} %  "
                  f"perdues(<50%)={100*r['star_surv_lt50']:.2f} %")
            print(f"    pic touche par deepCR : {100*r['star_peak_touched']:.2f} %  "
                  f"| ouverture touchee : {100*r['star_aper_touched']:.2f} %")
        if "cr_surv_med" in r:
            print(f"  survie RAYONS COSM med={100*r['cr_surv_med']:.1f} %  "
                  f"p90={100*r['cr_surv_p90']:.1f} %  survivants(>50%)={100*r['cr_surv_gt50']:.2f} %"
                  f"  | pic touche : {100*r['cr_peak_touched']:.2f} %")
        for k, v in r["tiers"].items():
            s = f"n={v['n']}"
            if "surv_med" in v:
                s += f" survie={100*v['surv_med']:.1f} %"
            if "surv_p10" in v:
                s += f" p10={100*v['surv_p10']:.1f} %"
            print(f"    {k:10s} {s}")
        print(f"  faibles 10-30sigma n={r['n_faint']} -> etoiles {r['n_faint_star']} "
              f"CR {r['n_faint_cr']}")
        if "faint_star_surv_med" in r:
            print(f"    survie etoiles faibles med={100*r['faint_star_surv_med']:.1f} % "
                  f"p10={100*r['faint_star_surv_p10']:.1f} % "
                  f"perdues={100*r['faint_star_surv_lt50']:.2f} %")
        if "faint_cr_surv_med" in r:
            print(f"    survie CR faibles med={100*r['faint_cr_surv_med']:.1f} %")

    with open("qc_tmp/s4_deepcr.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
