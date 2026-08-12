"""Section 6: useful-signal statistics per field, and the star/cosmic-ray
inversion the new data are supposed to deliver.

Two pictures are produced for each field:
  RAW  : among the >30 sigma peaks of an FLC, how many are real stars and how
         many are cosmic rays. This is the prior that taught run 5
         'bright point = cosmic ray'.
  CLEANED : the same census on the deepCR-cleaned chip, i.e. what the network
         will actually be trained on.
Old fields are measured the same way, for comparison.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_RAW, CAT_CLEAN, CAT_FULL, load_catalog, group_label,
                       open_chip, sky_and_sigma, local_maxima, psf_like, aper_flux)
from s3b_align_clean import measure

R = 3
NEW = [("NGC 4258/F814W/chip1", 0, [1, 4]),
       ("NGC 598/F606W/chip1", 0, [1, 2]),
       ("NGC 598/F814W/chip1", 0, [1, 2]),
       ("NGC 1569/F606W/chip1", 0, [1, 2]),
       ("NGC 1569/F606W/chip2", 0, [1, 2]),
       ("IC 10/F814W/chip1", 0, [1, 2]),
       ("IC 10/F814W/chip2", 0, [1, 2]),
       ("NGC 3031/F606W/chip1", 0, [1, 2]),
       ("NGC 3031/F606W/chip2", 0, [1, 2])]
OLD = ["NGC 628/F435W/chip1", "NGC 628/F814W/chip2", "NGC 1365/F814W/chip2",
       "NGC 1559/F814W/chip2"]

TIERS = ((10, 30, "10-30"), (30, 100, "30-100"), (100, 1e12, ">100"))


def census(files_raw, files_cln, ia, ibs, which):
    """which='raw' or 'clean': image whose peaks are counted. Classification
    always uses the RAW partner exposures (a cleaned partner would have had its
    own cosmic rays removed, biasing the test)."""
    A = open_chip(files_raw[ia])
    Ac = open_chip(files_cln[ia])
    da = A["data"] if which == "raw" else Ac["data"]
    hdr_wcs = A["wcs"]
    sky, sig = sky_and_sigma(da)
    h, w = da.shape

    ys, xs = local_maxima(da, sky + 10 * sig, box=3, edge=45)
    if len(ys) > 120000:
        o = np.random.default_rng(2).choice(len(ys), 120000, replace=False)
        ys, xs = ys[o], xs[o]
    pk = (da[ys, xs] - sky) / sig
    ispsf, _ = psf_like(da, ys, xs, sky, sig, 0.25, 0.95)
    fa = aper_flux(da, ys, xs, sky, R)

    rats = []
    for ib in ibs:
        B = open_chip(files_raw[ib])
        m = measure(open_chip(files_cln[ia]), open_chip(files_cln[ib]))
        off = (m["off_x"], m["off_y"]) if m else (0.0, 0.0)
        skyb, _ = sky_and_sigma(B["data"])
        sc = hdr_wcs.all_pix2world(np.column_stack([xs, ys]).astype(float), 0)
        xyb = B["wcs"].all_world2pix(sc, 0)
        xf, yf = xyb[:, 0] + off[0], xyb[:, 1] + off[1]
        inb = (xf > R + 2) & (xf < w - R - 3) & (yf > R + 2) & (yf < h - R - 3)
        xb = np.clip(np.round(xf).astype(int), R + 2, w - R - 3)
        yb = np.clip(np.round(yf).astype(int), R + 2, h - R - 3)
        fb = aper_flux(B["data"], yb, xb, skyb, R)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = fb / np.maximum(fa, 1e-6)
        rr[~inb] = np.nan
        rats.append(rr)
        del B
    M = np.vstack(rats)
    fin = np.all(np.isfinite(M), axis=0) & (fa > 0)
    star = ispsf & fin & np.all((M > 0.5) & (M < 2.0), axis=0)
    cr = fin & np.all(M <= 0.3, axis=0) & ~ispsf

    out = {"sigma": sig, "sky": sky, "n_peak": int(fin.sum())}
    for lo, hi, nm in TIERS:
        m = (pk >= lo) & (pk < hi) & fin
        ns, nc = int((star & m).sum()), int((cr & m).sum())
        out[nm] = {"star": ns, "cr": nc,
                   "ratio": ns / nc if nc else float("inf"),
                   "pct_star": 100.0 * ns / max(ns + nc, 1)}
    out["tot_star"] = sum(out[nm]["star"] for _, _, nm in TIERS)
    out["tot_cr"] = sum(out[nm]["cr"] for _, _, nm in TIERS)
    del A, Ac
    return out


def show(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'champ / groupe':24s} {'sigma':>6s} | "
          + " | ".join(f"{nm+' (n*/nCR, %*)':>26s}" for _, _, nm in TIERS))
    for gl, r in rows:
        cells = []
        for _, _, nm in TIERS:
            v = r[nm]
            cells.append(f"{v['star']:7d}/{v['cr']:<7d} {v['pct_star']:5.1f}%")
        print(f"{gl:24s} {r['sigma']:6.1f} | " + " | ".join(f"{c:>26s}" for c in cells))


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    full = {group_label(g): g for g in load_catalog(CAT_FULL)}

    res = {"new_raw": [], "new_clean": [], "old_raw": [], "old_clean": []}
    for gl, ia, ibs in NEW:
        fr, fc = raw[gl]["files"], cln[gl]["files"]
        res["new_raw"].append((gl, census(fr, fc, ia, ibs, "raw")))
        res["new_clean"].append((gl, census(fr, fc, ia, ibs, "clean")))
        print(f"  {gl} fait", flush=True)
    for gl in OLD:
        fc = full[gl]["files"]
        fr = [f.replace("training_data_crclean", "training_data") for f in fc]
        ibs = [1, 2] if len(fc) > 2 else [1]
        res["old_raw"].append((gl, census(fr, fc, 0, ibs, "raw")))
        res["old_clean"].append((gl, census(fr, fc, 0, ibs, "clean")))
        print(f"  {gl} fait", flush=True)

    show("NOUVEAUX CHAMPS - FLC BRUT (statistique vue par le run 5)", res["new_raw"])
    show("NOUVEAUX CHAMPS - APRES deepCR (statistique du futur entrainement)", res["new_clean"])
    show("ANCIENS CHAMPS - FLC BRUT", res["old_raw"])
    show("ANCIENS CHAMPS - APRES deepCR", res["old_clean"])

    print("\n--- agregat par jeu de donnees (etage >30 sigma, pics brillants) ---")
    for k, lbl in (("new_raw", "nouveaux, brut"), ("new_clean", "nouveaux, nettoyes"),
                   ("old_raw", "anciens, brut"), ("old_clean", "anciens, nettoyes")):
        s = sum(r["30-100"]["star"] + r[">100"]["star"] for _, r in res[k])
        c = sum(r["30-100"]["cr"] + r[">100"]["cr"] for _, r in res[k])
        print(f"  {lbl:22s} etoiles={s:7d}  rayons cosmiques={c:7d}  "
              f"ratio={s/max(c,1):6.2f}  part d'etoiles={100*s/max(s+c,1):5.1f} %")

    with open("qc_tmp/s6_stats.json", "w") as f:
        json.dump({k: [(gl, r) for gl, r in v] for k, v in res.items()}, f, indent=1)


if __name__ == "__main__":
    main()
