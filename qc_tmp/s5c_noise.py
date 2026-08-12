"""Section 5 (final): noise independence, interpolation-free, with a signal
contamination diagnostic.

Design
------
Any exposure of a group may serve as the reference; the three partners are
those whose WCS translation to it is within 0.15 px of an integer, so all four
patches are raw crops on the same grid and the common sky signal S cancels in
A-C and B-D. Estimator:

    rho_hat = 2 * corr(A-C, B-D)      -> 0 under mutual independence

Residual sub-pixel misregistration leaves a term proportional to the sky
gradient in BOTH differences, which biases rho_hat upward in crowded fields.
That contamination is measured per patch by the lag-1 spatial autocorrelation
of A-C: pure photon+read noise is white (acf1 ~ 0), a signal residual is not.
rho_hat is therefore reported both overall and restricted to the whitest
patches, and its trend against acf1 is the actual evidence.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip

PATCH = 384
INT_TOL = 0.15
TARGET_PATCH = 12

GROUPS = ["NGC 4258/F814W/chip1", "NGC 598/F606W/chip1", "NGC 598/F814W/chip1",
          "NGC 1569/F606W/chip1", "NGC 1569/F606W/chip2", "IC 10/F814W/chip1",
          "IC 10/F814W/chip2", "NGC 3031/F606W/chip1"]


def prep(chip):
    d = chip["data"] / float(chip["prim"].get("EXPTIME", 1.0) or 1.0)
    return d - float(np.median(d[::8, ::8]))


def shift_between(ref, other, y, x):
    rd = ref["wcs"].all_pix2world([[x, y]], 0)
    a = ref["wcs"].all_world2pix(rd, 0)[0]
    b = other["wcs"].all_world2pix(rd, 0)[0]
    return b[0] - a[0], b[1] - a[1]


def acf1(img2d):
    """Mean of the lag-1 autocorrelation in x and y. 0 for white noise."""
    a = img2d - img2d.mean()
    v = (a * a).mean()
    if v <= 0:
        return np.nan
    cx = (a[:, :-1] * a[:, 1:]).mean() / v
    cy = (a[:-1, :] * a[1:, :]).mean() / v
    return float(0.5 * (cx + cy))


def find_quads(chips):
    """(ref, [(j,tx,ty) x3]) combinations with near-integer shifts."""
    n = len(chips)
    cy, cx = chips[0]["data"].shape[0] // 2, chips[0]["data"].shape[1] // 2
    out = []
    for r in range(n):
        ok = []
        for j in range(n):
            if j == r:
                continue
            tx, ty = shift_between(chips[r], chips[j], cy, cx)
            if abs(tx - round(tx)) < INT_TOL and abs(ty - round(ty)) < INT_TOL:
                ok.append((j, int(round(tx)), int(round(ty))))
        if len(ok) >= 3:
            for tri in combinations(ok, 3):
                out.append((r, list(tri)))
    return out


def analyse(group, tag):
    chips = [open_chip(f) for f in group["files"]]
    for c in chips:
        c["data"] = prep(c)
    quads = find_quads(chips)
    rows = []
    rng = np.random.default_rng(17)
    h, w = chips[0]["data"].shape
    for (r, tri) in quads:
        if len(rows) >= TARGET_PATCH:
            break
        for _ in range(25):
            if len(rows) >= TARGET_PATCH:
                break
            y0 = int(rng.integers(96, h - PATCH - 96))
            x0 = int(rng.integers(96, w - PATCH - 96))
            P = [chips[r]["data"][y0:y0 + PATCH, x0:x0 + PATCH]]
            bad = False
            for (j, tx, ty) in tri:
                yy, xx = y0 + ty, x0 + tx
                if yy < 0 or xx < 0 or yy + PATCH > h or xx + PATCH > w:
                    bad = True
                    break
                P.append(chips[j]["data"][yy:yy + PATCH, xx:xx + PATCH])
            if bad or not all(np.isfinite(p).all() for p in P):
                continue
            P = [p.astype(np.float64) for p in P]
            mask = np.ones_like(P[0], bool)
            for p in P:
                s = np.median(p)
                sg = 1.4826 * np.median(np.abs(p - s))
                mask &= ~maximum_filter(np.abs(p - s) > 4 * sg, size=9)
            if mask.mean() < 0.05:
                continue
            A2, B2, C2, D2 = P
            AC2 = A2 - C2
            a, b, c_, d = A2[mask], B2[mask], C2[mask], D2[mask]
            ac, bd = a - c_, b - d
            if np.std(ac) == 0 or np.std(bd) == 0:
                continue
            # whiteness measured on the masked difference, holes filled with 0
            m2 = np.where(mask, AC2 - AC2[mask].mean(), 0.0)
            rows.append({"rho": 2 * float(np.corrcoef(ac, bd)[0, 1]),
                         "acf1": acf1(m2),
                         "naive": float(np.corrcoef(a, b)[0, 1]),
                         "frac": float(mask.mean()),
                         "vtest": float(np.var(a - b) /
                                        max(np.var(ac) + np.var(b - c_) - np.var(a - b), 1e-12))})
    for c in chips:
        c["data"] = None
    return rows, len(quads)


def main():
    out = {}
    for tag, catp in (("brut", CAT_RAW), ("nettoye", CAT_CLEAN)):
        cat = {group_label(g): g for g in load_catalog(catp)}
        print(f"\n================ donnees {tag} ================")
        print(f"{'groupe':24s} {'quad':>5s} {'npatch':>6s} {'%fond':>6s} {'acf1':>7s} "
              f"{'rho tous':>9s} {'rho 1/3 + blancs':>17s} {'corr naive':>11s} {'test var':>9s}")
        recs = []
        for gl in GROUPS:
            rows, nq = analyse(cat[gl], tag)
            if not rows:
                print(f"{gl:24s} {nq:5d}  aucun patch exploitable")
                recs.append({"group": gl, "tag": tag, "n": 0, "nquad": nq})
                continue
            rho = np.array([r["rho"] for r in rows])
            ac = np.array([r["acf1"] for r in rows])
            nv = np.array([r["naive"] for r in rows])
            vt = np.array([r["vtest"] for r in rows])
            fr = np.array([r["frac"] for r in rows])
            k = max(1, len(rows) // 3)
            best = np.argsort(ac)[:k]
            print(f"{gl:24s} {nq:5d} {len(rows):6d} {100*fr.mean():5.0f}% "
                  f"{np.median(ac):7.3f} {np.median(rho):9.4f} "
                  f"{np.median(rho[best]):9.4f} (acf1={np.median(ac[best]):.3f}) "
                  f"{np.median(nv):11.4f} {np.nanmedian(vt):9.3f}", flush=True)
            recs.append({"group": gl, "tag": tag, "n": len(rows), "nquad": nq,
                         "rho_all": float(np.median(rho)),
                         "rho_white": float(np.median(rho[best])),
                         "acf1_all": float(np.median(ac)),
                         "acf1_white": float(np.median(ac[best])),
                         "naive": float(np.median(nv)),
                         "vtest": float(np.nanmedian(vt)),
                         "frac_bg": float(fr.mean()),
                         "per_patch": rows})
        out[tag] = recs
    with open("qc_tmp/s5c_noise.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\n--- rho en fonction de la blancheur du residu (tous groupes, donnees brutes) ---")
    allrows = [r for rec in out["brut"] for r in rec.get("per_patch", [])]
    if allrows:
        a = np.array([r["acf1"] for r in allrows])
        rh = np.array([r["rho"] for r in allrows])
        for lo, hi in ((-1, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 1.0)):
            m = (a >= lo) & (a < hi)
            if m.sum():
                print(f"  acf1 dans [{lo:.2f},{hi:.2f}) : n={m.sum():3d}  "
                      f"rho median={np.median(rh[m]):+.4f}")
        if len(a) > 5:
            print(f"  pente rho vs acf1 : {np.polyfit(a, rh, 1)[0]:+.3f} "
                  f"(extrapolation a acf1=0 : rho={np.polyval(np.polyfit(a, rh, 1), 0):+.4f})")


if __name__ == "__main__":
    main()
