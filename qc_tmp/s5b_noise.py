"""Section 5 (redone): noise independence, WITHOUT any interpolation.

The first attempt was biased: three of the four exposures were bilinearly
resampled onto the reference grid and the fourth was not, so the common sky
signal did not cancel in the differences and the estimator returned values
above 1, which is impossible for a correlation.

Here only pairs whose WCS translation is within 0.12 px of an INTEGER are used,
so every exposure is a raw crop, treated identically, and the common signal S
cancels exactly in a difference:

    A - C = n_A - n_C ,  B - D = n_B - n_D          (S gone)
    corr(A-C, B-D) = 0 under mutual independence
                   ~ rho_AB / 2 if A and B share noise

reported as rho_hat = 2 * corr(A-C, B-D).

Second, model-free check on the pair itself:
    Var(A-B) / [Var(A-C) + Var(B-C) - Var(A-B)]   -> 1 under independence,
and the naive correlation of A and B is printed only to show how misleading it
is (it measures the common sky, not the noise).
"""

import json
import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip

PATCH = 384
N_PATCH = 8
INT_TOL = 0.12

GROUPS = ["NGC 4258/F814W/chip1", "NGC 598/F606W/chip1", "NGC 598/F814W/chip1",
          "NGC 1569/F606W/chip1", "IC 10/F814W/chip1", "NGC 3031/F606W/chip1"]


def prep(chip):
    d = chip["data"] / float(chip["prim"].get("EXPTIME", 1.0) or 1.0)
    return d - float(np.median(d[::8, ::8]))


def shift_between(ref, other, y, x):
    rd = ref["wcs"].all_pix2world([[x, y]], 0)
    a = ref["wcs"].all_world2pix(rd, 0)[0]
    b = other["wcs"].all_world2pix(rd, 0)[0]
    return b[0] - a[0], b[1] - a[1]


def near_integer(chips, ref_i, cand, y, x):
    """Indices among `cand` whose shift w.r.t. chips[ref_i] is near-integer."""
    ok = []
    for j in cand:
        tx, ty = shift_between(chips[ref_i], chips[j], y, x)
        if (abs(tx - round(tx)) < INT_TOL) and (abs(ty - round(ty)) < INT_TOL):
            ok.append((j, int(round(tx)), int(round(ty))))
    return ok


def main():
    out = {}
    for tag, catp in (("brut", CAT_RAW), ("nettoye", CAT_CLEAN)):
        cat = {group_label(g): g for g in load_catalog(catp)}
        print(f"\n================ donnees {tag} ================")
        print(f"{'groupe':24s} {'quadruplets':>11s} {'npatch':>6s} {'%fond':>6s} "
              f"{'rho_bruit':>10s} {'[min,max]':>18s} {'corr naive':>11s} {'test var':>9s}")
        recs = []
        for gl in GROUPS:
            g = cat[gl]
            chips = [open_chip(f) for f in g["files"]]
            for c in chips:
                c["data"] = prep(c)
            n = len(chips)
            cy, cx = chips[0]["data"].shape[0] // 2, chips[0]["data"].shape[1] // 2
            grp = near_integer(chips, 0, range(1, n), cy, cx)
            # quadruple: reference 0 + three exposures with integer shift to 0
            rows = []
            nquad = 0
            if len(grp) >= 3:
                rng = np.random.default_rng(5)
                for a, b, c_ in list(combinations(range(len(grp)), 3))[:4]:
                    nquad += 1
                    (jb, tbx, tby) = grp[a]
                    (jc, tcx, tcy) = grp[b]
                    (jd, tdx, tdy) = grp[c_]
                    tries = 0
                    got = 0
                    while got < N_PATCH // 2 and tries < 40:
                        tries += 1
                        y0 = int(rng.integers(80, chips[0]["data"].shape[0] - PATCH - 80))
                        x0 = int(rng.integers(80, chips[0]["data"].shape[1] - PATCH - 80))
                        P = [chips[0]["data"][y0:y0 + PATCH, x0:x0 + PATCH]]
                        bad = False
                        for (j, tx, ty) in ((jb, tbx, tby), (jc, tcx, tcy), (jd, tdx, tdy)):
                            yy, xx = y0 + ty, x0 + tx
                            if yy < 0 or xx < 0 or yy + PATCH > chips[j]["data"].shape[0] \
                                    or xx + PATCH > chips[j]["data"].shape[1]:
                                bad = True
                                break
                            P.append(chips[j]["data"][yy:yy + PATCH, xx:xx + PATCH])
                        if bad:
                            continue
                        P = [p.astype(np.float64) for p in P]
                        if not all(np.isfinite(p).all() for p in P):
                            continue
                        mask = np.ones_like(P[0], bool)
                        for p in P:
                            s = np.median(p)
                            sg = 1.4826 * np.median(np.abs(p - s))
                            mask &= ~maximum_filter(np.abs(p - s) > 4 * sg, size=9)
                        if mask.mean() < 0.05:
                            continue
                        A, B, C, D = (p[mask] for p in P)
                        ac, bd = A - C, B - D
                        rho = 2 * float(np.corrcoef(ac, bd)[0, 1])
                        vab, vac, vbc = np.var(A - B), np.var(A - C), np.var(B - C)
                        denom = vac + vbc - vab
                        rows.append({"rho": rho,
                                     "naive": float(np.corrcoef(A, B)[0, 1]),
                                     "vtest": float(vab / denom) if denom > 0 else np.nan,
                                     "frac": float(mask.mean())})
                        got += 1
            for c in chips:
                c["data"] = None
            if not rows:
                print(f"{gl:24s} {len(grp):11d}  aucun quadruplet a decalage entier exploitable")
                recs.append({"group": gl, "tag": tag, "n": 0,
                             "n_intshift": len(grp)})
                continue
            rho = np.array([r["rho"] for r in rows])
            nv = np.array([r["naive"] for r in rows])
            vt = np.array([r["vtest"] for r in rows])
            fr = np.array([r["frac"] for r in rows])
            print(f"{gl:24s} {nquad:11d} {len(rows):6d} {100*fr.mean():5.0f}% "
                  f"{np.median(rho):10.4f} [{rho.min():+.4f},{rho.max():+.4f}] "
                  f"{np.median(nv):11.4f} {np.nanmedian(vt):9.3f}", flush=True)
            recs.append({"group": gl, "tag": tag, "n": len(rows), "n_intshift": len(grp),
                         "rho_med": float(np.median(rho)), "rho_min": float(rho.min()),
                         "rho_max": float(rho.max()), "naive_med": float(np.median(nv)),
                         "vtest_med": float(np.nanmedian(vt)), "frac_bg": float(fr.mean())})
        out[tag] = recs
    with open("qc_tmp/s5b_noise.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nrho_bruit : 0 = bruits independants (hypothese N2N respectee).")
    print("test var  : Var(A-B)/[Var(A-C)+Var(B-C)-Var(A-B)] -> 1 si independants, "
          "<<1 revelerait un doublon.")
    print("corr naive: correlation brute de A et B sur le fond = signal commun, "
          "sans rapport avec l'hypothese N2N.")


if __name__ == "__main__":
    main()
