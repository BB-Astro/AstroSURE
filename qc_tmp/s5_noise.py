"""Section 5: are the noises of the two exposures of a pair independent?

The naive pixel-to-pixel correlation between two background patches is useless
here: both contain the SAME astrophysical background (unresolved stars, galaxy
light), which is common signal, not noise, and drives the correlation to a large
positive value with no bearing on the Noise2Noise hypothesis.

Signal-free estimator, using four distinct exposures A, B, C, D of a group:

    X_i = S + n_i          (S = common sky signal, identical in all four)
    A - C = n_A - n_C      (S cancels exactly)
    B - D = n_B - n_D
    corr(A-C, B-D) = [Cov(nA,nB) - Cov(nA,nD) - Cov(nC,nB) + Cov(nC,nD)] / ...

which is 0 for mutually independent noises, and ~= rho_AB / 2 if only A and B
share noise. Reported as rho_hat = 2 * corr(A-C, B-D).

Cross-checks:
  * naive corr(A, B) on the background, for reference (expected large, harmless);
  * variance ratio Var(A-B) / Var(A-C): a duplicate file would give ~0;
  * the same test on the CR-cleaned data (deepCR inpainting is a shared,
    deterministic operation and could in principle introduce correlation).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift, maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import (CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip,
                       sky_and_sigma)
from s3b_align_clean import measure

PATCH = 512
N_PATCH = 6

# one group per field, four exposures known to be in the same visit where
# possible (alignment matters here)
CASES = [
    ("NGC 4258/F814W/chip1", [0, 1, 4, 5]),
    ("NGC 598/F606W/chip1", [0, 1, 2, 3]),
    ("NGC 1569/F606W/chip1", [0, 1, 2, 3]),
    ("IC 10/F814W/chip1", [0, 1, 2, 3]),
    ("NGC 3031/F606W/chip1", [0, 1, 2, 3]),
]


def resample_to_ref(ref, other, y0, x0, n):
    """Extract `other` on ref's pixel grid over the [y0:y0+n, x0:x0+n] window,
    using the same WCS + bilinear machinery as dataset_n2n._extract_pair."""
    ra_dec = ref["wcs"].all_pix2world([[x0 + n / 2, y0 + n / 2]], 0)
    xy_r = ref["wcs"].all_world2pix(ra_dec, 0)[0]
    xy_o = other["wcs"].all_world2pix(ra_dec, 0)[0]
    tx, ty = xy_o[0] - xy_r[0], xy_o[1] - xy_r[1]
    by0f, bx0f = y0 + ty, x0 + tx
    by0, bx0 = int(np.floor(by0f)), int(np.floor(bx0f))
    fy, fx = by0f - by0, bx0f - bx0
    h, w = other["data"].shape
    if by0 - 1 < 0 or bx0 - 1 < 0 or by0 + n + 2 > h or bx0 + n + 2 > w:
        return None
    big = other["data"][by0 - 1:by0 + n + 2, bx0 - 1:bx0 + n + 2]
    return nd_shift(big, (-fy, -fx), order=1, mode="nearest")[1:1 + n, 1:1 + n]


def prep(chip):
    """e-/s and per-exposure sky removal, exactly like the dataset."""
    d = chip["data"] / float(chip["prim"].get("EXPTIME", 1.0) or 1.0)
    d = d - float(np.median(d[::8, ::8]))
    return d


def analyse(group, idxs, tag):
    chips = [open_chip(group["files"][i]) for i in idxs]
    for c in chips:
        c["data"] = prep(c)
    ref = chips[0]
    sky0, sig0 = sky_and_sigma(ref["data"])
    rng = np.random.default_rng(11)
    rows = []
    tries = 0
    while len(rows) < N_PATCH and tries < 60:
        tries += 1
        y0 = int(rng.integers(64, ref["data"].shape[0] - PATCH - 64))
        x0 = int(rng.integers(64, ref["data"].shape[1] - PATCH - 64))
        P = [ref["data"][y0:y0 + PATCH, x0:x0 + PATCH]]
        bad = False
        for c in chips[1:]:
            q = resample_to_ref(ref, c, y0, x0, PATCH)
            if q is None or not np.isfinite(q).all():
                bad = True
                break
            P.append(q)
        if bad or not np.isfinite(P[0]).all():
            continue
        P = [p.astype(np.float64) for p in P]
        # background mask: reject any pixel within 4 px of a >4 sigma peak in
        # any of the four exposures, so only sky-dominated pixels remain
        mask = np.ones_like(P[0], bool)
        for p in P:
            s, sg = np.median(p), 1.4826 * np.median(np.abs(p - np.median(p)))
            hot = maximum_filter((p - s) > 4 * sg, size=9)
            mask &= ~hot
        if mask.mean() < 0.15:
            continue
        A, B, C, D = P
        a, b, c_, d = A[mask], B[mask], C[mask], D[mask]
        ac, bd = a - c_, b - d
        rho_half = float(np.corrcoef(ac, bd)[0, 1])
        naive = float(np.corrcoef(a, b)[0, 1])
        vab = float(np.var(a - b))
        vac = float(np.var(ac))
        rows.append({"y0": y0, "x0": x0, "frac_bg": float(mask.mean()),
                     "rho_hat": 2 * rho_half, "corr_ACBD": rho_half,
                     "naive_AB": naive, "var_ratio": vab / vac,
                     "sd_a": float(np.std(a)), "sd_diff": float(np.std(a - b))})
    for c in chips:
        c["data"] = None
    return rows


def main():
    out = {}
    for tag, catp in (("brut", CAT_RAW), ("nettoye", CAT_CLEAN)):
        cat = {group_label(g): g for g in load_catalog(catp)}
        print(f"\n================ donnees {tag} ================")
        print(f"{'groupe':24s} {'npatch':>6s} {'%fond':>6s} {'rho_bruit':>10s} "
              f"{'[min,max]':>18s} {'corr naive A,B':>15s} {'Var(A-B)/Var(A-C)':>18s}")
        recs = []
        for gl, idxs in CASES:
            rows = analyse(cat[gl], idxs, tag)
            if not rows:
                print(f"{gl:24s}  aucun patch de fond exploitable")
                continue
            rho = np.array([r["rho_hat"] for r in rows])
            nv = np.array([r["naive_AB"] for r in rows])
            vr = np.array([r["var_ratio"] for r in rows])
            fb = np.array([r["frac_bg"] for r in rows])
            print(f"{gl:24s} {len(rows):6d} {100*fb.mean():5.0f}% {np.median(rho):10.4f} "
                  f"[{rho.min():+.4f},{rho.max():+.4f}] {np.median(nv):15.4f} "
                  f"{np.median(vr):18.3f}", flush=True)
            recs.append({"group": gl, "tag": tag, "n": len(rows),
                         "rho_med": float(np.median(rho)),
                         "rho_min": float(rho.min()), "rho_max": float(rho.max()),
                         "naive_med": float(np.median(nv)),
                         "var_ratio_med": float(np.median(vr)),
                         "frac_bg": float(fb.mean())})
        out[tag] = recs
    with open("qc_tmp/s5_noise.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nLecture : rho_bruit = correlation estimee des bruits de A et B, "
          "signal commun elimine.\n         0 = hypothese N2N respectee. "
          "Var(A-B)/Var(A-C) ~ 1 = A et B aussi independants que A et C ; "
          "~0 revelerait un doublon.")


if __name__ == "__main__":
    main()
