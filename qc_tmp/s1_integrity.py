"""Section 1: file integrity of the 116 new raw chips and their 116 cleaned twins."""

import json
import sys
from collections import defaultdict

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from qc_common import CAT_RAW, CAT_CLEAN, load_catalog, group_label, clean_wcs_header


def check(path, expect_clean):
    rec = {"path": path, "ok": True, "problems": []}
    try:
        with fits.open(path, memmap=False) as hdul:
            try:
                sci = hdul["SCI"]
            except KeyError:
                rec["ok"] = False
                rec["problems"].append("no SCI extension")
                return rec
            rec["extver"] = sci.header.get("EXTVER")
            rec["nhdu"] = len(hdul)
            d = sci.data
            rec["shape"] = tuple(d.shape)
            rec["dtype"] = str(d.dtype)
            if d.shape != (2048, 4096):
                rec["ok"] = False
                rec["problems"].append(f"shape {d.shape} != (2048,4096)")
            if d.dtype.itemsize != 4 or d.dtype.kind != "f":
                rec["ok"] = False
                rec["problems"].append(f"dtype {d.dtype} not 4-byte float")
            a = np.asarray(d, dtype=np.float32)
            n_nan = int(np.isnan(a).sum())
            n_inf = int(np.isinf(a).sum())
            rec["n_nan"] = n_nan
            rec["n_inf"] = n_inf
            if n_nan or n_inf:
                rec["ok"] = False
                rec["problems"].append(f"{n_nan} NaN / {n_inf} Inf")
            fin = a[np.isfinite(a)]
            rec["min"] = float(fin.min())
            rec["max"] = float(fin.max())
            rec["med"] = float(np.median(a[::8, ::8]))
            exptime = hdul[0].header.get("EXPTIME", None)
            rec["exptime"] = exptime
            if exptime is None:
                rec["ok"] = False
                rec["problems"].append("EXPTIME missing in PRIMARY")
            elif not (exptime > 0):
                rec["ok"] = False
                rec["problems"].append(f"EXPTIME={exptime} not > 0")
            rec["filter1"] = hdul[0].header.get("FILTER1")
            rec["filter2"] = hdul[0].header.get("FILTER2")
            rec["rootname"] = hdul[0].header.get("ROOTNAME")
            rec["dateobs"] = hdul[0].header.get("DATE-OBS")
            rec["postarg1"] = hdul[0].header.get("POSTARG1")
            rec["postarg2"] = hdul[0].header.get("POSTARG2")
            rec["pa_v3"] = hdul[0].header.get("PA_V3")
            rec["ncrpix"] = sci.header.get("NCRPIX")
            rec["deepcr"] = sci.header.get("DEEPCR")
            rec["crthresh"] = sci.header.get("CRTHRESH")
            for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRVAL1", "CRVAL2",
                      "CRPIX1", "CRPIX2"):
                if k not in sci.header:
                    rec["ok"] = False
                    rec["problems"].append(f"WCS keyword {k} missing")
            if expect_clean:
                if rec["ncrpix"] is None:
                    rec["ok"] = False
                    rec["problems"].append("NCRPIX missing in SCI header")
                if not rec["deepcr"]:
                    rec["ok"] = False
                    rec["problems"].append("DEEPCR flag missing/false")
            else:
                if rec["ncrpix"] is not None:
                    rec["problems"].append("NCRPIX present in a RAW file (unexpected)")
            # WCS constructible?
            try:
                from astropy.wcs import WCS
                w = WCS(clean_wcs_header(sci.header))
                _ = w.all_pix2world([[10, 10]], 0)
                rec["wcs_ok"] = True
            except Exception as e:
                rec["wcs_ok"] = False
                rec["ok"] = False
                rec["problems"].append(f"WCS failed: {e}")
    except Exception as e:
        rec["ok"] = False
        rec["problems"].append(f"open failed: {type(e).__name__}: {e}")
    return rec


def main():
    out = {}
    for tag, cat_path, expect_clean in (("raw", CAT_RAW, False),
                                        ("clean", CAT_CLEAN, True)):
        cat = load_catalog(cat_path)
        recs = []
        for g in cat:
            for f in g["files"]:
                r = check(f, expect_clean)
                r["group"] = group_label(g)
                r["galaxy"] = g["galaxy"]
                r["cat_filter"] = g["filter"]
                recs.append(r)
                if not r["ok"]:
                    print(f"  !! {tag} {r['path']}: {r['problems']}", flush=True)
        out[tag] = recs
        nbad = sum(1 for r in recs if not r["ok"])
        print(f"{tag}: {len(recs)} files, {nbad} with problems", flush=True)

    with open("qc_tmp/s1_integrity.json", "w") as f:
        json.dump(out, f, indent=1)

    # summary per galaxy
    print("\n--- per field ---")
    for tag in ("raw", "clean"):
        agg = defaultdict(list)
        for r in out[tag]:
            agg[r["galaxy"]].append(r)
        for gal, rs in agg.items():
            exps = sorted({r["exptime"] for r in rs})
            print(f"{tag:5s} {gal:10s} n={len(rs):3d} exptime={exps} "
                  f"nan={sum(r.get('n_nan',0) for r in rs)} "
                  f"inf={sum(r.get('n_inf',0) for r in rs)} "
                  f"minmax=[{min(r['min'] for r in rs):.1f},{max(r['max'] for r in rs):.1f}]")


if __name__ == "__main__":
    main()
