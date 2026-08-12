"""Pre-clean cosmic rays from training chips with deepCR (ACS-WFC model).

Reads every unique chip referenced by pairs_catalog.json, runs deepCR
mask + inpaint on the SCI extension, and writes the cleaned chip to a
mirror tree under training_data_crclean/, preserving headers. Finally
writes pairs_catalog_crclean.json with rewritten paths.

Run with the dedicated deepCR venv (deepCR only builds on python 3.10/3.11):
    ~/.bb-astro/deepcr_venv/bin/python3 download_mast/clean_cr_deepcr.py [--file CHIP]

Rationale (2026-08-05): training Noise2Noise on CR-laden FLC frames
conflates denoising with CR rejection and sacrifices bright stars (run 5:
point sources > ~30 sigma erased). Cleaning the training data removes the
star/CR dilemma at the root and matches BB's real images, which are
DeepCR-cleaned by the same model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "training_data" / "pairs_catalog.json"
OUT_ROOT = ROOT / "training_data_crclean"
SRC_ROOT = ROOT / "training_data"


def unique_files(catalog):
    seen = []
    for group in catalog:
        for f in group["files"]:
            if f not in seen:
                seen.append(f)
    return seen


def clean_one(mdl, src_path: Path, dst_path: Path, threshold: float, n_jobs: int):
    with fits.open(src_path) as hdul:
        data = hdul["SCI"].data.astype(np.float32)
        mask, cleaned = mdl.clean(
            data, threshold=threshold, inpaint=True, segment=True,
            patch=256, n_jobs=n_jobs,
        )
        hdul["SCI"].data = cleaned.astype(np.float32)
        hdul["SCI"].header["DEEPCR"] = (True, "deepCR ACS-WFC mask+inpaint applied")
        hdul["SCI"].header["CRTHRESH"] = (threshold, "deepCR detection threshold")
        hdul["SCI"].header["NCRPIX"] = (int(mask.sum()), "cosmic-ray pixels inpainted")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(dst_path, overwrite=True)
    return int(mask.sum()), data.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(CATALOG))
    parser.add_argument("--out-catalog", default=None,
                        help="Where to write the rewritten catalog "
                             "(default: training_data_crclean/pairs_catalog_crclean.json). "
                             "Set it when --catalog is not the main catalog, "
                             "otherwise the main cleaned catalog gets overwritten.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="deepCR threshold (ACS-WFC default 0.5)")
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--file", default=None,
                        help="Process only this chip (validation mode)")
    args = parser.parse_args()

    from deepCR import deepCR
    mdl = deepCR(mask="ACS-WFC", device="CPU")

    with open(args.catalog) as f:
        catalog = json.load(f)

    files = [args.file] if args.file else unique_files(catalog)
    print(f"{len(files)} chip(s) to clean, threshold={args.threshold}")

    for i, f in enumerate(files):
        src = Path(f)
        dst = OUT_ROOT / src.relative_to(SRC_ROOT)
        if dst.exists():
            print(f"[{i+1}/{len(files)}] {src.name}: already done, skipped")
            continue
        t0 = time.time()
        ncr, npix = clean_one(mdl, src, dst, args.threshold, args.n_jobs)
        print(f"[{i+1}/{len(files)}] {src.name}: {ncr} px CR ({ncr/npix*100:.2f}%) "
              f"en {time.time()-t0:.0f}s", flush=True)

    if not args.file:
        out_catalog = []
        for group in catalog:
            g = dict(group)
            g["files"] = [str(OUT_ROOT / Path(f).relative_to(SRC_ROOT)) for f in group["files"]]
            out_catalog.append(g)
        out_path = Path(args.out_catalog) if args.out_catalog \
            else OUT_ROOT / "pairs_catalog_crclean.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out_catalog, f, indent=1)
        print(f"Catalogue écrit: {out_path}")


if __name__ == "__main__":
    main()
