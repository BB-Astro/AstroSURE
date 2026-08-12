"""Batch driver: AstroSURE v2 chain on the Arp SNAP series (program 15446).

For each ~/DocM2max/Astro/ArpNNN_MAST/ target directory, the canonical
sequence (PLAN_ENTRAINEMENT.md "ORDRE CANONIQUE", validated by BB on
10 Aug 2026):
  1. deepcr_flc.py   : CR mask + masked-median repair, partner-flux
     clump veto protecting star cores            -> *_flc_dcr.fits
  2. destripe_flc.py : StripeField engine, theta=0 in detector frame
                                                 -> *_flc_dcr_dsf.fits
  3. asure2_flc.py   : 2-channel N2N denoise     -> *_..._asure9e20b.fits
  4. drizzle of the denoised pair AND of the non-denoised pair (noise
     donor for the blend) onto the combined DRC grid
  5. blend BLEND*denoised + (1-BLEND)*donor
       -> ArpNNN_dsf_blend075_drizzled.fits      (produit final)
  6. drizzle of the RAW pair (same engine/grid, classic reference)
       -> ArpNNN_raw_drizzled.fits
The official MAST DRC stays in mastDownload/ as the classic-chain reference.

Only hst_*_flc.fits are used (short-rootname duplicates in Arp070_MAST have
a different WCS, documented pitfall).

Usage: python pipeline/batch_arp.py [ArpNNN ...]   (default: all Arp*_MAST)
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

PY = sys.executable
HERE = Path(__file__).resolve().parent
ASTRO = Path.home() / "DocM2max" / "Astro"
# Output suffixes identify the chain: _dcr = deepCR pre-clean, _dsf =
# StripeField destripe, _asure9e20b = run-9 epoch 20 + masked-median DQ fill.
DCR_SUFFIX = "_dcr"
DSF_SUFFIX = "_dsf"
SUFFIX = "_asure9e20b"
# Denoise strength chosen by BB (10 Aug): noise reduced ~3.4x.
BLEND = 0.75


def run(cmd):
    r = subprocess.run([PY] + [str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"failed: {cmd[0]}")
    return r.stdout


def process_target(tdir: Path):
    name = tdir.name.replace("_MAST", "")
    flc = sorted(p for p in tdir.glob("mastDownload/HST/*/hst_*_flc.fits")
                 if "_asure" not in p.name)
    drc = sorted(tdir.glob("mastDownload/HST/*/hst_15446_*f606w_*_drc.fits"))
    drc = [d for d in drc if "skycell" not in d.name]
    if len(flc) != 2 or not drc:
        print(f"{name}: SKIP (flc={len(flc)}, drc={len(drc)})", flush=True)
        return False
    ref = drc[0]
    print(f"{name}: {flc[0].name} + {flc[1].name} -> grid {ref.name}", flush=True)

    run([HERE / "deepcr_flc.py", "--suffix", DCR_SUFFIX, *flc])
    dcr = [p.with_name(p.stem + DCR_SUFFIX + ".fits") for p in flc]

    run([HERE / "destripe_flc.py", "--suffix", DSF_SUFFIX, *dcr])
    dsf = [p.with_name(p.stem + DSF_SUFFIX + ".fits") for p in dcr]

    run([HERE / "asure2_flc.py", "--suffix", SUFFIX, *dsf])
    asure = [p.with_name(p.stem + SUFFIX + ".fits") for p in dsf]

    den_out = tdir / f"{name}{DCR_SUFFIX}{DSF_SUFFIX}{SUFFIX}_drizzled.fits"
    run([HERE / "drizzle_asure.py", ref, den_out, *asure])
    donor_out = tdir / f"{name}{DCR_SUFFIX}{DSF_SUFFIX}_drizzled.fits"
    run([HERE / "drizzle_asure.py", ref, donor_out, *dsf])

    den_hdul = fits.open(den_out)
    donor = fits.getdata(donor_out).astype(np.float32)
    blend = BLEND * den_hdul[0].data.astype(np.float32) + (1 - BLEND) * donor
    hdr = den_hdul[0].header.copy()
    hdr["HISTORY"] = (f"blend {BLEND}*asure + {1-BLEND:.2f}*dcr_dsf "
                      f"(denoise strength)")
    blend_out = tdir / f"{name}{DSF_SUFFIX}_blend{int(BLEND*100):03d}_drizzled.fits"
    fits.writeto(blend_out, blend.astype(np.float32), hdr, overwrite=True)
    den_hdul.close()
    print(f"  ecrit: {blend_out.name}", flush=True)

    raw_out = tdir / f"{name}_raw_drizzled.fits"
    if not raw_out.exists():
        run([HERE / "drizzle_asure.py", ref, raw_out, *flc])
        print(f"  ecrit: {name}_raw_drizzled.fits", flush=True)
    return True


def main():
    if len(sys.argv) > 1:
        dirs = [ASTRO / f"{a}_MAST" for a in sys.argv[1:]]
    else:
        dirs = sorted(ASTRO.glob("Arp*_MAST"))
    done = 0
    for tdir in dirs:
        try:
            done += bool(process_target(tdir))
        except Exception as e:
            print(f"{tdir.name}: ERROR {e}", flush=True)
    print(f"batch done: {done}/{len(dirs)} targets", flush=True)
    if done < len(dirs):
        # A skipped or failed target must not look like success to a
        # caller that only checks the exit code.
        sys.exit(1)


if __name__ == "__main__":
    main()
