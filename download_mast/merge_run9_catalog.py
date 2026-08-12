"""
merge_run9_catalog.py — Assemble le catalogue d'entrainement du run 9.

  pairs_catalog_run9.json = pairs_catalog_full_raw.json (inchange)
                          + les unites galaxies validees par la QC

et complete training_data/wcs_offsets.json avec les decalages de zero point
WCS mesures sur les nouveaux chips (sauvegarde prealable en .bak_run9).

La QC (qc_tmp/s7_align_run9.py) ne rend pas un simple verdict oui/non par
groupe : quand un groupe porte un gradient de champ entre VISITES, elle le
decoupe par visite (chaque visite HST est rigide en interne) et rend une
unite par visite exploitable. Le catalogue est donc construit a partir des
"subgroups" de la QC, pas des groupes bruts.

Les poses ecartees et les groupes exclus partent dans
training_data/_excluded/<raison>/. Rien n'est jamais supprime.

Usage :
    PY=/opt/homebrew/Caskroom/miniconda/base/envs/dip/bin/python
    $PY download_mast/merge_run9_catalog.py --dry-run
    $PY download_mast/merge_run9_catalog.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
TD = ROOT / "training_data"
CAT_FULL = TD / "pairs_catalog_full_raw.json"
CAT_NEW = TD / "pairs_catalog_run9_new.json"
CAT_OUT = TD / "pairs_catalog_run9.json"
OFFSETS = TD / "wcs_offsets.json"
OFFSETS_BAK = TD / "wcs_offsets.json.bak_run9"
QC = ROOT / "qc_tmp" / "s7_align_run9.json"
EXCLUDED = TD / "_excluded"

MIN_GROUP = 4


def slug_for(reason):
    if "gradient" in reason:
        return "gradient_wcs"
    if "roulis" in reason:
        return "roulis_relatif"
    if "mesure" in reason:
        return "mesure_wcs_impossible"
    return "groupe_trop_petit"


def date_obs(path):
    with fits.open(path, memmap=False) as hdul:
        return str(hdul[0].header.get("DATE-OBS", "")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    full = json.loads(CAT_FULL.read_text())
    new = json.loads(CAT_NEW.read_text())
    qc = json.loads(QC.read_text())

    meta = {f"{g['galaxy']}/{g['filter']}/chip{g['chip']}": g for g in new}

    keep = []
    for sg in qc["subgroups"]:
        base = meta[f"{sg['galaxy']}/{sg['filter']}/chip{sg['chip']}"]
        files = sg["files"]
        n = len(files)
        if n < MIN_GROUP:
            continue
        dates = sorted(date_obs(f) for f in files)
        keep.append({
            "galaxy": base["galaxy"],
            "filter": base["filter"],
            "ra_targ": base["ra_targ"],
            "dec_targ": base["dec_targ"],
            "n_exposures": n,
            "n_pairs": n * (n - 1) // 2,
            "proposals": base["proposals"],
            "date_range": f"{dates[0]} → {dates[-1]}",
            "chip": base["chip"],
            "files": files,
            "note": base.get("note", ""),
            "scope": sg["scope"],
        })

    catalogued = {Path(f).name for g in keep for f in g["files"]}
    moved = []
    for gl, f, why in qc["dropped_exposures"]:
        if f in catalogued:
            continue
        moved.append((f, slug_for(why), why))
    for gl, why in qc["excluded"]:
        base = meta.get(gl)
        if not base:
            continue
        for f in base["files"]:
            if Path(f).name not in catalogued:
                moved.append((Path(f).name, "gradient_wcs", why))

    # locate the files to move, one entry per file (a chip can be listed both
    # as a dropped exposure and as part of an excluded group)
    index = {Path(f).name: Path(f) for g in new for f in g["files"]}
    seen, uniq = set(), []
    for n, slug, why in moved:
        if n in seen or n not in index:
            continue
        seen.add(n)
        uniq.append((index[n], slug, why))
    moved = uniq

    missing = [f for g in full + keep for f in g["files"] if not Path(f).exists()]
    if missing:
        print(f"✗ {len(missing)} fichiers du catalogue absents du disque :")
        for m in missing[:10]:
            print(f"    {m}")
        return 1

    merged = full + keep
    print(f"Catalogue run 9 : {len(full)} groupes existants "
          f"+ {len(keep)} nouvelles unites = {len(merged)}")
    print(f"  nouveau : {sum(g['n_exposures'] for g in keep)} chips, "
          f"{sum(g['n_pairs'] for g in keep)} paires")
    for g in sorted(keep, key=lambda g: (g["galaxy"], g["filter"], g["chip"], g["scope"])):
        print(f"    {g['galaxy']:10s} {g['filter']} chip{g['chip']} "
              f"{g['n_exposures']:2d} poses {g['n_pairs']:3d} paires  ({g['scope']})")
    if moved:
        print(f"\n  chips vers _excluded : {len(moved)}")
        for src, slug, why in moved:
            print(f"    {slug}/{src.name} : {why}")

    old_off = json.loads(OFFSETS.read_text()) if OFFSETS.exists() else {}
    add = {k: v for k, v in qc["offsets"].items() if k in catalogued}
    print(f"\nOffsets WCS : {len(old_off)} existants + {len(add)} nouveaux")

    if args.dry_run:
        print("\n[DRY-RUN] Rien ecrit.")
        return 0

    for src, slug, why in moved:
        if not src.exists():
            continue
        dest = EXCLUDED / slug
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest / src.name))

    if OFFSETS.exists() and not OFFSETS_BAK.exists():
        shutil.copy2(OFFSETS, OFFSETS_BAK)
        print(f"Sauvegarde : {OFFSETS_BAK}")
    merged_off = dict(old_off)
    merged_off.update(add)
    OFFSETS.write_text(json.dumps(merged_off, indent=1))

    CAT_OUT.write_text(json.dumps(merged, indent=2))
    print(f"\nEcrit : {CAT_OUT}")
    print(f"        {len(merged)} groupes, "
          f"{sum(g['n_exposures'] for g in merged)} chips, "
          f"{sum(g['n_pairs'] for g in merged)} paires")
    print(f"Ecrit : {OFFSETS} ({len(merged_off)} chips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
