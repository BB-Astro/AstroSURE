"""
download_hubble_extra.py — Extension ciblee du jeu d'entrainement AstroSURE.

Script frere de download_hubble_pairs.py. Meme pipeline (strip_sci -> chips),
mais trois differences qui evitent les pieges MAST payes en bande passante :

  1. Pre-validation par en-tetes : chaque FLC candidat est sonde par une
     requete HTTP Range de 64 ko (PRIMARY + SCI,1). On connait donc EXPTIME,
     RA/DEC_TARG, POSTARG et surtout l'angle de roulis reel
     (arctan2(CD1_2, CD1_1), le critere exact de dataset_n2n.py) AVANT de
     depenser 168 Mo par fichier. Le sondage est mis en cache.
  2. Deduplication AVANT telechargement : MAST expose chaque exposition sous
     deux noms (court jXXXXXXXq_flc.fits et long hst_*_flc.fits). Les deux
     font 168 Mo et n'ont PAS la meme WCS (le produit long est realigne).
     On telecharge un seul variant, le long, comme le catalogue existant.
  3. Budget d'octets strict : le telechargement s'arrete avant depassement.

Sortie : nouveaux dossiers training_data/<GALAXIE>/<FILTRE>/*_chip[12].fits
et un catalogue SEPARE training_data/pairs_catalog_new.json. Ni
pairs_catalog.json ni training_data_crclean/ ne sont touches.

Usage :
    PY=/opt/homebrew/Caskroom/miniconda/base/envs/dip/bin/python
    $PY download_mast/download_hubble_extra.py --dry-run
    $PY download_mast/download_hubble_extra.py --yes --budget-gb 9.8
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from astroquery.mast import Observations

from split_chips import split_file
from strip_sci import strip_to_sci

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = ROOT / "training_data"
CATALOG_PATH = TRAINING_DATA / "pairs_catalog_new.json"
CACHE_PATH = Path(__file__).resolve().parent / "header_cache.json"

MAST_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/{}"
PROBE_BYTES = 65536          # PRIMARY + SCI,1 tiennent dans 64 ko
MAX_POSTARG_ARCSEC = 10.0    # meme seuil que clean_flc.py
POINTING_TOL_ARCSEC = 10.0   # meme seuil que clean_flc.py
MAX_ROTATION_DEG = 0.1       # meme seuil que training/dataset_n2n.py

# ──────────────────────────────────────────────────────────────
# CHAMPS RETENUS
# Selection issue d'un recensement MAST de tout ACS/WFC F606W/F814W sur
# cibles resolues (voir le rapport). Criteres : galaxie proche resolue,
# absente du jeu existant, >= 8 expositions au meme pointage / meme roulis /
# meme EXPTIME, pose moderee (densite de rayons cosmiques raisonnable).
# ──────────────────────────────────────────────────────────────

FIELDS = [
    # dossier    proposal  cible MAST        filtre  exptime  n a prendre
    ("NGC_4258", "9810",  "NGC4258-INNER",  "F814W",  400.0, 10),
    ("NGC_598",  "10190", "M33-DISK2",      "F606W", 1240.0, 10),
    ("NGC_598",  "10190", "M33-DISK2",      "F814W", 1240.0, 10),
    ("NGC_1569", "10885", "NGC1569",        "F606W", 1223.0, 10),
    ("IC_10",    "10242", "IC10-BAR",       "F814W",  595.0, 10),
    ("NGC_3031", "10915", "M81-DEEP",       "F606W", 2708.0,  8),
]

KEYS_PRIMARY = ("ROOTNAME", "RA_TARG", "DEC_TARG", "PROPOSID", "EXPTIME",
                "POSTARG1", "POSTARG2", "FILTER1", "FILTER2", "DATE-OBS",
                "TARGNAME")
KEYS_SCI = ("CD1_1", "CD1_2", "NAXIS1", "NAXIS2")


# ──────────────────────────────────────────────────────────────
# ETAPE 1 : lister les FLC candidats (un seul nom par exposition)
# ──────────────────────────────────────────────────────────────

def root8(filename):
    """Cle de deduplication : 8 premiers caracteres du rootname.

    Le nom long hst_*_<rootname sans le dernier caractere>_flc.fits ne porte
    que 8 des 9 caracteres de l'ipppssoot, d'ou la troncature commune.
    """
    stem = filename.replace("_flc.fits", "")
    base = stem.split("_")[-1] if stem.startswith("hst_") else stem
    return base[:8].lower()


def flc_candidates(proposal, filt, target_prefix):
    """{root8: {'file': nom long de preference, 'size': octets}} pour un champ."""
    obs = Observations.query_criteria(
        obs_collection="HST", proposal_id=proposal, instrument_name="ACS/WFC",
        filters=filt, dataproduct_type="image", intentType="science")
    keep = [o for o in obs if str(o["target_name"]).startswith(target_prefix)]
    if not keep:
        return {}

    products = Observations.get_product_list(
        Observations.query_criteria(
            obs_collection="HST", proposal_id=proposal,
            instrument_name="ACS/WFC", filters=filt, dataproduct_type="image",
            intentType="science", obs_id=[str(o["obs_id"]) for o in keep]))
    flc = Observations.filter_products(
        products, productSubGroupDescription="FLC", extension="fits")

    out = {}
    for row in flc:
        name = str(row["productFilename"])
        key = root8(name)
        is_long = name.startswith("hst_")
        if key not in out or (is_long and not out[key]["long"]):
            out[key] = {"file": name, "long": is_long, "size": int(row["size"])}
    return out


# ──────────────────────────────────────────────────────────────
# ETAPE 2 : sonder les en-tetes par requete HTTP Range
# ──────────────────────────────────────────────────────────────

def parse_fits_headers(buf):
    """Extrait PRIMARY puis SCI,1 d'un buffer FITS tronque."""
    out, offset, hdu, keys = {}, 0, 0, KEYS_PRIMARY
    while offset + 2880 <= len(buf):
        block = buf[offset:offset + 2880]
        offset += 2880
        end = False
        for j in range(0, 2880, 80):
            card = block[j:j + 80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                end = True
                break
            if key in keys and "=" in card:
                out[f"h{hdu}_{key}"] = card[10:].split("/")[0].strip().strip("'").strip()
        if end:
            hdu += 1
            keys = KEYS_SCI
            if hdu > 1:
                break
    return out


def probe(filename):
    try:
        r = requests.get(MAST_URL.format(filename),
                         headers={"Range": f"bytes=0-{PROBE_BYTES - 1}"},
                         timeout=90)
        if r.status_code not in (200, 206):
            return filename, None, 0
        return filename, parse_fits_headers(r.content), len(r.content)
    except requests.RequestException as e:
        print(f"    sondage echoue {filename}: {e}", file=sys.stderr)
        return filename, None, 0


def record_from_header(filename, size, hdr):
    """Enregistrement normalise, ou None si l'en-tete est inexploitable."""
    if not hdr or "h1_CD1_1" not in hdr or "h1_CD1_2" not in hdr:
        return None
    try:
        f1 = hdr.get("h0_FILTER1", "").strip()
        f2 = hdr.get("h0_FILTER2", "").strip()
        filt = f2 if f1.startswith("CLEAR") else f1
        return {
            "file": filename,
            "size": size,
            "root": root8(filename),
            "rootname": hdr.get("h0_ROOTNAME", "").strip().lower(),
            "targname": hdr.get("h0_TARGNAME", "").strip(),
            "proposid": int(float(hdr["h0_PROPOSID"])),
            "filter": filt.strip(),
            "exptime": float(hdr["h0_EXPTIME"]),
            "ra_targ": float(hdr["h0_RA_TARG"]),
            "dec_targ": float(hdr["h0_DEC_TARG"]),
            "postarg1": float(hdr["h0_POSTARG1"]),
            "postarg2": float(hdr["h0_POSTARG2"]),
            "date_obs": hdr.get("h0_DATE-OBS", "").strip(),
            "cd_angle": float(np.degrees(np.arctan2(float(hdr["h1_CD1_2"]),
                                                    float(hdr["h1_CD1_1"])))),
            "naxis1": int(float(hdr["h1_NAXIS1"])),
            "naxis2": int(float(hdr["h1_NAXIS2"])),
        }
    except (KeyError, ValueError):
        return None


def probe_all(candidates, cache, workers=8):
    """Sonde ce qui manque au cache. Retourne (records, octets sondes)."""
    todo = [c["file"] for c in candidates.values() if c["file"] not in cache]
    probed_bytes = 0
    if todo:
        print(f"    sondage de {len(todo)} en-tetes "
              f"({len(todo) * PROBE_BYTES / 1e6:.1f} Mo)...", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for filename, hdr, nbytes in ex.map(probe, todo):
                probed_bytes += nbytes
                if hdr:
                    cache[filename] = hdr

    records = []
    for cand in candidates.values():
        hdr = cache.get(cand["file"])
        rec = record_from_header(cand["file"], cand["size"], hdr)
        if rec:
            records.append(rec)
    return records, probed_bytes


# ──────────────────────────────────────────────────────────────
# ETAPE 3 : groupes N2N valides (pointage, POSTARG, EXPTIME, roulis)
# ──────────────────────────────────────────────────────────────

def valid_groups(records, exptime):
    """Groupes ou TOUTE paire respecte pointage < 10" et roulis < 0.1 deg."""
    usable = [r for r in records
              if r["naxis1"] == 4096 and r["naxis2"] == 2048
              and abs(r["exptime"] - exptime) < 1.0
              and abs(r["postarg1"]) <= MAX_POSTARG_ARCSEC
              and abs(r["postarg2"]) <= MAX_POSTARG_ARCSEC]
    usable.sort(key=lambda r: (r["date_obs"], r["root"]))

    groups, used = [], set()
    for i, a in enumerate(usable):
        if i in used:
            continue
        group = [a]
        used.add(i)
        for j, b in enumerate(usable):
            if j in used:
                continue
            ok = True
            for c in group:
                dra = (b["ra_targ"] - c["ra_targ"]) * np.cos(np.radians(c["dec_targ"])) * 3600
                ddec = (b["dec_targ"] - c["dec_targ"]) * 3600
                if np.hypot(dra, ddec) >= POINTING_TOL_ARCSEC:
                    ok = False
                    break
                if abs(b["cd_angle"] - c["cd_angle"]) > MAX_ROTATION_DEG:
                    ok = False
                    break
            if ok:
                group.append(b)
                used.add(j)
        if len(group) >= 2:
            groups.append(group)
    groups.sort(key=len, reverse=True)
    return groups


# ──────────────────────────────────────────────────────────────
# ETAPE 4 : telechargement avec budget strict
# ──────────────────────────────────────────────────────────────

def download_flc(filename, dest):
    """Telecharge un FLC. Retourne le nombre d'octets ecrits."""
    tmp = dest.with_suffix(".part")
    written = 0
    with requests.get(MAST_URL.format(filename), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
    tmp.replace(dest)
    return written


# ──────────────────────────────────────────────────────────────
# ETAPE 5 : catalogue
# ──────────────────────────────────────────────────────────────

def build_catalog(selected):
    """selected : liste de (galaxy_dir, filtre, records, chips). -> catalogue."""
    catalog = []
    for galaxy_dir, filt, recs, chips in selected:
        for chip in (1, 2):
            files = [chips[r["root"]][chip - 1] for r in recs
                     if r["root"] in chips and chips[r["root"]][chip - 1]]
            if len(files) < 2:
                continue
            n = len(files)
            catalog.append({
                "galaxy": galaxy_dir.replace("_", " "),
                "filter": filt,
                "ra_targ": round(recs[0]["ra_targ"], 5),
                "dec_targ": round(recs[0]["dec_targ"], 5),
                "n_exposures": n,
                "n_pairs": n * (n - 1) // 2,
                "proposals": sorted(set(str(r["proposid"]) for r in recs)),
                "date_range": f"{min(r['date_obs'] for r in recs)} → "
                              f"{max(r['date_obs'] for r in recs)}",
                "chip": chip,
                "files": [str(f) for f in files],
            })
    catalog.sort(key=lambda x: -x["n_pairs"])
    return catalog


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-gb", type=float, default=9.8,
                   help="Plafond de telechargement des FLC, en Go decimaux")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="Ne pas demander confirmation")
    p.add_argument("--output-dir", type=Path, default=TRAINING_DATA)
    return p.parse_args()


def main():
    args = parse_args()
    budget = args.budget_gb * 1e9
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    probed_total = 0

    print(f"=== Extension AstroSURE — budget {args.budget_gb:.2f} Go ===\n")

    plan = []
    for galaxy_dir, proposal, target, filt, exptime, n_take in FIELDS:
        print(f"[{galaxy_dir} {filt}] GO-{proposal} {target} {exptime:.0f}s")
        candidates = flc_candidates(proposal, filt, target)
        print(f"    {len(candidates)} expositions uniques chez MAST")
        records, nbytes = probe_all(candidates, cache)
        probed_total += nbytes
        groups = valid_groups(records, exptime)
        if not groups:
            print("    aucun groupe valide — champ ignore")
            continue
        best = groups[0]
        angles = [r["cd_angle"] for r in best]
        chosen = best[:n_take]
        print(f"    groupe retenu : {len(best)} exp valides "
              f"(roulis max {max(angles) - min(angles):.4f} deg), "
              f"{len(chosen)} prises -> {len(chosen) * (len(chosen) - 1) // 2} paires/chip")
        plan.append((galaxy_dir, filt, chosen))
        time.sleep(0.2)

    CACHE_PATH.write_text(json.dumps(cache))
    if probed_total:
        print(f"\nSondage des en-tetes : {probed_total / 1e6:.1f} Mo")

    total_bytes = sum(r["size"] for _, _, recs in plan for r in recs)
    total_files = sum(len(recs) for _, _, recs in plan)
    total_pairs = sum(len(recs) * (len(recs) - 1) // 2 for _, _, recs in plan)
    print(f"\n{'=' * 58}")
    print(f"Plan : {total_files} FLC, {total_bytes / 1e9:.2f} Go, "
          f"{total_pairs} paires/chip ({total_pairs * 2} au total)")
    for galaxy_dir, filt, recs in plan:
        print(f"  {galaxy_dir:10s} {filt}  {len(recs):2d} exp  "
              f"{len(recs) * (len(recs) - 1) // 2:3d} paires/chip  "
              f"{sum(r['size'] for r in recs) / 1e9:.2f} Go")

    if total_bytes > budget:
        print(f"\n✗ Le plan depasse le budget ({total_bytes / 1e9:.2f} > "
              f"{args.budget_gb:.2f} Go). Reduire FIELDS.")
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] Rien telecharge.")
        return 0
    if not args.yes:
        if input("\nLancer le telechargement ? [y/N] ").lower() != "y":
            print("Annule.")
            return 0

    downloaded = 0
    selected = []
    for galaxy_dir, filt, recs in plan:
        target_dir = args.output_dir / galaxy_dir / filt
        target_dir.mkdir(parents=True, exist_ok=True)
        chips, kept = {}, []
        for rec in recs:
            c1 = target_dir / f"{rec['file'].replace('_flc.fits', '')}_chip1.fits"
            c2 = target_dir / f"{rec['file'].replace('_flc.fits', '')}_chip2.fits"
            if c1.exists() and c2.exists():
                chips[rec["root"]] = (c1, c2)
                kept.append(rec)
                continue
            if downloaded + rec["size"] > budget:
                print(f"  ⚠ budget atteint ({downloaded / 1e9:.2f} Go) — arret")
                break
            dest = target_dir / rec["file"]
            t0 = time.time()
            nbytes = download_flc(rec["file"], dest)
            downloaded += nbytes
            strip_to_sci(dest)
            result = split_file(dest)
            if result is None:
                print(f"    ✗ split impossible : {dest.name}")
                continue
            chips[rec["root"]] = result
            kept.append(rec)
            print(f"  [{downloaded / 1e9:5.2f}/{args.budget_gb:.2f} Go] "
                  f"{rec['file']}  {nbytes / 1e6:.0f} Mo en {time.time() - t0:.0f}s",
                  flush=True)
        if len(kept) >= 2:
            selected.append((galaxy_dir, filt, kept, chips))

    catalog = build_catalog(selected)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2))

    print(f"\n{'=' * 58}")
    print(f"Telecharge : {downloaded / 1e9:.2f} Go "
          f"(+ {probed_total / 1e6:.1f} Mo de sondage)")
    print(f"Catalogue  : {CATALOG_PATH}")
    print(f"             {len(catalog)} groupes, "
          f"{sum(g['n_pairs'] for g in catalog)} paires N2N")
    return 0


if __name__ == "__main__":
    sys.exit(main())
