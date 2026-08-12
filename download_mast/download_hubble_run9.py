"""
download_hubble_run9.py — Extension ciblee GALAXIES du jeu AstroSURE (run 9).

Suite de download_hubble_extra.py. Meme mecanique (sondage d'en-tetes par
requete HTTP Range, deduplication avant telechargement, budget strict,
strip_sci -> split_chips), mais une selection differente :

  - CHAMPS DE GALAXIES uniquement (spirale / vue par la tranche / elliptique
    de fusion / paire en superposition / spirales d'amas). Le run 7 a montre
    qu'elargir le jeu avec des champs stellaires denses degrade la QC en
    champ de galaxie ; l'extension precedente en avait ajoute cinq
    (M33, IC 10, NGC 1569, M81, NGC 4258). Aucun champ dense ici.
  - GROUPES DE >= 4 EXPOSITIONS au meme pointage ET au meme roulis, ce que
    le dataset run 8 (dataset_n2n_stack.py) exige pour former une unite
    (2 entrees + >= 1 cible). Les groupes retenus font 8 a 10 poses : la
    cible empilee y est sqrt(M) fois plus calme.
  - EXPTIME individuel dans [300, 1300] s.
  - Fenetre temporelle la plus serree possible dans chaque groupe (la
    plupart des groupes tiennent en 0 a 4 jours) : moins de derive
    astrometrique inter-visite et moins de variabilite des sources.

Les champs ont ete choisis apres un recensement MAST complet de tout
ACS/WFC F606W/F814W (57 790 observations), croise avec une liste de
galaxies proches structurees, puis sonde par en-tetes.

Sortie : training_data/<GALAXIE>/<FILTRE>/*_chip[12].fits et le catalogue
des NOUVEAUX groupes training_data/pairs_catalog_run9_new.json. Aucun
fichier ni catalogue existant n'est touche.

Usage :
    PY=/opt/homebrew/Caskroom/miniconda/base/envs/dip/bin/python
    $PY download_mast/download_hubble_run9.py --dry-run
    $PY download_mast/download_hubble_run9.py --yes --budget-gb 19.5
"""

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import requests
from astroquery.mast import Observations

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_hubble_extra import (MAST_URL, MAX_POSTARG_ARCSEC,  # noqa: E402
                                   MAX_ROTATION_DEG, POINTING_TOL_ARCSEC,
                                   probe_all, root8)
from split_chips import split_file  # noqa: E402
from strip_sci import strip_to_sci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = ROOT / "training_data"
CATALOG_PATH = TRAINING_DATA / "pairs_catalog_run9_new.json"
CACHE_PATH = Path(__file__).resolve().parent / "header_cache.json"

EXPTIME_MIN, EXPTIME_MAX = 300.0, 1300.0
ROLL_MATCH_DEG = 0.02  # tolerance pour retrouver le groupe vise par son roulis

# ──────────────────────────────────────────────────────────────
# SELECTION
# (dossier, proposal, filtre, regex sur target_name, roulis du groupe,
#  nombre de poses a prendre, note)
# Le roulis identifie le groupe de facon unique : c'est la seule cle stable
# entre deux executions (les listes de fichiers MAST peuvent evoluer).
# ──────────────────────────────────────────────────────────────

FIELDS = [
    ("NGC_1316", "13691", "F814W", r"^NGC-?1316", -52.135, 10,
     "Fornax A : elliptique de fusion, coquilles et voies de poussiere"),
    ("NGC_1448", "13691", "F814W", r"^NGC-?1448", 43.630, 10,
     "spirale vue par la tranche"),
    ("NGC_4921", "10842", "F606W", r"^NGC4921", -119.548, 10,
     "grande spirale de Coma, voies de poussiere"),
    ("NGC_4921", "10842", "F814W", r"^NGC4921", -119.560, 10,
     "grande spirale de Coma, voies de poussiere"),
    ("NGC_4911", "10842", "F606W", r"^NGC4911", -117.399, 10,
     "spirale de Coma"),
    ("NGC_3314", "9445",  "F606W", r"^NGC3314", 88.300, 10,
     "paire de spirales en superposition sur la ligne de visee"),
    ("NGC_4639", "17079", "F814W", r"^NGC-?4639", 51.378, 10,
     "spirale de la Vierge (meme programme que NGC 1559)"),
    ("NGC_4639", "17079", "F606W", r"^NGC-?4639", 51.374, 8,
     "spirale de la Vierge (meme programme que NGC 1559)"),
    ("NGC_5194", "14704", "F814W", r"^M-?51", -156.944, 8,
     "M51 : spirale grand design, poussiere et regions HII"),
    ("NGC_5194", "14704", "F606W", r"^M-?51", -156.945, 8,
     "M51 : spirale grand design, poussiere et regions HII"),
    ("NGC_4486", "10543", "F814W", r"^M87", -105.331, 8,
     "M87 : elliptique geante, jet et amas globulaires"),
    ("NGC_3370", "9351",  "F814W", r"^NGC3370", 69.913, 8,
     "spirale hote de SN 1994ae"),
]


# ──────────────────────────────────────────────────────────────
# ETAPE 1 : candidats FLC (un seul variant de nom par exposition)
# ──────────────────────────────────────────────────────────────

def flc_candidates(proposal, filt, targ_re):
    """{root8: {'file', 'long', 'size'}} pour un programme/filtre/cible."""
    obs = Observations.query_criteria(
        obs_collection="HST", proposal_id=proposal, instrument_name="ACS/WFC",
        filters=filt, dataproduct_type="image", intentType="science")
    rx = re.compile(targ_re, re.I)
    keep = [o for o in obs
            if rx.match(str(o["target_name"])) and len(str(o["obs_id"])) == 9]
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
# ETAPE 2 : groupes meme pointage + meme roulis
# ──────────────────────────────────────────────────────────────

def group_by_pointing_roll(records):
    """Groupes ou TOUTE paire partage pointage (< 10") et roulis (< 0.1 deg).

    EXPTIME individuel dans [300, 1300] s, POSTARG <= 10", plein format WFC.
    """
    usable = [r for r in records
              if r["naxis1"] == 4096 and r["naxis2"] == 2048
              and EXPTIME_MIN <= r["exptime"] <= EXPTIME_MAX
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
        groups.append(group)
    groups.sort(key=len, reverse=True)
    return groups


def download_flc_resume(filename, dest, tries=8):
    """Telechargement repris par requete HTTP Range.

    MAST coupe regulierement un transfert de 168 Mo en cours de route
    (ChunkedEncodingError / IncompleteRead). Reprendre a zero gaspille tout
    l'octet deja recu : sur ce run les coupures arrivaient vers 50-150 Mo,
    soit presque autant de perdu que de recupere. On garde donc le .part et
    on redemande la suite avec Range: bytes=<recu>-. Le serveur fait
    autorite sur la taille totale (Content-Length ou Content-Range), et le
    fichier n'est promu en .fits que complet : strip_sci ne verra jamais un
    FITS tronque.

    Retourne le nombre d'octets reellement transferes (reprises comprises),
    pour que la comptabilite du budget reste honnete.
    """
    tmp = dest.with_suffix(".part")
    url = MAST_URL.format(filename)
    added = 0
    total = None
    last_err = None

    for k in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=(30, 300)) as r:
                if have and r.status_code == 200:
                    # Le serveur ignore Range : on repart proprement de zero.
                    tmp.unlink(missing_ok=True)
                    have = 0
                r.raise_for_status()
                if r.status_code == 206:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr:
                        total = int(cr.rsplit("/", 1)[1])
                elif "Content-Length" in r.headers:
                    total = int(r.headers["Content-Length"])
                with open(tmp, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        added += len(chunk)
        except Exception as e:
            last_err = e
            got = tmp.stat().st_size if tmp.exists() else 0
            print(f"    reprise {k + 1}/{tries - 1} depuis {got / 1e6:.0f} Mo "
                  f"({type(e).__name__})", flush=True)
            time.sleep(3 * (k + 1))
            continue
        if total is not None and tmp.stat().st_size >= total:
            break

    size = tmp.stat().st_size if tmp.exists() else 0
    if total is None or size < total:
        raise IOError(f"{filename}: {size}/{total} octets apres {tries} "
                      f"tentatives ({last_err})")
    tmp.replace(dest)
    return added


def ordinal(date_obs):
    y, m, d = (int(v) for v in date_obs.split("-"))
    return date(y, m, d).toordinal()


def tightest_window(group, n_take):
    """Les n_take poses les plus rapprochees dans le temps."""
    g = sorted(group, key=lambda r: (r["date_obs"], r["root"]))
    if len(g) <= n_take:
        return g
    best, best_span = None, None
    for i in range(len(g) - n_take + 1):
        w = g[i:i + n_take]
        span = ordinal(w[-1]["date_obs"]) - ordinal(w[0]["date_obs"])
        if best_span is None or span < best_span:
            best, best_span = w, span
    return best


# ──────────────────────────────────────────────────────────────
# ETAPE 3 : catalogue
# ──────────────────────────────────────────────────────────────

def build_catalog(selected):
    """selected : [(galaxy_dir, filtre, note, records, chips)] -> catalogue."""
    catalog = []
    for galaxy_dir, filt, note, recs, chips in selected:
        for chip in (1, 2):
            files = [chips[r["root"]][chip - 1] for r in recs
                     if r["root"] in chips and chips[r["root"]][chip - 1]]
            if len(files) < 2:
                continue
            n = len(files)
            catalog.append({
                "galaxy": galaxy_dir.replace("_", " "),
                "filter": filt,
                "ra_targ": round(float(np.mean([r["ra_targ"] for r in recs])), 5),
                "dec_targ": round(float(np.mean([r["dec_targ"] for r in recs])), 5),
                "n_exposures": n,
                "n_pairs": n * (n - 1) // 2,
                "proposals": sorted({str(r["proposid"]) for r in recs}),
                "date_range": f"{min(r['date_obs'] for r in recs)} → "
                              f"{max(r['date_obs'] for r in recs)}",
                "chip": chip,
                "files": [str(f) for f in files],
                "note": note,
            })
    catalog.sort(key=lambda x: -x["n_pairs"])
    return catalog


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-gb", type=float, default=19.5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--output-dir", type=Path, default=TRAINING_DATA)
    p.add_argument("--fields", default="",
                   help="Indices de FIELDS a traiter, separes par des virgules. "
                        "Permet de lancer plusieurs instances sur des "
                        "sous-ensembles DISJOINTS (aucun fichier commun, donc "
                        "aucune course sur le meme .part). Defaut : tous.")
    p.add_argument("--catalog", type=Path, default=CATALOG_PATH,
                   help="Chemin du catalogue ecrit en sortie.")
    return p.parse_args()


def main():
    args = parse_args()
    budget = args.budget_gb * 1e9
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    probed_total = 0

    print(f"=== AstroSURE run 9 — champs de galaxies — budget "
          f"{args.budget_gb:.2f} Go ===\n")

    fields = FIELDS
    if args.fields:
        want = [int(i) for i in args.fields.split(",")]
        fields = [FIELDS[i] for i in want]
        print(f"Sous-ensemble : champs {want}\n")

    plan = []
    for galaxy_dir, proposal, filt, targ_re, roll, n_take, note in fields:
        print(f"[{galaxy_dir} {filt}] GO-{proposal} roulis {roll:+.3f} deg")
        cands = flc_candidates(proposal, filt, targ_re)
        records, nbytes = probe_all(cands, cache)
        probed_total += nbytes
        groups = group_by_pointing_roll(records)
        match = [g for g in groups
                 if abs(g[0]["cd_angle"] - roll) < ROLL_MATCH_DEG and len(g) >= 4]
        if not match:
            print("    ✗ groupe introuvable au roulis vise — champ ignore")
            continue
        if len(match) > 1:
            print(f"    ✗ {len(match)} groupes au meme roulis — ambigu, ignore")
            continue
        group = match[0]
        chosen = tightest_window(group, n_take)
        if len(chosen) < 4:
            print(f"    ✗ seulement {len(chosen)} poses — ignore")
            continue
        if not all(r["file"].startswith("hst_") for r in chosen):
            # Piege MAST n°1 : les deux variants de nom n'ont pas la meme WCS.
            print("    ✗ variants de nom melanges dans le groupe — ignore")
            continue
        span = (ordinal(max(r["date_obs"] for r in chosen))
                - ordinal(min(r["date_obs"] for r in chosen)))
        exps = sorted({round(r["exptime"]) for r in chosen})
        print(f"    groupe {len(group)} poses -> {len(chosen)} prises, "
              f"etalement {span} j, EXPTIME {exps}, "
              f"{sum(r['size'] for r in chosen) / 1e9:.2f} Go")
        plan.append((galaxy_dir, filt, note, chosen))
        time.sleep(0.2)

    CACHE_PATH.write_text(json.dumps(cache))
    if probed_total:
        print(f"\nSondage des en-tetes : {probed_total / 1e6:.1f} Mo")

    total_bytes = sum(r["size"] for _, _, _, recs in plan for r in recs)
    total_files = sum(len(recs) for _, _, _, recs in plan)
    print(f"\n{'=' * 62}")
    print(f"Plan : {total_files} FLC, {total_bytes / 1e9:.2f} Go, "
          f"{len(plan)} groupes (x2 chips = {2 * len(plan)} unites)")
    for galaxy_dir, filt, _, recs in plan:
        print(f"  {galaxy_dir:10s} {filt}  {len(recs):2d} poses  "
              f"{sum(r['size'] for r in recs) / 1e9:.2f} Go")

    if total_bytes > budget:
        print(f"\n✗ Plan hors budget ({total_bytes / 1e9:.2f} > "
              f"{args.budget_gb:.2f} Go). Reduire FIELDS.")
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] Rien telecharge.")
        return 0
    if not args.yes and input("\nLancer le telechargement ? [y/N] ").lower() != "y":
        print("Annule.")
        return 0

    downloaded = 0
    selected = []
    for galaxy_dir, filt, note, recs in plan:
        target_dir = args.output_dir / galaxy_dir / filt
        target_dir.mkdir(parents=True, exist_ok=True)
        chips, kept = {}, []
        for rec in recs:
            stem = rec["file"].replace("_flc.fits", "")
            c1 = target_dir / f"{stem}_chip1.fits"
            c2 = target_dir / f"{stem}_chip2.fits"
            if c1.exists() and c2.exists():
                chips[rec["root"]] = (c1, c2)
                kept.append(rec)
                continue
            if downloaded + rec["size"] > budget:
                print(f"  ⚠ budget atteint ({downloaded / 1e9:.2f} Go) — arret")
                break
            dest = target_dir / rec["file"]
            t0 = time.time()
            try:
                nbytes = download_flc_resume(rec["file"], dest)
            except Exception as e:
                # Un champ perdu ne doit pas emporter les suivants.
                print(f"    ✗ telechargement abandonne {rec['file']}: {e}",
                      flush=True)
                continue
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
        if len(kept) >= 4:
            selected.append((galaxy_dir, filt, note, kept, chips))
        elif kept:
            print(f"  ⚠ {galaxy_dir} {filt} : {len(kept)} poses seulement, "
                  f"groupe non catalogue")

    catalog = build_catalog(selected)
    args.catalog.write_text(json.dumps(catalog, indent=2))

    print(f"\n{'=' * 62}")
    print(f"Telecharge : {downloaded / 1e9:.2f} Go "
          f"(+ {probed_total / 1e6:.1f} Mo de sondage)")
    print(f"Catalogue  : {args.catalog}")
    print(f"             {len(catalog)} groupes, "
          f"{sum(g['n_pairs'] for g in catalog)} paires N2N")
    return 0


if __name__ == "__main__":
    sys.exit(main())
