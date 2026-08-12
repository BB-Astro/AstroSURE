"""
Téléchargement de paires d'images Hubble (ACS/WFC) depuis MAST
pour entraîner AstroSURE en Noise2Noise.

Stratégie : partir d'une liste curatée de galaxies connues bien imagées par Hubble
(PHANGS-HST, ANGST, Heritage...), puis chercher les paires d'expositions.

Usage:
    source venv/bin/activate
    python download_mast/download_hubble_pairs.py --dry-run
    python download_mast/download_hubble_pairs.py --max-targets 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from astroquery.mast import Observations

from strip_sci import strip_to_sci

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

DOWNLOAD_DIR = Path(__file__).parent.parent / "training_data"
FILTERS = ["F435W", "F475W", "F555W", "F606W", "F625W", "F775W", "F814W"]
MIN_EXPOSURES = 3
MAX_EXPOSURES = 6

# ──────────────────────────────────────────────────────────────
# LISTE CURATÉE : galaxies bien imagées par Hubble
# Sources : PHANGS-HST, ANGST, Hubble Heritage, programmes classiques
# Toutes observées plusieurs fois en ACS/WFC avec filtres large-bande
# ──────────────────────────────────────────────────────────────

GALAXY_LIST = [
    # PHANGS-HST — galaxies spirales proches, multi-expositions ACS/WFC
    "NGC 628",   "NGC 1365",  "NGC 1559",  "NGC 1566",  "NGC 1672",
    "NGC 2442",  "NGC 2903",  "NGC 3351",  "NGC 3627",  "NGC 4254",
    "NGC 4303",  "NGC 4321",  "NGC 4535",  "NGC 4536",  "NGC 4548",
    "NGC 4569",  "NGC 4571",  "NGC 4654",  "NGC 4689",  "NGC 4826",
    "NGC 5068",  "NGC 5194",  "NGC 5248",  "NGC 6744",  "NGC 7496",
    "NGC 3344",  "NGC 3621",  "NGC 4038",  "NGC 4039",  "NGC 4571",
    # Hubble Heritage — galaxies iconiques avec beaucoup d'expositions
    "M51",       "M101",      "M81",       "M82",       "M104",
    "NGC 1300",  "NGC 4594",  "NGC 4676",  "NGC 2207",  "NGC 7293",
    # ANGST — galaxies naines proches (Nearby Galaxy Survey)
    "NGC 300",   "NGC 2403",  "NGC 247",   "NGC 253",   "NGC 55",
    "NGC 3031",  "NGC 4244",  "NGC 4258",  "NGC 4605",  "NGC 4736",
    # Autres classiques bien observés
    "NGC 891",   "NGC 1232",  "NGC 2841",  "NGC 3310",  "NGC 3982",
    "NGC 4125",  "NGC 4414",  "NGC 4449",  "NGC 4490",  "NGC 4522",
    "NGC 4526",  "NGC 4560",  "NGC 5055",  "NGC 5128",  "NGC 5195",
    "NGC 5457",  "NGC 6503",  "NGC 7331",  "NGC 7742",  "UGC 9128",
]


# ──────────────────────────────────────────────────────────────
# RECHERCHE PAR NOM DE GALAXIE
# ──────────────────────────────────────────────────────────────

def find_pairs_for_galaxy(galaxy_name):
    """
    Pour une galaxie donnée, trouve toutes les paires d'expositions ACS/WFC
    dans les filtres large-bande.
    Retourne une liste de dicts {galaxy, filter, obsids, n_exposures}.
    """
    try:
        results = Observations.query_object(
            galaxy_name,
            radius="3 arcmin",
        )
    except Exception as e:
        print(f"    Erreur requête '{galaxy_name}': {e}")
        return []

    if results is None or len(results) == 0:
        return []

    # Filtrer : ACS/WFC, imagerie science, filtres cibles
    mask = np.array([
        str(row["instrument_name"]) == "ACS/WFC" and
        str(row["dataproduct_type"]) == "image" and
        str(row["intentType"]) == "science" and
        str(row["filters"]) in FILTERS
        for row in results
    ])
    results = results[mask]

    if len(results) == 0:
        return []

    # Grouper par filtre
    pairs = []
    filters_present = set(str(r["filters"]) for r in results)

    for filt in filters_present:
        filt_mask = np.array([str(r["filters"]) == filt for r in results])
        filt_obs = results[filt_mask]

        if len(filt_obs) >= MIN_EXPOSURES:
            obsids = [str(r["obsid"]) for r in filt_obs[:MAX_EXPOSURES]]
            n_pairs = len(obsids) * (len(obsids) - 1) // 2
            pairs.append({
                "galaxy": galaxy_name,
                "filter": filt,
                "obsids": obsids,
                "n_exposures": len(obsids),
                "n_pairs": n_pairs,
                "ra": float(filt_obs["s_ra"][0]),
                "dec": float(filt_obs["s_dec"][0]),
            })

    return pairs


def find_all_pairs(galaxy_list, max_targets):
    """Parcourt la liste de galaxies et collecte les paires."""
    all_targets = []
    total_pairs = 0

    for i, galaxy in enumerate(galaxy_list):
        if len(all_targets) >= max_targets:
            break

        print(f"  [{i+1}/{len(galaxy_list)}] {galaxy}...", end=" ", flush=True)
        pairs = find_pairs_for_galaxy(galaxy)

        if pairs:
            n = sum(p["n_pairs"] for p in pairs)
            filters_found = [p["filter"] for p in pairs]
            print(f"OK — {len(pairs)} filtre(s) {filters_found}, {n} paires")
            for p in pairs:
                if len(all_targets) < max_targets:
                    all_targets.append(p)
                    total_pairs += p["n_pairs"]
        else:
            print("pas de paires suffisantes")

        time.sleep(0.3)  # politesse MAST

    print(f"\nTotal : {len(all_targets)} entrées (filtre×galaxie), {total_pairs} paires N2N")
    return all_targets


# ──────────────────────────────────────────────────────────────
# TÉLÉCHARGEMENT
# ──────────────────────────────────────────────────────────────

def download_target(target, download_dir, dry_run=False):
    """Télécharge les fichiers FLC pour un groupe (galaxie × filtre).
    Structure de sortie : download_dir/NGC_628/F606W/fichier_flc.fits
    """
    galaxy_slug = target["galaxy"].replace(" ", "_")
    target_dir = download_dir / galaxy_slug / target["filter"]
    target_dir.mkdir(parents=True, exist_ok=True)

    # Skip si ce dossier contient déjà des FLC
    existing = list(target_dir.glob("*_flc.fits"))
    if existing:
        print(f"      → {len(existing)} fichier(s) déjà présents — ignoré")
        return 0

    downloaded = 0
    for obsid in target["obsids"]:
        try:
            products = Observations.get_product_list(obsid)

            for subgroup in ["FLC", "FLT"]:
                flc = Observations.filter_products(
                    products,
                    productSubGroupDescription=subgroup,
                    extension="fits",
                )
                if len(flc) > 0:
                    break

            if len(flc) == 0:
                print(f"      ⚠ Pas de FLC/FLT pour obsid {obsid}")
                continue

            if dry_run:
                print(f"      [DRY-RUN] {len(flc)} fichier(s) — obsid {obsid}")
            else:
                # Télécharger dans un tmp puis aplatir vers target_dir
                tmp_dir = target_dir / "_tmp"
                Observations.download_products(
                    flc,
                    download_dir=str(tmp_dir),
                    cache=True,
                )
                # Déplacer les FLC directement dans target_dir + strip SCI
                import shutil
                for fits_file in tmp_dir.rglob("*_flc.fits"):
                    dest = target_dir / fits_file.name
                    if not dest.exists():
                        fits_file.rename(dest)
                        strip_to_sci(dest)  # ne garder que SCI
                # Nettoyer le tmp
                shutil.rmtree(tmp_dir, ignore_errors=True)

            downloaded += len(flc)
            time.sleep(0.5)

        except Exception as e:
            print(f"      ✗ Erreur obsid {obsid}: {e}")

    return downloaded


def download_all(targets, download_dir=DOWNLOAD_DIR, dry_run=False):
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = download_dir / "targets_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(targets, f, indent=2)
    print(f"Catalogue : {catalog_path}\n")

    total_files = 0
    for i, t in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {t['galaxy']} {t['filter']} "
              f"({t['n_exposures']} exp, {t['n_pairs']} paires)")
        n = download_target(t, download_dir, dry_run=dry_run)
        total_files += n
        print(f"  → {n} fichier(s) {'(simulé)' if dry_run else 'téléchargé(s)'}")

    size_gb = total_files * 170 / 1024
    print(f"\n{'='*50}")
    print(f"Total : {total_files} fichiers FLC (~{size_gb:.1f} GB)")
    print(f"Dossier : {download_dir}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-targets", type=int, default=10)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output-dir", type=Path, default=DOWNLOAD_DIR)
    p.add_argument("--yes", action="store_true", help="Ne pas demander confirmation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dry_run:
        print("=== MODE DRY-RUN ===\n")

    print(f"Recherche de paires dans {len(GALAXY_LIST)} galaxies cibles...\n")
    targets = find_all_pairs(GALAXY_LIST, max_targets=args.max_targets)

    if not targets:
        print("Aucune cible trouvée.")
        exit(1)

    print("\nRésumé des cibles :")
    total_pairs = 0
    for t in targets:
        print(f"  {t['galaxy']:15s} {t['filter']}  {t['n_exposures']} exp  {t['n_pairs']} paires")
        total_pairs += t["n_pairs"]
    print(f"\n  → {total_pairs} paires N2N au total")

    if not args.dry_run and not args.yes:
        confirm = input(f"\nLancer le téléchargement ? [y/N] ")
        if confirm.lower() != "y":
            print("Annulé.")
            exit(0)

    download_all(targets, download_dir=args.output_dir, dry_run=args.dry_run)
