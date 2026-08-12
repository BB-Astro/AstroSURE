"""
Nettoyage des fichiers FLC téléchargés depuis MAST.

Trois problèmes corrigés :
  1. Doublons — MAST livre le même fichier sous deux noms (court + long)
  2. Pointages multiples — grouper par RA_TARG + DEC_TARG à < 10 arcsec
  3. POSTARG hors limites — exclure |POSTARG| > 10 arcsec

Sortie : pairs_catalog.json avec les groupes valides et leurs paires N2N.

Usage:
    source venv/bin/activate
    python download_mast/clean_flc.py
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits

TRAINING_DATA = Path(__file__).parent.parent / "training_data"
MAX_POSTARG_ARCSEC = 10.0
MIN_GROUP_SIZE = 2        # au moins 2 fichiers pour former 1 paire
POINTING_TOL_ARCSEC = 10  # deux expositions = même pointage si < 10"


# ──────────────────────────────────────────────────────────────
# ÉTAPE 1 : collecter tous les FLC
# ──────────────────────────────────────────────────────────────

def find_all_flc(base_dir):
    flc_files = [f for f in base_dir.rglob("*_flc.fits")
                 if "_excluded" not in f.parts]
    print(f"{len(flc_files)} fichiers FLC trouvés sous {base_dir}")
    return flc_files


# ──────────────────────────────────────────────────────────────
# ÉTAPE 2 : lire les headers et extraire les métadonnées
# ──────────────────────────────────────────────────────────────

def _extract_filter(header):
    """Extrait le vrai filtre science (ignore les roues CLEAR*)."""
    for key in ["FILTER1", "FILTER2"]:
        val = (header.get(key) or "").strip()
        if val and not val.startswith("CLEAR"):
            return val
    # Fallback
    return (header.get("FILTER1") or header.get("FILTER2") or "UNKNOWN").strip()


def read_header(filepath):
    try:
        with fits.open(filepath, memmap=True) as hdul:
            h = hdul[0].header
            return {
                "path": str(filepath),
                "rootname": h.get("ROOTNAME", "").strip().lower(),
                "proposid": h.get("PROPOSID", 0),
                "filter": _extract_filter(h),
                "date_obs": h.get("DATE-OBS", ""),
                "exptime": float(h.get("EXPTIME", 0)),
                "ra_targ": float(h.get("RA_TARG", 0)),
                "dec_targ": float(h.get("DEC_TARG", 0)),
                "postarg1": float(h.get("POSTARG1", 0)),
                "postarg2": float(h.get("POSTARG2", 0)),
            }
    except Exception as e:
        print(f"  ✗ Impossible de lire {filepath.name}: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# ÉTAPE 3 : déduplication par rootname
# ──────────────────────────────────────────────────────────────

def deduplicate(headers):
    """Garde un seul fichier par rootname (préfère le nom long hst_*)."""
    by_rootname = defaultdict(list)
    for h in headers:
        by_rootname[h["rootname"]].append(h)

    kept = []
    removed = []
    for rootname, group in by_rootname.items():
        if len(group) == 1:
            kept.append(group[0])
        else:
            # Préférer le fichier avec le nom long (hst_*) — plus explicite
            long_names = [h for h in group if Path(h["path"]).name.startswith("hst_")]
            chosen = long_names[0] if long_names else group[0]
            kept.append(chosen)
            for h in group:
                if h is not chosen:
                    removed.append(h)

    print(f"  Déduplication : {len(kept)} gardés, {len(removed)} doublons supprimés")
    return kept, removed


# ──────────────────────────────────────────────────────────────
# ÉTAPE 4 : filtrer par POSTARG
# ──────────────────────────────────────────────────────────────

def filter_postarg(headers, max_arcsec=MAX_POSTARG_ARCSEC):
    kept = []
    removed = []
    for h in headers:
        if abs(h["postarg1"]) <= max_arcsec and abs(h["postarg2"]) <= max_arcsec:
            kept.append(h)
        else:
            removed.append(h)
    print(f"  Filtre POSTARG (< {max_arcsec}\"): {len(kept)} gardés, {len(removed)} exclus")
    if removed:
        for h in removed:
            print(f"    ✗ {Path(h['path']).name}  POSTARG=({h['postarg1']:.1f}\", {h['postarg2']:.1f}\")")
    return kept, removed


# ──────────────────────────────────────────────────────────────
# ÉTAPE 5 : grouper par (filtre, RA_TARG, DEC_TARG)
# ──────────────────────────────────────────────────────────────

def group_by_pointing(headers, tol_arcsec=POINTING_TOL_ARCSEC):
    """Groupe les headers par filtre + pointage identique (< tol_arcsec)."""
    groups = []
    used = set()

    for i, hi in enumerate(headers):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j, hj in enumerate(headers):
            if j in used or j == i:
                continue
            if hi["filter"] != hj["filter"]:
                continue
            dra = (hj["ra_targ"] - hi["ra_targ"]) * np.cos(np.radians(hi["dec_targ"])) * 3600
            ddec = (hj["dec_targ"] - hi["dec_targ"]) * 3600
            dist = np.sqrt(dra**2 + ddec**2)
            if dist < tol_arcsec:
                group.append(j)
                used.add(j)
        groups.append([headers[k] for k in group])

    valid = [g for g in groups if len(g) >= MIN_GROUP_SIZE]
    rejected = [g for g in groups if len(g) < MIN_GROUP_SIZE]

    print(f"  Groupes par pointage : {len(valid)} valides (≥{MIN_GROUP_SIZE} exp), "
          f"{len(rejected)} rejetés (trop peu d'expositions)")
    return valid, rejected


# ──────────────────────────────────────────────────────────────
# ÉTAPE 6 : construire le catalogue de paires
# ──────────────────────────────────────────────────────────────

def build_pairs_catalog(valid_groups):
    catalog = []
    total_pairs = 0

    for g in valid_groups:
        files = sorted(g, key=lambda h: (h["date_obs"], h["proposid"]))
        n = len(files)
        n_pairs = n * (n - 1) // 2
        total_pairs += n_pairs

        # Extraire le nom de galaxie depuis le chemin (training_data/<Galaxy>/<Filter>/...)
        galaxy = Path(files[0]["path"]).parts[-3].replace("_", " ")

        catalog.append({
            "galaxy": galaxy,
            "filter": files[0]["filter"],
            "ra_targ": round(files[0]["ra_targ"], 5),
            "dec_targ": round(files[0]["dec_targ"], 5),
            "n_exposures": n,
            "n_pairs": n_pairs,
            "proposals": sorted(set(str(f["proposid"]) for f in files)),
            "date_range": f"{min(f['date_obs'] for f in files)} → {max(f['date_obs'] for f in files)}",
            "files": [f["path"] for f in files],
        })

    catalog.sort(key=lambda x: -x["n_pairs"])
    return catalog, total_pairs


# ──────────────────────────────────────────────────────────────
# ÉTAPE 7 : supprimer les fichiers inutiles du disque
# ──────────────────────────────────────────────────────────────

EXCLUDED_DIR = TRAINING_DATA / "_excluded"

def move_files(headers, reason, label="fichiers"):
    """Déplace les fichiers exclus dans _excluded/<raison>/ au lieu de supprimer."""
    dest_dir = EXCLUDED_DIR / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for h in headers:
        src = Path(h["path"])
        if src.exists():
            dst = dest_dir / src.name
            src.rename(dst)
            moved += 1
    print(f"  Déplacé {moved} {label} → _excluded/{reason}/")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Nettoyage des FLC dans {TRAINING_DATA} ===\n")

    # 1. Trouver tous les FLC
    all_files = find_all_flc(TRAINING_DATA)
    if not all_files:
        print("Aucun fichier FLC trouvé.")
        exit(1)

    # 2. Lire les headers
    print("\nLecture des headers...")
    headers = [read_header(f) for f in all_files]
    headers = [h for h in headers if h is not None]
    print(f"  {len(headers)} headers lus")

    # 3. Déduplication
    print("\nDéduplication...")
    headers, dupes = deduplicate(headers)

    # 4. Filtre POSTARG
    print("\nFiltre POSTARG...")
    headers, bad_postarg = filter_postarg(headers)

    # 5. Groupement par pointage
    print("\nGroupement par pointage...")
    valid_groups, rejected_groups = group_by_pointing(headers)

    # 6. Catalogue
    catalog, total_pairs = build_pairs_catalog(valid_groups)

    print(f"\n{'='*55}")
    print(f"Résultat final : {sum(g['n_exposures'] for g in catalog)} fichiers utiles")
    print(f"                {len(catalog)} groupes valides")
    print(f"                {total_pairs} paires N2N disponibles")
    print(f"\nDétail des groupes :")
    for g in catalog:
        props = "+".join(g["proposals"])
        print(f"  {g['galaxy']:12s} {g['filter']:8s} {g['n_exposures']:2d} exp  "
              f"{g['n_pairs']:3d} paires  [{props}]")

    # 7. Sauvegarder le catalogue
    catalog_path = TRAINING_DATA / "pairs_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"\nCatalogue sauvegardé : {catalog_path}")

    # 8. Déplacer les fichiers exclus (jamais supprimer)
    print("\nDéplacement des fichiers exclus vers _excluded/...")
    move_files(dupes, "doublons")
    move_files(bad_postarg, "postarg_hors_limites")
    rejected_files = [h for g in rejected_groups for h in g]
    move_files(rejected_files, "groupe_trop_petit")

    # Nettoyer les dossiers vides (sauf _excluded)
    for d in sorted(TRAINING_DATA.rglob("*"), reverse=True):
        if d.is_dir() and d.name != "_excluded" and not any(d.iterdir()):
            d.rmdir()
    print("  Dossiers vides supprimés")

    print("\nTerminé.")
