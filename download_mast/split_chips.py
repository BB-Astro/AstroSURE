"""
split_chips.py — Sépare chaque FLC ACS/WFC en deux fichiers, un par chip CCD.

Un FLC ACS/WFC (après strip_sci.py) contient :
  [0] PRIMARY
  [1] SCI ver=1  (WFC chip 1, 4096×2048)
  [2] SCI ver=2  (WFC chip 2, 4096×2048)

Les deux chips pointent vers des zones de ciel différentes (~100 arcsec d'écart).
On ne peut pas les mélanger dans une paire N2N.

Ce script crée :
  hst_..._flc.fits  →  hst_..._chip1.fits  +  hst_..._chip2.fits

Puis met à jour pairs_catalog.json : chaque entrée devient 2 entrées
(une par chip), avec les chemins mis à jour.

Usage :
    python download_mast/split_chips.py            # tout training_data/
    python download_mast/split_chips.py --dry-run  # aperçu sans modifier
"""

import argparse
import json
import sys
from pathlib import Path

from astropy.io import fits


TRAINING_DATA = Path(__file__).parent.parent / "training_data"


def chip_paths(src: Path):
    """Retourne (chip1_path, chip2_path) pour un fichier source."""
    stem = src.stem.replace("_flc", "")
    return (
        src.parent / f"{stem}_chip1.fits",
        src.parent / f"{stem}_chip2.fits",
    )


def split_file(src: Path, dry_run: bool = False) -> tuple[Path, Path] | None:
    """
    Sépare src en deux fichiers chip1/chip2.
    Retourne (chip1_path, chip2_path) si succès, None si déjà fait ou erreur.
    """
    chip1, chip2 = chip_paths(src)

    # Déjà splitté
    if chip1.exists() and chip2.exists():
        return chip1, chip2

    try:
        with fits.open(src, memmap=False) as hdul:
            sci_hdus = [h for h in hdul if h.name == "SCI"]
            if len(sci_hdus) < 2:
                print(f"  ⚠ {src.name} : moins de 2 SCI — ignoré")
                return None

            primary = fits.PrimaryHDU(header=hdul[0].header)

            if not dry_run:
                fits.HDUList([primary,
                              fits.ImageHDU(data=sci_hdus[0].data,
                                            header=sci_hdus[0].header,
                                            name="SCI")]).writeto(chip1)
                fits.HDUList([primary,
                              fits.ImageHDU(data=sci_hdus[1].data,
                                            header=sci_hdus[1].header,
                                            name="SCI")]).writeto(chip2)
                src.unlink()

    except Exception as e:
        print(f"  ✗ {src.name}: {e}")
        return None

    return chip1, chip2


def update_catalog(catalog_path: Path, split_map: dict[str, tuple[str, str]]):
    """
    Met à jour pairs_catalog.json : chaque fichier original → 2 entrées chip.
    split_map : {str(original_path): (str(chip1_path), str(chip2_path))}
    """
    with open(catalog_path) as f:
        catalog = json.load(f)

    new_catalog = []
    for group in catalog:
        # Construire les listes chip1 et chip2 séparément
        chip1_files = []
        chip2_files = []
        for fpath in group["files"]:
            if fpath in split_map:
                c1, c2 = split_map[fpath]
                chip1_files.append(c1)
                chip2_files.append(c2)
            else:
                # Fichier déjà splitté ou manquant — chercher les chips
                src = Path(fpath)
                c1, c2 = chip_paths(src)
                if c1.exists():
                    chip1_files.append(str(c1))
                if c2.exists():
                    chip2_files.append(str(c2))

        base = {k: v for k, v in group.items() if k != "files"}

        if len(chip1_files) >= 2:
            n = len(chip1_files)
            new_catalog.append({**base,
                                 "chip": 1,
                                 "n_exposures": n,
                                 "n_pairs": n * (n - 1) // 2,
                                 "files": chip1_files})
        if len(chip2_files) >= 2:
            n = len(chip2_files)
            new_catalog.append({**base,
                                 "chip": 2,
                                 "n_exposures": n,
                                 "n_pairs": n * (n - 1) // 2,
                                 "files": chip2_files})

    new_catalog.sort(key=lambda x: -x["n_pairs"])

    with open(catalog_path, "w") as f:
        json.dump(new_catalog, f, indent=2)

    total_pairs = sum(g["n_pairs"] for g in new_catalog)
    print(f"  pairs_catalog.json mis à jour : {len(new_catalog)} groupes, "
          f"{total_pairs} paires N2N")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Trouver tous les FLC non encore splittés
    flc_files = [f for f in TRAINING_DATA.rglob("*_flc.fits")
                 if "_excluded" not in f.parts]

    if not flc_files:
        print("Aucun fichier *_flc.fits à traiter (déjà tous splittés ?).")
        sys.exit(0)

    print(f"{len(flc_files)} fichiers FLC à séparer en chip1/chip2\n")

    split_map = {}
    ok = 0
    for src in sorted(flc_files):
        c1, c2 = chip_paths(src)
        size = src.stat().st_size / 1e6
        if args.dry_run:
            print(f"  [DRY] {src.name} → {c1.name} + {c2.name}  ({size:.0f} MB)")
            split_map[str(src)] = (str(c1), str(c2))
            ok += 1
        else:
            result = split_file(src)
            if result:
                split_map[str(src)] = (str(result[0]), str(result[1]))
                ok += 1

    if args.dry_run:
        print(f"\n[DRY-RUN] {ok} fichiers seraient splittés.")
        return

    print(f"\n{ok}/{len(flc_files)} fichiers splittés.")

    # Mettre à jour le catalogue
    catalog_path = TRAINING_DATA / "pairs_catalog.json"
    if catalog_path.exists():
        print("\nMise à jour du catalogue...")
        update_catalog(catalog_path, split_map)
    else:
        print("  ⚠ pairs_catalog.json introuvable — relancer clean_flc.py")

    # Compter le résultat
    chip_files = list(TRAINING_DATA.rglob("*_chip[12].fits"))
    total_gb = sum(f.stat().st_size for f in chip_files) / 1e9
    print(f"\n{len(chip_files)} fichiers chip ({total_gb:.1f} GB)")
    print("Terminé.")


if __name__ == "__main__":
    main()
