"""
strip_sci.py — Réduit chaque FLC FITS à ses extensions SCI uniquement.

Un FLC ACS/WFC contient 20+ extensions (ERR, DQ, D2IMARR, WCSDVARR, HDRLET,
WCSCORR…) inutiles pour l'entraînement. On ne garde que :
  - [0] PRIMARY  (header de provenance)
  - [SCI,1]      chip 1  (4096×2048 float32)
  - [SCI,2]      chip 2  (4096×2048 float32)

Taille avant : ~168 MB → après : ~66 MB (économie de 60 %).

Usage :
    python download_mast/strip_sci.py                         # tout training_data/
    python download_mast/strip_sci.py training_data/NGC_628   # un sous-dossier
"""

import argparse
import sys
from pathlib import Path

from astropy.io import fits


def strip_to_sci(filepath: Path, dry_run: bool = False) -> bool:
    """
    Écrase le fichier FITS en ne conservant que PRIMARY + SCI.
    Retourne True si le fichier a été modifié.
    """
    try:
        with fits.open(filepath, memmap=False) as hdul:
            # Compter les SCI avant de filtrer
            sci_hdus = [hdu for hdu in hdul if hdu.name == "SCI"]
            if not sci_hdus:
                print(f"  ⚠ Pas de SCI dans {filepath.name} — ignoré")
                return False

            already_stripped = (
                len(hdul) == 1 + len(sci_hdus)  # PRIMARY + SCI seulement
            )
            if already_stripped:
                return False  # déjà propre, rien à faire

            if dry_run:
                n_before = len(hdul)
                n_after = 1 + len(sci_hdus)
                size_mb = filepath.stat().st_size / 1e6
                print(f"  [DRY] {filepath.name}: {n_before}→{n_after} ext, {size_mb:.0f} MB")
                return True

            # Construire le nouveau HDUList : PRIMARY + SCI
            new_hdul = fits.HDUList()
            new_hdul.append(fits.PrimaryHDU(header=hdul[0].header))
            for sci in sci_hdus:
                new_hdul.append(fits.ImageHDU(data=sci.data, header=sci.header,
                                              name="SCI"))

        # Écraser le fichier original
        tmp = filepath.with_suffix(".tmp.fits")
        new_hdul.writeto(tmp, overwrite=True)
        tmp.replace(filepath)
        return True

    except Exception as e:
        print(f"  ✗ Erreur sur {filepath.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Strip FITS to SCI-only")
    parser.add_argument("paths", nargs="*", type=Path,
                        default=[Path(__file__).parent.parent / "training_data"],
                        help="Fichiers ou dossiers à traiter")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche ce qui serait fait sans modifier les fichiers")
    args = parser.parse_args()

    fits_files = []
    for p in args.paths:
        if p.is_file() and p.suffix == ".fits":
            fits_files.append(p)
        elif p.is_dir():
            # Exclure le dossier _excluded
            fits_files.extend(
                f for f in p.rglob("*_flc.fits")
                if "_excluded" not in f.parts
            )
        else:
            print(f"Chemin inconnu : {p}")

    if not fits_files:
        print("Aucun fichier FITS trouvé.")
        sys.exit(0)

    total_before = sum(f.stat().st_size for f in fits_files) / 1e9
    print(f"{len(fits_files)} fichiers FLC — {total_before:.2f} GB au total\n")

    modified = 0
    for i, f in enumerate(fits_files, 1):
        size_before = f.stat().st_size / 1e6
        changed = strip_to_sci(f, dry_run=args.dry_run)
        if changed and not args.dry_run:
            size_after = f.stat().st_size / 1e6
            print(f"  [{i}/{len(fits_files)}] {f.name}: {size_before:.0f}→{size_after:.0f} MB")
            modified += 1

    if not args.dry_run:
        total_after = sum(f.stat().st_size for f in fits_files) / 1e9
        print(f"\n{'='*55}")
        print(f"Fichiers modifiés : {modified}/{len(fits_files)}")
        print(f"Taille totale : {total_before:.2f} GB → {total_after:.2f} GB "
              f"(-{(total_before - total_after):.2f} GB)")
    else:
        print("\n[DRY-RUN] Aucun fichier modifié.")


if __name__ == "__main__":
    main()
