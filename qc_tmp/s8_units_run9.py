"""Section 8: what the run-8/9 stacked dataset actually sees in the merged catalog.

Reproduces N2NStackDataset's unit construction (files of one catalog group
sharing a roll angle, >= 3 members) on pairs_catalog_run9.json and compares
it with the previous catalog, so the gain can be read before launching a run.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
MAX_ROTATION_DEG = 0.1

_angles = {}


def cd_angle(path):
    if path not in _angles:
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul["SCI"].header
        _angles[path] = float(np.degrees(np.arctan2(hdr["CD1_2"], hdr["CD1_1"])))
    return _angles[path]


def units_of(catalog):
    """Same clustering as N2NStackDataset.__init__."""
    units, small = [], 0
    for group in catalog:
        files = sorted(group["files"], key=cd_angle)
        clusters = []
        for f in files:
            if clusters and abs(cd_angle(f) - cd_angle(clusters[-1][0])) <= MAX_ROTATION_DEG:
                clusters[-1].append(f)
            else:
                clusters.append([f])
        for c in clusters:
            if len(c) >= 3:
                units.append((group, c))
            else:
                small += 1
    return units, small


def report(name, path):
    cat = json.loads(Path(path).read_text())
    units, small = units_of(cat)
    sizes = Counter(len(c) for _, c in units)
    print(f"\n=== {name} : {len(cat)} groupes, {len(units)} unites "
          f"(>=3 poses au meme roulis), {small} amas < 3 ecartes ===")
    for s in sorted(sizes):
        print(f"   {s:2d} poses : {sizes[s]:3d} unites")
    print(f"   unites >= 4 poses : {sum(v for k, v in sizes.items() if k >= 4)}")
    by_gal = Counter()
    for g, c in units:
        by_gal[g["galaxy"]] += 1
    print("   par galaxie : " + ", ".join(f"{k}={v}" for k, v in sorted(by_gal.items())))
    return units


def main():
    old = report("AVANT (pairs_catalog_full_raw)",
                 ROOT / "training_data" / "pairs_catalog_full_raw.json")
    new = report("APRES (pairs_catalog_run9)",
                 ROOT / "training_data" / "pairs_catalog_run9.json")
    print(f"\nGain : {len(new) - len(old)} unites "
          f"({len(old)} -> {len(new)})")


if __name__ == "__main__":
    sys.exit(main())
