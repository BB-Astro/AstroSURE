"""Control: is the per-visit WCS offset also present in the 260 pre-existing chips?"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import CAT_FULL, load_catalog, group_label, open_chip
from s3b_align_clean import measure

NEW = {"NGC 4258", "NGC 598", "NGC 1569", "IC 10", "NGC 3031"}


def main():
    cat = load_catalog(CAT_FULL)
    old = [g for g in cat if g["galaxy"] not in NEW and len(g["files"]) >= 4]
    rng = np.random.default_rng(3)
    pick = rng.choice(len(old), size=min(6, len(old)), replace=False)
    rows = []
    print(f"{'group':30s} {'file':>4s} {'nmatch':>7s} {'off_x':>7s} {'off_y':>7s} {'|off|':>7s} {'scat':>7s}")
    for k in pick:
        g = old[k]
        files = g["files"][:6]
        A = open_chip(files[0])
        for j in range(1, len(files)):
            B = open_chip(files[j])
            m = measure(A, B)
            if m is None:
                print(f"{group_label(g):30s} {j:4d}  pas assez d'appariements")
                continue
            print(f"{group_label(g):30s} {j:4d} {m['n_match']:7d} {m['off_x']:7.3f} "
                  f"{m['off_y']:7.3f} {m['off_r']:7.3f} {m['med_scatter']:7.3f}"
                  f"{'  <<<' if m['off_r']>0.3 else ''}", flush=True)
            m["group"] = group_label(g)
            m["galaxy"] = g["galaxy"]
            m["file"] = Path(files[j]).name
            m["ref"] = Path(files[0]).name
            rows.append(m)
            del B
        del A
    with open("qc_tmp/s3d_old_control.json", "w") as f:
        json.dump(rows, f, indent=1)
    offs = np.array([r["off_r"] for r in rows])
    print(f"\nANCIENS CHAMPS: n={len(offs)} mesures, |off| median={np.median(offs):.3f} px, "
          f"max={offs.max():.3f} px, >0.3px: {int((offs>0.3).sum())}/{len(offs)} "
          f"({100*(offs>0.3).mean():.0f} %)")


if __name__ == "__main__":
    main()
