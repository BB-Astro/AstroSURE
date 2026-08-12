"""Section 2: replay the dataset's pair-validity criteria on the new catalog."""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import CAT_RAW, CAT_CLEAN, load_catalog, group_label, rootname_of

MAX_ROTATION_DEG = 0.1  # dataset_n2n.MAX_ROTATION_DEG


def main():
    integ = json.load(open("qc_tmp/s1_integrity.json"))
    by_path = {r["path"]: r for r in integ["raw"]}
    by_path_clean = {r["path"]: r for r in integ["clean"]}

    for tag, cat_path, table in (("RAW", CAT_RAW, by_path),
                                 ("CLEAN", CAT_CLEAN, by_path_clean)):
        cat = load_catalog(cat_path)
        print(f"\n================ {tag} : {cat_path.name} ================")
        print(f"{'group':28s} {'n':>3s} {'C2':>4s} {'kept':>4s} {'rot_max':>8s} "
              f"{'exptime':>9s} {'filters':>18s} {'dup':>4s} {'sky_min/max':>16s}")
        tot_pairs = tot_kept = 0
        anomalies = []
        for g in cat:
            files = g["files"]
            recs = [table[f] for f in files]
            # roll angles from CD matrix
            angles = []
            for f in files:
                import astropy.io.fits as fits
                with fits.open(f, memmap=False) as h:
                    hd = h["SCI"].header
                    angles.append(float(np.degrees(np.arctan2(hd["CD1_2"], hd["CD1_1"]))))
            angles = np.array(angles)
            n = len(files)
            c2 = n * (n - 1) // 2
            kept = 0
            worst = 0.0
            for a, b in combinations(range(n), 2):
                d = abs(angles[a] - angles[b])
                worst = max(worst, d)
                if d <= MAX_ROTATION_DEG:
                    kept += 1
            tot_pairs += c2
            tot_kept += kept

            exps = sorted({r["exptime"] for r in recs})
            filt = sorted({(r["filter1"], r["filter2"]) for r in recs})
            filt_s = "/".join(f"{a}+{b}" for a, b in filt)
            roots = [rootname_of(f) for f in files]
            dup = len(roots) - len(set(roots))
            skies = [r["med"] for r in recs]
            hdr_roots = [r["rootname"] for r in recs]
            dup_hdr = len(hdr_roots) - len(set(hdr_roots))

            print(f"{group_label(g):28s} {n:3d} {c2:4d} {kept:4d} {worst:8.4f} "
                  f"{str(exps):>9s} {filt_s:>18s} {dup:4d} "
                  f"{min(skies):7.1f}/{max(skies):7.1f}")

            if len(exps) > 1:
                anomalies.append((group_label(g), "BLOQUANT", f"EXPTIME non uniforme: {exps}"))
            if len(filt) > 1:
                anomalies.append((group_label(g), "BLOQUANT", f"filtres mixtes: {filt_s}"))
            if dup or dup_hdr:
                anomalies.append((group_label(g), "BLOQUANT",
                                  f"doublons rootname: fichier={dup} header={dup_hdr}"))
            if worst > MAX_ROTATION_DEG:
                anomalies.append((group_label(g), "info",
                                  f"{c2-kept}/{c2} paires rejetees par rotation (max {worst:.3f} deg)"))
            if max(skies) > 4 * max(min(skies), 1e-3):
                anomalies.append((group_label(g), "attention",
                                  f"ciel variable x{max(skies)/max(min(skies),1e-3):.1f} "
                                  f"({min(skies):.1f} -> {max(skies):.1f} e-)"))
            # catalog filter vs header filter
            hdr_filters = set()
            for a, b in filt:
                hdr_filters.add(a if a != "CLEAR1L" and not str(a).startswith("CLEAR") else b)
            if g["filter"] not in hdr_filters:
                anomalies.append((group_label(g), "BLOQUANT",
                                  f"filtre catalogue {g['filter']} != header {hdr_filters}"))

        print(f"TOTAL {tag}: {tot_pairs} paires C(n,2), {tot_kept} retenues apres filtre rotation")

        # cross-group duplicate rootnames within the same filter+chip
        allroots = Counter()
        for g in cat:
            for f in g["files"]:
                allroots[(g["filter"], g["chip"], rootname_of(f))] += 1
        cross = {k: v for k, v in allroots.items() if v > 1}
        if cross:
            anomalies.append(("<global>", "BLOQUANT", f"rootname duplique: {cross}"))

        print("\nAnomalies:")
        if not anomalies:
            print("  aucune")
        for a in anomalies:
            print(f"  [{a[1]}] {a[0]}: {a[2]}")


if __name__ == "__main__":
    main()
