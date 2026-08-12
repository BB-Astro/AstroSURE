"""Section 4f: look at the actual pixels of the 'destroyed stars'."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qc_common import CAT_RAW, CAT_CLEAN, load_catalog, group_label, open_chip, sky_and_sigma

SPOTS = [
    ("NGC 4258/F814W/chip1", 0, 1867, 1597),
    ("NGC 4258/F814W/chip1", 0, 2702, 1425),
    ("NGC 4258/F814W/chip1", 0, 350, 550),
    ("NGC 3031/F606W/chip1", 0, 1867, 1607),
    ("NGC 3031/F606W/chip1", 0, 2736, 1569),
    ("NGC 3031/F606W/chip1", 1, 309, 889),
    ("IC 10/F814W/chip2", 0, 3405, 242),
    ("NGC 4258/F814W/chip2", 0, 3405, 242),
]


def show(a, sky, label):
    print(f"      {label}")
    for row in a:
        print("        " + " ".join(f"{v - sky:8.0f}" for v in row))


def main():
    raw = {group_label(g): g for g in load_catalog(CAT_RAW)}
    cln = {group_label(g): g for g in load_catalog(CAT_CLEAN)}
    for gl, ia, x, y in SPOTS:
        A = open_chip(raw[gl]["files"][ia])
        Ac = open_chip(cln[gl]["files"][ia])
        sky, sig = sky_and_sigma(A["data"])
        print(f"\n### {gl} exposition {ia}  pixel ({x},{y})  ciel={sky:.1f} sigma={sig:.1f}")
        h = 3
        show(A["data"][y - h:y + h + 1, x - h:x + h + 1], sky, "BRUT (ciel soustrait)")
        show(Ac["data"][y - h:y + h + 1, x - h:x + h + 1], sky, "NETTOYE")
        # vertical extent of the bright feature along the column
        col = A["data"][:, x] - sky
        hot = np.nonzero(col > 30 * sig)[0]
        near = hot[(hot > y - 60) & (hot < y + 60)]
        print(f"      colonne x={x}: {len(hot)} px >30sigma sur 2048 "
              f"(mediane du chip: voir ci-dessous) ; autour de y: "
              f"{near.min() if len(near) else '-'}..{near.max() if len(near) else '-'} "
              f"({len(near)} px)")
        cnt = ((A["data"] - sky) > 30 * sig).sum(axis=0)
        print(f"      ce chip: colonne x={x} a {cnt[x]} px >30sigma ; "
              f"mediane sur toutes colonnes = {np.median(cnt):.0f}, "
              f"p99 = {np.percentile(cnt, 99):.0f}, max = {cnt.max()}")
        # is the same column hot in another, unrelated field?
        others = [g for g in raw if g.endswith(gl[-5:]) and not g.startswith(gl.split('/')[0])]
        for og in others[:2]:
            O = open_chip(raw[og]["files"][0])
            so, sgo = sky_and_sigma(O["data"])
            c2 = ((O["data"] - so) > 30 * sgo).sum(axis=0)
            print(f"      champ sans rapport {og:22s}: colonne x={x} -> {c2[x]} px "
                  f"(mediane {np.median(c2):.0f}, p99 {np.percentile(c2,99):.0f})")
            del O
        del A, Ac


if __name__ == "__main__":
    main()
