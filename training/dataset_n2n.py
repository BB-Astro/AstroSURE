"""Noise2Noise dataset with on-the-fly WCS alignment for Hubble ACS/WFC chips."""

import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import shift as nd_shift
from torch.utils.data import Dataset

PATCH_SIZE = 256
MARGIN = 10  # pixels from edge to avoid partial patches
MAX_ROTATION_DEG = 0.1  # pairs with a larger relative roll cannot be aligned by translation
ASINH_BETA = 1.0  # dynamics compression knee, in scale units (~4.5 sigma of background)
ROOT_OFFSETS = Path(__file__).resolve().parent.parent / "training_data" / "wcs_offsets.json"


class N2NDataset(Dataset):
    """
    Each sample = one random 256x256 patch from a random pair in the catalog.

    Pairs are formed from all C(n,2) combinations within each group.
    Alignment is done via WCS: pick a random sky position in the overlap region,
    project to pixel coords in both images, extract patches.
    """

    def __init__(
        self,
        catalog_path: str,
        patches_per_pair: int = 8,
        augment: bool = True,
        groups: list | None = None,
    ):
        self.patch_size = PATCH_SIZE
        self.augment = augment
        self.patches_per_pair = patches_per_pair

        with open(catalog_path) as f:
            catalog = json.load(f)

        if groups is not None:
            catalog = [catalog[i] for i in groups]

        # Build flat list of (file_a, file_b) pairs. Pairs whose exposures
        # have a different roll angle (e.g. NGC 628: proposals at ~-74 vs
        # ~+100 deg) are dropped: patch extraction aligns by translation
        # only, a rotated target teaches the network to erase everything
        # that is not rotation-invariant, starting with point sources.
        angles = {}
        def cd_angle(path):
            if path not in angles:
                with fits.open(path) as hdul:
                    hdr = hdul["SCI"].header
                angles[path] = float(np.degrees(np.arctan2(hdr["CD1_2"], hdr["CD1_1"])))
            return angles[path]

        self.pairs = []
        n_rotated = 0
        for group in catalog:
            files = group["files"]
            for a, b in combinations(range(len(files)), 2):
                if abs(cd_angle(files[a]) - cd_angle(files[b])) > MAX_ROTATION_DEG:
                    n_rotated += 1
                    continue
                self.pairs.append((files[a], files[b]))
        if n_rotated:
            print(f"Dropped {n_rotated} rotated pairs (relative roll > {MAX_ROTATION_DEG} deg)")

        # Cache: path -> (data, wcs, footprint corners, wcs offset)
        self._cache = {}

        # Per-exposure WCS zero-point corrections (pixels), measured by star
        # centroids against a per-group reference (QC 2026-08-05): the
        # absolute astrometry shifts by up to 1.5 px BETWEEN VISITS, while
        # intra-visit alignment is < 0.1 px. Keyed by file basename so the
        # same table serves the raw and CR-cleaned trees. Files absent from
        # the table (original 5 galaxies, median 0.03 px) get (0, 0).
        self._wcs_offsets = {}
        offsets_path = ROOT_OFFSETS
        if offsets_path.exists():
            with open(offsets_path) as f:
                self._wcs_offsets = json.load(f)
            n_big = sum(1 for v in self._wcs_offsets.values()
                        if (v[0] ** 2 + v[1] ** 2) ** 0.5 > 0.3)
            print(f"WCS offsets loaded: {len(self._wcs_offsets)} exposures "
                  f"({n_big} above 0.3 px)")

    def __len__(self):
        return len(self.pairs) * self.patches_per_pair

    @staticmethod
    def _clean_wcs_header(header):
        """Remove lookup-table distortion keywords that reference stripped extensions."""
        hdr = header.copy()
        for kw in list(hdr.keys()):
            if "D2IM" in kw or "CPDIS" in kw or "DP" in kw:
                del hdr[kw]
        return hdr

    def _load(self, path):
        if path not in self._cache:
            hdul = fits.open(path, memmap=True)
            data = hdul["SCI"].data.astype(np.float32)
            # Align the pair's underlying signal (run 3). FLC counts are in
            # electrons accumulated over EXPTIME (54% of pairs mix exposure
            # times, up to 360 vs 590 s) and the sky glow differs between
            # visits (up to 4x within a group): convert to e-/s and remove
            # each exposure's own sky median so both images of a pair share
            # the same target signal, as Noise2Noise requires.
            exptime = float(hdul[0].header.get("EXPTIME", 0.0)) or 1.0
            data = data / exptime
            data -= float(np.median(data[::8, ::8]))
            hdr = self._clean_wcs_header(hdul["SCI"].header)
            wcs = WCS(hdr)
            # Compute sky footprint of valid region (with margin)
            h, w = data.shape
            corners_pix = np.array([
                [MARGIN, MARGIN],
                [w - MARGIN, MARGIN],
                [w - MARGIN, h - MARGIN],
                [MARGIN, h - MARGIN],
            ], dtype=float)
            corners_sky = wcs.all_pix2world(corners_pix, 0)  # (4, 2) ra/dec
            off = tuple(self._wcs_offsets.get(Path(path).name, (0.0, 0.0)))
            self._cache[path] = (data, wcs, corners_sky, off)
        return self._cache[path]

    def _overlap_bbox(self, sky_a, sky_b):
        """Return (ra_min, ra_max, dec_min, dec_max) of overlap."""
        ra_min = max(sky_a[:, 0].min(), sky_b[:, 0].min())
        ra_max = min(sky_a[:, 0].max(), sky_b[:, 0].max())
        dec_min = max(sky_a[:, 1].min(), sky_b[:, 1].min())
        dec_max = min(sky_a[:, 1].max(), sky_b[:, 1].max())
        if ra_min >= ra_max or dec_min >= dec_max:
            return None
        return ra_min, ra_max, dec_min, dec_max

    def _extract_pair(self, data_a, wcs_a, off_a, data_b, wcs_b, off_b, ra, dec):
        """Extract an aligned patch pair centred on (ra, dec).

        The input patch A is a raw crop on its own pixel grid (its noise
        stays untouched). The target patch B is resampled (bilinear) onto
        A's exact grid: HST dither offsets are deliberately half-integer,
        and a >=0.5 px misregistration of an undersampled PSF makes stars
        inconsistent between input and target, which a robust loss then
        learns to erase (run 3: star photometry 0.5%). Only the target is
        interpolated, at the price of slightly correlated target noise.
        Returns (patch_a, patch_b) or None if out of bounds / non-finite.
        """
        xy_a = wcs_a.all_world2pix([[ra, dec]], 0)[0]
        xy_b = wcs_b.all_world2pix([[ra, dec]], 0)[0]
        ps = self.patch_size
        half = ps // 2
        h_a, w_a = data_a.shape
        h_b, w_b = data_b.shape

        cxa, cya = int(round(xy_a[0])), int(round(xy_a[1]))
        ya0, xa0 = cya - half, cxa - half
        if ya0 < 0 or xa0 < 0 or ya0 + ps > h_a or xa0 + ps > w_a:
            return None
        pa = data_a[ya0:ya0 + ps, xa0:xa0 + ps].copy()
        if not np.isfinite(pa).all():
            return None

        # Local translation A grid -> B grid (rotated pairs were dropped),
        # corrected by the measured per-exposure WCS zero-point shifts:
        # a star's true position in an image is the WCS prediction plus
        # that exposure's offset, so the pair's true translation is the
        # WCS translation plus the offset difference.
        tx = xy_b[0] - xy_a[0] + (off_b[0] - off_a[0])
        ty = xy_b[1] - xy_a[1] + (off_b[1] - off_a[1])
        by0f, bx0f = ya0 + ty, xa0 + tx
        by0, bx0 = int(np.floor(by0f)), int(np.floor(bx0f))
        fy, fx = by0f - by0, bx0f - bx0
        if by0 - 1 < 0 or bx0 - 1 < 0 or by0 + ps + 2 > h_b or bx0 + ps + 2 > w_b:
            return None
        big = data_b[by0 - 1:by0 + ps + 2, bx0 - 1:bx0 + ps + 2]
        if not np.isfinite(big).all():
            return None
        # out[k] = big[k + f]: sample B at A's grid positions
        pb = nd_shift(big, (-fy, -fx), order=1, mode="nearest")[1:1 + ps, 1:1 + ps]
        return pa, pb

    @staticmethod
    def _normalize_pair(a, b):
        """Scale both patches with the SAME joint statistics.

        Scale only, no offset: the sky is already removed per exposure at
        load time. No clipping either: run 2 capped every star core and
        cosmic ray to 1.0 during training while inference fed the network
        values in the hundreds, an untrained regime.
        """
        both = np.concatenate([a.ravel(), b.ravel()])
        p1, p99 = np.percentile(both, [1, 99])
        scale = p99 - p1 + 1e-8
        a = a / scale
        b = b / scale
        # Compress the dynamics (asinh, linear below ASINH_BETA). Without
        # it a 30+ unit star swamps the LeakyReLU features and the shape
        # information is lost: the run-4 probe kept faint PSF stars at
        # 95-97% but erased bright ones (>~45 sigma) like cosmic rays.
        return (np.arcsinh(a / ASINH_BETA) * ASINH_BETA,
                np.arcsinh(b / ASINH_BETA) * ASINH_BETA)

    def _apply_augment(self, a, b):
        """Same random flip/rot90 applied to both patches."""
        if random.random() > 0.5:
            a, b = np.flip(a, 0).copy(), np.flip(b, 0).copy()
        if random.random() > 0.5:
            a, b = np.flip(a, 1).copy(), np.flip(b, 1).copy()
        k = random.randint(0, 3)
        if k:
            a, b = np.rot90(a, k).copy(), np.rot90(b, k).copy()
        return a, b

    def __getitem__(self, idx):
        pair_idx = idx // self.patches_per_pair
        path_a, path_b = self.pairs[pair_idx]
        # Pairs are unordered: a random role swap cancels any leftover
        # directional bias (acquisition order, drifting sky). Swapped at
        # file level so the target is always the resampled one.
        if random.random() < 0.5:
            path_a, path_b = path_b, path_a

        data_a, wcs_a, sky_a, off_a = self._load(path_a)
        data_b, wcs_b, sky_b, off_b = self._load(path_b)
        bbox = self._overlap_bbox(sky_a, sky_b)

        # Try up to 50 random sky positions to get a valid patch pair
        for _ in range(50):
            if bbox is None:
                break
            ra = random.uniform(bbox[0], bbox[1])
            dec = random.uniform(bbox[2], bbox[3])
            patches = self._extract_pair(data_a, wcs_a, off_a, data_b, wcs_b, off_b, ra, dec)
            if patches is not None:
                pa, pb = patches
                pa, pb = self._normalize_pair(pa, pb)
                if self.augment:
                    pa, pb = self._apply_augment(pa, pb)
                return (
                    torch.from_numpy(pa[np.newaxis]).float(),
                    torch.from_numpy(pb[np.newaxis]).float(),
                )

        # Fallback: return zero patches (shouldn't happen often)
        z = torch.zeros(1, self.patch_size, self.patch_size)
        return z, z


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--catalog",
        default=str(
            Path(__file__).resolve().parent.parent / "training_data" / "pairs_catalog.json"
        ),
    )
    args = parser.parse_args()

    if args.test:
        ds = N2NDataset(args.catalog, patches_per_pair=4, augment=True)
        print(f"Dataset: {len(ds)} samples from {len(ds.pairs)} pairs")

        t0 = time.time()
        for i in range(min(8, len(ds))):
            a, b = ds[i]
            print(f"  sample {i}: A {a.shape} [{a.min():.3f}, {a.max():.3f}]  "
                  f"B {b.shape} [{b.min():.3f}, {b.max():.3f}]")
        dt = time.time() - t0
        print(f"Loaded {min(8, len(ds))} samples in {dt:.1f}s")
