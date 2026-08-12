"""Run-8 dataset: 2-channel input pair, stacked target (PLAN, CHANTIER RUN 8).

For each group of N >= 3 same-roll exposures:
  input  = [A raw crop, B resampled onto A's grid]  (2 channels)
  target = mean of the remaining exposures, each resampled onto A's grid,
           cosmic rays rejected against the stack.

The model sees twice the photons and learns from a sqrt(M)-quieter target,
which moves the noise/detail trade-off without any new data. Sampling is
balanced per group unit (same number of samples whether the unit has 3 or
20 exposures): run 7 showed that letting dense star fields dominate the
sample count degrades the galaxy-field QC.
"""

import random

import numpy as np
import torch

from dataset_n2n import MAX_ROTATION_DEG, N2NDataset

MAX_PARTNERS = 12  # cap target stack size (noise gain is sqrt(M), CPU cost linear)
CLIP_SIGMA = 5.0   # positive-outlier rejection threshold against the stack


class N2NStackDataset(N2NDataset):
    """Group-level samples. Reuses N2NDataset's loading, WCS-offset table,
    subpixel extraction and joint normalization machinery."""

    def __init__(self, catalog_path: str, samples_per_unit: int = 128,
                 augment: bool = True):
        # Parent __init__ builds the pair list (unused here) but also the
        # cache, the rotation-angle helper and the WCS offset table.
        super().__init__(catalog_path, patches_per_pair=1, augment=augment)
        self.samples_per_unit = samples_per_unit

        import json
        with open(catalog_path) as f:
            catalog = json.load(f)

        from astropy.io import fits
        angles = {}

        def cd_angle(path):
            if path not in angles:
                with fits.open(path) as hdul:
                    hdr = hdul["SCI"].header
                angles[path] = float(np.degrees(np.arctan2(hdr["CD1_2"], hdr["CD1_1"])))
            return angles[path]

        # A unit = files of one group sharing the same roll angle (compared
        # to the cluster anchor), with at least 3 members: 2 inputs + >= 1
        # target. Two-file groups cannot feed the 2-channel model.
        self.units = []
        n_small = 0
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
                    self.units.append(c)
                else:
                    n_small += 1
        sizes = sorted(len(u) for u in self.units)
        print(f"Stack dataset: {len(self.units)} units (>=3 same-roll exposures), "
              f"sizes {sizes[0]}-{sizes[-1]}, {n_small} clusters dropped (<3 files)")

    def __len__(self):
        return len(self.units) * self.samples_per_unit

    @staticmethod
    def _stack_target(patches):
        """Mean of the partner patches with cosmic rays rejected.

        Diluting a CR into a mean is worse than leaving it whole: L1 ignores
        a rare huge outlier but follows a small bias present in the mean, so
        the stack must be cleaned before averaging. CRs are positive-only.

        The rejection threshold has a signal-proportional term: on a bright
        star core, Poisson noise and subpixel resampling differences exceed
        the BACKGROUND sigma by far, and a background-only threshold clips
        the positive fluctuations while keeping the negative ones, a
        systematic flux bias the model then learns (measured -4.2% median,
        -9% p10 at >50 sigma pixels; seen as 92-94% bright-star photometry
        in the epoch-30 QC probe of the first run-8 attempt).
        """
        stack = np.stack(patches)
        m = len(patches)
        if m == 1:
            return stack[0]
        if m == 2:
            diff = stack[0] - stack[1]
            level = np.maximum(stack.mean(0), 0.0)
            sigma = 1.4826 * float(np.median(np.abs(diff))) + 1e-8
            thresh = CLIP_SIGMA * sigma + 0.5 * level
            out = stack.mean(0)
            bad = np.abs(diff) > thresh
            # A large A/B disagreement is a CR in one of the two; CRs being
            # positive, the smaller value is the clean one.
            out[bad] = stack.min(0)[bad]
            return out
        med = np.median(stack, 0)
        resid = stack - med
        sigma = 1.4826 * float(np.median(np.abs(resid))) + 1e-8
        thresh = CLIP_SIGMA * sigma + 0.5 * np.maximum(med, 0.0)
        cleaned = np.where(resid > thresh, med, stack)
        return cleaned.mean(0)

    @staticmethod
    def _normalize_group(arrays):
        """Joint scale over every channel and the target, then asinh.
        Same statistics for everything, as Noise2Noise requires."""
        from dataset_n2n import ASINH_BETA
        both = np.concatenate([a.ravel() for a in arrays])
        p1, p99 = np.percentile(both, [1, 99])
        scale = p99 - p1 + 1e-8
        return [np.arcsinh(a / scale / ASINH_BETA) * ASINH_BETA for a in arrays]

    def _apply_augment_multi(self, arrays):
        """Same random flip/rot90 applied to every array."""
        if random.random() > 0.5:
            arrays = [np.flip(a, 0).copy() for a in arrays]
        if random.random() > 0.5:
            arrays = [np.flip(a, 1).copy() for a in arrays]
        k = random.randint(0, 3)
        if k:
            arrays = [np.rot90(a, k).copy() for a in arrays]
        return arrays

    def __getitem__(self, idx):
        unit = self.units[idx // self.samples_per_unit]

        for _ in range(50):
            a_idx, b_idx = random.sample(range(len(unit)), 2)
            path_a, path_b = unit[a_idx], unit[b_idx]
            partners = [p for i, p in enumerate(unit) if i not in (a_idx, b_idx)]
            if len(partners) > MAX_PARTNERS:
                partners = random.sample(partners, MAX_PARTNERS)

            data_a, wcs_a, sky_a, off_a = self._load(path_a)
            data_b, wcs_b, sky_b, off_b = self._load(path_b)
            bbox = self._overlap_bbox(sky_a, sky_b)
            if bbox is None:
                continue
            ra = random.uniform(bbox[0], bbox[1])
            dec = random.uniform(bbox[2], bbox[3])

            res = self._extract_pair(data_a, wcs_a, off_a, data_b, wcs_b, off_b, ra, dec)
            if res is None:
                continue
            pa, pb = res

            tpatches = []
            for p in partners:
                data_p, wcs_p, _, off_p = self._load(p)
                r = self._extract_pair(data_a, wcs_a, off_a, data_p, wcs_p, off_p, ra, dec)
                if r is not None:
                    tpatches.append(r[1])
            if not tpatches:
                continue

            tgt = self._stack_target(tpatches)
            pa, pb, tgt = self._normalize_group([pa, pb, tgt])
            if self.augment:
                pa, pb, tgt = self._apply_augment_multi([pa, pb, tgt])
            return (
                torch.from_numpy(np.stack([pa, pb])).float(),
                torch.from_numpy(tgt[np.newaxis]).float(),
            )

        z = torch.zeros(1, self.patch_size, self.patch_size)
        return torch.zeros(2, self.patch_size, self.patch_size), z


if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--catalog",
        default=str(Path(__file__).resolve().parent.parent
                    / "training_data" / "pairs_catalog_full_raw.json"),
    )
    args = parser.parse_args()

    if args.test:
        ds = N2NStackDataset(args.catalog, samples_per_unit=4)
        print(f"Dataset: {len(ds)} samples from {len(ds.units)} units")
        t0 = time.time()
        for i in range(min(8, len(ds))):
            inp, tgt = ds[i]
            print(f"  sample {i}: in {tuple(inp.shape)} [{inp.min():.3f}, {inp.max():.3f}]  "
                  f"tgt {tuple(tgt.shape)} [{tgt.min():.3f}, {tgt.max():.3f}]")
        print(f"Loaded {min(8, len(ds))} samples in {time.time() - t0:.1f}s")
