"""Full-image inference for the run-8 2-channel model.

Inputs are two single-chip mono FITS files ALREADY on the same pixel grid:
  ref.fits     = raw reference chip (EXPTIME in header, electrons)
  partner.fits = partner exposure(s) resampled onto the reference grid
                 (produced by pipeline/asure2_flc.py, e-/s, sky-subtracted,
                 EXPTIME=1.0)
The output is the denoised reference chip in the reference units.

Preprocessing mirrors dataset_n2n_stack: each channel to e-/s minus its
own sky median, ONE joint scale over both channels, asinh compression.
"""

import argparse

import numpy as np
import torch
from astropy.io import fits

from infer import ASINH_BETA, pad_to_multiple
from unet_model import UNet


def load_chip(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data.astype(np.float32)
                exptime = float(hdul[0].header.get("EXPTIME", 0.0)) or 1.0
                return data, exptime
    raise ValueError(f"no image data in {path}")


def infer2(ref_path, partner_path, model_path, output_path, device="mps",
           half_split=True):
    device = torch.device(device)

    raw, exptime = load_chip(ref_path)
    partner, exptime_p = load_chip(partner_path)
    if raw.shape != partner.shape:
        raise ValueError(f"grid mismatch: {raw.shape} vs {partner.shape}")

    rate = raw / exptime
    sky = float(np.median(rate[np.isfinite(rate)]))
    ch0 = rate - sky
    rate_p = partner / exptime_p
    ch1 = rate_p - float(np.median(rate_p[np.isfinite(rate_p)]))

    # Training normalized per 256x256 patch: on a chip hosting a bright
    # galaxy, the FULL-CHIP p99-p1 is inflated (measured 2.2x on Arp 141
    # SCI2) which compresses the background below the network's working
    # amplitude and collapses the denoising (sigma /3 instead of /12).
    # The median of per-tile scales matches the training statistics.
    scales = []
    tile = 256
    for y in range(0, ch0.shape[0] - tile + 1, tile):
        for x in range(0, ch0.shape[1] - tile + 1, tile):
            both = np.concatenate([ch0[y:y + tile, x:x + tile].ravel(),
                                   ch1[y:y + tile, x:x + tile].ravel()])
            p1, p99 = np.percentile(both, [1, 99])
            scales.append(p99 - p1)
    scale = float(np.median(scales)) + 1e-8
    ch0 = np.arcsinh(ch0 / scale / ASINH_BETA) * ASINH_BETA
    ch1 = np.arcsinh(ch1 / scale / ASINH_BETA) * ASINH_BETA

    pad0, (pad_h, pad_w) = pad_to_multiple(ch0.astype(np.float32), 32)
    pad1, _ = pad_to_multiple(ch1.astype(np.float32), 32)
    stacked = np.stack([pad0, pad1])  # (2, H, W)

    model = UNet(in_channels=2, out_channels=1, load_from=model_path).to(device)
    model.eval()

    with torch.no_grad():
        if half_split and stacked.shape[1] > 2048:
            mid = (stacked.shape[1] // 2 // 32) * 32
            top = torch.from_numpy(stacked[:, :mid][np.newaxis]).to(device)
            bot = torch.from_numpy(stacked[:, mid:][np.newaxis]).to(device)
            out = np.concatenate([model(top).cpu().numpy()[0, 0],
                                  model(bot).cpu().numpy()[0, 0]], axis=0)
        else:
            tensor = torch.from_numpy(stacked[np.newaxis]).to(device)
            out = model(tensor).cpu().numpy()[0, 0]

    if pad_h > 0:
        out = out[:-pad_h]
    if pad_w > 0:
        out = out[:, :-pad_w]
    result = (np.sinh(out / ASINH_BETA) * ASINH_BETA * scale + sky) * exptime

    hdr = fits.Header()
    hdr["EXPTIME"] = exptime
    fits.writeto(output_path, result.astype(np.float32), hdr, overwrite=True)
    print(f"Saved {output_path} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run-8 two-channel inference")
    parser.add_argument("ref", help="reference chip FITS (raw grid)")
    parser.add_argument("partner", help="partner chip FITS aligned on the reference grid")
    parser.add_argument("model", help="checkpoint (.pth), trained with in_channels=2")
    parser.add_argument("output", help="output FITS")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--no-split", action="store_true")
    args = parser.parse_args()

    infer2(args.ref, args.partner, args.model, args.output,
           device=args.device, half_split=not args.no_split)
