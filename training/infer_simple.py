"""Inference for plain mono FITS images (no SCI extension).

Same preprocessing chain as infer.py (e-/s, sky removal, joint-style scale,
asinh compression), for processed images such as drizzled mosaics. EXPTIME
is taken from the header when present, else 1.0; either way it cancels
exactly through the adaptive scale, so absolute units do not matter.
"""

import argparse

import numpy as np
import torch
from astropy.io import fits

from infer import ASINH_BETA, denormalize, normalize, pad_to_multiple
from unet_model import UNet


def infer_simple(input_path, model_path, output_path, device="mps",
                 tile_scale=False):
    device = torch.device(device)
    hdul = fits.open(input_path)
    hdu = next(h for h in hdul if h.data is not None)
    raw = hdu.data.astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a mono 2D image, got shape {raw.shape}")

    exptime = float(hdu.header.get("EXPTIME", 0.0)) or 1.0
    data_norm, sky, scale = normalize(raw, exptime)
    if tile_scale:
        # Median of per-256px-tile p99-p1: the training statistics. The
        # full-image percentiles are inflated by a bright galaxy, which
        # compresses the background below the network's working amplitude
        # (same fix as infer2.py; opt-in to leave the certified run-5
        # chain behavior untouched).
        x = raw / exptime - sky
        scales = []
        tile = 256
        for y in range(0, x.shape[0] - tile + 1, tile):
            for xx in range(0, x.shape[1] - tile + 1, tile):
                p1, p99 = np.percentile(x[y:y + tile, xx:xx + tile], [1, 99])
                scales.append(p99 - p1)
        scale = float(np.median(scales)) + 1e-8
        data_norm = np.arcsinh(x / scale / ASINH_BETA) * ASINH_BETA
    data_pad, (pad_h, pad_w) = pad_to_multiple(data_norm.astype(np.float32), 32)

    model = UNet(in_channels=1, out_channels=1, load_from=model_path).to(device)
    model.eval()

    with torch.no_grad():
        if data_pad.shape[0] > 2048:
            mid = (data_pad.shape[0] // 2 // 32) * 32
            top = torch.from_numpy(data_pad[:mid][np.newaxis, np.newaxis]).to(device)
            bot = torch.from_numpy(data_pad[mid:][np.newaxis, np.newaxis]).to(device)
            out = np.concatenate([model(top).cpu().numpy()[0, 0],
                                  model(bot).cpu().numpy()[0, 0]], axis=0)
        else:
            tensor = torch.from_numpy(data_pad[np.newaxis, np.newaxis]).to(device)
            out = model(tensor).cpu().numpy()[0, 0]

    if pad_h > 0:
        out = out[:-pad_h]
    if pad_w > 0:
        out = out[:, :-pad_w]
    result = denormalize(out, sky, scale, exptime)

    out_hdu = fits.PrimaryHDU(data=result.astype(np.float32), header=hdu.header.copy())
    out_hdu.header["HISTORY"] = "AstroSURE N2N denoising (infer_simple)"
    out_hdu.writeto(output_path, overwrite=True)
    print(f"Saved {output_path} ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a plain mono FITS image")
    parser.add_argument("input")
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--tile-scale", action="store_true",
                        help="normalization scale from 256px-tile statistics "
                             "(robust to a bright galaxy in the field)")
    args = parser.parse_args()
    infer_simple(args.input, args.model, args.output, device=args.device,
                 tile_scale=args.tile_scale)
