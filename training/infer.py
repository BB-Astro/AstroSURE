"""Full-image inference with reflective padding (no grid artifacts)."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from unet_model import UNet


def pad_to_multiple(data, multiple=32):
    """Pad with reflect so dimensions are divisible by `multiple`. Returns padded array and padding."""
    h, w = data.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = np.pad(data, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (pad_h, pad_w)


ASINH_BETA = 1.0  # must match dataset_n2n.ASINH_BETA


def normalize(data, exptime):
    """Mirror the run-5 training preprocessing: e-/s, own sky removed,
    joint-style scale, asinh dynamics compression."""
    rate = data / exptime
    sky = float(np.median(rate[np.isfinite(rate)]))
    x = rate - sky
    p1, p99 = np.percentile(x[np.isfinite(x)], [1, 99])
    scale = p99 - p1 + 1e-8
    x = np.arcsinh(x / scale / ASINH_BETA) * ASINH_BETA
    return x, sky, scale


def denormalize(data, sky, scale, exptime):
    x = np.sinh(data / ASINH_BETA) * ASINH_BETA
    return (x * scale + sky) * exptime


def infer(input_path, model_path, output_path, device="mps", half_split=True):
    """Denoise a single chip FITS file."""
    device = torch.device(device)

    # Load image
    hdul = fits.open(input_path)
    raw = hdul["SCI"].data.astype(np.float32)
    h_orig, w_orig = raw.shape

    # Normalize (same signal alignment as training, run 3)
    exptime = float(hdul[0].header.get("EXPTIME", 0.0)) or 1.0
    data_norm, sky, scale = normalize(raw, exptime)

    # Pad
    data_pad, (pad_h, pad_w) = pad_to_multiple(data_norm.astype(np.float32), 32)

    # Load model
    model = UNet(in_channels=1, out_channels=1, load_from=model_path).to(device)
    model.eval()

    with torch.no_grad():
        if half_split and data_pad.shape[0] > 2048:
            # Split vertically to fit in memory
            mid = data_pad.shape[0] // 2
            # Ensure mid is divisible by 32
            mid = (mid // 32) * 32

            top = torch.from_numpy(data_pad[:mid][np.newaxis, np.newaxis]).to(device)
            bot = torch.from_numpy(data_pad[mid:][np.newaxis, np.newaxis]).to(device)

            out_top = model(top).cpu().numpy()[0, 0]
            out_bot = model(bot).cpu().numpy()[0, 0]
            out = np.concatenate([out_top, out_bot], axis=0)
        else:
            tensor = torch.from_numpy(data_pad[np.newaxis, np.newaxis]).to(device)
            out = model(tensor).cpu().numpy()[0, 0]

    # Remove padding & denormalize
    if pad_h > 0:
        out = out[:-pad_h]
    if pad_w > 0:
        out = out[:, :-pad_w]
    result = denormalize(out, sky, scale, exptime)

    # Save: preserve header from original
    hdu_primary = hdul["PRIMARY"].copy()
    hdu_sci = fits.ImageHDU(data=result.astype(np.float32), header=hdul["SCI"].header)
    hdu_sci.name = "SCI"
    out_hdul = fits.HDUList([hdu_primary, hdu_sci])
    out_hdul.writeto(output_path, overwrite=True)
    print(f"Saved {output_path} ({result.shape[1]}x{result.shape[0]})")

    hdul.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a Hubble ACS/WFC chip image")
    parser.add_argument("input", help="Input FITS chip file")
    parser.add_argument("model", help="Path to model checkpoint (.pth)")
    parser.add_argument("output", help="Output FITS file")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--no-split", action="store_true", help="Process full image at once")
    args = parser.parse_args()

    infer(args.input, args.model, args.output, device=args.device, half_split=not args.no_split)
