"""Training loop for AstroSURE Noise2Noise."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset_n2n import N2NDataset
from dataset_n2n_stack import N2NStackDataset
from unet_model import UNet

# Force unbuffered output
print = lambda *a, **kw: (sys.stdout.write(" ".join(map(str, a)) + kw.get("end", "\n")), sys.stdout.flush())

# ── Defaults ──────────────────────────────────────────────────────────────────
PATCH_SIZE = 256
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 50
PATCHES_PER_PAIR = 8
VAL_FRACTION = 0.2
CKPT_EVERY = 5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "training_data" / "pairs_catalog.json"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse.item())


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    n_batches = len(loader)
    for i, (inp, tgt) in enumerate(loader):
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inp.size(0)
        n += inp.size(0)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  batch {i+1}/{n_batches}  loss={loss.item():.6f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp)
        total_loss += criterion(out, tgt).item() * inp.size(0)
        total_psnr += psnr(out, tgt) * inp.size(0)
        n += inp.size(0)
    return total_loss / max(n, 1), total_psnr / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default=str(CATALOG))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patches-per-pair", type=int, default=PATCHES_PER_PAIR)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mode", choices=["pairs", "stack2"], default="pairs",
                        help="pairs = run-5 recipe (1 ch); stack2 = run-8 "
                             "recipe (2-ch input, stacked target)")
    parser.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    parser.add_argument("--samples-per-unit", type=int, default=128,
                        help="stack2 mode: samples per group unit per epoch")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if ckpt_dir.is_symlink():
        # training/checkpoints is a symlink to the champion run's directory:
        # writing through it would silently destroy the reference best.pth.
        sys.exit(f"ERROR: {ckpt_dir} is a symlink ({ckpt_dir.resolve().name}); "
                 f"pass an explicit --ckpt-dir for a new run.")
    ckpt_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device} | mode: {args.mode} | checkpoints: {ckpt_dir}")

    # ── Dataset & split ───────────────────────────────────────────────────────
    # A slot = one pair (pairs mode) or one group unit (stack2 mode); the
    # train/val split happens at slot level.
    if args.mode == "stack2":
        full_ds = N2NStackDataset(args.catalog, samples_per_unit=args.samples_per_unit)
        n_slots = len(full_ds.units)
        spg = args.samples_per_unit
        in_channels = 2
    else:
        full_ds = N2NDataset(args.catalog, patches_per_pair=args.patches_per_pair)
        n_slots = len(full_ds.pairs)
        spg = args.patches_per_pair
        in_channels = 1
    indices = list(range(n_slots))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(n_slots * (1 - VAL_FRACTION))

    train_slot_idx = indices[:split]
    val_slot_idx = indices[split:]

    # Expand slot indices to sample indices
    train_indices = [p * spg + k for p in train_slot_idx for k in range(spg)]
    val_indices = [p * spg + k for p in val_slot_idx for k in range(spg)]

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    print(f"Slots: {n_slots} total, {len(train_slot_idx)} train, {len(val_slot_idx)} val")
    print(f"Samples: {len(train_ds)} train, {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(in_channels=in_channels, out_channels=1, load_from=args.resume).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    # L1: the target's cosmic rays (~2% of pixels on a 1200 s ACS exposure)
    # dominate a quadratic loss and bias the mean predictor; the median
    # predictor (L1) is immune to them.
    criterion = torch.nn.L1Loss()

    best_val = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_p = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        dt = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | "
            f"PSNR {val_p:.2f} dB | lr {lr_now:.1e} | {dt:.0f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> saved best.pth (val_loss={best_val:.6f})")

        if epoch % CKPT_EVERY == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

    # Save final
    torch.save(model.state_dict(), ckpt_dir / "final.pth")
    print("Training complete.")


if __name__ == "__main__":
    main()
