"""
Phase 2 - baseline U-Net training loop.

Purpose: exercise the whole pipeline (data -> model -> loss -> checkpointing)
end to end on a small, fast model, before Phase 3 swaps in the transformer
backbone. Everything here (AMP, grad accumulation, checkpoint format, CSV
logging) is meant to carry forward unchanged into that phase.

Usage:
    python -m src.train_baseline --data-root data/LOLdataset --epochs 50

Data interface (confirmed against the actual Phase 1 module):
    make_splits(root, crop_size=256, val_fraction=0.1, seed=42)
        -> tuple[LOLDataset, LOLDataset, LOLDataset]   # train, val, test
`make_splits` already returns fully-constructed `LOLDataset` instances
(augmentation pipeline wired in internally via `PairedTransform`/
`AugmentConfig`), so `build_dataloaders()` below only wraps them in
DataLoaders -- no need to touch `PairedTransform` or `LOLDataset` directly.

Each `LOLDataset` item is a dict: {"low": Tensor(3,H,W), "high": Tensor(3,H,W),
"filename": str}. The default collate stacks "low"/"high" into (B,3,H,W)
batches and "filename" into a list of strings -- train_one_epoch()/validate()
index into the batch dict rather than unpacking a tuple.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.unet import UNetBaseline
from src.losses import ReconstructionLoss
from src.metrics import batch_psnr, batch_ssim


def build_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    """Wrap the Phase 1 datasets in DataLoaders. `make_splits()` does all
    the actual dataset construction (including augmentation); nothing else
    needed here."""
    from src.data.lol_dataset import make_splits

    train_ds, val_ds, _test_ds = make_splits(args.data_root, crop_size=args.crop_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best_val_psnr: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_psnr": best_val_psnr,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scaler, device):
    # weights_only=False is explicit (not just the current default) because
    # this checkpoint dict carries optimizer/scaler state alongside the raw
    # tensors, not just a state_dict -- weights_only=True would reject that.
    # Safe here since these are always our own checkpoints, never loaded
    # from an untrusted source.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt["epoch"] + 1, ckpt["best_val_psnr"]


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, accum_steps: int, use_amp: bool,
                     profile: bool = False) -> float:
    """Set profile=True to print a breakdown of where epoch time actually
    goes: waiting for a batch from the DataLoader ("data"), host->device
    transfer ("h2d"), forward+loss ("fwd"), backward ("bwd"), optimizer
    step ("step").

    This inserts torch.cuda.synchronize() at every stage boundary to get an
    honest split -- CUDA calls are async, so without synchronizing, a "fast"
    stage might just be one that queued work for the GPU without waiting for
    it to finish. That synchronization is exactly what normal training
    avoids (it lets CPU and GPU overlap), so a profiled epoch will run
    noticeably slower than an unprofiled one. Use the printed *percentages*
    to find the bottleneck -- don't use the profiled epoch's wall-clock time
    as a benchmark of normal training speed.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    timings = {"data": 0.0, "h2d": 0.0, "fwd": 0.0, "bwd": 0.0, "step": 0.0}
    t_prev = time.time()

    for i, batch in enumerate(loader):
        if profile:
            torch.cuda.synchronize()
            timings["data"] += time.time() - t_prev

        t0 = time.time()
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        if profile:
            torch.cuda.synchronize()
            timings["h2d"] += time.time() - t0

        t0 = time.time()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(low)
            loss = loss_fn(pred, high) / accum_steps
        if profile:
            torch.cuda.synchronize()
            timings["fwd"] += time.time() - t0

        t0 = time.time()
        scaler.scale(loss).backward()
        if profile:
            torch.cuda.synchronize()
            timings["bwd"] += time.time() - t0

        t0 = time.time()
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if profile:
            torch.cuda.synchronize()
            timings["step"] += time.time() - t0

        running_loss += loss.item() * accum_steps
        n_batches += 1

        if profile:
            t_prev = time.time()

    if profile and n_batches > 0:
        total = sum(timings.values()) or 1e-9
        breakdown = " | ".join(f"{k}={v:.1f}s ({v / total * 100:.0f}%)" for k, v in timings.items())
        print(f"    [profile] {breakdown} | tracked_total={total:.1f}s over {n_batches} batches "
              f"(sync overhead inflates this vs. an unprofiled epoch)")

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> tuple[float, float, float]:
    """Runs in fp32 unconditionally -- NOT gated by the --amp flag.

    Diagnosed on this project's GTX 1650: a batch of 4 full-resolution
    (400x600, uncropped) validation images under fp16 autocast produced a
    NaN prediction across every element of the batch, while the identical
    images processed one at a time under fp16 -- and the identical batch
    under fp32 -- were both clean. That's the signature of cuDNN selecting
    a numerically unstable fp16 convolution algorithm (commonly Winograd)
    for this batch-size/resolution combination, an issue documented on
    Turing-generation cards. Validation has no backward pass or optimizer
    state, so it isn't the VRAM bottleneck AMP exists for -- there's no
    real cost to just running it in full precision.
    """
    model.eval()
    running_loss, running_psnr, running_ssim = 0.0, 0.0, 0.0
    n_batches = 0

    for batch in loader:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        pred = model(low)
        loss = loss_fn(pred, high)

        # Model no longer clamps internally (see src/models/unet.py) -- clamp
        # here since PSNR/SSIM assume a valid [0, 1] max_val range.
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        running_loss += loss.item()
        running_psnr += batch_psnr(pred_clamped.float(), high.float())
        running_ssim += batch_ssim(pred_clamped.float(), high.float())
        n_batches += 1

    n_batches = max(n_batches, 1)
    return running_loss / n_batches, running_psnr / n_batches, running_ssim / n_batches


def main():
    p = argparse.ArgumentParser(description="Phase 2 baseline U-Net training")
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--output-dir", type=str, default="runs/phase2_baseline")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=1,
                    help="Gradient accumulation steps. Scaffolding for Phase 3/4 "
                         "when the transformer backbone needs a smaller per-step "
                         "batch to fit in 4GB VRAM; leave at 1 for the baseline.")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", action="store_true",
                    help="Print a data/h2d/forward/backward/step time breakdown each epoch. "
                         "Adds cuda.synchronize() calls, so epoch time will look worse than "
                         "an unprofiled run -- read the percentages, not the absolute seconds.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.csv"
    last_ckpt = output_dir / "last.pt"
    best_ckpt = output_dir / "best.pt"

    train_loader, val_loader = build_dataloaders(args)

    model = UNetBaseline().to(device)
    loss_fn = ReconstructionLoss()  # baseline default: Charbonnier only
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val_psnr = 0, float("-inf")
    if args.resume:
        start_epoch, best_val_psnr = load_checkpoint(Path(args.resume), model, optimizer, scaler, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val_psnr={best_val_psnr:.2f}")

    is_new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "seconds"])

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                          device, args.accum_steps, use_amp, profile=args.profile)
            t_train = time.time() - t0

            t1 = time.time()
            val_loss, val_psnr, val_ssim = validate(model, val_loader, loss_fn, device)
            t_val = time.time() - t1

            elapsed = t_train + t_val

            print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_psnr={val_psnr:.2f} val_ssim={val_ssim:.3f} "
                  f"({elapsed:.1f}s = train {t_train:.1f}s + val {t_val:.1f}s)")
            if train_loss != train_loss:  # NaN check without importing math
                print("  /!\\ train_loss is NaN -- GradScaler is silently skipping every "
                      "optimizer step (inf/nan gradients detected), so the model is NOT "
                      "training. This project has already hit this exact failure mode once "
                      "(cuDNN picking an unstable fp16 conv algorithm for a specific "
                      "batch/shape combo on this GPU's non-Tensor-Core fp16 path). "
                      "Try --no-amp, especially at smaller --crop-size values where VRAM "
                      "headroom is no longer the constraint AMP was solving for.")
            writer.writerow([epoch, train_loss, val_loss, val_psnr, val_ssim, f"{elapsed:.1f}"])
            f.flush()

            save_checkpoint(last_ckpt, model, optimizer, scaler, epoch, best_val_psnr)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, best_val_psnr)
                print(f"  -> new best (val_psnr={best_val_psnr:.2f}), saved to {best_ckpt}")


if __name__ == "__main__":
    main()
