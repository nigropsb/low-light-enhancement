"""
Phase 3 - Restormer training loop.

Reuses build_dataloaders / save_checkpoint / load_checkpoint / train_one_epoch
/ validate unchanged from src.train_baseline. Phase 2 already proved those are
model-agnostic: they operate purely on batch["low"]/batch["high"] tensors and
call model(low), with no UNetBaseline-specific logic anywhere in them. So the
only things this script changes are (a) which model gets built, and (b) the
default hyperparameters suited to a much heavier model on the same 4GB-VRAM
GTX 1650.

Differences from train_baseline.py's defaults, and why:
- batch-size=2, accum-steps=2 (effective batch 4, matching the Phase 2
  baseline's batch_size=4 for a fair comparison in the Phase 6 results
  table). Restormer's transformer blocks -- and the 8x channel widening at
  the bottleneck -- are far more memory-hungry per sample than the
  baseline's plain convolutions.
- crop-size=128, not 256, as the starting point -- same VRAM-conservatism
  that led Phase 2 to `--crop-size 128 --no-amp`. Revisit 256 only once this
  is confirmed stable (scripts/verify_phase3.py reports peak VRAM at these
  defaults on the real GPU).
- amp defaults to OFF. Phase 2 found this GTX 1650 (Turing, no Tensor Cores)
  picks numerically unstable fp16 conv algorithms under AMP; attention's
  softmax is an *additional* fp16 overflow risk on top of that, so the same
  "AMP only if VRAM actually forces it" decision rule applies here, if
  anything more strongly. Try --amp only if training hits CUDA OOM at
  batch-size=2/crop-size=128, and reuse scripts/diagnose_val_nan.py's
  bisection approach (raw data -> single-sample -> batch, fp16 vs fp32) if
  train_loss goes NaN.
- ssim-weight=0.2 (baseline used the Charbonnier-only default). losses.py's
  SSIM term was wired into ReconstructionLoss back in Phase 2 specifically
  for this moment: turned on now because Restormer's larger receptive field
  and attention mechanism should actually be able to exploit the structural
  signal, whereas the shallow 3-level baseline U-Net had less to gain from
  it.

Usage:
    python -m src.train_restormer --data-root data/LOLdataset --epochs 100
"""

import argparse
import csv
import time
from pathlib import Path

import torch

from src.models.restormer import Restormer
from src.losses import ReconstructionLoss
from src.train_baseline import (
    build_dataloaders,
    save_checkpoint,
    load_checkpoint,
    train_one_epoch,
    validate,
)


def main():
    p = argparse.ArgumentParser(description="Phase 3 Restormer training")
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--output-dir", type=str, default="runs/phase3_restormer")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--accum-steps", type=int, default=2,
        help="Gradient accumulation steps. Combined with --batch-size=2 gives "
             "an effective batch size of 4, matching the Phase 2 baseline.",
    )
    p.add_argument(
        "--amp", action="store_true", default=False,
        help="Off by default on this GPU -- see module docstring for why.",
    )
    p.add_argument("--dim", type=int, default=24, help="Base embedding dimension.")
    p.add_argument(
        "--ssim-weight", type=float, default=0.2,
        help="Weight of the SSIM term in ReconstructionLoss (0.0 = Charbonnier only).",
    )
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--profile", action="store_true",
        help="Print a data/h2d/forward/backward/step time breakdown each epoch "
             "(see train_one_epoch in src/train_baseline.py).",
    )
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

    model = Restormer(dim=args.dim).to(device)
    loss_fn = ReconstructionLoss(ssim_weight=args.ssim_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val_psnr = 0, float("-inf")
    if args.resume:
        start_epoch, best_val_psnr = load_checkpoint(Path(args.resume), model, optimizer, scaler, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val_psnr={best_val_psnr:.2f}")

    n_params = sum(pm.numel() for pm in model.parameters() if pm.requires_grad)
    print(
        f"Restormer params: {n_params:,} | device={device} | amp={use_amp} | "
        f"effective batch size={args.batch_size * args.accum_steps} "
        f"(batch={args.batch_size} x accum={args.accum_steps})"
    )

    is_new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "seconds"])

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scaler,
                device, args.accum_steps, use_amp, profile=args.profile,
            )
            t_train = time.time() - t0

            t1 = time.time()
            val_loss, val_psnr, val_ssim = validate(model, val_loader, loss_fn, device)
            t_val = time.time() - t1

            elapsed = t_train + t_val
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_psnr={val_psnr:.2f} val_ssim={val_ssim:.3f} "
                f"({elapsed:.1f}s = train {t_train:.1f}s + val {t_val:.1f}s)"
            )
            if train_loss != train_loss:  # NaN check without importing math
                print(
                    "  /!\\ train_loss is NaN -- same GradScaler-silently-skips-every-step "
                    "failure mode diagnosed in Phase 2 (scripts/diagnose_val_nan.py): cuDNN/"
                    "attention picking an unstable fp16 algorithm on this Turing card. If "
                    "--amp was passed, drop it. If it persists in fp32 too, reduce "
                    "--crop-size or --batch-size next."
                )
            writer.writerow([epoch, train_loss, val_loss, val_psnr, val_ssim, f"{elapsed:.1f}"])
            f.flush()

            save_checkpoint(last_ckpt, model, optimizer, scaler, epoch, best_val_psnr)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_checkpoint(best_ckpt, model, optimizer, scaler, epoch, best_val_psnr)
                print(f"  -> new best (val_psnr={best_val_psnr:.2f}), saved to {best_ckpt}")


if __name__ == "__main__":
    main()
