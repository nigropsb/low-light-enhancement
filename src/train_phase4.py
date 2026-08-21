"""
Phase 4 - Restormer + DINOv2 perceptual loss training loop.

Reuses build_dataloaders / save_checkpoint / load_checkpoint /
train_one_epoch / validate unchanged from src.train_baseline, and reuses
the Restormer architecture unchanged from Phase 3 (src/models/restormer.py)
-- the only thing this script changes relative to src/train_restormer.py is
the loss function: ReconstructionLoss now also carries a
perceptual_weight > 0 term (frozen DINOv2 features; see
src/perceptual_loss.py for the design rationale).

Differences from train_restormer.py:
- --perceptual-weight, default 0.05 (changed from an initial guess of 0.1
  after Run 1 -- see project plan's Phase 4 log for the full diagnosis).
  At 0.1, perceptual's weighted loss contribution ended up ~2x ssim's
  despite the smaller weight value, and Run 1's best SSIM (0.817) never
  reached Phase 3's 0.839 across all 100 epochs. 0.05 targets a weighted
  contribution roughly matching ssim_weight=0.2's own -- see
  scripts/verify_phase4.py's loss-magnitude breakdown before trusting this
  on a different crop-size/architecture, and retune from the real run's
  printed numbers if the component balance still looks off.
- loss_fn.to(device) is now required: DINOv2's pretrained weights are real
  parameters that must live on the training device, unlike Phase 2/3's
  CharbonnierLoss/SSIMLoss, which are stateless. See ReconstructionLoss's
  docstring in src/losses.py.
- Same batch-size=2 / crop-size=128 / accum-steps=2 / amp=off defaults as
  Phase 3, since the Restormer side of the computation is unchanged -- but
  DINOv2's extra forward pass (twice per step: pred with grad, target
  without) is new VRAM pressure on top of that budget, and extra compute
  during validation too (validate() runs on full native 400x600 images,
  now with a perceptual-loss forward pass added). Run
  scripts/verify_phase4.py first; drop crop-size or batch-size further if
  it reports peak VRAM uncomfortably close to 4GB.
- --init-from lets Phase 4 optionally warm-start model weights from a
  Phase 3 checkpoint (same architecture, so state_dict loads directly)
  instead of training from random init. Off by default: training from
  scratch keeps the Phase 3 vs. Phase 4 comparison apples-to-apples (same
  epoch budget, same starting point), matching how Phase 3 itself was
  trained from scratch rather than fine-tuned from Phase 2's weights.
  Mainly useful for a quick qualitative look at what the perceptual term
  does to an already-decent model, not for the final results-table run.

Usage:
    python -m src.train_phase4 --data-root data/LOLdataset --epochs 100
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
    p = argparse.ArgumentParser(description="Phase 4 Restormer + DINOv2 perceptual loss training")
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--output-dir", type=str, default="runs/phase4_perceptual")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--accum-steps", type=int, default=2,
        help="Gradient accumulation steps. Same effective batch size (4) as Phase 3.",
    )
    p.add_argument(
        "--amp", action="store_true", default=False,
        help="Off by default -- same Turing/no-Tensor-Core reasoning as Phase 3, now with "
             "DINOv2's own attention softmax as a further fp16 overflow risk on this GPU.",
    )
    p.add_argument("--dim", type=int, default=24, help="Restormer base embedding dimension.")
    p.add_argument(
        "--ssim-weight", type=float, default=0.2,
        help="Weight of the SSIM term in ReconstructionLoss (carried over from Phase 3).",
    )
    p.add_argument(
        "--perceptual-weight", type=float, default=0.05,
        help="Weight of the frozen-DINOv2 feature term in ReconstructionLoss. 0.0 reproduces "
             "Phase 3's loss exactly. Changed from an initial guess of 0.1 to 0.05 after Run 1 "
             "(see project plan): at 0.1, perceptual's raw magnitude (~1.38, vs. ~0.36 for both "
             "charbonnier and ssim per verify_phase4.py) meant its *weighted* contribution "
             "(~0.138) was nearly 2x ssim's (~0.071) despite the smaller weight value -- and "
             "Run 1's best SSIM (0.817) never reached Phase 3's 0.839. 0.05 targets a weighted "
             "contribution roughly matching ssim_weight=0.2's own (~0.069 vs ~0.071). Re-check "
             "against scripts/verify_phase4.py's printed magnitudes before trusting this blindly "
             "on a different crop-size/architecture.",
    )
    p.add_argument(
        "--perceptual-feature-size", type=int, default=224,
        help="Resolution images are resized to before the frozen DINOv2 forward pass "
             "(must be divisible by 14; 224 is DINOv2's standard eval resolution).",
    )
    p.add_argument(
        "--init-from", type=str, default=None,
        help="Optional path to a Phase 3 Restormer checkpoint to warm-start model weights "
             "from (weights only -- optimizer/scaler/epoch state are NOT restored, this is "
             "not the same as --resume). Leave unset to train from scratch.",
    )
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a Phase 4 checkpoint to resume training from (full state).")
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

    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state"])
        print(f"Warm-started model weights from {args.init_from} "
              f"(epoch {init_ckpt['epoch']} of that run; optimizer/scaler state NOT restored)")

    # NOTE: .to(device) is required here -- unlike Phase 2/3, this loss now
    # holds real pretrained parameters (frozen DINOv2) that must live on the
    # training device. See ReconstructionLoss's docstring in src/losses.py.
    loss_fn = ReconstructionLoss(
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        perceptual_feature_size=args.perceptual_feature_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val_psnr = 0, float("-inf")
    if args.resume:
        start_epoch, best_val_psnr = load_checkpoint(Path(args.resume), model, optimizer, scaler, device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_val_psnr={best_val_psnr:.2f}")

    n_params = sum(pm.numel() for pm in model.parameters() if pm.requires_grad)
    print(
        f"Restormer params: {n_params:,} (unchanged from Phase 3) | device={device} | amp={use_amp} | "
        f"ssim_weight={args.ssim_weight} | perceptual_weight={args.perceptual_weight} | "
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
                    "failure mode diagnosed in Phase 2/3 (scripts/diagnose_val_nan.py). If "
                    "--amp was passed, drop it first. If it persists in fp32, try lowering "
                    "--perceptual-weight before touching crop-size/batch-size -- that isolates "
                    "whether the new DINOv2 term itself is the source of instability."
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
