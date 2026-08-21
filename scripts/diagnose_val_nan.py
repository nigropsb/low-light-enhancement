"""
One-off diagnostic for the val_loss=nan issue.

train_loss is clean and decreasing every epoch; val_loss is nan on *every*
epoch. Since the val split isn't shuffled, the same sample order repeats
each epoch -- that points at something specific to one validation sample or
to the eval-mode data path, rather than a generic fp16/AMP instability
(which would be expected to eventually show up in training too, given train
gets far more exposure: 437 images vs. 48).

This script checks, in order:
  1. Are any raw validation tensors (before the model touches them) already
     NaN/Inf? -> points at LOLDataset/PairedTransform in eval mode.
  2. If inputs are clean, does the model produce NaN/Inf on any validation
     sample, under fp16 (autocast) vs fp32? -> isolates AMP as the trigger,
     and tells you exactly which file.
  3. If single images are clean (as they were), bisect BATCHING itself:
     manual batch-of-4 (no DataLoader) -> DataLoader with 0 workers ->
     DataLoader with real workers. Pinpoints whether it's a batch-size-
     dependent numerical effect vs. the DataLoader/worker processes, and at
     which stage (input, model output, or loss/metric) it first appears.

Usage:
    python scripts/diagnose_val_nan.py --data-root data/LOLdataset \
        --checkpoint runs/phase2_baseline/last.pt
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unet import UNetBaseline


def nan_inf_count(t: torch.Tensor) -> tuple[int, int]:
    return torch.isnan(t).sum().item(), torch.isinf(t).sum().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/LOLdataset")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--checkpoint", default="runs/phase2_baseline/last.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.data.lol_dataset import make_splits
    _train_ds, val_ds, _test_ds = make_splits(args.data_root, crop_size=args.crop_size)
    print(f"val_ds has {len(val_ds)} samples\n")

    # --- Step 1: raw data, before the model sees it at all ---
    print("--- Step 1: checking raw validation tensors for NaN/Inf ---")
    bad_inputs = []
    for i in range(len(val_ds)):
        item = val_ds[i]
        low, high, fname = item["low"], item["high"], item["filename"]
        n_nan_low, n_inf_low = nan_inf_count(low)
        n_nan_high, n_inf_high = nan_inf_count(high)
        if n_nan_low or n_inf_low or n_nan_high or n_inf_high:
            bad_inputs.append(fname)
            print(f"  [{i}] {fname}: low nan={n_nan_low} inf={n_inf_low} "
                  f"(range [{low.min():.3f}, {low.max():.3f}]) | "
                  f"high nan={n_nan_high} inf={n_inf_high}")

    if bad_inputs:
        print(f"\n-> {len(bad_inputs)} validation sample(s) are corrupt BEFORE the model runs: "
              f"{bad_inputs}")
        print("   This is a LOLDataset/PairedTransform (eval-mode) bug, not a model issue.")
        return
    print("-> all validation inputs clean. Checking the model forward pass next.\n")

    # --- Step 2: model forward, fp16 vs fp32, to isolate AMP as the trigger ---
    model = UNetBaseline().to(device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded weights from {ckpt_path} (epoch {ckpt['epoch']})\n")
    else:
        print(f"No checkpoint at {ckpt_path}; using random init weights.\n")
    model.eval()

    for use_amp, label in [(True, "fp16 (autocast, matches training)"), (False, "fp32")]:
        print(f"--- Step 2: forward pass under {label} ---")
        first_bad = None
        with torch.no_grad():
            for i in range(len(val_ds)):
                item = val_ds[i]
                low = item["low"].unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    pred = model(low)
                n_nan, n_inf = nan_inf_count(pred)
                if n_nan or n_inf:
                    first_bad = (i, item["filename"], n_nan, n_inf, low.min().item(), low.max().item())
                    break
        if first_bad:
            i, fname, n_nan, n_inf, lo_min, lo_max = first_bad
            print(f"  -> FIRST bad output at index {i} ({fname}): nan={n_nan} inf={n_inf} "
                  f"| input range [{lo_min:.3f}, {lo_max:.3f}]")
        else:
            print(f"  -> all {len(val_ds)} outputs clean under {label}")
        print()

    # --- Step 3: single images were clean -- now bisect batching itself.
    # Three variants, isolating one axis at a time: manual batch-of-4 (no
    # DataLoader at all) -> DataLoader batch-of-4 with 0 workers -> DataLoader
    # batch-of-4 with the real worker count. If (a) is already bad, it's a
    # batch-size-dependent numerical effect (e.g. cuDNN picking a different,
    # less stable fp16 conv algorithm for batch>1). If (a) is clean but (b)/(c)
    # break, it's the DataLoader collate or worker processes, not the model. ---
    from src.losses import ReconstructionLoss
    from src.metrics import batch_psnr, batch_ssim
    loss_fn = ReconstructionLoss()
    use_amp = device.type == "cuda"

    def run_batch(low, high, fnames, tag, use_amp):
        low, high = low.to(device), high.to(device)
        n_nan_low, n_inf_low = nan_inf_count(low)
        n_nan_high, n_inf_high = nan_inf_count(high)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(low)
            loss = loss_fn(pred, high)
        n_nan_pred, n_inf_pred = nan_inf_count(pred)
        pred_c = torch.clamp(pred, 0.0, 1.0)
        psnr_v = batch_psnr(pred_c.float(), high.float())
        ssim_v = batch_ssim(pred_c.float(), high.float())
        status = "OK"
        import math
        if n_nan_low or n_inf_low or n_nan_high or n_inf_high:
            status = "BAD INPUT"
        elif n_nan_pred or n_inf_pred:
            status = "BAD PRED"
        elif math.isnan(loss.item()) or math.isnan(psnr_v) or math.isnan(ssim_v):
            status = "BAD LOSS/METRIC"
        print(f"  [{tag}] {status} | files={fnames} | nan_low={n_nan_low} nan_high={n_nan_high} "
              f"nan_pred={n_nan_pred} inf_pred={n_inf_pred} loss={loss.item():.4f} "
              f"psnr={psnr_v:.2f} ssim={ssim_v:.3f}")
        return status == "OK"

    print("--- Step 3a: manual batch-of-4, fp16 (no DataLoader at all) ---")
    all_ok = True
    for start in range(0, len(val_ds), 4):
        items = [val_ds[i] for i in range(start, min(start + 4, len(val_ds)))]
        low = torch.stack([it["low"] for it in items])
        high = torch.stack([it["high"] for it in items])
        fnames = [it["filename"] for it in items]
        ok = run_batch(low, high, fnames, f"manual batch @ {start}", use_amp=True)
        all_ok = all_ok and ok
    print(f"  -> {'all clean' if all_ok else 'FOUND A BAD BATCH ABOVE'}\n")

    print("--- Step 3c: SAME manual batch-of-4, but fp32 (isolates precision vs. batch size) ---")
    all_ok = True
    for start in range(0, len(val_ds), 4):
        items = [val_ds[i] for i in range(start, min(start + 4, len(val_ds)))]
        low = torch.stack([it["low"] for it in items])
        high = torch.stack([it["high"] for it in items])
        fnames = [it["filename"] for it in items]
        ok = run_batch(low, high, fnames, f"manual batch @ {start} (fp32)", use_amp=False)
        all_ok = all_ok and ok
    print(f"  -> {'all clean' if all_ok else 'FOUND A BAD BATCH ABOVE'}\n")

    from torch.utils.data import DataLoader
    for num_workers in [0, 4]:
        print(f"--- Step 3b: real DataLoader, batch_size=4, num_workers={num_workers}, fp16 ---")
        loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=num_workers)
        all_ok = True
        for batch in loader:
            ok = run_batch(batch["low"], batch["high"], batch["filename"],
                            f"DataLoader(nw={num_workers})", use_amp=True)
            all_ok = all_ok and ok
        print(f"  -> {'all clean' if all_ok else 'FOUND A BAD BATCH ABOVE'}\n")


if __name__ == "__main__":
    main()
