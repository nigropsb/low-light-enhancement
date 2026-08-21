"""
Qualitative check (Phase 4): visual before/after grids on the real eval15
test set, extending scripts/visualize_predictions.py (Phase 3) to compare
across all three Restormer checkpoints trained so far -- Phase 3
(Charbonnier + SSIM) and the two Phase 4 perceptual-loss runs
(perceptual_weight=0.1 and 0.05) -- against the same input/ground-truth
pairs, side by side.

Why this script exists, not just a re-run of visualize_predictions.py:
Phase 4's numeric results were a genuine mixed verdict (see project plan --
Run 2 edged past Phase 3 on PSNR by +0.12 dB but fell short on SSIM, 0.826
vs. 0.839). Perceptual losses are specifically expected to trade some
pixel/structural-similarity score for perceptual sharpness or texture that
PSNR/SSIM don't reward -- so the numbers alone can't settle whether the
perceptual term was worth adding. This script is the tiebreaker: look at
the actual images.

All three checkpoints share the same Restormer architecture (dim=24), so
loading them is just three state_dicts into three otherwise-identical
model instances -- no architecture-specific handling needed per run.

Usage:
    python scripts/visualize_predictions_phase4.py --data-root data/LOLdataset
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lol_dataset import make_splits
from src.models.restormer import Restormer
from src.metrics import batch_psnr, batch_ssim


def load_model_if_exists(ckpt_path: Path, device, dim: int = 24):
    if not ckpt_path.exists():
        return None
    model = Restormer(dim=dim)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["epoch"]


def to_numpy_img(t: torch.Tensor):
    """CHW float tensor in [0, 1] -> HWC array for imshow."""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--phase3-ckpt", type=str, default="runs/phase3_restormer/best.pt")
    p.add_argument("--run1-ckpt", type=str, default="runs/phase4_perceptual/best.pt",
                    help="Phase 4 Run 1 (perceptual_weight=0.1 -- SSIM regressed, see project plan)")
    p.add_argument("--run2-ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt",
                    help="Phase 4 Run 2 (perceptual_weight=0.05, rebalanced)")
    p.add_argument("--output-dir", type=str, default="runs/phase4_perceptual_run2/qualitative")
    p.add_argument("--num-samples", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_ds, _val_ds, test_ds = make_splits(args.data_root)
    n = min(args.num_samples, len(test_ds))
    # Evenly spaced indices across eval15 (not just the first N) for a more
    # representative spread of scenes -- same choice as Phase 3's script.
    indices = [round(i * (len(test_ds) - 1) / max(n - 1, 1)) for i in range(n)]

    # --- Load whichever checkpoints are actually present. Phase 3 is
    # treated as required (there's no comparison without it); the two
    # Phase 4 runs degrade gracefully to "not shown" if a checkpoint is
    # missing, so this script still works mid-experiment (e.g. before Run 2
    # exists yet) rather than hard-failing. ---
    phase3 = load_model_if_exists(Path(args.phase3_ckpt), device)
    if phase3 is None:
        raise FileNotFoundError(f"No Phase 3 checkpoint at {args.phase3_ckpt} -- required as the baseline.")
    phase3, phase3_epoch = phase3

    run1 = load_model_if_exists(Path(args.run1_ckpt), device)
    if run1 is None:
        print(f"No Run 1 checkpoint at {args.run1_ckpt} -- omitting from the comparison.")
        run1_epoch = None
    else:
        run1, run1_epoch = run1

    run2 = load_model_if_exists(Path(args.run2_ckpt), device)
    if run2 is None:
        print(f"No Run 2 checkpoint at {args.run2_ckpt} -- omitting from the comparison.")
        run2_epoch = None
    else:
        run2, run2_epoch = run2

    # Column order: input, Phase 3, Run 1 (if present), Run 2 (if present), ground truth.
    models = [("Phase 3", phase3, phase3_epoch)]
    if run1 is not None:
        models.append(("Run 1 (w=0.1)", run1, run1_epoch))
    if run2 is not None:
        models.append(("Run 2 (w=0.05)", run2, run2_epoch))

    n_cols = 2 + len(models)  # input + each model + ground truth
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes[None, :]

    header = f"{'file':<14}" + "".join(f"{name:>18}" for name, _, _ in models)
    print(header)

    # Running averages, to complement the per-image printout with the same
    # kind of summary number the training log already reports -- on just
    # this small sample, not a substitute for the full val-set metrics.
    running = {name: {"psnr": 0.0, "ssim": 0.0} for name, _, _ in models}

    for row, idx in enumerate(indices):
        item = test_ds[idx]
        low = item["low"].unsqueeze(0).to(device)
        high = item["high"].unsqueeze(0).to(device)

        col = 0
        axes[row, col].imshow(to_numpy_img(low[0]))
        axes[row, col].set_title(f"{item['filename']}\ninput (low-light)")
        col += 1

        row_summary = f"{item['filename']:<14}"
        with torch.no_grad():
            for name, model, epoch in models:
                pred = torch.clamp(model(low), 0, 1)
                psnr_v = batch_psnr(pred, high)
                ssim_v = batch_ssim(pred, high)
                running[name]["psnr"] += psnr_v
                running[name]["ssim"] += ssim_v

                axes[row, col].imshow(to_numpy_img(pred[0]))
                axes[row, col].set_title(f"{name} (ep {epoch})\nPSNR {psnr_v:.1f} / SSIM {ssim_v:.2f}")
                col += 1
                row_summary += f"{f'{psnr_v:.1f}/{ssim_v:.2f}':>18}"

        axes[row, col].imshow(to_numpy_img(high[0]))
        axes[row, col].set_title("ground truth")

        print(row_summary)

        for c in range(n_cols):
            axes[row, c].axis("off")

    mean_row = f"{'mean':<14}"
    for name, _, _ in models:
        mean_psnr = running[name]["psnr"] / n
        mean_ssim = running[name]["ssim"] / n
        mean_row += f"{f'{mean_psnr:.1f}/{mean_ssim:.2f}':>18}"
    print(f"\n{mean_row}  (this small sample only -- not the full 15-image eval15 average)")

    plt.tight_layout()
    out_path = output_dir / "phase4_comparison_grid.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved qualitative comparison grid ({n} images, {len(models)} models) to {out_path}")


if __name__ == "__main__":
    main()
