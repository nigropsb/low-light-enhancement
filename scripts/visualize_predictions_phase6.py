"""
Phase 6 -- final qualitative before/after grid for the README: input vs.
each of the three trained models vs. ground truth, side by side. Extends
scripts/visualize_predictions_phase4.py (which compared Phase 3 against
both Phase 4 runs) by swapping in the Phase 2 U-Net baseline as the
leftmost model column and dropping Run 1 (already excluded from
comparisons -- see the Phase 4 plan-log entry, its best.pt is epoch 24
where val_psnr peaked early and never recovered).

Column order: input -> baseline (U-Net) -> Restormer (Phase 3) ->
Restormer+perceptual (Phase 4 Run 2) -> ground truth. This mirrors
benchmark_phase6.py's model list and ordering exactly, so the qualitative
grid and the numeric results table tell a consistent story read
side by side.

Titles annotate PSNR/SSIM only (not LPIPS/MANIQA) -- same choice
visualize_predictions_phase4.py made, to keep each subplot title readable.
The full four-metric numbers live in benchmark_phase6.py's output
(runs/phase6_benchmark/phase6_results_table.md); this script is a
lightweight qualitative sanity check on a small sample, not a second copy
of the full-set numeric benchmark, and deliberately doesn't load
LPIPS/MANIQA to stay that way.

Usage:
    python scripts/visualize_predictions_phase6.py --data-root data/LOLdataset
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lol_dataset import make_splits
from src.models.unet import UNetBaseline
from src.models.restormer import Restormer
from src.metrics import batch_psnr, batch_ssim


def load_model_if_exists(ckpt_path: Path, model_class, device, **model_kwargs):
    if not ckpt_path.exists():
        return None
    model = model_class(**model_kwargs)
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
    p.add_argument("--phase2-ckpt", type=str, default="runs/phase2_baseline/best.pt")
    p.add_argument("--phase3-ckpt", type=str, default="runs/phase3_restormer/best.pt")
    p.add_argument("--phase4-ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt",
                    help="Phase 4 Run 2 (perceptual_weight=0.05, the kept run).")
    p.add_argument("--output-dir", type=str, default="runs/phase6_benchmark")
    p.add_argument("--num-samples", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_ds, _val_ds, test_ds = make_splits(args.data_root)
    n = min(args.num_samples, len(test_ds))
    # Evenly spaced indices across eval15 -- same choice as Phase 3/4's scripts.
    indices = [round(i * (len(test_ds) - 1) / max(n - 1, 1)) for i in range(n)]

    checkpoints = [
        ("baseline (U-Net)", Path(args.phase2_ckpt), UNetBaseline, {}),
        ("Restormer", Path(args.phase3_ckpt), Restormer, {"dim": 24}),
        ("Restormer+perceptual", Path(args.phase4_ckpt), Restormer, {"dim": 24}),
    ]
    models = []
    for name, ckpt_path, model_class, kwargs in checkpoints:
        loaded = load_model_if_exists(ckpt_path, model_class, device, **kwargs)
        if loaded is None:
            print(f"No checkpoint at {ckpt_path} -- omitting '{name}' from the grid.")
            continue
        model, epoch = loaded
        models.append((name, model, epoch))

    if not models:
        raise FileNotFoundError("No checkpoints found for any of Phase 2/3/4 -- nothing to plot.")

    n_cols = 2 + len(models)  # input + each model + ground truth
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes[None, :]

    header = f"{'file':<14}" + "".join(f"{name:>22}" for name, _, _ in models)
    print(header)

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
                row_summary += f"{f'{psnr_v:.1f}/{ssim_v:.2f}':>22}"

        axes[row, col].imshow(to_numpy_img(high[0]))
        axes[row, col].set_title("ground truth")

        print(row_summary)

        for c in range(n_cols):
            axes[row, c].axis("off")

    mean_row = f"{'mean':<14}"
    for name, _, _ in models:
        mean_psnr = running[name]["psnr"] / n
        mean_ssim = running[name]["ssim"] / n
        mean_row += f"{f'{mean_psnr:.1f}/{mean_ssim:.2f}':>22}"
    print(f"\n{mean_row}  (this small sample only -- see phase6_results_table.md for the full eval15 average)")

    plt.tight_layout()
    out_path = output_dir / "phase6_comparison_grid.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved qualitative comparison grid ({n} images, {len(models)} models) to {out_path}")


if __name__ == "__main__":
    main()
