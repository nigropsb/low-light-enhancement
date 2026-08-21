"""
Quick qualitative check (Phase 3): visual before/after grids on the real
eval15 test set, not synthetic tensors.

Purpose: PSNR/SSIM are averages that can hide real artifacts (color casts,
blockiness, hallucinated detail) that only show up by looking at an actual
image. Cheap insurance to run once before building Phase 4 (DINOv2 perceptual
loss) on top of this architecture -- and the saved grid doubles as README-
ready before/after material for the eventual Phase 9 documentation pass.

This intentionally does NOT duplicate Phase 6 (full PSNR/SSIM/LPIPS/NR-IQA
benchmarking across all three models on the whole test set) -- it's a small,
fast spot-check on a handful of images, not the final results table.

Usage:
    python scripts/visualize_predictions.py --data-root data/LOLdataset
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


def load_model_if_exists(model: torch.nn.Module, ckpt_path: Path, device):
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def to_numpy_img(t: torch.Tensor):
    """CHW float tensor in [0, 1] -> HWC array for imshow."""
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--restormer-ckpt", type=str, default="runs/phase3_restormer/best.pt")
    p.add_argument("--baseline-ckpt", type=str, default="runs/phase2_baseline/best.pt")
    p.add_argument("--output-dir", type=str, default="runs/phase3_restormer/qualitative")
    p.add_argument("--num-samples", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_ds, _val_ds, test_ds = make_splits(args.data_root)
    n = min(args.num_samples, len(test_ds))
    # Evenly spaced indices across eval15 (not just the first N) for a more
    # representative spread of scenes.
    indices = [round(i * (len(test_ds) - 1) / max(n - 1, 1)) for i in range(n)]

    restormer = load_model_if_exists(Restormer(), Path(args.restormer_ckpt), device)
    if restormer is None:
        raise FileNotFoundError(f"No Restormer checkpoint at {args.restormer_ckpt}")

    baseline = load_model_if_exists(UNetBaseline(), Path(args.baseline_ckpt), device)
    if baseline is None:
        print(f"No baseline checkpoint at {args.baseline_ckpt} -- showing Restormer only.")

    n_cols = 4 if baseline is not None else 3
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
    if n == 1:
        axes = axes[None, :]

    print(f"{'file':<14} {'baseline':>16} {'restormer':>16}")
    for row, idx in enumerate(indices):
        item = test_ds[idx]
        low = item["low"].unsqueeze(0).to(device)
        high = item["high"].unsqueeze(0).to(device)

        with torch.no_grad():
            restormer_pred = torch.clamp(restormer(low), 0, 1)
            r_psnr = batch_psnr(restormer_pred, high)
            r_ssim = batch_ssim(restormer_pred, high)

            col = 0
            axes[row, col].imshow(to_numpy_img(low[0]))
            axes[row, col].set_title(f"{item['filename']}\ninput (low-light)")
            col += 1

            b_txt = "n/a"
            if baseline is not None:
                baseline_pred = torch.clamp(baseline(low), 0, 1)
                b_psnr = batch_psnr(baseline_pred, high)
                b_ssim = batch_ssim(baseline_pred, high)
                b_txt = f"{b_psnr:.1f}/{b_ssim:.2f}"
                axes[row, col].imshow(to_numpy_img(baseline_pred[0]))
                axes[row, col].set_title(f"U-Net baseline\nPSNR {b_psnr:.1f} / SSIM {b_ssim:.2f}")
                col += 1

            axes[row, col].imshow(to_numpy_img(restormer_pred[0]))
            axes[row, col].set_title(f"Restormer\nPSNR {r_psnr:.1f} / SSIM {r_ssim:.2f}")
            col += 1

            axes[row, col].imshow(to_numpy_img(high[0]))
            axes[row, col].set_title("ground truth")

        print(f"{item['filename']:<14} {b_txt:>16} {f'{r_psnr:.1f}/{r_ssim:.2f}':>16}")

        for c in range(n_cols):
            axes[row, c].axis("off")

    plt.tight_layout()
    out_path = output_dir / "before_after_grid.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved qualitative grid ({n} images) to {out_path}")


if __name__ == "__main__":
    main()
