"""
Phase 6 -- full benchmarking pass across all three models plus the raw
input, on the whole eval15 test set: PSNR, SSIM (full-reference, vs.
ground truth), LPIPS (full-reference, vs. ground truth), and NR-IQA/MANIQA
(no-reference). This is the formal three-way comparison table every
earlier phase deferred (Phase 3, 4, and 5's plan-log entries all
explicitly pointed here) and the source for the README results table and
qualitative writeup.

Models compared:
  - raw low-light input (the "do nothing" reference point)
  - Phase 2 baseline (U-Net)
  - Phase 3 (Restormer, Charbonnier + SSIM)
  - Phase 4 Run 2 (Restormer + DINOv2 perceptual loss, perceptual_weight=
    0.05 -- the kept run; Run 1 regressed and was already excluded from
    comparisons per the Phase 4 plan-log entry)
  - ground truth (NR-IQA only -- PSNR/SSIM/LPIPS against itself are
    trivial/undefined and not informative; same convention
    evaluate_nriqa_phase5.py used for its ground-truth column)

VRAM-conscious loading order, distinct from evaluate_nriqa_phase5.py and
visualize_predictions_phase4.py (which load every checkpoint up front,
fine there since neither also carries LPIPS *and* MANIQA concurrently):
this script loads restoration checkpoints ONE AT A TIME, running that
model's full eval15 pass before freeing it and moving to the next. LPIPS
(AlexNet, small) and MANIQA (Phase 5's NRIQAMetric, ~2.65 GB per
verify_phase5.py's own measurement) stay resident throughout, since every
model's predictions need both. Phase 5's plan-log entry already flagged
MANIQA + one Restormer checkpoint concurrently as ~3.5 GB estimated,
uncomfortably close to the GTX 1650's 4 GB ceiling -- this script never
holds more than [LPIPS + MANIQA + one restoration checkpoint] at once, by
design, rather than adding a second or third checkpoint on top of that
combination.

Usage:
    python scripts/benchmark_phase6.py --data-root data/LOLdataset
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lol_dataset import make_splits
from src.models.unet import UNetBaseline
from src.models.restormer import Restormer
from src.metrics import batch_psnr, batch_ssim
from src.lpips_metric import LPIPSMetric
from src.quality_metrics import NRIQAMetric

METRIC_ORDER = ["psnr", "ssim", "lpips", "nriqa"]


def load_model_if_exists(ckpt_path: Path, model_class, device, **model_kwargs):
    if not ckpt_path.exists():
        return None
    model = model_class(**model_kwargs)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["epoch"]


def evaluate_model(name, model, epoch, test_ds, device, lpips_metric, nriqa_metric):
    """Runs one model across the whole eval15 set, returns per-image rows
    plus the mean row. `model=None` means "raw input" (identity -- score
    the low-light input itself, no forward pass)."""
    rows = []
    sums = {k: 0.0 for k in METRIC_ORDER}

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        low = item["low"].unsqueeze(0).to(device)
        high = item["high"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = low if model is None else torch.clamp(model(low), 0, 1)
            values = {
                "psnr": batch_psnr(pred, high),
                "ssim": batch_ssim(pred, high),
                "lpips": lpips_metric.batch_mean(pred, high),
                "nriqa": nriqa_metric.batch_mean(pred),
            }

        for k in METRIC_ORDER:
            sums[k] += values[k]
        rows.append({"model": name, "filename": item["filename"], **values})

    n = len(test_ds)
    mean = {k: sums[k] / n for k in METRIC_ORDER}
    print(
        f"{name:<32} (ep {epoch if epoch is not None else '-'}): "
        f"PSNR {mean['psnr']:.2f} | SSIM {mean['ssim']:.3f} | "
        f"LPIPS {mean['lpips']:.4f} | MANIQA {mean['nriqa']:.3f}"
    )
    return rows, mean


def evaluate_ground_truth_nriqa(test_ds, device, nriqa_metric):
    """Ground truth only gets an NR-IQA number -- PSNR/SSIM/LPIPS against
    itself are trivially perfect (inf / 1.0 / 0.0) and not informative;
    same choice evaluate_nriqa_phase5.py made for its ground-truth column,
    applied here to the full four-metric table instead of NR-IQA alone."""
    total = 0.0
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        high = item["high"].unsqueeze(0).to(device)
        with torch.no_grad():
            total += nriqa_metric.batch_mean(high)
    mean_nriqa = total / len(test_ds)
    print(f"{'ground truth':<32} (NR-IQA only): MANIQA {mean_nriqa:.3f}")
    return mean_nriqa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--phase2-ckpt", type=str, default="runs/phase2_baseline/best.pt")
    p.add_argument("--phase3-ckpt", type=str, default="runs/phase3_restormer/best.pt")
    p.add_argument("--phase4-ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt",
                    help="Phase 4 Run 2 (perceptual_weight=0.05, the kept run).")
    p.add_argument("--output-dir", type=str, default="runs/phase6_benchmark")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _train_ds, _val_ds, test_ds = make_splits(args.data_root)
    print(f"eval15 test set: {len(test_ds)} images")
    print(f"Device: {device}\n")

    # Metrics loaded once, stay resident for the whole run -- see module
    # docstring re: VRAM budget.
    lpips_metric = LPIPSMetric(net="alex", device=device)
    nriqa_metric = NRIQAMetric(metric_name="maniqa", device=device)

    all_rows = []
    means = {}

    # --- raw input, no model ---
    rows, mean = evaluate_model(
        "input (low-light)", None, None, test_ds, device, lpips_metric, nriqa_metric
    )
    all_rows += rows
    means["input (low-light)"] = mean

    # --- Phase 2 baseline (U-Net), Phase 3 Restormer, Phase 4 Restormer:
    # loaded and freed one at a time -- see module docstring. ---
    checkpoints = [
        ("baseline (U-Net, Phase 2)", Path(args.phase2_ckpt), UNetBaseline, {}),
        ("Restormer (Phase 3)", Path(args.phase3_ckpt), Restormer, {"dim": 24}),
        ("Restormer+perceptual (Phase 4)", Path(args.phase4_ckpt), Restormer, {"dim": 24}),
    ]
    for name, ckpt_path, model_class, kwargs in checkpoints:
        loaded = load_model_if_exists(ckpt_path, model_class, device, **kwargs)
        if loaded is None:
            print(f"No checkpoint at {ckpt_path} -- omitting '{name}' from the benchmark.")
            continue
        model, epoch = loaded
        rows, mean = evaluate_model(name, model, epoch, test_ds, device, lpips_metric, nriqa_metric)
        all_rows += rows
        means[name] = mean
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- ground truth, NR-IQA only ---
    gt_nriqa = evaluate_ground_truth_nriqa(test_ds, device, nriqa_metric)

    # --- Save per-image CSV (all metrics, all models) ---
    per_image_path = output_dir / "phase6_per_image.csv"
    with open(per_image_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "filename"] + METRIC_ORDER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved per-image results to {per_image_path}")

    # --- Save summary CSV + a ready-to-paste markdown table for the README ---
    summary_path = output_dir / "phase6_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "psnr_db", "ssim", "lpips", "maniqa"])
        for name, mean in means.items():
            writer.writerow([name, f"{mean['psnr']:.2f}", f"{mean['ssim']:.3f}",
                              f"{mean['lpips']:.4f}", f"{mean['nriqa']:.3f}"])
        writer.writerow(["ground truth", "", "", "", f"{gt_nriqa:.3f}"])
    print(f"Saved summary results to {summary_path}")

    md_path = output_dir / "phase6_results_table.md"
    with open(md_path, "w") as f:
        f.write("| Model | PSNR (dB) | SSIM | LPIPS \u2193 | MANIQA \u2191 |\n")
        f.write("|---|---|---|---|---|\n")
        for name, mean in means.items():
            f.write(f"| {name} | {mean['psnr']:.2f} | {mean['ssim']:.3f} | "
                     f"{mean['lpips']:.4f} | {mean['nriqa']:.3f} |\n")
        f.write(f"| ground truth | \u2014 | \u2014 | \u2014 | {gt_nriqa:.3f} |\n")
    print(f"Saved README-ready markdown table to {md_path}")

    print(
        "\nNote for the README/plan-log writeup: DINOv2 (Phase 4's training loss) "
        "and LPIPS (this table's eval metric) are architecturally and "
        "methodologically distinct perceptual constructs -- see "
        "src/lpips_metric.py's module docstring before describing them as "
        '"the same perceptual check, twice."'
    )


if __name__ == "__main__":
    main()
