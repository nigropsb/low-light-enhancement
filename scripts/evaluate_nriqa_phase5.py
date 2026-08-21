"""
Phase 5 real-data check -- MANIQA scores across the actual eval15 test set.

Closes out Phase 5 the same way every earlier phase closed out: a synthetic
smoke test (verify_phase5.py) first, then a real-data run to confirm the new
component behaves sensibly outside synthetic tensors, before calling the
phase done. This is deliberately NOT the formal three-way benchmark table
(that's Phase 6, alongside LPIPS, on the full protocol) -- this script's job
is narrower: does MANIQA produce plausible, non-degenerate scores on real
low-light photos and real model output, checked against the one directional
prior we can state with actual confidence up front (see the interpretation
note below).

Reuses the same checkpoint-loading pattern as
scripts/visualize_predictions_phase4.py (load_model_if_exists) --
duplicated rather than imported, matching that script's own choice to stay
a self-contained, one-off diagnostic rather than something other modules
import from.

Interpretation note, read before trusting the printed numbers: MANIQA is a
*general* photographic-quality/aesthetic prior, not an exposure or
brightness detector specifically. Raw eval15 low-light inputs are expected
to score worse than ground truth on *some* combination of noise/contrast/
detail visibility that MANIQA's training data teaches it to associate with
"bad photo" -- but that association is empirical, not guaranteed, and
that's exactly why this project isn't relying on PSNR/SSIM alone: they
can't capture this dimension of quality at all, for better or worse. If
low-light < ground-truth doesn't hold cleanly here, that's a real, useful
finding about MANIQA's sensitivity on this dataset, not necessarily a bug
in this script -- note it in the Phase 5 plan-log entry either way rather
than silently assuming it must be a mistake.

Usage:
    python scripts/evaluate_nriqa_phase5.py --data-root data/LOLdataset
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lol_dataset import make_splits
from src.models.restormer import Restormer
from src.quality_metrics import NRIQAMetric


def load_model_if_exists(ckpt_path: Path, device, dim: int = 24):
    if not ckpt_path.exists():
        return None
    model = Restormer(dim=dim)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["epoch"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--phase3-ckpt", type=str, default="runs/phase3_restormer/best.pt")
    p.add_argument("--phase4-ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt",
                    help="Phase 4 Run 2 (perceptual_weight=0.05, the one kept for comparisons).")
    p.add_argument("--metric", type=str, default="maniqa")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = NRIQAMetric(metric_name=args.metric, device=device)
    print(f"Metric: {metric.metric_name} (lower_better={metric.lower_better})\n")

    _train_ds, _val_ds, test_ds = make_splits(args.data_root)
    print(f"eval15 test set: {len(test_ds)} images\n")

    phase3 = load_model_if_exists(Path(args.phase3_ckpt), device)
    if phase3 is None:
        raise FileNotFoundError(f"No Phase 3 checkpoint at {args.phase3_ckpt} -- required as the baseline.")
    phase3, phase3_epoch = phase3

    phase4 = load_model_if_exists(Path(args.phase4_ckpt), device)
    if phase4 is None:
        print(f"No Phase 4 checkpoint at {args.phase4_ckpt} -- omitting from the comparison.")
        phase4_epoch = None
    else:
        phase4, phase4_epoch = phase4

    columns = ["input (low-light)", f"Phase 3 (ep {phase3_epoch})"]
    if phase4 is not None:
        columns.append(f"Phase 4 (ep {phase4_epoch})")
    columns.append("ground truth")

    header = f"{'file':<14}" + "".join(f"{c:>22}" for c in columns)
    print(header)

    running = {c: 0.0 for c in columns}

    for idx in range(len(test_ds)):
        item = test_ds[idx]
        low = item["low"].unsqueeze(0).to(device)
        high = item["high"].unsqueeze(0).to(device)

        row_scores = {}
        with torch.no_grad():
            row_scores["input (low-light)"] = metric.batch_mean(low)

            pred3 = torch.clamp(phase3(low), 0, 1)
            row_scores[f"Phase 3 (ep {phase3_epoch})"] = metric.batch_mean(pred3)

            if phase4 is not None:
                pred4 = torch.clamp(phase4(low), 0, 1)
                row_scores[f"Phase 4 (ep {phase4_epoch})"] = metric.batch_mean(pred4)

            row_scores["ground truth"] = metric.batch_mean(high)

        row = f"{item['filename']:<14}"
        for c in columns:
            v = row_scores[c]
            running[c] += v
            row += f"{v:>22.4f}"
        print(row)

    n = len(test_ds)
    mean_row = f"{'mean':<14}"
    for c in columns:
        mean_row += f"{running[c] / n:>22.4f}"
    print(f"\n{mean_row}")
    print(
        f"\n({'lower' if metric.lower_better else 'higher'}-is-better) -- see this script's "
        "docstring before treating the low-light-vs-ground-truth ordering as a given."
    )


if __name__ == "__main__":
    main()
