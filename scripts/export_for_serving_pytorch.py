"""
Phase 8 -- stages a lightweight, inference-only checkpoint for the PyTorch
serving backend (src/serve_pytorch.py). Companion to
scripts/export_for_serving_onnx.py, same "single source of truth is the raw
checkpoint" philosophy, but for the PyTorch-CPU comparison image instead.

Strips the AdamW optimizer state and GradScaler state before saving -- the
full training checkpoint (37.68 MB) is ~2.8x the weights alone precisely
because it carries `exp_avg`/`exp_avg_sq` momentum buffers (see Phase 7's
own size analysis in the plan log) that inference never reads. Re-saving
weights + epoch only should land close to ONNX fp32's 13.29 MB, which is
itself a useful cross-check: if this comes out meaningfully larger than that,
something beyond "optimizer state" is being carried along and is worth
investigating before trusting the size comparison in benchmark_phase8.py.

Usage:
    python scripts/export_for_serving_pytorch.py \
        --ckpt runs/phase4_perceptual_run2/best.pt \
        --output-dir deploy/model
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt")
    p.add_argument("--output-dir", type=str, default="deploy/model")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stripped = {"model_state": full_ckpt["model_state"], "epoch": full_ckpt["epoch"]}

    out_path = output_dir / "restormer_phase4_inference.pt"
    torch.save(stripped, out_path)

    full_mb = ckpt_path.stat().st_size / (1024 ** 2)
    stripped_mb = out_path.stat().st_size / (1024 ** 2)
    print(
        f"Staged {out_path} ({stripped_mb:.2f} MB, epoch {stripped['epoch']}) "
        f"for the Docker build. Full training checkpoint was {full_mb:.2f} MB "
        f"({100 * (1 - stripped_mb / full_mb):.0f}% reduction from stripping "
        "optimizer/scaler state)."
    )
    print("Next: docker build -f Dockerfile.pytorch -t low-light-enhancement:pytorch . "
          "(from the repo root, with deploy/model/ present)")


if __name__ == "__main__":
    main()
