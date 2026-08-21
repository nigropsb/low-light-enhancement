"""
Phase 8 -- stages the Phase 7 ONNX artifact + a small provenance sidecar
into the Docker build context, so `docker build` doesn't need access to
runs/phase7_deploy/ (which typically sits outside what you'd want in a
build context anyway).

Deliberately re-exports from the checkpoint rather than just copying
runs/phase7_deploy/restormer_phase4_fp32.onnx as-is: this keeps a single
source of truth (the checkpoint) and guarantees the epoch sidecar always
matches the exported weights, rather than trusting two independently-
produced files to stay in sync.

Companion to scripts/export_for_serving_pytorch.py, which stages the other
comparison backend's artifact from the same checkpoint.

Usage:
    python scripts/export_for_serving_onnx.py \
        --ckpt runs/phase4_perceptual_run2/best.pt \
        --output-dir deploy/model
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.onnx_export import load_restormer_for_export, export_onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt")
    p.add_argument("--output-dir", type=str, default="deploy/model")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    wrapper, epoch = load_restormer_for_export(ckpt_path, "cpu")
    onnx_path = output_dir / "restormer_phase4_fp32.onnx"
    export_onnx(wrapper, onnx_path, dummy_hw=(400, 600), device="cpu")

    epoch_path = onnx_path.with_suffix(".epoch")
    epoch_path.write_text(str(epoch))

    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"Staged {onnx_path} ({size_mb:.2f} MB, epoch {epoch}) for the Docker build.")
    print(f"Next: docker build -f Dockerfile.onnx -t low-light-enhancement:onnx . "
          f"(from the repo root, with {output_dir} present)")


if __name__ == "__main__":
    main()
