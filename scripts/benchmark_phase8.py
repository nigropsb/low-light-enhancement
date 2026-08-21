"""
Phase 8 -- builds both serving containers (ONNX Runtime CPU and PyTorch CPU),
and measures the three things the Dockerfile.onnx/Dockerfile.pytorch choice
actually rests on: image size, cold-start time, and per-request latency
through the real HTTP layer (not the bare model-call numbers Phase 7
measured -- request parsing, image decode/encode, and the FastAPI/uvicorn
stack all add overhead neither of those numbers included).

This script exists because Phase 7's ~18%-faster-on-CPU PyTorch number was
being used to justify an ONNX choice based on an UNVERIFIED container-size
assumption -- this replaces that assumption with a real measurement, same
"measure, don't assume" principle as Phase 4's perceptual-loss-weight check.

Requires:
  - Docker installed and running (this script shells out to the `docker` CLI)
  - deploy/model/restormer_phase4_fp32.onnx + .epoch
        (from scripts/export_for_serving_onnx.py)
  - deploy/model/restormer_phase4_inference.pt
        (from scripts/export_for_serving_pytorch.py)
  - requests (pip install requests if not already present)

Usage:
    python scripts/benchmark_phase8.py \
        --low-image data/LOLdataset/eval15/low/1.png \
        --high-image data/LOLdataset/eval15/high/1.png
"""
import argparse
import csv
import io
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import batch_psnr, batch_ssim

BACKENDS = [
    {"name": "onnx", "dockerfile": "Dockerfile.onnx", "tag": "low-light-enhancement:onnx", "port": 8081},
    {"name": "pytorch", "dockerfile": "Dockerfile.pytorch", "tag": "low-light-enhancement:pytorch", "port": 8082},
]


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _docker_build(dockerfile: str, tag: str) -> None:
    print(f"  building {tag} from {dockerfile} ...")
    result = subprocess.run(
        ["docker", "build", "-f", dockerfile, "-t", tag, "."],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # Bug fix: check=True raises CalledProcessError with stdout/stderr
        # attached, but main()'s except-and-print(e) only shows the repr,
        # not the actual build log -- exactly the "silent failure" pattern
        # this project's own verify_*.py scripts are built to avoid. Print
        # the real docker output before raising so a build failure is
        # diagnosable from this script's own output, not a second manual run.
        print(f"  --- docker build stdout ---\n{result.stdout}")
        print(f"  --- docker build stderr ---\n{result.stderr}")
        raise RuntimeError(f"docker build failed for {tag} (exit {result.returncode})")


def _image_size_mb(tag: str) -> float:
    out = subprocess.run(
        ["docker", "image", "inspect", tag, "--format={{.Size}}"],
        check=True, capture_output=True, text=True,
    )
    return int(out.stdout.strip()) / (1024 ** 2)


def _wait_for_health(url: str, timeout_s: float = 60.0) -> float:
    """Returns seconds elapsed until /health responds 200 -- a cold-start proxy."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return time.perf_counter() - t0
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"{url}/health did not return 200 within {timeout_s}s")


def _run_one_backend(backend: dict, low_path: Path, high_path: Path, n_requests: int) -> dict:
    tag, port = backend["tag"], backend["port"]
    url = f"http://localhost:{port}"

    print(f"\n=== {backend['name']} ===")
    _docker_build(backend["dockerfile"], tag)
    size_mb = _image_size_mb(tag)
    print(f"  image size: {size_mb:.1f} MB")

    container_name = f"phase8-bench-{backend['name']}"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    proc = subprocess.Popen(
        ["docker", "run", "--rm", "--name", container_name, "-p", f"{port}:8080", tag]
    )
    try:
        cold_start_s = _wait_for_health(url)
        print(f"  cold start: {cold_start_s:.2f} s")

        low_img = Image.open(low_path)
        buf = io.BytesIO()
        low_img.convert("RGB").save(buf, format="PNG")
        raw = buf.getvalue()

        round_trip_ms = []
        server_inference_ms = []
        last_out_img = None
        for _ in range(n_requests):
            t0 = time.perf_counter()
            r = requests.post(
                f"{url}/enhance",
                files={"image": ("input.png", io.BytesIO(raw), "image/png")},
                timeout=60,
            )
            round_trip_ms.append((time.perf_counter() - t0) * 1000.0)
            r.raise_for_status()
            server_inference_ms.append(float(r.headers.get("X-Inference-Ms", "nan")))
            last_out_img = Image.open(io.BytesIO(r.content)).convert("RGB")

        high_img = Image.open(high_path)
        pred_t, high_t = _to_tensor(last_out_img), _to_tensor(high_img)
        psnr = batch_psnr(pred_t, high_t)
        ssim = batch_ssim(pred_t, high_t)

        return {
            "backend": backend["name"],
            "image_size_mb": round(size_mb, 1),
            "cold_start_s": round(cold_start_s, 2),
            "mean_round_trip_ms": round(float(np.mean(round_trip_ms)), 1),
            "mean_server_inference_ms": round(float(np.mean(server_inference_ms)), 1),
            "psnr": round(float(psnr), 2),
            "ssim": round(float(ssim), 3),
        }
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True)
        proc.wait(timeout=15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--low-image", type=str, required=True)
    p.add_argument("--high-image", type=str, required=True)
    p.add_argument("--n-requests", type=int, default=10)
    p.add_argument("--output", type=str, default="phase8_results.csv")
    args = p.parse_args()

    low_path, high_path = Path(args.low_image), Path(args.high_image)
    if not low_path.exists() or not high_path.exists():
        raise SystemExit(f"Missing --low-image/--high-image: {low_path}, {high_path}")

    results = []
    for backend in BACKENDS:
        try:
            results.append(_run_one_backend(backend, low_path, high_path, args.n_requests))
        except Exception as e:
            print(f"  FAILED: {backend['name']}: {type(e).__name__}: {e}")
            results.append({"backend": backend["name"], "error": str(e)})

    print("\n=== Summary ===")
    for r in results:
        print(r)

    fieldnames = [
        "backend", "image_size_mb", "cold_start_s",
        "mean_round_trip_ms", "mean_server_inference_ms", "psnr", "ssim", "error",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
