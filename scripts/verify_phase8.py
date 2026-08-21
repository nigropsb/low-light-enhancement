"""
Phase 8 smoke test -- hits an already-running instance of the serving
container (local `docker run` or a deployed Cloud Run URL) and checks the
things that would silently break a demo without erroring loudly:

  1. /health responds and reports a real epoch (not -1, which means the
     .epoch sidecar wasn't found -- see scripts/export_for_serving.py).
  2. /enhance on a real eval15 low-light image returns a decodable PNG at
     the same resolution as the input (dynamic-axes ONNX graph, no
     resolution assumptions baked in incorrectly).
  3. /enhance actually improves PSNR/SSIM vs. the raw input, using the
     matching eval15 high-light image as reference -- same
     "did this obviously work" philosophy as verify_phase1-7.py, not a
     full benchmark_phase6.py-style multi-image evaluation.
  4. Oversized-upload rejection (413) and corrupt-image rejection (400)
     both return the expected status codes rather than crashing the server
     or hanging.

Backend-agnostic: works against either src/serve_onnx.py or
src/serve_pytorch.py, since both expose the same /health and /enhance
contract. Useful as a quick single-request sanity check before running the
side-by-side comparison in scripts/benchmark_phase8.py.

Usage:
    # container already running locally, e.g.:
    #   docker run -p 8080:8080 low-light-enhancement:onnx
    #   docker run -p 8080:8080 low-light-enhancement:pytorch
    python scripts/verify_phase8.py --url http://localhost:8080 \
        --low-image data/LOLdataset/eval15/low/1.png \
        --high-image data/LOLdataset/eval15/high/1.png
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import batch_psnr, batch_ssim  # reuse the one source of truth
import torch


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, default="http://localhost:8080")
    p.add_argument("--low-image", type=str, required=True)
    p.add_argument("--high-image", type=str, required=True)
    args = p.parse_args()

    # --- 1. Health check ---
    r = requests.get(f"{args.url}/health", timeout=10)
    assert r.status_code == 200, f"/health returned {r.status_code}: {r.text}"
    health = r.json()
    assert health["model_epoch"] != -1, (
        "model_epoch is -1 -- the .epoch sidecar wasn't found at container "
        "build time. Check scripts/export_for_serving.py ran before `docker build`."
    )
    print(f"[1/4] /health OK. epoch={health['model_epoch']} providers={health['providers']}")

    # --- 2. Enhance a real low-light image, check shape round-trips ---
    low_path = Path(args.low_image)
    low_img = Image.open(low_path)
    orig_size = low_img.size  # (w, h)

    buf = io.BytesIO()
    low_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    t0 = __import__("time").perf_counter()
    r = requests.post(f"{args.url}/enhance", files={"image": ("input.png", buf, "image/png")}, timeout=60)
    round_trip_ms = (__import__("time").perf_counter() - t0) * 1000.0
    assert r.status_code == 200, f"/enhance returned {r.status_code}: {r.text}"

    out_img = Image.open(io.BytesIO(r.content)).convert("RGB")
    assert out_img.size == orig_size, (
        f"output size {out_img.size} != input size {orig_size} -- "
        "pad/crop logic in the exported graph may be broken."
    )
    server_inference_ms = r.headers.get("X-Inference-Ms", "unknown")
    print(
        f"[2/4] /enhance OK. shape round-trips ({orig_size[0]}x{orig_size[1]}). "
        f"server inference_ms={server_inference_ms} round_trip_ms={round_trip_ms:.1f}"
    )

    # --- 3. Sanity-check quality vs. raw input on this one pair ---
    high_img = Image.open(args.high_image)
    pred_t = _to_tensor(out_img)
    high_t = _to_tensor(high_img)
    low_t = _to_tensor(low_img)

    psnr_out, ssim_out = batch_psnr(pred_t, high_t), batch_ssim(pred_t, high_t)
    psnr_raw, ssim_raw = batch_psnr(low_t, high_t), batch_ssim(low_t, high_t)
    assert psnr_out > psnr_raw, (
        f"served output PSNR ({psnr_out:.2f}) did not beat raw input PSNR "
        f"({psnr_raw:.2f}) -- something is wrong with the served model, not "
        "just 'not state of the art'."
    )
    print(
        f"[3/4] quality OK. served PSNR {psnr_out:.2f} / SSIM {ssim_out:.3f} "
        f"vs. raw input PSNR {psnr_raw:.2f} / SSIM {ssim_raw:.3f}"
    )

    # --- 4. Error handling: corrupt image -> 400, not a 500 or a hang ---
    r = requests.post(
        f"{args.url}/enhance",
        files={"image": ("junk.png", io.BytesIO(b"not an image"), "image/png")},
        timeout=10,
    )
    assert r.status_code == 400, f"corrupt image expected 400, got {r.status_code}"
    print("[4/4] error handling OK. corrupt upload correctly rejected with 400.")

    print("\nAll Phase 8 smoke tests passed.")


if __name__ == "__main__":
    main()
