"""
Phase 8 -- repeated-sampling benchmark against an ALREADY-DEPLOYED Cloud Run
(or any live) URL. Written specifically because the config-tuning back and
forth (--cpu=2 -> --cpu=4 -> ORT_INTRA_OP_THREADS=2) was being judged off
single verify_phase8.py calls -- a live serverless environment has real
request-to-request variance (network path, host scheduling, TLS handshake)
that one sample can't separate from a genuine config effect. This applies
the same mean +/- std discipline benchmark_phase7.py and benchmark_phase8.py
already use locally, to the deployed service instead.

Does NOT build, push, or deploy anything -- purely measures whatever is
currently live at --url. Run this after any `gcloud run services update` to
get a real read before deciding the change helped, hurt, or did nothing.

Saves per-request results to a CSV (default: phase8_cloud_run_results.csv),
matching the phase6_per_image.csv / phase7_latency.csv pattern used
elsewhere in this project -- every other phase's plan.md numbers are backed
by a real data file, and this one originally wasn't.

Usage:
    python scripts/benchmark_cloud_run.py \
        --url https://low-light-enhancement-il7b3gcc4a-uc.a.run.app \
        --low-image data/LOLdataset/eval15/low/1.png \
        --high-image data/LOLdataset/eval15/high/1.png \
        --n-requests 10
"""
import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import batch_psnr, batch_ssim


def _to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, required=True)
    p.add_argument("--low-image", type=str, required=True)
    p.add_argument("--high-image", type=str, required=True)
    p.add_argument("--n-requests", type=int, default=10)
    p.add_argument("--output", type=str, default="phase8_cloud_run_results.csv")
    args = p.parse_args()

    r = requests.get(f"{args.url}/health", timeout=10)
    r.raise_for_status()
    print(f"/health OK: {r.json()}")

    low_img = Image.open(args.low_image)
    buf = io.BytesIO()
    low_img.convert("RGB").save(buf, format="PNG")
    raw = buf.getvalue()

    round_trip_ms, server_inference_ms = [], []
    last_out_img = None
    print(f"\nSending {args.n_requests} requests to {args.url}/enhance ...")
    for i in range(args.n_requests):
        t0 = time.perf_counter()
        r = requests.post(
            f"{args.url}/enhance",
            files={"image": ("input.png", io.BytesIO(raw), "image/png")},
            timeout=60,
        )
        rt_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        inf_ms = float(r.headers.get("X-Inference-Ms", "nan"))
        round_trip_ms.append(rt_ms)
        server_inference_ms.append(inf_ms)
        last_out_img = Image.open(io.BytesIO(r.content)).convert("RGB")
        print(f"  [{i+1}/{args.n_requests}] round_trip={rt_ms:.1f}ms server_inference={inf_ms:.1f}ms")

    high_img = Image.open(args.high_image)
    pred_t, high_t = _to_tensor(last_out_img), _to_tensor(high_img)
    psnr, ssim = float(batch_psnr(pred_t, high_t)), float(batch_ssim(pred_t, high_t))

    def _stats(vals):
        return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

    rt_mean, rt_std, rt_min, rt_max = _stats(round_trip_ms)
    inf_mean, inf_std, inf_min, inf_max = _stats(server_inference_ms)

    # Per-request rows, not just a summary -- lets anyone re-derive mean/std
    # themselves rather than trusting a single pre-computed number, same
    # granularity as phase6_per_image.csv. psnr/ssim repeated per row
    # (computed once, from a fixed input/model -- deterministic here) so the
    # CSV is self-contained without needing the console log alongside it.
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_num", "round_trip_ms", "server_inference_ms", "psnr", "ssim"])
        for i, (rt, inf) in enumerate(zip(round_trip_ms, server_inference_ms), start=1):
            w.writerow([i, round(rt, 1), round(inf, 1), round(psnr, 2), round(ssim, 3)])
    print(f"\nSaved: {args.output}")

    print(f"\n=== Summary over {args.n_requests} requests ===")
    print(f"round_trip_ms:       mean={rt_mean:.1f} std={rt_std:.1f} min={rt_min:.1f} max={rt_max:.1f}")
    print(f"server_inference_ms: mean={inf_mean:.1f} std={inf_std:.1f} min={inf_min:.1f} max={inf_max:.1f}")
    print(f"quality: PSNR {psnr:.2f} / SSIM {ssim:.3f}")


if __name__ == "__main__":
    main()
