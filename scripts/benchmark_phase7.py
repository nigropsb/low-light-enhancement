"""
Phase 7 -- ONNX export, dynamic INT8 quantization, and latency/size/quality
benchmarking of the Phase 4 checkpoint (Restormer + DINOv2 perceptual loss,
the Phase 6 winner across all four metrics: PSNR 21.33 / SSIM 0.800 /
LPIPS 0.1857 / MANIQA 0.293 on eval15). This is the "what does it cost to
deploy the model that won Phase 6" companion to benchmark_phase6.py's
"which model wins" comparison.

What's benchmarked:
  - Size on disk: raw .pt checkpoint vs. ONNX fp32 vs. ONNX INT8 (dynamic/
    weights-only quantization, attempted via
    src/onnx_export.py:quantize_dynamic_int8 -- no calibration set needed,
    unlike static quantization; see that function's docstring for why this
    project's Restormer is a hard case for it: entirely Conv-based, with
    many grouped/depthwise convolutions that ONNX Runtime's CPU
    ConvInteger kernel may not support. The INT8 rows below are only
    populated if quantize_dynamic_int8 actually verifies a runnable
    result -- if it's not viable on your onnxruntime install, this script
    still completes and reports fp32-only results, with the gap logged as
    an open item rather than silently skipped or forced through).
  - Latency at eval15's native resolution (400x600, batch=1):
        PyTorch CPU | PyTorch CUDA (if available) |
        ONNX Runtime CPU fp32 | ONNX Runtime CPU INT8 (if viable) |
        ONNX Runtime CUDA fp32 (if available)
    INT8 is CPU-only even when it works: ONNX Runtime's CUDAExecutionProvider
    doesn't consume dynamically-quantized (QOperator) INT8 graphs the way
    the CPU EP does -- real GPU INT8 would need a TensorRT-based export
    path, explicitly out of scope for this phase (open item, not a silent gap).
  - Quality delta from export/quantization on the REAL eval15 test set:
    PSNR/SSIM for PyTorch fp32 (should reproduce Phase 6's own 21.33/0.800
    as a sanity check that the deployment wrapper's pad/clamp logic didn't
    change predictions) vs. ONNX fp32 vs. ONNX INT8 (if viable) -- so "does
    export change predictions" and "what does quantization cost" are both
    answered with real numbers, not assumed to be free.

Usage:
    python scripts/benchmark_phase7.py --data-root data/LOLdataset
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.lol_dataset import make_splits
from src.metrics import batch_psnr, batch_ssim
from src.onnx_export import load_restormer_for_export, export_onnx, quantize_dynamic_int8

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit(
        "pip install onnx onnxruntime (add onnxruntime-gpu instead if you want the "
        "ONNX Runtime CUDA fp32 row -- CPU-only onnxruntime is enough otherwise)"
    ) from e


def _time_calls(fn, n_warmup: int = 5, n_runs: int = 20) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) over n_runs, after n_warmup untimed calls."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000.0
    return float(times.mean()), float(times.std())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/LOLdataset")
    p.add_argument("--ckpt", type=str, default="runs/phase4_perceptual_run2/best.pt",
                    help="Phase 4 Run 2 checkpoint -- the Phase 6 winner across all four metrics.")
    p.add_argument("--output-dir", type=str, default="runs/phase7_deploy")
    p.add_argument("--n-runs", type=int, default=20)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path} -- Phase 4 must be trained first.")

    cpu = torch.device("cpu")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}\n")

    # --- Export (traced on CPU; the resulting graph is hardware-agnostic --
    # both CPU and CUDA execution providers below load the same file) ---
    wrapper_cpu, epoch = load_restormer_for_export(ckpt_path, cpu)
    print(f"Loaded checkpoint from epoch {epoch}: {ckpt_path}")

    onnx_fp32_path = output_dir / "restormer_phase4_fp32.onnx"
    export_onnx(wrapper_cpu, onnx_fp32_path, dummy_hw=(400, 600), device=cpu)
    onnx_int8_path = output_dir / "restormer_phase4_int8.onnx"
    print("\nAttempting dynamic INT8 quantization:")
    # sample_hw kept small (64x64) for the internal load+run check -- this
    # is just verifying the quantized graph is executable at all, not
    # measuring anything at this shape; the real latency/quality numbers
    # below always use eval15's native 400x600.
    int8_ok = quantize_dynamic_int8(onnx_fp32_path, onnx_int8_path, sample_hw=(64, 64))

    size_pt_mb = ckpt_path.stat().st_size / (1024 ** 2)
    size_fp32_mb = onnx_fp32_path.stat().st_size / (1024 ** 2)
    print(f"\nSizes: .pt checkpoint {size_pt_mb:.2f} MB | ONNX fp32 {size_fp32_mb:.2f} MB", end="")
    if int8_ok:
        size_int8_mb = onnx_int8_path.stat().st_size / (1024 ** 2)
        print(
            f" | ONNX int8 {size_int8_mb:.2f} MB "
            f"({100 * (1 - size_int8_mb / size_fp32_mb):.0f}% smaller than fp32 ONNX)"
        )
    else:
        print(" | ONNX int8: N/A (dynamic quantization not viable on this graph -- see above)")

    # --- Latency, native eval15 resolution (400x600), batch=1 ---
    print("\nLatency (400x600, batch=1):")
    dummy = torch.rand(1, 3, 400, 600)
    dummy_np = dummy.numpy()
    latency_rows = []

    def _pt_cpu_call():
        with torch.no_grad():
            return wrapper_cpu(dummy)

    mean_ms, std_ms = _time_calls(_pt_cpu_call, n_runs=args.n_runs)
    latency_rows.append(("PyTorch CPU", mean_ms, std_ms))
    print(f"  PyTorch CPU:            {mean_ms:7.1f} +/- {std_ms:5.1f} ms")

    if cuda_available:
        cuda = torch.device("cuda")
        wrapper_cuda, _ = load_restormer_for_export(ckpt_path, cuda)
        dummy_cuda = dummy.to(cuda)

        def _pt_cuda_call():
            with torch.no_grad():
                out = wrapper_cuda(dummy_cuda)
            torch.cuda.synchronize()
            return out

        mean_ms, std_ms = _time_calls(_pt_cuda_call, n_runs=args.n_runs)
        latency_rows.append(("PyTorch CUDA", mean_ms, std_ms))
        print(f"  PyTorch CUDA:           {mean_ms:7.1f} +/- {std_ms:5.1f} ms")

    sess_cpu_fp32 = ort.InferenceSession(str(onnx_fp32_path), providers=["CPUExecutionProvider"])
    mean_ms, std_ms = _time_calls(
        lambda: sess_cpu_fp32.run(None, {"input": dummy_np}), n_runs=args.n_runs
    )
    latency_rows.append(("ONNX Runtime CPU fp32", mean_ms, std_ms))
    print(f"  ONNX Runtime CPU fp32:  {mean_ms:7.1f} +/- {std_ms:5.1f} ms")

    sess_cpu_int8 = None
    if int8_ok:
        sess_cpu_int8 = ort.InferenceSession(str(onnx_int8_path), providers=["CPUExecutionProvider"])
        mean_ms, std_ms = _time_calls(
            lambda: sess_cpu_int8.run(None, {"input": dummy_np}), n_runs=args.n_runs
        )
        latency_rows.append(("ONNX Runtime CPU int8", mean_ms, std_ms))
        print(f"  ONNX Runtime CPU int8:  {mean_ms:7.1f} +/- {std_ms:5.1f} ms")
    else:
        print("  ONNX Runtime CPU int8:  skipped -- dynamic quantization not viable (see above)")

    if cuda_available and "CUDAExecutionProvider" in ort.get_available_providers():
        sess_cuda_fp32 = ort.InferenceSession(str(onnx_fp32_path), providers=["CUDAExecutionProvider"])
        mean_ms, std_ms = _time_calls(
            lambda: sess_cuda_fp32.run(None, {"input": dummy_np}), n_runs=args.n_runs
        )
        latency_rows.append(("ONNX Runtime CUDA fp32", mean_ms, std_ms))
        print(f"  ONNX Runtime CUDA fp32: {mean_ms:7.1f} +/- {std_ms:5.1f} ms")
    elif cuda_available:
        print(
            "  ONNX Runtime CUDA fp32: skipped -- CUDA is available but the installed "
            "onnxruntime has no CUDAExecutionProvider (you have 'onnxruntime', not "
            "'onnxruntime-gpu'). pip install onnxruntime-gpu to get this row."
        )

    # --- Quality on the real eval15 set: PyTorch fp32 vs ONNX fp32 vs ONNX int8 ---
    _train_ds, _val_ds, test_ds = make_splits(args.data_root)
    print(f"\neval15 quality check ({len(test_ds)} images):")

    def _pt_predict(low_np):
        with torch.no_grad():
            return wrapper_cpu(torch.from_numpy(low_np)).numpy()

    quality_targets = [
        ("PyTorch fp32", _pt_predict),
        ("ONNX fp32", lambda low_np: sess_cpu_fp32.run(None, {"input": low_np})[0]),
    ]
    if int8_ok:
        quality_targets.append(
            ("ONNX int8", lambda low_np: sess_cpu_int8.run(None, {"input": low_np})[0])
        )
    else:
        print("  ONNX int8:       skipped -- dynamic quantization not viable (see above)")

    quality_rows = []
    for name, predict_fn in quality_targets:
        psnr_sum, ssim_sum = 0.0, 0.0
        for idx in range(len(test_ds)):
            item = test_ds[idx]
            low_np = item["low"].unsqueeze(0).numpy()
            high = item["high"].unsqueeze(0)
            pred = torch.from_numpy(predict_fn(low_np))
            psnr_sum += batch_psnr(pred, high)
            ssim_sum += batch_ssim(pred, high)
        n = len(test_ds)
        mean_psnr, mean_ssim = psnr_sum / n, ssim_sum / n
        quality_rows.append((name, mean_psnr, mean_ssim))
        print(f"  {name:<16} PSNR {mean_psnr:.2f} | SSIM {mean_ssim:.3f}")

    print(
        "  (compare 'PyTorch fp32' above to Phase 6's own 21.33 / 0.800 for this "
        "checkpoint -- they should match closely; a meaningful gap would mean the "
        "deployment wrapper's pad/clamp logic changed predictions, not just packaged them.)"
    )

    # --- Save results ---
    size_path = output_dir / "phase7_sizes.csv"
    with open(size_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "size_mb"])
        w.writerow([".pt checkpoint", f"{size_pt_mb:.2f}"])
        w.writerow(["ONNX fp32", f"{size_fp32_mb:.2f}"])
        w.writerow(["ONNX int8", f"{size_int8_mb:.2f}" if int8_ok else "N/A"])

    latency_path = output_dir / "phase7_latency.csv"
    with open(latency_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["runtime", "mean_ms", "std_ms"])
        w.writerows([(n, f"{m:.2f}", f"{s:.2f}") for n, m, s in latency_rows])

    quality_path = output_dir / "phase7_quality.csv"
    with open(quality_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["runtime", "psnr_db", "ssim"])
        w.writerows([(n, f"{p:.2f}", f"{s:.3f}") for n, p, s in quality_rows])

    print(f"\nSaved: {size_path}\n       {latency_path}\n       {quality_path}")

    open_items = []
    if int8_ok:
        open_items.append(
            "Dynamic quantization only touches weights; activations stay fp32 at "
            "runtime, capping the latency win vs. what static (calibrated) INT8 could give."
        )
    else:
        open_items.append(
            "Dynamic INT8 quantization was NOT viable on this architecture/onnxruntime "
            "combination -- Restormer is entirely Conv-based (no nn.Linear/MatMul-with-"
            "weight ops) and ONNX Runtime's ConvInteger CPU kernel doesn't support several "
            "of its grouped/depthwise conv configurations. Static (calibrated, QDQ-format) "
            "quantization is the documented ORT-recommended path for CNN-heavy models and "
            "is the natural next thing to try here, not attempted in this phase."
        )
    open_items.append(
        "GPU INT8 not benchmarked -- ONNX Runtime's CUDA EP doesn't run dynamically-"
        "quantized graphs; a real GPU INT8 path would need TensorRT."
    )
    open_items.append("No FastAPI/serving wrapper yet -- that's Phase 8 (cloud deployment).")
    print(
        "\nOpen items carried forward (not addressed in this phase):\n"
        + "\n".join(f"  - {item}" for item in open_items)
    )


if __name__ == "__main__":
    main()
