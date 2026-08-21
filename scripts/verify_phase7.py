"""
Phase 7 smoke test -- ONNX export + dynamic INT8 quantization, on a
freshly initialized (untrained) Restormer. Same "synthetic tensors, no
real checkpoint" philosophy as verify_phase2/3/4.py: exercise the
deployment wrapper and export/quantize pipeline in isolation before
scripts/benchmark_phase7.py touches the real Phase 4 checkpoint and the
real eval15 set.

New dependency this phase, not needed by any earlier phase:
    pip install onnx onnxruntime
(onnxruntime-gpu instead of onnxruntime if you want the CUDA execution
provider row in benchmark_phase7.py -- CPU-only onnxruntime is enough for
this smoke test.)

Usage: python scripts/verify_phase7.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.restormer import Restormer
from src.onnx_export import DeployRestormer, export_onnx, quantize_dynamic_int8

try:
    import onnx
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit(
        "Phase 7 needs onnx + onnxruntime installed (new this phase). "
        "Run: pip install onnx onnxruntime"
    ) from e


def main():
    device = torch.device("cpu")  # export is traced on CPU; the resulting
    # graph is hardware-agnostic afterward -- benchmark_phase7.py runs it
    # under both CPU and CUDA execution providers from the same file.
    torch.manual_seed(0)

    model = Restormer(dim=24).to(device).eval()
    wrapper = DeployRestormer(model).to(device).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        onnx_path = tmpdir / "restormer_fp32.onnx"

        # --- 1. Export at a multiple-of-8 shape, checker validates the graph ---
        export_onnx(wrapper, onnx_path, dummy_hw=(128, 128), device=device)
        onnx.checker.check_model(str(onnx_path))
        size_fp32_mb = onnx_path.stat().st_size / (1024 ** 2)
        print(f"[1/7] export OK. onnx.checker passed. fp32 ONNX size: {size_fp32_mb:.2f} MB")

        # --- 2. Numerical equivalence vs PyTorch, at the export shape ---
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = torch.rand(1, 3, 128, 128)
        with torch.no_grad():
            torch_out = wrapper(x).numpy()
        onnx_out = session.run(None, {"input": x.numpy()})[0]
        max_diff = np.abs(torch_out - onnx_out).max()
        assert max_diff < 1e-3, f"ONNX output diverges from PyTorch: max abs diff {max_diff:.2e}"
        print(f"[2/7] numerical equivalence OK. max abs diff vs PyTorch: {max_diff:.2e}")

        # --- 3. Clamp holds at the graph level, even for out-of-range input ---
        x_extreme = torch.rand(1, 3, 128, 128) * 5 - 2  # deliberately outside [0, 1]
        onnx_out_extreme = session.run(None, {"input": x_extreme.numpy()})[0]
        assert onnx_out_extreme.min() >= 0.0 and onnx_out_extreme.max() <= 1.0, (
            f"clamp not enforced in the exported graph: range "
            f"[{onnx_out_extreme.min():.3f}, {onnx_out_extreme.max():.3f}]"
        )
        print("[3/7] output clamp OK. range within [0, 1] even for out-of-range input.")

        # --- 4. Non-multiple-of-8 shape -- the pad/crop wrapper's actual reason to exist ---
        x_odd = torch.rand(1, 3, 133, 191)  # neither dim divisible by 8
        with torch.no_grad():
            torch_out_odd = wrapper(x_odd).numpy()
        onnx_out_odd = session.run(None, {"input": x_odd.numpy()})[0]
        assert onnx_out_odd.shape == (1, 3, 133, 191), f"shape mismatch: {onnx_out_odd.shape}"
        max_diff_odd = np.abs(torch_out_odd - onnx_out_odd).max()
        assert max_diff_odd < 1e-3, f"odd-shape ONNX output diverges: {max_diff_odd:.2e}"
        print(
            f"[4/7] non-multiple-of-8 input (133x191) OK. output shape correct, "
            f"max abs diff {max_diff_odd:.2e} -- confirms the reflect-pad/crop wrapper "
            "does its job without re-exporting, unlike the raw Restormer.forward() assert."
        )

        # --- 5. Dynamic axes: a second, different resolution on the SAME session ---
        x2 = torch.rand(1, 3, 96, 160)
        onnx_out2 = session.run(None, {"input": x2.numpy()})[0]
        assert onnx_out2.shape == (1, 3, 96, 160)
        print(
            "[5/7] dynamic axes OK. a second resolution ran on the same exported "
            "session without re-exporting."
        )

        # --- 6. Dynamic INT8 quantization round-trip. quantize_dynamic_int8
        # verifies the result actually LOADS AND RUNS under CPUExecutionProvider
        # (not just that a file got written) -- see its docstring in
        # src/onnx_export.py for a concrete case (ConvInteger + this model's
        # grouped/depthwise convs) where quantize_dynamic() alone silently
        # produces an unrunnable graph. A False return here is a legitimate,
        # documented outcome for a Conv-only architecture on some
        # onnxruntime versions, not a smoke-test failure -- benchmark_phase7.py
        # (and this script) handle it by skipping int8-dependent checks
        # rather than crashing. ---
        onnx_int8_path = tmpdir / "restormer_int8.onnx"
        int8_ok = quantize_dynamic_int8(onnx_path, onnx_int8_path, sample_hw=(128, 128))
        if int8_ok:
            size_int8_mb = onnx_int8_path.stat().st_size / (1024 ** 2)
            session_int8 = ort.InferenceSession(str(onnx_int8_path), providers=["CPUExecutionProvider"])
            onnx_int8_out = session_int8.run(None, {"input": x.numpy()})[0]
            assert onnx_int8_out.shape == torch_out.shape
            quant_diff = np.abs(torch_out - onnx_int8_out).max()
            print(
                f"[6/7] dynamic INT8 quantization OK. size {size_fp32_mb:.2f} MB -> "
                f"{size_int8_mb:.2f} MB ({100 * (1 - size_int8_mb / size_fp32_mb):.0f}% smaller). "
                f"max abs diff vs fp32 PyTorch on an UNTRAINED model: {quant_diff:.3f} "
                "(expect this to be a real, visible number, not ~0 -- random-init weights "
                "quantize worse than a real trained model typically does; the number that "
                "actually matters is benchmark_phase7.py's PSNR/SSIM comparison on the real "
                "Phase 4 checkpoint, not this one)."
            )
        else:
            print(
                "[6/7] dynamic INT8 quantization not viable on this onnxruntime install "
                "(see the reason printed above) -- benchmark_phase7.py will skip the int8 "
                "rows and report fp32-only results. Not treated as a smoke-test failure."
            )

        # --- 7. Size context vs. the raw PyTorch checkpoint on disk ---
        pt_path = tmpdir / "model.pt"
        torch.save({"model_state": model.state_dict(), "epoch": 0}, pt_path)
        size_pt_mb = pt_path.stat().st_size / (1024 ** 2)
        print(
            f"[7/7] size context: PyTorch .pt state_dict {size_pt_mb:.2f} MB "
            "(untrained Restormer -- state_dict size doesn't depend on training, so "
            "this number is directly comparable to the real checkpoint's size)."
        )

    print("\nAll Phase 7 smoke tests passed.")


if __name__ == "__main__":
    main()
