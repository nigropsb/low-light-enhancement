"""
Phase 7 -- deployment wrapper + ONNX export helpers for the Restormer model.

Two things every real caller of the raw Phase 4 checkpoint would otherwise
have to reimplement themselves, baked into the exported graph instead:

1. Pad/crop to satisfy Restormer's H,W-divisible-by-8 requirement.
   restormer.py's forward() *asserts* this rather than handling it --
   correctly so for training, where PairedTransform's eval-mode center-crop
   already guarantees it (see restormer.py's docstring). A deployment
   caller has no such guarantee (an arbitrary uploaded photo won't
   generally have both dimensions divisible by 8), so DeployRestormer
   reflect-pads up to the next multiple of 8, runs the model, then crops
   the padding back off. Reflect (not zero) padding, since zero-padding a
   photo's border introduces a hard black edge right where the model's
   receptive field would otherwise see continuous image content.

2. Clamp the output to [0, 1]. Restormer.forward() intentionally returns
   `x + residual` UNCLAMPED (see its docstring: clamping inside forward()
   was found in Phase 2 to zero gradients at initialization). That's
   correct for training, but every eval-time caller so far
   (benchmark_phase6.py, visualize_predictions*.py) has clamped the raw
   output at the call site -- a project-specific convention that a future
   ONNX Runtime / FastAPI consumer has no way to know about unless it's
   baked into the graph itself.

Usage:
    from src.onnx_export import load_restormer_for_export, export_onnx

    wrapper, epoch = load_restormer_for_export("runs/phase4_perceptual_run2/best.pt", "cpu")
    export_onnx(wrapper, "runs/phase7_deploy/restormer_phase4_fp32.onnx", dummy_hw=(400, 600))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.restormer import Restormer


class DeployRestormer(nn.Module):
    """Inference-only wrapper: pad-to-multiple-of-8 -> Restormer -> crop -> clamp[0,1].

    Wraps an already-constructed Restormer (trained or freshly initialized
    -- verify_phase7.py's smoke test deliberately uses an untrained one, same
    "synthetic tensors, no real checkpoint needed" philosophy as
    verify_phase2/3/4.py).
    """

    def __init__(self, model: Restormer, pad_multiple: int = 8):
        super().__init__()
        self.model = model
        self.pad_multiple = pad_multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        m = self.pad_multiple
        pad_h = (m - h % m) % m
        pad_w = (m - w % m) % m

        # NOTE: pad/crop unconditionally -- do NOT gate this behind
        # `if pad_h or pad_w:`. Under torch.onnx.export's tracer, a Python
        # `if` on a shape-derived value is resolved using the concrete
        # value seen at trace time, not recorded as a graph branch: trace
        # with an already-multiple-of-8 dummy input (pad=0) and the whole
        # pad step is baked OUT of the exported graph, permanently -- the
        # graph would then just be missing the padding for every future
        # non-multiple-of-8 input, however dynamic_axes is configured.
        # (Confirmed the hard way: verify_phase7.py's [4/7] check, a
        # 133x191 input, failed inside Restormer's internal PixelUnshuffle
        # with a non-divisible reshape once this happened.)
        # F.pad's reflect mode requires pad < corresponding input dim;
        # pad_h/pad_w are at most m-1=7 here, so a zero-valued pad (the
        # already-divisible case, e.g. eval15's 400x600) is always a valid,
        # cheap no-op -- there's no reason to special-case it away.
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        out = self.model(x)
        out = out[:, :, :h, :w]
        return torch.clamp(out, 0.0, 1.0)


def load_restormer_for_export(ckpt_path: str | Path, device) -> tuple[DeployRestormer, int]:
    """Loads a trained Restormer checkpoint (dim=24, matching the
    convention every other phase used -- see benchmark_phase6.py) in eval
    mode, wrapped for deployment. Returns (wrapper, epoch)."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = Restormer(dim=24).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    wrapper = DeployRestormer(model).to(device).eval()
    return wrapper, ckpt["epoch"]


def export_onnx(
    wrapper: nn.Module,
    onnx_path: str | Path,
    dummy_hw: tuple[int, int] = (400, 600),
    device: str | torch.device = "cpu",
    opset: int = 17,
) -> Path:
    """Exports `wrapper` to ONNX with dynamic batch/height/width axes, so
    the graph isn't hardcoded to one resolution -- eval15's images are all
    400x600, but a deployment endpoint shouldn't assume every future input
    is. `dummy_hw` only fixes the *trace* shape; dynamic_axes is what makes
    the exported graph actually resolution-agnostic afterward (checked
    explicitly in verify_phase7.py's [5/7] step)."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, *dummy_hw, device=device)
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return onnx_path


def quantize_dynamic_int8(
    fp32_path: str | Path,
    int8_path: str | Path,
    sample_hw: tuple[int, int] = (64, 64),
    min_size_reduction: float = 0.05,
) -> bool:
    """Attempts dynamic (weights-only) INT8 quantization, and -- unlike a
    bare call to onnxruntime.quantization.quantize_dynamic() -- verifies
    the result is both RUNNABLE and an actual, meaningful compression
    before trusting it. Two separate failure modes were found the hard
    way during Phase 7 development, and both are checked for here:

    1. quantize_dynamic() can complete without raising anything while
       still producing a graph the runtime can't execute at all.
       Restormer has zero nn.Linear layers; every weight-bearing op,
       including the 1x1 "pointwise" projections in MDTA and GDFN, is
       Conv2d. Quantizing with op_types_to_quantize=["Conv", "MatMul"]
       therefore quantizes the Conv layers, which ONNX Runtime represents
       as ConvInteger nodes -- quantize_dynamic() itself completes
       cleanly, but the CPU execution provider's ConvInteger kernel
       doesn't implement several of the *grouped/depthwise* conv
       configurations this architecture is full of (MDTA's qkv_dwconv,
       GDFN's dwconv, etc, all groups > 1). The failure only surfaces as
       NOT_IMPLEMENTED at InferenceSession *load* time.

    2. Falling back to MatMul-only quantization to dodge (1) "succeeds"
       in the sense that the graph loads and runs -- but since this model
       has NO weight-bearing MatMul ops either (every candidate op is
       Conv2d, see above), there's nothing for it to actually quantize.
       Confirmed directly: the resulting file came out *larger* than
       fp32 (quantization scaffolding overhead with zero compression
       payoff), with a 0.000 numerical diff from the fp32 output --
       proof nothing was actually quantized, not evidence of a good
       result. A bare "did it load and run" check reports this as a
       false-positive success; this function additionally requires at
       least `min_size_reduction` (default 5%) smaller than the fp32
       file before accepting a result.

    Tries progressively narrower op sets, falling back rather than
    crashing:
        1. Conv + MatMul (the ideal case for this Conv-heavy model)
        2. MatMul only (falls back if Conv's ConvInteger path is unusable)
        3. onnxruntime's own defaults

    Returns True with int8_path containing a verified-runnable,
    meaningfully-smaller model, or False (int8_path may not exist, or may
    exist but should not be trusted) if none of the above worked --
    callers must check the return value, not just file existence.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnxruntime as ort

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    fp32_size = fp32_path.stat().st_size

    def _attempt(op_types: list[str] | None) -> float:
        kwargs = dict(model_input=str(fp32_path), model_output=str(int8_path),
                      weight_type=QuantType.QInt8)
        if op_types is not None:
            kwargs["op_types_to_quantize"] = op_types
        quantize_dynamic(**kwargs)
        # Check 1: does CPUExecutionProvider actually load and run this
        # graph, not just "did a file get written."
        session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        dummy = np.random.rand(1, 3, *sample_hw).astype(np.float32)
        session.run(None, {"input": dummy})
        # Check 2: did it actually compress anything, or just add
        # quantization scaffolding around ops it had nothing to quantize.
        int8_size = int8_path.stat().st_size
        reduction = 1 - (int8_size / fp32_size)
        if reduction < min_size_reduction:
            raise RuntimeError(
                f"graph loads and runs, but is only {reduction * 100:.1f}% smaller than fp32 "
                f"({int8_size / 1e6:.2f} MB vs {fp32_size / 1e6:.2f} MB) -- not a meaningful "
                f"quantization, most likely because op_types={op_types} matched no real "
                "weight-bearing ops in this graph. Rejecting rather than reporting a hollow "
                "'success'."
            )
        return reduction

    for op_types, label in [
        (["Conv", "MatMul"], "Conv+MatMul"),
        (["MatMul"], "MatMul only"),
        (None, "onnxruntime defaults"),
    ]:
        try:
            reduction = _attempt(op_types)
            print(f"  dynamic INT8 quantization OK with op_types={label} "
                  f"({reduction * 100:.0f}% smaller than fp32).")
            return True
        except Exception as e:
            print(f"  dynamic INT8 quantization with op_types={label} failed/rejected: "
                  f"{type(e).__name__}: {e}")

    print(
        "  Dynamic quantization isn't viable for this graph on the installed "
        "onnxruntime: Restormer is entirely Conv-based (no nn.Linear/MatMul-with-"
        "weight ops) and ONNX Runtime's ConvInteger CPU kernel doesn't support "
        "several of its grouped/depthwise conv configurations -- and the MatMul-only "
        "fallback has nothing real to quantize either. Static (calibrated, QDQ-format) "
        "quantization is the documented path for CNN-heavy models like this one -- "
        "carried forward as an open item rather than forced here."
    )
    return False
