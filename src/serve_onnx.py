"""
Phase 8 -- FastAPI serving wrapper around the Phase 7 ONNX fp32 artifact
(restormer_phase4_fp32.onnx, exported from runs/phase4_perceptual_run2/best.pt,
the Phase 6 winner: PSNR 21.33 / SSIM 0.800 / LPIPS 0.1857 / MANIQA 0.293 on
eval15).

This is one of TWO comparison backends -- see src/serve_pytorch.py for the
PyTorch-CPU counterpart, and scripts/benchmark_phase8.py for what actually
decides between them. Deliberately NOT assuming this one wins: Phase 7's own
benchmark showed ONNX Runtime CPU fp32 (~3.0s/image at 400x600) is ~18%
SLOWER per-request than PyTorch CPU (~2.5s). This backend's only *candidate*
advantage is a smaller, torch-free container image and faster cold start on
Cloud Run's scale-to-zero model -- unverified until benchmark_phase8.py
actually measures both images side by side. Do not treat this file's
existence as an endorsement; read the benchmark_phase8.py output before
picking one for the real deployment.

The pad-to-multiple-of-8 / crop / clamp[0,1] logic lives inside the exported
ONNX graph itself (src/onnx_export.py:DeployRestormer), so this file does NOT
reimplement it -- any resolution goes in, no assumptions needed here about
divisibility by 8.

Usage (local):
    uvicorn src.serve_onnx:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health   -> {"status": "ok", "model_epoch": int, "onnx_opset": int}
    POST /enhance   -> multipart/form-data image upload, returns PNG bytes
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

# --- Structured logging: plain key=value lines to stdout, which Cloud Run
# captures automatically into Cloud Logging without any extra setup. Avoids
# a JSON-logging dependency for what's currently a handful of fields. ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("serve")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/restormer_phase4_fp32.onnx"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
# Hard cap on input megapixels -- an unbounded-resolution upload on a CPU-only
# service is a real cost/DoS risk, not just a latency inconvenience. 400x600
# (eval15's native resolution, ~0.24 MP) is the only regime this model has
# actually been *evaluated* at; this cap is set generously above that for
# real photo uploads while still bounding worst-case request cost.
MAX_MEGAPIXELS = float(os.environ.get("MAX_MEGAPIXELS", 4.0))

app = FastAPI(title="Low-Light Enhancement API")

_session: ort.InferenceSession | None = None
_model_epoch: int | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _session, _model_epoch
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No ONNX model at {MODEL_PATH} -- copy "
            "runs/phase7_deploy/restormer_phase4_fp32.onnx into the image "
            "(see Dockerfile) or set MODEL_PATH."
        )
    # Single-threaded intra-op by default keeps memory/CPU predictable under
    # Cloud Run's per-instance CPU allocation; raise via ORT_INTRA_OP_THREADS
    # if profiling later shows headroom (open item, not tuned here -- see
    # Phase 7's own CPU-vs-ORT latency gap, not yet root-caused at the op
    # level, before assuming more threads is the right knob).
    so = ort.SessionOptions()
    threads = os.environ.get("ORT_INTRA_OP_THREADS")
    if threads:
        so.intra_op_num_threads = int(threads)
    _session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=so, providers=["CPUExecutionProvider"]
    )
    # Epoch isn't embedded in the ONNX graph itself (only weights are) --
    # sidecar file written by scripts/export_for_serving.py alongside the
    # .onnx so /health can report real provenance instead of "unknown".
    epoch_file = MODEL_PATH.with_suffix(".epoch")
    _model_epoch = int(epoch_file.read_text().strip()) if epoch_file.exists() else -1
    logger.info(
        f"model loaded path={MODEL_PATH} epoch={_model_epoch} "
        f"providers={_session.get_providers()}"
    )


@app.get("/health")
def health() -> dict:
    if _session is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_epoch": _model_epoch,
        "onnx_opset": _session.get_modelmeta().custom_metadata_map or "unknown",
        "providers": _session.get_providers(),
    }


def _decode_and_validate(raw: bytes, request_id: str) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode image: {e}") from e

    w, h = img.size
    megapixels = (w * h) / 1_000_000
    if megapixels > MAX_MEGAPIXELS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"image is {megapixels:.1f} MP, exceeds the {MAX_MEGAPIXELS} MP "
                "limit for this CPU-only endpoint"
            ),
        )
    logger.info(f"id={request_id} decoded w={w} h={h} megapixels={megapixels:.2f}")

    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = arr.transpose(2, 0, 1)[None, ...]  # -> NCHW
    return np.ascontiguousarray(arr)


@app.post("/enhance")
async def enhance(image: UploadFile = File(...)) -> Response:
    if _session is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    request_id = uuid.uuid4().hex[:12]
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload is {len(raw)} bytes, exceeds {MAX_UPLOAD_BYTES} byte limit",
        )

    low_np = _decode_and_validate(raw, request_id)

    t0 = time.perf_counter()
    out = _session.run(None, {"input": low_np})[0]  # NCHW, [0,1], already clamped
    inference_ms = (time.perf_counter() - t0) * 1000.0

    out = np.clip(out[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)  # HWC uint8
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")

    logger.info(
        f"id={request_id} status=ok inference_ms={inference_ms:.1f} "
        f"in_bytes={len(raw)} out_bytes={buf.tell()}"
    )
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"X-Request-Id": request_id, "X-Inference-Ms": f"{inference_ms:.1f}"},
    )


@app.exception_handler(HTTPException)
async def _log_http_errors(request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"status=error code={exc.status_code} detail={exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
