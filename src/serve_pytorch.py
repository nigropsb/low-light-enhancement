"""
Phase 8 -- FastAPI serving wrapper around the Phase 4 checkpoint served
directly with PyTorch (CPU), no ONNX involved.

This is one of TWO comparison backends -- see src/serve_onnx.py for the
ONNX Runtime counterpart, and scripts/benchmark_phase8.py for what actually
decides between them. Phase 7's own benchmark showed this backend is ~18%
FASTER per-request than ONNX Runtime CPU at 400x600 (2473ms vs 3027ms) --
the open question this file exists to help answer is whether that per-request
win survives once container size and cold-start time are measured too,
which benchmark_phase8.py does for real rather than assuming either way.

Reuses DeployRestormer from src/onnx_export.py directly -- the pad-to-
multiple-of-8 / crop / clamp[0,1] wrapper logic was written framework-
agnostic in Phase 7 specifically so it didn't need reimplementing here.

Loads a STRIPPED checkpoint (weights + epoch only, no optimizer/scaler
state) produced by scripts/export_for_serving_pytorch.py -- shipping the
full training checkpoint (37.68 MB, ~2.8x the weights alone, see Phase 7's
own size analysis) into a serving image would be carrying AdamW momentum
buffers that inference never touches.

Usage (local):
    uvicorn src.serve_pytorch:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /health   -> {"status": "ok", "model_epoch": int, "framework": "pytorch"}
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
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.onnx_export import DeployRestormer
from src.models.restormer import Restormer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s level=%(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("serve_pytorch")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/restormer_phase4_inference.pt"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
# Same cap and same rationale as src/serve_onnx.py -- kept identical across
# both backends so the comparison in benchmark_phase8.py isn't confounded
# by different request-shaping rules.
MAX_MEGAPIXELS = float(os.environ.get("MAX_MEGAPIXELS", 4.0))

app = FastAPI(title="Low-Light Enhancement API (PyTorch backend)")

_wrapper: DeployRestormer | None = None
_model_epoch: int | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _wrapper, _model_epoch
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"No checkpoint at {MODEL_PATH} -- run "
            "scripts/export_for_serving_pytorch.py to stage a stripped, "
            "inference-only checkpoint (see Dockerfile.pytorch) or set MODEL_PATH."
        )
    threads = os.environ.get("TORCH_NUM_THREADS")
    if threads:
        torch.set_num_threads(int(threads))
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    model = Restormer(dim=24).eval()
    model.load_state_dict(ckpt["model_state"])
    _wrapper = DeployRestormer(model).eval()
    _model_epoch = ckpt["epoch"]
    logger.info(
        f"model loaded path={MODEL_PATH} epoch={_model_epoch} "
        f"torch_threads={torch.get_num_threads()}"
    )


@app.get("/health")
def health() -> dict:
    if _wrapper is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_epoch": _model_epoch,
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def _decode_and_validate(raw: bytes, request_id: str) -> torch.Tensor:
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
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # NCHW
    return tensor


@app.post("/enhance")
async def enhance(image: UploadFile = File(...)) -> Response:
    if _wrapper is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    request_id = uuid.uuid4().hex[:12]
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"upload is {len(raw)} bytes, exceeds {MAX_UPLOAD_BYTES} byte limit",
        )

    low_t = _decode_and_validate(raw, request_id)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_t = _wrapper(low_t)  # already padded/cropped/clamped to [0,1]
    inference_ms = (time.perf_counter() - t0) * 1000.0

    out = (out_t[0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
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
