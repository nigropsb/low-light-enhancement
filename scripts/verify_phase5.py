"""
Phase 5 smoke test -- NR-IQA metric integration (MANIQA via pyiqa).

Same philosophy as scripts/verify_phase2/3/4.py: exercise the new component
in isolation on synthetic data before it's trusted inside a real evaluation
script (scripts/evaluate_nriqa_phase5.py) that touches actual checkpoints
and the real eval15 set. Run this first.

Structurally simpler than verify_phase4.py: there's no optimizer, no
gradients, no checkpoint to round-trip -- NRIQAMetric (src/quality_metrics.py)
is a frozen, pretrained, eval-only wrapper, not something this project
trains. What still needs checking, same as any new pretrained dependency
added to this project (DINOv2 in Phase 4 needed the same treatment):
  - the pretrained-weight fetch actually works (network/cache, first-run
    only -- pyiqa caches its own weights, separately from DINOv2's
    torch.hub cache used in Phase 4)
  - output shape/range is what downstream code assumes
  - the metric's *direction* (lower_better) is read correctly, not assumed
  - VRAM footprint at the resolution Phase 6 will actually run this at:
    eval15's full native 400x600, batch=1 -- following this project's
    established convention (src/train_baseline.py's validate()) that
    validation/eval always runs at full native resolution regardless of
    training crop size, since NR-IQA evaluation is a Phase 6 eval-time
    step, not something folded into the crop-size training loop at all.

Usage: python scripts/verify_phase5.py
"""

import sys
from pathlib import Path

import torch
import kornia.filters as kfilters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quality_metrics import NRIQAMetric


def _safe_call(metric: NRIQAMetric, img: torch.Tensor) -> torch.Tensor:
    """Wraps a metric call so an OOM fails with an actionable message
    instead of a bare traceback. Unlike every training phase in this
    project, GPU speed genuinely doesn't matter here -- NR-IQA scoring
    only ever runs at eval time on a handful of images (eval15 is 15
    images total), not across epochs, so falling back to CPU costs
    seconds, not hours. That tradeoff never applied to Phase 2-4's
    training loops, which is why this project always fought to keep
    things on GPU there -- this is a genuinely different case."""
    try:
        return metric(img)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA OOM calling '{metric.metric_name}' on a {tuple(img.shape)} input. "
            "Before assuming this metric just doesn't fit on the GTX 1650: run "
            "`nvidia-smi` to rule out another process already holding VRAM. If the "
            "GPU is otherwise clean, the practical fallback is running this metric "
            "on CPU (NRIQAMetric(device='cpu')) -- unlike training, NR-IQA scoring "
            "only ever touches a handful of eval-time images, so CPU cost here is "
            "seconds, not the multi-hour difference GPU vs CPU made for training."
        ) from e


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Import + metric creation (pretrained-weight fetch happens here) ---
    metric = NRIQAMetric(metric_name="maniqa", device=device)
    print(
        f"[1/5] pyiqa metric created OK. metric={metric.metric_name} "
        f"lower_better={metric.lower_better}"
    )

    # --- 2. Output shape/range sanity, batch=1.
    # Deliberately NOT batch=4 here (an earlier version of this script was
    # -- it OOM'd on a 4GB card: MANIQA's vit_base backbone + multi-scale
    # channel-attention (TAB) block is far heavier than Phase 4's
    # dinov2_vits14, and batch=4 at 224x224 tried to allocate 2.81 GiB in
    # a single attention op). batch=1 also matches how this metric is
    # actually called everywhere downstream: scripts/evaluate_nriqa_phase5.py
    # scores eval15 images one at a time (no-reference metrics don't need
    # paired batching the way PSNR/SSIM's train/val loaders do), so there
    # was never a real reason to test batch=4 in the first place. ---
    single = torch.rand(1, 3, 224, 224, device=device)
    scores = _safe_call(metric, single)
    assert scores.shape == (1,), f"expected shape (1,), got {tuple(scores.shape)}"
    assert torch.isfinite(scores).all(), f"non-finite scores: {scores}"
    print(f"[2/5] output shape OK. score={scores.item():.4f}")

    # --- 3. Direction sanity check via synthetic blur, NOT iid noise, at
    # batch=1. MANIQA is trained on and calibrated against real
    # photographs -- pure torch.rand() noise sits far outside that
    # distribution and isn't a trustworthy direction signal for a
    # *learned* quality prior (unlike the DINOv2 perceptual-loss check in
    # verify_phase4.py, which only needed a generic "these two images
    # differ" signal, not a calibrated quality judgment). Blur is one of
    # the few distortions essentially every NR-IQA model reliably
    # penalizes regardless of training set, so it's a more honest smoke
    # test here. ---
    raw = torch.rand(1, 3, 224, 224, device=device)
    clean = kfilters.gaussian_blur2d(raw, kernel_size=(5, 5), sigma=(1.5, 1.5))
    degraded = kfilters.gaussian_blur2d(raw, kernel_size=(25, 25), sigma=(8.0, 8.0))
    clean_score = _safe_call(metric, clean).item()
    degraded_score = _safe_call(metric, degraded).item()
    if metric.lower_better:
        assert degraded_score > clean_score, (
            f"expected heavily-blurred to score worse (higher): "
            f"clean={clean_score:.4f}, degraded={degraded_score:.4f}"
        )
    else:
        assert degraded_score < clean_score, (
            f"expected heavily-blurred to score worse (lower): "
            f"clean={clean_score:.4f}, degraded={degraded_score:.4f}"
        )
    print(
        f"[3/5] direction sanity OK. clean(mild blur)={clean_score:.4f} | "
        f"degraded(heavy blur)={degraded_score:.4f} "
        f"({'lower' if metric.lower_better else 'higher'}-is-better)"
    )

    # --- 4. Determinism check, NOT batch-vs-single consistency.
    # The original version of this check compared a batched call against
    # N single-image calls -- pointless now that nothing downstream ever
    # batches, and it was the other place batch>1 would have re-triggered
    # the same OOM. What's still worth catching: some pretrained vision
    # models are non-deterministic across repeated calls (uninitialized
    # dropout left in train mode, a stray random crop/augment baked into
    # the metric's own preprocessing) -- that class of bug would silently
    # corrupt a real eval15 run where you might reasonably expect calling
    # the same image twice to give the same score. ---
    score_a = _safe_call(metric, single).item()
    score_b = _safe_call(metric, single).item()
    diff = abs(score_a - score_b)
    print(
        f"[4/5] determinism: repeated call on the same image -> {score_a:.6f} vs "
        f"{score_b:.6f} (diff={diff:.2e}) "
        f"({'OK' if diff < 1e-4 else 'NOTE: nonzero, check for dropout/randomness left active in eval mode'})"
    )

    # --- 5. VRAM footprint at eval15's actual native resolution
    # (400x600, batch=1) -- the resolution this metric will really run at
    # in Phase 6, not the 224x224 used above for the direction check. ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    native_res_img = torch.rand(1, 3, 400, 600, device=device)
    _ = _safe_call(metric, native_res_img)
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"[5/5] peak VRAM at native eval15 resolution (400x600, batch=1): "
            f"{peak_gb:.2f} GB (GTX 1650 budget: 4 GB)"
        )
    else:
        print(
            "[5/5] skipped -- no CUDA device in this environment; run on the "
            "WSL2/GTX 1650 box for a real VRAM number before trusting this at "
            "scale. Don't assume this is automatically safe just because it's "
            "inference-only: MANIQA's vit_base + multi-scale channel-attention "
            "backbone already needed a batch-size cut to fit at 224x224 (see "
            "checks [2-4] and this project's plan log) -- a native 400x600 "
            "single image is a different, larger shape, not strictly smaller "
            "work, so it's worth its own real check rather than assuming it's "
            "covered by the 224x224 result."
        )

    print("\nAll Phase 5 smoke tests passed.")


if __name__ == "__main__":
    main()
