"""
Phase 6 smoke test -- LPIPS metric integration (AlexNet via `lpips`).

Same philosophy as scripts/verify_phase4/5.py: exercise the new component
in isolation on synthetic data before it's trusted inside the real
benchmarking script (scripts/benchmark_phase6.py) that touches all three
trained checkpoints and the real eval15 set. Run this first.

Structurally closer to verify_phase5.py than verify_phase4.py -- no
optimizer, no gradients, no checkpoint to round-trip, LPIPSMetric
(src/lpips_metric.py) is a frozen eval-only wrapper. One real difference
from Phase 5's smoke test: LPIPS is full-reference (needs pred + target,
like batch_psnr/batch_ssim), so the direction check can use an actual
identical-vs-degraded-relative-to-target pair rather than needing a
domain-calibrated distortion like Phase 5's blur trick -- and the
normalize=True convention (see src/lpips_metric.py's docstring) gets its
own explicit check here, since getting it wrong doesn't crash, it just
silently scores wrong numbers (confirmed ~65% off on a synthetic pair
during development -- see the project plan's Phase 6 entry).

Usage: python scripts/verify_phase6.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lpips_metric import LPIPSMetric


def _safe_call(metric: LPIPSMetric, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Same OOM-message wrapper convention as verify_phase5.py's
    _safe_call -- GPU speed doesn't matter here either, LPIPS only ever
    runs at eval time on a handful of images."""
    try:
        return metric(pred, target)
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            f"CUDA OOM calling LPIPS (net='{metric.net}') on a {tuple(pred.shape)} input. "
            "Run `nvidia-smi` to rule out another process holding VRAM first. If the GPU "
            "is otherwise clean, fall back to CPU (LPIPSMetric(device='cpu')) -- LPIPS "
            "scoring only ever touches a handful of eval-time images, so CPU cost here "
            "is seconds, not the multi-hour difference GPU vs CPU made for training."
        ) from e


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Import + metric creation (pretrained-weight fetch happens here) ---
    metric = LPIPSMetric(net="alex", device=device)
    print(f"[1/5] lpips metric created OK. net={metric.net} (lower-is-better, always)")

    # --- 2. Output shape/range sanity, batch=1, [0,1]-range input. ---
    a = torch.rand(1, 3, 224, 224, device=device)
    b = torch.rand(1, 3, 224, 224, device=device)
    scores = _safe_call(metric, a, b)
    assert scores.shape == (1,), f"expected shape (1,), got {tuple(scores.shape)}"
    assert torch.isfinite(scores).all(), f"non-finite scores: {scores}"
    print(f"[2/5] output shape OK. score={scores.item():.4f}")

    # --- 3. normalize=True correctness + direction check, combined.
    # An identical pair (pred is literally target) must score ~0 -- if
    # this is instead some large nonzero number, that's the normalize
    # convention silently mismatched (see src/lpips_metric.py's docstring:
    # [0,1] data fed to a [-1,1]-expecting network doesn't crash, it just
    # scores wrong -- measured ~65% off on a synthetic degraded pair
    # during development). A meaningfully-degraded pred vs. the same
    # target must score higher than the identical case. ---
    target = torch.rand(1, 3, 224, 224, device=device)
    identical = target.clone()
    degraded = torch.clamp(target + 0.3 * torch.randn_like(target), 0, 1)
    identical_score = _safe_call(metric, identical, target).item()
    degraded_score = _safe_call(metric, degraded, target).item()
    assert identical_score < 0.05, (
        f"identical pred vs. target scored {identical_score:.4f}, expected near 0 -- "
        "check the normalize=True convention in src/lpips_metric.py"
    )
    assert degraded_score > identical_score, (
        f"expected degraded to score worse (higher): "
        f"identical={identical_score:.4f}, degraded={degraded_score:.4f}"
    )
    print(
        f"[3/5] normalize + direction OK. identical={identical_score:.4f} | "
        f"degraded={degraded_score:.4f}"
    )

    # --- 4. Determinism check -- same pair called twice should agree. ---
    score_a = _safe_call(metric, degraded, target).item()
    score_b = _safe_call(metric, degraded, target).item()
    diff = abs(score_a - score_b)
    print(
        f"[4/5] determinism: repeated call on the same pair -> {score_a:.6f} vs "
        f"{score_b:.6f} (diff={diff:.2e}) "
        f"({'OK' if diff < 1e-4 else 'NOTE: nonzero, check for dropout/randomness left active in eval mode'})"
    )

    # --- 5. VRAM footprint at eval15's actual native resolution
    # (400x600, batch=1) -- the resolution this metric will really run at
    # in Phase 6's benchmark script, alongside a loaded Restormer/U-Net
    # checkpoint (not profiled here in combination -- see this script's
    # printed note). ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    native_pred = torch.rand(1, 3, 400, 600, device=device)
    native_target = torch.rand(1, 3, 400, 600, device=device)
    _ = _safe_call(metric, native_pred, native_target)
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"[5/5] peak VRAM at native eval15 resolution (400x600, batch=1): "
            f"{peak_gb:.2f} GB (GTX 1650 budget: 4 GB). AlexNet is far lighter than "
            "Phase 5's MANIQA vit_base, so this number alone should be comfortable -- "
            "but scripts/benchmark_phase6.py also holds a Restormer/U-Net checkpoint "
            "in memory concurrently, same unprofiled-combination caveat Phase 5 flagged "
            "for MANIQA + Restormer together."
        )
    else:
        print(
            "[5/5] skipped -- no CUDA device in this environment; run on the "
            "WSL2/GTX 1650 box for a real VRAM number before trusting this at scale."
        )

    print("\nAll Phase 6 smoke tests passed.")


if __name__ == "__main__":
    main()
