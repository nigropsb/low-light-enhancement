"""
Frozen pretrained no-reference IQA metric (Phase 5) -- MANIQA via pyiqa.

Eval-only, not a training loss: unlike src/perceptual_loss.py's DINOv2 term
(a training-time loss with gradients flowing into the model), this is a
reporting metric computed under torch.no_grad(), same convention as
batch_psnr/batch_ssim in src/metrics.py. The key structural difference from
those two: PSNR/SSIM are full-reference (need pred + ground truth), while
MANIQA is no-reference (NR) -- it scores a single image's perceived quality
with no ground truth involved at all. That's the whole reason it's in the
plan: it's the only metric in this project that can, in principle, score a
real low-light photo with no paired ground truth, which PSNR/SSIM/LPIPS
structurally cannot.

Why MANIQA over the older classics (NIQE, BRISQUE): those are hand-crafted
natural-scene-statistics models from the pre-deep-learning IQA era. MANIQA
is a ViT-based, human-judgment-calibrated NR-IQA model -- same "modern,
transformer-based, foundation-adjacent" positioning as Restormer/DINOv2
elsewhere in this project, and the one the project plan already named.

Why this is eval-only (the "lighter" Phase 5 option, not a trained
regression head): MANIQA ships pretrained, so integration is just a
forward pass through someone else's already-calibrated model -- no new
dataset (KonIQ-10k), no new training loop, no new failure surface on the
GTX 1650. See the project plan's Phase 5 entry for the full lighter-vs-
fuller reasoning.

Direction caveat: NR-IQA metrics are not all oriented the same way. MANIQA
is higher-is-better, but plenty of NR-IQA metrics (NIQE, BRISQUE) are
lower-is-better. Always read `.lower_better` off the instance rather than
hardcoding a direction -- this module is written to work with any pyiqa NR
metric via the `metric_name` argument, not just MANIQA specifically.
"""
from __future__ import annotations

import torch


class NRIQAMetric:
    """Thin wrapper around a pyiqa no-reference metric (default: MANIQA).

    Deliberately a plain class, not an nn.Module: there's no gradient path
    through this (eval-only, see module docstring), so none of the buffer/
    parameter/.train() machinery src/perceptual_loss.py needed for its
    DINOv2 branch applies here. pyiqa's create_metric() already handles
    device placement internally via its `device` argument.
    """

    def __init__(self, metric_name: str = "maniqa", device: torch.device | str = "cpu"):
        try:
            import pyiqa
        except ImportError as e:
            raise ImportError(
                "pyiqa is not installed. Run `pip install pyiqa` first -- see "
                "scripts/verify_phase5.py for the smoke test that exercises this "
                "import plus the first-run pretrained-weight download."
            ) from e

        try:
            self.metric = pyiqa.create_metric(metric_name, device=device)
        except Exception as e:
            # Same failure mode as DINOv2 in src/perceptual_loss.py: first
            # call fetches pretrained weights over the network (pyiqa's own
            # cache dir, not torch.hub's) -- fail loudly with actionable
            # info instead of a bare traceback mid-eval-loop.
            raise RuntimeError(
                f"Could not create pyiqa metric '{metric_name}'. This needs network "
                "access on first run to fetch pretrained weights (cached afterward, "
                "so subsequent runs work offline). "
                f"Original error: {e}"
            ) from e

        self.metric_name = metric_name
        # Some NR-IQA metrics (NIQE, BRISQUE) are lower-is-better; MANIQA is
        # higher-is-better. Read it off the instance so downstream sorting/
        # reporting code never has to hardcode a direction.
        self.lower_better = bool(getattr(self.metric, "lower_better", False))

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) float tensor in [0, 1], RGB -- same convention
        as every other tensor in this pipeline (src/data/lol_dataset.py's
        _load_image). Returns a (B,) tensor of per-image scores; pyiqa's
        raw output shape varies slightly by metric, so this flattens it
        rather than assuming (B, 1).

        No reference image needed -- that's the point of "no-reference."
        Don't confuse this call signature with batch_psnr/batch_ssim in
        src/metrics.py, which both take (pred, target) pairs.
        """
        return self.metric(img).flatten()

    @torch.no_grad()
    def batch_mean(self, img: torch.Tensor) -> float:
        """Convenience wrapper matching src/metrics.py's batch_psnr/
        batch_ssim signature (returns a plain python float, mean over the
        batch)."""
        return self(img).mean().item()


if __name__ == "__main__":
    # Lightweight standalone check, same philosophy as
    # src/perceptual_loss.py's __main__ block: needs network access on
    # first run to fetch MANIQA's pretrained weights.
    #
    # Deliberately NOT using torch.rand() iid noise as the "clean" sample
    # here (unlike the DINOv2 perceptual-loss check, which only needed a
    # generic "images differ" signal). MANIQA is trained on and calibrated
    # against real photographs; uniform random noise is far outside that
    # distribution and isn't a reliable direction check for a *learned*
    # aesthetic/quality prior. Instead: build a synthetic natural-image-ish
    # pattern via a mild Gaussian blur (low-pass random noise, roughly
    # approximating a real photo's 1/f spectral falloff), then blur that
    # much harder for the "degraded" version. Blur is one of the few
    # distortions essentially every NR-IQA model reliably penalizes,
    # regardless of training set -- a stronger, more honest sanity check
    # than a noise-direction test would be here.
    import kornia.filters as kfilters

    # batch=1 throughout: MANIQA's vit_base + multi-scale channel-attention
    # backbone is far heavier than Phase 4's dinov2_vits14, and this
    # metric is always called one image at a time downstream anyway
    # (scripts/evaluate_nriqa_phase5.py) -- see scripts/verify_phase5.py's
    # comments for the batch=4 OOM that established this.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = NRIQAMetric(device=device)

    raw = torch.rand(1, 3, 224, 224, device=device)
    clean = kfilters.gaussian_blur2d(raw, kernel_size=(5, 5), sigma=(1.5, 1.5))
    degraded = kfilters.gaussian_blur2d(raw, kernel_size=(25, 25), sigma=(8.0, 8.0))

    clean_score = metric.batch_mean(clean)
    degraded_score = metric.batch_mean(degraded)

    if metric.lower_better:
        assert degraded_score > clean_score, (
            f"expected degraded (heavily blurred) to score worse (higher, "
            f"lower_better=True) than clean: clean={clean_score:.4f}, "
            f"degraded={degraded_score:.4f}"
        )
    else:
        assert degraded_score < clean_score, (
            f"expected degraded (heavily blurred) to score worse (lower, "
            f"lower_better=False) than clean: clean={clean_score:.4f}, "
            f"degraded={degraded_score:.4f}"
        )

    direction = "lower" if metric.lower_better else "higher"
    print(
        f"OK. metric={metric.metric_name} (lower_better={metric.lower_better}, "
        f"{direction}-is-better) | clean(mild blur)={clean_score:.4f} | "
        f"degraded(heavy blur)={degraded_score:.4f}"
    )
