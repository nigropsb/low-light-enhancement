"""
Frozen pretrained full-reference perceptual metric (Phase 6) -- LPIPS via
the `lpips` package.

Full-reference, unlike Phase 5's MANIQA: LPIPS needs both pred and a
ground-truth target, same call signature as batch_psnr/batch_ssim in
src/metrics.py, not NRIQAMetric's single-image signature. Structurally
this sits between those two precedents: full-reference like PSNR/SSIM, but
a pretrained-network feature-space comparison like Phase 4's
DINOv2PerceptualLoss -- eval-only though, computed under torch.no_grad(),
same convention as NRIQAMetric (src/quality_metrics.py), not a training
loss.

LPIPS vs. DINOv2 (src/perceptual_loss.py) -- these are NOT the same
perceptual construct, despite both being "pretrained-network feature
distance." Do not conflate them in the README:
  - DINOv2 (Phase 4, training-time loss): self-supervised ViT features, no
    human-judgment calibration at all -- shaped by a contrastive/
    distillation objective over unlabeled images.
  - LPIPS (Phase 6, eval-only metric): CNN features (AlexNet/VGG) with a
    small linear-calibration layer fit to the BAPPS dataset of actual
    human perceptual-similarity judgments. It's a metric built
    specifically to correlate with human "which image looks more similar"
    preferences -- DINOv2 was never calibrated against human judgments at
    all.
  Training against DINOv2 and then evaluating with LPIPS is a deliberate
  design choice (independent perceptual signals), not duplicated effort --
  but it means "the perceptual loss went down" and "LPIPS improved" are
  two different claims that happen to often move together, not the same
  claim twice. State this explicitly in the README per the Phase 5
  plan-log note.

Backbone choice: AlexNet (`net='alex'`), not VGG. The LPIPS paper itself
found AlexNet/SqueezeNet correlate at least as well as VGG with human
judgments while being far cheaper -- and since this is an eval-only metric
computed once per checkpoint over eval15 (not a training loss shaping
gradients, where VGG's smoother features are sometimes preferred), there's
no reason to pay VGG's extra compute here. `net='alex'` is also the
`lpips` package's own default.

Normalization trap, worth flagging explicitly: the `lpips` package's
default `forward()` assumes input already in [-1, 1] (`normalize=False`).
Every tensor in this pipeline is [0, 1] (src/data/lol_dataset.py's
_load_image, same convention DINOv2PerceptualLoss and NRIQAMetric both
already follow) -- calling LPIPS without `normalize=True` would silently
score wrong numbers (not crash, not NaN -- just wrong, since [0,1] data
fed to a [-1,1]-expecting network is a valid-looking but shifted input).
This wrapper always passes `normalize=True` internally so callers never
have to remember the convention mismatch. scripts/verify_phase6.py checks
this explicitly (identical pred/target must score ~0).
"""
from __future__ import annotations

import torch


class LPIPSMetric:
    """Thin wrapper around `lpips.LPIPS`, full-reference, eval-only.

    Deliberately not exposed as an nn.Module subclass at this layer (even
    though `lpips.LPIPS` itself is one internally) -- same rationale as
    NRIQAMetric: nothing here trains, so there's no reason for this
    wrapper's own interface to imply a gradient path or expose .train()/
    .parameters(). Always lower-is-better (0 = identical) -- unlike
    NRIQAMetric's per-metric-dependent direction, no `.lower_better` flag
    needed here.
    """

    def __init__(self, net: str = "alex", device: torch.device | str = "cpu"):
        try:
            import lpips
        except ImportError as e:
            raise ImportError(
                "lpips is not installed. Run `pip install lpips` first -- see "
                "scripts/verify_phase6.py for the smoke test that exercises this "
                "import plus the first-run pretrained-weight download."
            ) from e

        try:
            self.model = lpips.LPIPS(net=net).to(device)
        except Exception as e:
            # Same failure mode as DINOv2 (Phase 4) and MANIQA (Phase 5):
            # first call fetches pretrained weights over the network
            # (cached afterward).
            raise RuntimeError(
                f"Could not create LPIPS metric (net='{net}'). This needs network "
                "access on first run to fetch pretrained weights (cached afterward, "
                f"so subsequent runs work offline). Original error: {e}"
            ) from e

        self.model.eval()
        self.net = net
        self.device = torch.device(device) if isinstance(device, str) else device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: (B, 3, H, W) float tensors in [0, 1] -- same
        convention as batch_psnr/batch_ssim (src/metrics.py). Returns a
        (B,) tensor of per-image LPIPS distances (0 = identical, no fixed
        upper bound, but in practice rarely exceeds ~1 for natural
        images).

        `normalize=True` is passed explicitly on every call -- see the
        module docstring's normalization-trap note. Don't remove this.
        """
        return self.model(pred, target, normalize=True).flatten()

    @torch.no_grad()
    def batch_mean(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Convenience wrapper matching NRIQAMetric.batch_mean /
        src/metrics.py's batch_psnr/batch_ssim signature style."""
        return self(pred, target).mean().item()


if __name__ == "__main__":
    # Lightweight standalone check: identical images -> ~0, degraded ->
    # higher. Unlike NRIQAMetric's __main__ block, plain torch.rand() noise
    # is a valid direction check here -- LPIPS just needs "these two images
    # differ," not a calibrated real-photo aesthetic judgment, since it's
    # comparing pred against a *specific* target rather than scoring one
    # image against a learned quality prior.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = LPIPSMetric(device=device)

    target = torch.rand(1, 3, 224, 224, device=device)
    identical = target.clone()
    degraded = torch.clamp(target + 0.3 * torch.randn_like(target), 0, 1)

    identical_score = metric.batch_mean(identical, target)
    degraded_score = metric.batch_mean(degraded, target)
    assert degraded_score > identical_score, (
        f"expected degraded to score higher (worse): "
        f"identical={identical_score:.4f}, degraded={degraded_score:.4f}"
    )
    print(
        f"OK. net={metric.net} (lower-is-better) | "
        f"identical={identical_score:.4f} | degraded={degraded_score:.4f}"
    )
