"""
Frozen DINOv2 perceptual loss (Phase 4).

Adds a feature-space term to the reconstruction loss: instead of (or in
addition to) comparing pred/target pixel-by-pixel (Charbonnier) or in local
windows (SSIM), this compares their frozen DINOv2 patch-token embeddings.
The intuition carried over from the broader perceptual-loss literature
(originally popularized with VGG features) is that a pretrained vision
backbone's intermediate features correlate better with human-perceived
image similarity than raw pixel differences -- two images can differ a lot
in exact pixel values while still looking like the "same photo, evenly
lit," and a perceptual loss rewards that case more than Charbonnier alone
would.

DINOv2 specifically (over the more traditional VGG16) because:
  1. It's the backbone this project's plan already names for Phase 5's
     quality-assessment head -- reusing it here means one pretrained
     dependency, not two.
  2. DINOv2 is self-supervised (trained with no classification labels at
     all), so its features are shaped purely by visual structure/texture
     similarity rather than by what's discriminative for ImageNet
     categories -- arguably a better match for "does this look like a
     correctly-exposed version of the same scene" than a classifier's
     features.

Model size (dinov2_vits14, ~21M params, the smallest DINOv2 release) is a
deliberate choice, not a default: this project's 4GB-VRAM GTX 1650 already
needed batch-size/crop-size compromises to fit a 3.3M-param Restormer
(Phase 3) comfortably. Adding a second frozen network's forward pass (twice
per step -- once for pred with gradients, once for target without) is new
VRAM pressure on top of that regardless of how small the extra network is,
so starting with the smallest variant is the only reasonable first move.
Run scripts/verify_phase4.py for an actual peak-VRAM number at the training
defaults before committing to a long run -- same workflow Phase 3 used
before its own long run.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2PerceptualLoss(nn.Module):
    """L1 distance between frozen DINOv2 patch-token features of pred vs. target.

    Input is CHW-batched, float, notionally in [0, 1] (values may stray
    slightly outside that range coming from the restoration models'
    unclamped residual output -- consistent with how SSIMLoss is already
    used on unclamped `pred` elsewhere in this project; see
    ReconstructionLoss in src/losses.py). Internally:
      1. Resized (bilinear) to `feature_size` x `feature_size` -- DINOv2's
         patch-14 stride requires H, W divisible by 14, which this
         project's 128x128 training crops don't satisfy, and 224 is
         DINOv2's standard evaluation resolution.
      2. Normalized with ImageNet mean/std (what DINOv2 was pretrained
         with) -- independent of any normalization elsewhere in the
         pipeline, which works directly in [0, 1] pixel space.
      3. Passed through the frozen backbone; only patch tokens are
         compared (not the CLS token), since patch tokens retain
         per-region structure -- the relevant signal for a pixel-
         restoration task, where *where* something looks wrong matters,
         unlike CLS, which is trained for whole-image discrimination.

    Only `pred`'s branch keeps gradients; `target` is fixed ground truth
    and its features are computed under torch.no_grad() to avoid wasting
    memory on an activation graph that will never be backpropagated
    through.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, feature_size: int = 224, model_name: str = "dinov2_vits14"):
        super().__init__()
        self.feature_size = feature_size
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as e:  # network/cache failure -- fail loudly with actionable info
            raise RuntimeError(
                f"Could not load '{model_name}' via torch.hub. This needs network "
                "access on first run to fetch pretrained weights (cached afterward "
                "in ~/.cache/torch/hub, so subsequent runs work offline). "
                f"Original error: {e}"
            ) from e

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(self.feature_size, self.feature_size),
            mode="bilinear", align_corners=False,
        )
        return (x - self.mean) / self.std

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        return self.model.forward_features(x)["x_norm_patchtokens"]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat = self._features(pred)
        with torch.no_grad():
            target_feat = self._features(target)
        return F.l1_loss(pred_feat, target_feat)

    def train(self, mode: bool = True):
        # Defensive: keep the frozen backbone in eval() even if something
        # upstream calls .train() on the parent loss module -- otherwise
        # DINOv2's own dropout/stochastic-depth (if enabled in this hub
        # variant) would make the "fixed target" half of this loss
        # nondeterministic across calls.
        super().train(mode)
        self.model.eval()
        return self


if __name__ == "__main__":
    # Lightweight standalone check -- needs network access on first run to
    # fetch DINOv2 weights via torch.hub. Not run as part of the offline
    # verify_phaseN.py suite for that reason; scripts/verify_phase4.py
    # exercises this in the full training-machinery context instead.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = DINOv2PerceptualLoss().to(device)

    target = torch.rand(2, 3, 128, 128, device=device)
    pred = torch.clamp(target + 0.05 * torch.randn_like(target), 0, 1)
    identical_loss = loss_fn(target, target)
    perturbed_loss = loss_fn(pred, target)

    assert identical_loss.item() < 1e-5, "identical images should have ~zero perceptual distance"
    assert perturbed_loss.item() > identical_loss.item(), "perturbed image should score higher"
    print(
        f"OK. identical={identical_loss.item():.6f} | "
        f"perturbed={perturbed_loss.item():.4f}"
    )
