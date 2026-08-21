"""
Reconstruction loss for Phase 2 (baseline), extended in Phase 3 (SSIM) and
Phase 4 (DINOv2 perceptual).

Kept as a single module by design: rather than swapping loss functions
between phases, each phase adds an optional weighted term (ssim_weight,
then perceptual_weight), all default-off so earlier phases' behavior is
reproduced exactly when the new weight is left at 0. This means the
training-loop call site -- `loss = loss_fn(pred, target)` in
train_one_epoch()/validate() (src/train_baseline.py) -- has never needed to
change as the loss grew from Phase 2 through Phase 4.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Smooth L1 variant used throughout restoration literature (Restormer,
    MPRNet, etc.): behaves like L1 away from zero, but is differentiable at
    zero, giving slightly better-behaved gradients early in training than
    plain L1.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps ** 2).mean()


class ReconstructionLoss(nn.Module):
    """Charbonnier + optional SSIM term + optional DINOv2 perceptual term.

    ssim_weight=0.0, perceptual_weight=0.0 (both default) -> baseline
    behavior: pure Charbonnier, matching Phase 2 exactly.

    Set ssim_weight > 0 (Phase 3) to blend in structural similarity.
    Set perceptual_weight > 0 (Phase 4) to additionally blend in a frozen
    DINOv2 feature-space term -- see src/perceptual_loss.py for the design
    rationale (why DINOv2, why the smallest variant, why patch tokens).

    Each optional term only imports/constructs its dependency when its
    weight is > 0, so earlier phases don't pick up new dependencies just by
    importing this module: kornia is only needed if ssim_weight > 0;
    torch.hub's DINOv2 fetch only happens if perceptual_weight > 0.

    IMPORTANT (new in Phase 4): unlike CharbonnierLoss and kornia's
    SSIMLoss, which are both stateless, the DINOv2 branch holds real
    pretrained weights that must live on the same device as the input
    tensors. `ReconstructionLoss` is an nn.Module specifically so that
    `.to(device)` moves everything -- including the frozen DINOv2
    submodule -- at once. Callers that build this with perceptual_weight
    > 0 MUST call `.to(device)` on the returned loss object, e.g.:

        loss_fn = ReconstructionLoss(perceptual_weight=0.1).to(device)

    This wasn't needed in Phase 2/3 (CharbonnierLoss has no parameters,
    SSIMLoss is stateless -- both operate purely on whatever device their
    input tensors already happen to be on), so it's an easy thing to forget
    when copying those call sites forward. src/train_phase4.py does this
    explicitly and comments on why.
    """

    def __init__(
        self,
        ssim_weight: float = 0.0,
        charbonnier_eps: float = 1e-3,
        ssim_window: int = 11,
        perceptual_weight: float = 0.0,
        perceptual_feature_size: int = 224,
    ):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps)

        self.ssim_weight = ssim_weight
        if ssim_weight > 0:
            import kornia.losses as klosses
            self._ssim_loss = klosses.SSIMLoss(window_size=ssim_window, reduction="mean")

        self.perceptual_weight = perceptual_weight
        if perceptual_weight > 0:
            from src.perceptual_loss import DINOv2PerceptualLoss
            self._perceptual_loss = DINOv2PerceptualLoss(feature_size=perceptual_feature_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.charbonnier(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * self._ssim_loss(pred, target)
        if self.perceptual_weight > 0:
            loss = loss + self.perceptual_weight * self._perceptual_loss(pred, target)
        return loss


if __name__ == "__main__":
    pred = torch.rand(4, 3, 64, 64)
    target = torch.rand(4, 3, 64, 64)
    loss_fn = ReconstructionLoss()  # baseline default: Charbonnier only
    loss = loss_fn(pred, target)
    assert loss.dim() == 0 and loss.item() > 0
    print(f"OK. Charbonnier-only loss: {loss.item():.4f}")
