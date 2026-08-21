"""
Baseline U-Net for low-light image enhancement (Phase 2).

Purpose: validate the full pipeline (data -> model -> loss -> checkpointing)
with something simple and fast to iterate on, before Phase 3 brings in the
Restormer/SwinIR transformer backbone.

Design choices, tuned for a 4GB-VRAM GTX 1650 at 256x256 crops:
- base_channels=32 (half of the original U-Net's 64) and 3 downsampling
  stages instead of 4 -- enough capacity to prove the pipeline works,
  without the deepest feature map + activations eating VRAM.
- No BatchNorm. With batch_size=4 (see Phase 1 verified shapes), BatchNorm's
  running statistics are noisy; skipping it avoids that failure mode
  entirely rather than reaching for GroupNorm as a workaround.
- Residual formulation: the network predicts a correction added to the
  input rather than the absolute output. For low-light enhancement this is
  the easier function to learn (mostly "add back the light that's already
  structurally there") and trains more stably as a baseline.

Input: (B, 3, H, W) in [0, 1]. H and W must be divisible by 8 (three 2x
downsamples) -- the 256x256 crops from Phase 1 satisfy this.

Output: (B, 3, H, W), NOT clamped to [0, 1]. An earlier version clamped
inside forward(), but torch.clamp has zero gradient outside the clamped
range -- at random init a meaningful fraction of pixels land outside
[0, 1], so those pixels got no gradient at all, and training was visibly
unstable from step one. The Charbonnier loss against a target that IS in
[0, 1] pulls the raw output into range on its own during training. Clamp
explicitly at the call site when you need a displayable image or a valid
PSNR/SSIM input (see validate() in train_baseline.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convs + LeakyReLU. See module docstring for why no norm layer."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample, concat with the matching encoder skip, then convolve.

    `in_ch` is the channel count of the incoming feature map (before
    upsampling); it also equals the concatenated channel count after the
    skip connection, since each encoder stage doubles channels and halves
    spatial size symmetrically.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one size mismatches (odd input dims).
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.inc = ConvBlock(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)

        self.up1 = Up(c * 8, c * 4)
        self.up2 = Up(c * 4, c * 2)
        self.up3 = Up(c * 2, c)

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        y = self.up1(x4, x3)
        y = self.up2(y, x2)
        y = self.up3(y, x1)

        residual = self.outc(y)
        return x + residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape/param sanity check -- no GPU or dataset required.
    model = UNetBaseline()
    dummy = torch.rand(4, 3, 256, 256)
    out = model(dummy)
    assert out.shape == dummy.shape, f"shape mismatch: {out.shape} vs {dummy.shape}"
    print(f"OK. Output shape: {tuple(out.shape)} (range [{out.min():.2f}, {out.max():.2f}], "
          f"unclamped by design). Params: {count_parameters(model):,}")
