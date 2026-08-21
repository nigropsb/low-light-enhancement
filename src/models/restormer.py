"""
Restormer-style transformer backbone for low-light enhancement (Phase 3).

Own implementation of the general architecture described in the Restormer
paper (Zamir et al., "Restormer: Efficient Transformer for High-Resolution
Image Restoration") -- built independently against this project's own data
pipeline and conventions, not a port of any specific codebase. Chosen over
SwinIR's window attention specifically for VRAM reasons: SwinIR's attention
still scales with spatial window size, while Restormer's MDTA (Multi-DConv
Head Transposed Attention) computes attention *across channels* rather than
across pixels, so its cost is independent of image resolution -- the more
forgiving choice on a 4GB, non-Tensor-Core GTX 1650 than a pure spatial-
attention design.

Sizing, scaled down from the paper's reference config (dim=48,
num_blocks=[4,6,6,8], num_refinement_blocks=4, ffn_expansion_factor=2.66,
~26M params) to keep the model closer to the baseline's iteration budget:
    dim=24, num_blocks=(2,3,3,4), num_refinement_blocks=2,
    ffn_expansion_factor=2.0
This roughly quarters both the channel width and the block counts. Run
scripts/verify_phase3.py for the exact parameter count and (on the real GPU)
peak VRAM at the training crop size/batch size -- both are open questions
flagged in the project plan, not assumed here.

Design continuity with src/models/unet.py (Phase 2):
- Same residual formulation: forward() returns `x + residual`, unclamped.
  The Phase 2 postmortem (see UNetBaseline's docstring) already found that
  clamping inside forward() kills gradients on out-of-range pixels at random
  init; ReconstructionLoss pulls the raw output into [0, 1] during training
  either way, so there's no reason to relitigate that here.
- Same divisibility requirement: H and W must be divisible by 8 (three 2x
  downsamples). PairedTransform's eval-mode center-crop (src/data/
  transforms.py) already enforces this -- that code's docstring called out
  "transformer/window-based backbones like SwinIR" back in Phase 1,
  anticipating this exact constraint before Phase 3 existed.

Input: (B, 3, H, W) in [0, 1], H and W divisible by 8.
Output: (B, 3, H, W), NOT clamped to [0, 1] (see above).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) tensor.

    Restormer normalizes each pixel across its channels -- distinct from
    BatchNorm2d (per-channel, across the batch+spatial dims) and GroupNorm
    (per-sample, across channel groups). Implemented by permuting C to the
    last axis, applying a standard nn.LayerNorm, and permuting back, rather
    than a hand-rolled affine normalization -- this keeps autograd and
    numerics identical to a well-tested nn.LayerNorm.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.norm(x)
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention.

    Standard multi-head self-attention builds an (H*W) x (H*W) attention
    matrix -- quadratic in pixel count, and the main reason plain spatial
    attention is VRAM-hostile for restoration-sized images. MDTA instead
    transposes the roles: q/k/v are reshaped to (heads, C/heads, H*W) and
    the attention matrix is built over the (C/heads) x (C/heads) channel
    axis instead, with H*W acting as the reduction dimension inside the
    matmul. Cost is then independent of image resolution and quadratic only
    in the (small, fixed) per-head channel count -- the property that makes
    this workable on a 4GB card at 128x128+ crops.

    A depthwise 3x3 conv on q/k/v before attention (`qkv_dwconv`) injects the
    local spatial mixing that channel-only attention otherwise can't see.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=False
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        # L2-normalize along the H*W axis before the matmul -- keeps the
        # channel-vs-channel similarity scores well-conditioned regardless
        # of image size, same role as the 1/sqrt(d) scaling in standard
        # attention, just adapted to this transposed layout.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # (heads, C/heads, C/heads) attention over channels, not pixels.
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network.

    A standard transformer FFN (two 1x1 convs) widened with a depthwise 3x3
    conv in between for local context, then split into two halves gated
    multiplicatively (x1 * gelu(x2)) rather than passed through a plain
    nonlinearity -- lets the network suppress irrelevant feature channels
    per-pixel instead of just rescaling all of them together.
    """

    def __init__(self, dim: int, ffn_expansion_factor: float):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2) * x1
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Pre-norm MDTA + GDFN with residual connections around each sub-layer."""

    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float):
        super().__init__()
        self.norm1 = ChannelLayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = ChannelLayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """3x3 conv (halves channels) + PixelUnshuffle(2) (x4 channels, /2 spatial)
    -> net effect: channels x2, spatial /2. Chosen over strided conv/pooling
    because PixelUnshuffle is a pure reshape (no learned spatial aliasing),
    which tends to preserve high-frequency detail better for restoration."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    """3x3 conv (x2 channels) + PixelShuffle(2) (/4 channels, x2 spatial)
    -> net effect: channels /2, spatial x2. The learned-reshape counterpart
    to Downsample above, avoiding ConvTranspose2d's characteristic
    checkerboard artifacts."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    """3x3 conv stem instead of a non-overlapping patch-embed (e.g. ViT's
    strided patchify). Keeps neighboring-pixel context at the very first
    layer, which matters for a pixel-accurate restoration task in a way it
    doesn't for classification."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Restormer(nn.Module):
    """4-level encoder-decoder transformer, U-Net-shaped skip connections.

    Channel progression across levels 1-4: dim, dim*2, dim*4, dim*8, with
    heads doubling at each level (default (1,2,4,8)) so that channels-per-
    head stays constant throughout -- consistent with the paper's design and
    convenient for reasoning about MDTA's per-head cost independent of
    depth.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 24,
        num_blocks: tuple[int, int, int, int] = (2, 3, 3, 4),
        num_refinement_blocks: int = 2,
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.0,
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_channels, dim)

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor) for _ in range(num_blocks[3])]
        )

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor) for _ in range(num_blocks[0])]
        )

        self.refinement = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor) for _ in range(num_refinement_blocks)]
        )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        assert h % 8 == 0 and w % 8 == 0, (
            f"Restormer needs H, W divisible by 8 (three 2x downsamples); got {h}x{w}. "
            "PairedTransform's eval-mode center-crop already enforces this "
            "(see src/data/transforms.py)."
        )

        x1 = self.patch_embed(x)
        e1 = self.encoder_level1(x1)

        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        latent = self.latent(self.down3_4(e3))

        d3 = self.up4_3(latent)
        d3 = self.reduce_chan_level3(torch.cat([d3, e3], dim=1))
        d3 = self.decoder_level3(d3)

        d2 = self.up3_2(d3)
        d2 = self.reduce_chan_level2(torch.cat([d2, e2], dim=1))
        d2 = self.decoder_level2(d2)

        d1 = self.up2_1(d2)
        d1 = self.reduce_chan_level1(torch.cat([d1, e1], dim=1))
        d1 = self.decoder_level1(d1)

        r = self.refinement(d1)
        residual = self.output(r)
        return x + residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick shape/param sanity check -- no GPU or dataset required, mirrors
    # src/models/unet.py's own __main__ block.
    model = Restormer()
    dummy = torch.rand(2, 3, 128, 128)
    out = model(dummy)
    assert out.shape == dummy.shape, f"shape mismatch: {out.shape} vs {dummy.shape}"
    print(f"OK. Output shape: {tuple(out.shape)} (range [{out.min():.2f}, {out.max():.2f}], "
          f"unclamped by design). Params: {count_parameters(model):,}")
