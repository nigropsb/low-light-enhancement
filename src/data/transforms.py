"""
Paired augmentations for low-light enhancement.

Both the low-light input and the normal-light target must receive the *same*
geometric transform (crop, flip, rotation) but only the input should get any
photometric/noise perturbation intended to simulate additional degradation
diversity. Kornia is used because its transforms operate on tensors (GPU-ready)
and support deterministic paired application via manual seeding.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import kornia.augmentation as K
import torch


@dataclass
class AugmentConfig:
    crop_size: int = 256
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate_p: float = 0.5
    train: bool = True


class PairedTransform:
    """Applies identical geometric augmentation to a (low, high) image pair.

    Usage:
        transform = PairedTransform(AugmentConfig(crop_size=256, train=True))
        low_t, high_t = transform(low_tensor, high_tensor)

    Expects low/high as CHW float tensors in [0, 1], same H, W.
    """

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, low: torch.Tensor, high: torch.Tensor):
        if not self.cfg.train:
            return self._center_crop(low, high)

        low, high = self._random_crop_pair(low, high)

        # Random flips/rotation: draw once, apply identically to both images.
        if random.random() < self.cfg.hflip_p:
            low, high = torch.flip(low, dims=[-1]), torch.flip(high, dims=[-1])
        if random.random() < self.cfg.vflip_p:
            low, high = torch.flip(low, dims=[-2]), torch.flip(high, dims=[-2])
        if random.random() < self.cfg.rotate_p:
            k = random.choice([1, 2, 3])
            low, high = torch.rot90(low, k, dims=[-2, -1]), torch.rot90(high, k, dims=[-2, -1])

        return low, high

    def _random_crop_pair(self, low: torch.Tensor, high: torch.Tensor):
        _, h, w = low.shape
        cs = self.cfg.crop_size
        if h < cs or w < cs:
            # Pad if the source image is smaller than the crop size.
            pad_h, pad_w = max(0, cs - h), max(0, cs - w)
            low = torch.nn.functional.pad(low, (0, pad_w, 0, pad_h), mode="reflect")
            high = torch.nn.functional.pad(high, (0, pad_w, 0, pad_h), mode="reflect")
            _, h, w = low.shape
        top = random.randint(0, h - cs)
        left = random.randint(0, w - cs)
        low = low[:, top : top + cs, left : left + cs]
        high = high[:, top : top + cs, left : left + cs]
        return low, high

    def _center_crop(self, low: torch.Tensor, high: torch.Tensor):
        # Validation/test: no random crop, just ensure divisibility by 8
        # (common requirement for transformer/window-based backbones like SwinIR).
        _, h, w = low.shape
        h8, w8 = h - h % 8, w - w % 8
        return low[:, :h8, :w8], high[:, :h8, :w8]


def synthetic_low_light_degrade(
    clean: torch.Tensor,
    gamma_range: tuple[float, float] = (2.0, 3.5),
    noise_sigma_range: tuple[float, float] = (0.01, 0.05),
) -> torch.Tensor:
    """Generates a synthetic low-light version of a clean image.

    Applies gamma darkening + approximate Poisson-Gaussian sensor noise, per the
    synthetic-data augmentation strategy in the project plan (Phase 1, optional).
    `clean` is a CHW float tensor in [0, 1].
    """
    gamma = random.uniform(*gamma_range)
    darkened = torch.clamp(clean, 1e-4, 1.0) ** gamma

    # Poisson component (signal-dependent) approximated via scaled Gaussian,
    # plus a flat Gaussian read-noise term.
    sigma = random.uniform(*noise_sigma_range)
    poisson_like = torch.randn_like(darkened) * torch.sqrt(darkened.clamp(min=1e-4)) * sigma
    read_noise = torch.randn_like(darkened) * (sigma * 0.5)

    degraded = darkened + poisson_like + read_noise
    return torch.clamp(degraded, 0.0, 1.0)
