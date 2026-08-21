"""
Validation-time image quality metrics: PSNR and SSIM.

These are for logging/model-selection only (not backpropagated), so they're
computed under torch.no_grad() by the caller. Thin wrappers around kornia
(already in the Phase 0 stack) rather than reimplementations, to keep one
source of truth for the SSIM math shared with src/losses.py.
"""

import torch


@torch.no_grad()
def batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    import kornia.metrics as kmetrics
    return kmetrics.psnr(pred, target, max_val).item()


@torch.no_grad()
def batch_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
                max_val: float = 1.0) -> float:
    import kornia.metrics as kmetrics
    ssim_map = kmetrics.ssim(pred, target, window_size, max_val)
    return ssim_map.mean().item()


if __name__ == "__main__":
    a = torch.rand(4, 3, 64, 64)
    b = a.clone()
    b_noisy = torch.clamp(a + 0.05 * torch.randn_like(a), 0, 1)

    identical_psnr = batch_psnr(a, b)
    noisy_psnr = batch_psnr(a, b_noisy)
    assert identical_psnr > noisy_psnr, "PSNR should drop when images differ"

    identical_ssim = batch_ssim(a, b)
    noisy_ssim = batch_ssim(a, b_noisy)
    assert identical_ssim > noisy_ssim, "SSIM should drop when images differ"

    print(f"OK. identical: PSNR={identical_psnr:.1f} SSIM={identical_ssim:.3f} | "
          f"noisy: PSNR={noisy_psnr:.1f} SSIM={noisy_ssim:.3f}")
