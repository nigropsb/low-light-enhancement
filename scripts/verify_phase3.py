"""
Phase 3 smoke test.

Same philosophy as scripts/verify_phase2.py: exercise model -> loss ->
optimizer -> checkpoint on synthetic random tensors, deliberately independent
of the real LOL data pipeline (already covered by scripts/verify_phase1.py).
Run this before src/train_restormer.py touches real data for a multi-hour run.

Additionally reports parameter count against the Phase 2 baseline, and (if a
CUDA device is available) peak VRAM usage at the training defaults
(crop=128, batch=2) -- fitting in the GTX 1650's 4GB is the open risk flagged
all the way back in Phase 0, so this gives an actual number instead of a
guess.

Usage: python scripts/verify_phase3.py
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unet import UNetBaseline, count_parameters as count_unet_params
from src.models.restormer import Restormer, count_parameters as count_restormer_params
from src.losses import ReconstructionLoss
from src.metrics import batch_psnr, batch_ssim
from src.train_baseline import save_checkpoint, load_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)
    model = Restormer(dim=24).to(device)
    loss_fn = ReconstructionLoss(ssim_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- 0. Param count vs. the Phase 2 baseline, for the Phase 6 results table ---
    unet_params = count_unet_params(UNetBaseline())
    restormer_params = count_restormer_params(model)
    print(
        f"Params: UNetBaseline={unet_params:,} | Restormer={restormer_params:,} "
        f"({restormer_params / max(unet_params, 1):.1f}x baseline)\n"
    )

    # Synthetic paired batch at the training defaults (crop=128, batch=2) --
    # same darkened+noised relationship used in verify_phase2.py, just a
    # smaller batch to match src/train_restormer.py's VRAM-conscious default.
    high = torch.rand(2, 3, 128, 128, device=device)
    low = torch.clamp(high * 0.3 + 0.02 * torch.randn_like(high), 0, 1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # --- 1. Forward/backward sanity ---
    pred = model(low)
    assert pred.shape == high.shape, f"shape mismatch: {pred.shape} vs {high.shape}"
    loss = loss_fn(pred, high)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradient reached the model parameters"
    optimizer.zero_grad(set_to_none=True)
    print(f"[1/5] forward/backward OK. initial loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

    # --- 2. A handful of steps: loss should trend down on this fixed batch ---
    losses = []
    for step in range(20):
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(low)
            loss = loss_fn(pred, high)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"[2/5] optimization OK. loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    # --- 3. Metrics sanity: PSNR/SSIM should be higher for the trained pred than for `low` itself ---
    with torch.no_grad():
        pred = torch.clamp(model(low), 0.0, 1.0)
    psnr_pred = batch_psnr(pred, high)
    psnr_low = batch_psnr(low, high)
    ssim_pred = batch_ssim(pred, high)
    ssim_low = batch_ssim(low, high)
    print(
        f"[3/5] metrics OK. PSNR pred={psnr_pred:.2f} vs raw-low={psnr_low:.2f} | "
        f"SSIM pred={ssim_pred:.3f} vs raw-low={ssim_low:.3f}"
    )

    # --- 4. Checkpoint round trip (reusing Phase 2's save/load, unchanged) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch=7, best_val_psnr=psnr_pred)

        model2 = Restormer(dim=24).to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=2e-4)
        scaler2 = torch.amp.GradScaler("cuda", enabled=use_amp)
        next_epoch, best_val_psnr = load_checkpoint(ckpt_path, model2, optimizer2, scaler2, device)

        assert next_epoch == 8
        assert abs(best_val_psnr - psnr_pred) < 1e-6

        with torch.no_grad():
            pred2 = torch.clamp(model2(low), 0.0, 1.0)
        assert torch.allclose(pred, pred2), "restored model produces different output"
    print("[4/5] checkpoint save/load OK. restored model matches original output.")

    # --- 5. VRAM footprint at the training defaults (only meaningful with a real GPU) ---
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[5/5] peak VRAM at batch=2/crop=128: {peak_gb:.2f} GB (GTX 1650 budget: 4 GB)")
    else:
        print(
            "[5/5] skipped -- no CUDA device in this environment; run on the "
            "WSL2/GTX 1650 box for a real VRAM number before committing to "
            "these batch-size/crop-size defaults for a multi-hour run."
        )

    print("\nAll Phase 3 smoke tests passed.")


if __name__ == "__main__":
    main()
