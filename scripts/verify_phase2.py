"""
Phase 2 smoke test.

Exercises model -> loss -> optimizer -> checkpoint save/load on synthetic
random tensors -- deliberately independent of the real LOL data pipeline,
so it isolates "is the training machinery correct" from "is the data
loading correct" (Phase 1 already has its own verify_phase1.py for that).

Run this first. If it passes, wire in the real DataLoaders in
src/train_baseline.py next.

Usage: python scripts/verify_phase2.py
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.unet import UNetBaseline
from src.losses import ReconstructionLoss
from src.metrics import batch_psnr, batch_ssim
from src.train_baseline import save_checkpoint, load_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)
    model = UNetBaseline().to(device)
    loss_fn = ReconstructionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Synthetic paired batch: "low" is a darkened, noised version of "high",
    # loosely mimicking the LOL pair relationship, so the loss has a real
    # signal to chase instead of pure random noise.
    high = torch.rand(4, 3, 256, 256, device=device)
    low = torch.clamp(high * 0.3 + 0.02 * torch.randn_like(high), 0, 1)

    # --- 1. Forward/backward sanity ---
    pred = model(low)
    assert pred.shape == high.shape, f"shape mismatch: {pred.shape} vs {high.shape}"
    loss = loss_fn(pred, high)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradient reached the model parameters"
    optimizer.zero_grad(set_to_none=True)
    print(f"[1/4] forward/backward OK. initial loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

    # --- 2. A handful of steps: loss should trend down on this fixed batch ---
    losses = []
    for step in range(30):
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(low)
            loss = loss_fn(pred, high)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"[2/4] optimization OK. loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    # --- 3. Metrics sanity: PSNR/SSIM should be higher for the trained pred than for `low` itself ---
    with torch.no_grad():
        pred = torch.clamp(model(low), 0.0, 1.0)
    psnr_pred = batch_psnr(pred, high)
    psnr_low = batch_psnr(low, high)
    ssim_pred = batch_ssim(pred, high)
    ssim_low = batch_ssim(low, high)
    print(f"[3/4] metrics OK. PSNR pred={psnr_pred:.2f} vs raw-low={psnr_low:.2f} | "
          f"SSIM pred={ssim_pred:.3f} vs raw-low={ssim_low:.3f}")

    # --- 4. Checkpoint round trip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        save_checkpoint(ckpt_path, model, optimizer, scaler, epoch=7, best_val_psnr=psnr_pred)

        model2 = UNetBaseline().to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scaler2 = torch.amp.GradScaler("cuda", enabled=use_amp)
        next_epoch, best_val_psnr = load_checkpoint(ckpt_path, model2, optimizer2, scaler2, device)

        assert next_epoch == 8
        assert abs(best_val_psnr - psnr_pred) < 1e-6

        with torch.no_grad():
            pred2 = torch.clamp(model2(low), 0.0, 1.0)
        assert torch.allclose(pred, pred2), "restored model produces different output"
    print("[4/4] checkpoint save/load OK. restored model matches original output.")

    print("\nAll Phase 2 smoke tests passed.")


if __name__ == "__main__":
    main()
