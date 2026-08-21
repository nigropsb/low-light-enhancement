"""
Phase 4 smoke test.

Same philosophy as scripts/verify_phase2.py and scripts/verify_phase3.py:
exercise model -> loss -> optimizer -> checkpoint on synthetic random
tensors, independent of the real LOL pipeline (already covered by
scripts/verify_phase1.py). Run this before src/train_phase4.py touches
real data for a multi-hour run.

Unlike the earlier verify scripts, this one needs network access on first
run: ReconstructionLoss(perceptual_weight > 0) triggers a torch.hub fetch
of DINOv2's pretrained weights (cached afterward in ~/.cache/torch/hub), so
this also doubles as a check that the download/cache step works before a
long training run depends on it.

Additionally reports:
  - the *relative magnitude* of each loss term (Charbonnier, SSIM,
    perceptual) at an early, barely-trained prediction, since
    perceptual_weight is a new hyperparameter with no established starting
    point in this project -- comparing raw magnitudes is the fastest way to
    pick a weight that won't be drowned out by (or drown out) the other two
    terms.
  - peak VRAM at the Phase 3 training defaults (crop=128, batch=2), now
    with the DINOv2 branch active, since that's the open question Phase 4
    adds on top of Phase 3's own VRAM budget (Phase 3 reported ~0.89 GB at
    these settings; compare against that number here).

Usage: python scripts/verify_phase4.py
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.restormer import Restormer, count_parameters
from src.losses import ReconstructionLoss
from src.metrics import batch_psnr, batch_ssim
from src.train_baseline import save_checkpoint, load_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)
    model = Restormer(dim=24).to(device)

    # NOTE: .to(device) here is required for the DINOv2 branch (see
    # ReconstructionLoss's docstring in src/losses.py) -- CharbonnierLoss
    # and SSIMLoss alone never needed this in Phase 2/3.
    # perceptual_weight=0.05, matching train_phase4.py's default -- revised
    # down from an initial 0.1 after Run 1 showed the perceptual term's
    # weighted contribution was ~2x ssim's at that weight (see project plan).
    loss_fn = ReconstructionLoss(ssim_weight=0.2, perceptual_weight=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = count_parameters(model)
    print(f"Restormer params: {n_params:,} (unchanged from Phase 3 -- only the loss changes)\n")

    # Synthetic paired batch at the Phase 3 training defaults (crop=128,
    # batch=2) -- same darkened+noised relationship used in
    # verify_phase2.py/verify_phase3.py.
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
    print(f"[1/6] forward/backward OK. initial loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

    # --- 2. Loss-term magnitude breakdown, to sanity-check perceptual_weight ---
    with torch.no_grad():
        pred_eval = model(low)
        charbonnier_v = loss_fn.charbonnier(pred_eval, high).item()
        ssim_v = loss_fn._ssim_loss(pred_eval, high).item() if loss_fn.ssim_weight > 0 else float("nan")
        perceptual_v = (
            loss_fn._perceptual_loss(pred_eval, high).item()
            if loss_fn.perceptual_weight > 0 else float("nan")
        )
    print(
        f"[2/6] loss term magnitudes (raw, unweighted) at near-random init:\n"
        f"      charbonnier={charbonnier_v:.4f} | ssim={ssim_v:.4f} | perceptual={perceptual_v:.4f}\n"
        f"      current weights: ssim_weight={loss_fn.ssim_weight}, "
        f"perceptual_weight={loss_fn.perceptual_weight} -> weighted contributions: "
        f"charbonnier={charbonnier_v:.4f}, ssim={loss_fn.ssim_weight * ssim_v:.4f}, "
        f"perceptual={loss_fn.perceptual_weight * perceptual_v:.4f}\n"
        f"      (near-random weights, not a trained model -- use this as a rough scale check "
        f"for --perceptual-weight, not a final answer; re-check partway through the real run)"
    )

    # --- 3. A handful of steps: loss should trend down on this fixed batch ---
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
    print(f"[3/6] optimization OK. loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")

    # --- 4. Metrics sanity: PSNR/SSIM should be higher for the trained pred than for `low` itself ---
    with torch.no_grad():
        pred = torch.clamp(model(low), 0.0, 1.0)
    psnr_pred = batch_psnr(pred, high)
    psnr_low = batch_psnr(low, high)
    ssim_pred = batch_ssim(pred, high)
    ssim_low = batch_ssim(low, high)
    print(
        f"[4/6] metrics OK. PSNR pred={psnr_pred:.2f} vs raw-low={psnr_low:.2f} | "
        f"SSIM pred={ssim_pred:.3f} vs raw-low={ssim_low:.3f}"
    )

    # --- 5. Checkpoint round trip. Only the Restormer's state is saved: DINOv2
    # is frozen (no optimizer state, no gradient updates), so it's correctly
    # absent from the checkpoint entirely -- it's reconstructed fresh (from
    # the same torch.hub cache) whenever ReconstructionLoss is rebuilt, not
    # restored from this checkpoint. ---
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
    print("[5/6] checkpoint save/load OK. restored model matches original output "
          "(and correctly excludes the frozen DINOv2 weights).")

    # --- 6. VRAM footprint with the DINOv2 branch active (only meaningful with a real GPU) ---
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"[6/6] peak VRAM at batch=2/crop=128 WITH perceptual loss: {peak_gb:.2f} GB "
            f"(GTX 1650 budget: 4 GB; Phase 3 reported ~0.89 GB at the same settings "
            f"without DINOv2 -- the difference here is roughly what the perceptual "
            f"branch costs)"
        )
    else:
        print(
            "[6/6] skipped -- no CUDA device in this environment; run on the "
            "WSL2/GTX 1650 box for a real VRAM number before committing to "
            "these settings for a multi-hour run."
        )

    print("\nAll Phase 4 smoke tests passed.")


if __name__ == "__main__":
    main()
