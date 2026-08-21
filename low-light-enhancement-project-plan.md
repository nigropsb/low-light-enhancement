# Low-Light Image Enhancement — Portfolio Project Plan

## Progress Log

### Phase 0 — Setup: COMPLETE
- Environment: WSL2 (Ubuntu), confirmed via `wsl -l -v` (distro-level VERSION must read
  "2" — this is separate from the `wsl --version` platform/launcher version).
- Converted distro from WSL1 → WSL2. Initial `wsl --set-version` attempt failed with
  `HCS_E_SERVICE_NOT_AVAILABLE`; resolved by enabling the `VirtualMachinePlatform`
  Windows feature via DISM and rebooting:
  ```
  dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
  ```
- GPU driver: original NVIDIA driver was version 465.89 (~2021), which predates
  WSL2 CUDA passthrough support (requires 470.76+). Symptom was
  `/usr/lib/wsl/lib/nvidia-smi` containing a literal `<DISABLED>` placeholder stub
  instead of the real binary.
  - **Resolution:** installed current NVIDIA **Studio Driver** (chosen over Game
    Ready for its more conservative validation cycle against CUDA/compute
    workloads — relevant for long, unattended training runs). Now on driver
    610.57.01, CUDA UMD version 13.3.
  - Rationale for Studio vs Game Ready: functionally equivalent CUDA driver
    components either way; Studio prioritizes stability, Game Ready prioritizes
    faster/more frequent updates. Stability was judged the better fit here.
- Confirmed GPU recognized inside WSL: `nvidia-smi` shows **NVIDIA GeForce GTX
  1650, 4GB VRAM**. (Later relevant: the GTX 1650 is Turing *without* Tensor
  Cores — see Phase 2 AMP findings below.)
- Project directory created under the Linux filesystem (not `/mnt/c/...`):
  `~/projects/low-light-enhancement`.
- Python virtual environment (`venv`) created and activated.
- Installed CUDA-enabled PyTorch (`cu121` index build) plus `timm`, `kornia`,
  `numpy`, `pandas`, `opencv-python`, `matplotlib`.
  - Hit `libGL.so.1: cannot open shared object file` on first `cv2` import
    (standard `opencv-python` needs system GL libs, not present by default on
    headless WSL). **Resolved by installing system packages** (chosen over
    switching to `opencv-python-headless`, to preserve the ability to use
    `cv2.imshow` for visual debugging during development):
    ```
    sudo apt install -y libgl1 libglib2.0-0
    ```
  - Note: this is a system-level (`apt`) install, applied at the WSL/Ubuntu
    distro level — independent of the Python venv.
- Verified end-to-end: `torch.cuda.is_available()` → `True`,
  `torch.cuda.get_device_name(0)` → `NVIDIA GeForce GTX 1650`.
- `git init` completed in the project directory.
- **Open risk (not yet acted on):** 4GB VRAM may constrain Phase 3–4
  (transformer backbone + DINOv2 perceptual loss). Mitigations identified:
  small batch size, gradient accumulation, mixed precision (AMP), or offloading
  heavy training runs to a cloud GPU (Colab/Vertex AI) while keeping WSL for
  local dev and data work.

### Phase 1 — Data: COMPLETE
- Obtained the LOL dataset via **Kaggle** (`kaggle datasets download -d
  soumikrakshit/lol-dataset`), after resolving a Kaggle API token issue (token
  file needed to be named exactly `~/.kaggle/kaggle.json` in
  `{"username":..., "key":...}` JSON format, with `chmod 600` permissions).
- Confirmed dataset layout under `data/LOLdataset/`:
  `our485/{low,high}` (485 paired images) and `eval15/{low,high}` (15 paired
  images, held-out test set). Native image resolution is 400x600 — relevant
  later since Phase 2's validation split turned out to use this uncropped
  resolution (see below).
- Implemented data loading code:
  - `src/data/lol_dataset.py` — `LOLDataset` (paired low/high image Dataset,
    returns a dict: `{"low": Tensor(3,H,W), "high": Tensor(3,H,W), "filename": str}`)
    and `make_splits(root, crop_size=256, val_fraction=0.1, seed=42)`, which
    splits `our485` into train/val (90/10, seeded) and uses `eval15` as-is for
    test, returning three fully-constructed `LOLDataset` instances.
  - `src/data/transforms.py` — `PairedTransform` (identical random crop/flip/
    rotation applied to both low and high images to preserve alignment,
    train-mode only) and `synthetic_low_light_degrade()` (optional gamma +
    Poisson-Gaussian noise degradation, not yet used — reserved for later
    DIV2K expansion per the Synthetic Data plan below).
  - `scripts/verify_phase1.py` — end-to-end sanity check script.
- **Verified working end-to-end:**
  ```
  train: 437 | val: 48 | test: 15
  low shape: torch.Size([4, 3, 256, 256])  high shape: torch.Size([4, 3, 256, 256])
  low range: 0.0 - 0.839...
  ```
  Note: augmentation is applied on-the-fly per batch (not precomputed/saved to
  disk) — the 437/48/15 counts are filename-list splits over the same original
  image files; each epoch re-augments those same files with fresh random
  crops/flips. Validation (`train=False`) does **not** crop — it returns the
  full native 400x600 image, confirmed as the root cause of a Phase 2 bug
  (see below).

### Phase 2 — Baseline model (U-Net): COMPLETE
Goal: validate the full pipeline (data → model → loss → checkpointing) with a
simple, fast U-Net before committing to the Restormer/SwinIR transformer in
Phase 3.

**Built:**
- `src/models/unet.py` — `UNetBaseline`: 3 downsampling stages, `base_channels=32`
  (half of the original U-Net's 64 — enough capacity to prove the pipeline
  works without pushing VRAM/compute on the 1650), no BatchNorm (unreliable
  running stats at `batch_size=4`), residual formulation (network predicts a
  correction added to the input rather than the absolute output — the easier
  function to learn for low-light enhancement, and more stable to train).
- `src/losses.py` — `CharbonnierLoss` (smooth L1) via `ReconstructionLoss`.
  Charbonnier-only for the baseline (`ssim_weight=0` by default); the SSIM
  term is already wired in, ready for Phase 3/4 without rewriting the module.
- `src/metrics.py` — `batch_psnr`/`batch_ssim`, thin wrappers around
  `kornia.metrics` (one source of truth for the SSIM math, shared with the
  loss module).
- `src/train_baseline.py` — training loop: AdamW, AMP/GradScaler, gradient-
  accumulation scaffolding (`--accum-steps`, unused at 1 for the baseline but
  ready for Phase 3/4's tighter VRAM budget), CSV logging, `last.pt`/`best.pt`
  checkpointing with full resume support, `--profile` instrumentation
  (per-stage timing breakdown).
- `scripts/verify_phase2.py` — smoke test on synthetic random tensors,
  independent of the real dataset (isolates "is the training machinery
  correct" from "is the data loading correct").
- `scripts/diagnose_val_nan.py` — bisection tool: raw data → single-image
  forward → batched forward, fp16 vs fp32. Built to chase the val NaN bug
  below; kept as a reusable playbook for Phase 3 if precision issues recur.

**Bugs found and fixed, in order:**
1. **Output clamp killed early gradients.** `torch.clamp(x + residual, 0, 1)`
   inside `forward()` zeroed gradients on any pixel pushed outside `[0, 1]` at
   random init — visible in the synthetic smoke test as loss *increasing*
   instead of decreasing. Fix: removed the clamp from the model; clamping now
   happens only at call sites that need a bounded value (metrics/display),
   not inside the forward pass used for training.
2. Deprecated `torch.cuda.amp.GradScaler(...)` → `torch.amp.GradScaler("cuda", ...)`.
3. **Data-loading adapter mismatch.** First guess at wiring `PairedTransform`/
   `LOLDataset` construction into `train_baseline.py` was wrong
   (`TypeError: PairedTransform.__init__() got an unexpected keyword argument
   'crop_size'`). Confirmed the real signatures via `inspect.signature(...)`:
   `make_splits()` already returns fully-constructed `LOLDataset` train/val/test
   instances (augmentation wired in internally) — `build_dataloaders()`
   simplified to just wrapping those in `DataLoader`s.
4. `LOLDataset.__getitem__` returns a dict, not a `(low, high)` tuple — fixed
   the unpacking in both `train_one_epoch` and `validate`.
5. **`val_loss=nan` on every single epoch**, while `train_loss` stayed clean.
   Bisected with `diagnose_val_nan.py`: raw validation tensors clean →
   single-image forward clean (fp16 *and* fp32) → **a batch of 4 at the full
   native validation resolution (400x600, uncropped) came back 100% NaN under
   fp16**, even though the same images one at a time did not. Root cause:
   cuDNN selecting a numerically unstable fp16 convolution algorithm
   (commonly Winograd) for that batch-size/resolution/precision combination —
   a documented issue on Turing-generation cards, and the GTX 1650 has no
   Tensor Cores to fall back on. Fix: `validate()` now runs unconditionally in
   fp32, regardless of the `--amp` flag — it has no backward pass or optimizer
   state, so it was never the reason AMP existed in the first place.
6. Free efficiency win: added `persistent_workers=True` to both DataLoaders so
   worker processes aren't torn down and respawned every single epoch.

**Performance investigation:**
- Added `--profile` (per-stage timing: data / h2d / fwd / bwd / step) plus a
  train-phase-vs-val-phase split in the epoch printout.
- At `crop_size=256`: ~47–51s/epoch, almost entirely GPU-compute-bound
  (fwd+bwd ~98% of tracked time; data loading negligible — ruled out the
  DataLoader/WSL filesystem as a bottleneck before touching model/crop size).
- Tried `crop_size=128`: fwd+bwd time dropped ~3.6x, roughly matching the ~4x
  pixel reduction — but this re-triggered the *same class* of fp16/cuDNN
  instability, this time during training (`train_loss=nan`, weights frozen at
  random init since GradScaler silently skipped every optimizer step on
  inf/nan gradients).
- **Key hardware insight:** the GTX 1650 has no Tensor Cores (Turing without
  RT/Tensor cores), so AMP was only ever buying memory headroom on this card,
  never compute speed. Since `crop_size=128` already frees up plenty of VRAM,
  tried `--no-amp`: fully stable, and *faster* than the broken AMP run at the
  same crop size.
- Added an in-training NaN warning (checks `train_loss` each epoch, prints the
  known cause and suggests `--no-amp`) so this fails loudly instead of
  silently wasting epochs in the future.
- **Decision rule carried forward:** AMP is not a default-on choice on this
  GPU — enable it only if VRAM is the actual binding constraint at a given
  crop/batch size. Revisit for Phase 3, where the deeper transformer backbone
  may genuinely need the memory savings despite the instability risk.

**Verified working end-to-end** (`--crop-size 128 --no-amp`, 50 epochs):
```
train_loss: 0.2009 -> 0.1142 (clean, monotonic decrease, no instability)
best val:   PSNR 16.96 dB / SSIM 0.765  (epoch 46, saved to best.pt)
last.pt:    epoch 49
epoch time: ~49-51s (crop=256) -> ~6-7s (crop=128, no-amp); ~7-8x faster iteration
```
Raw unenhanced low-light images sit around 7–8 dB PSNR (confirmed via the
smoke test's `raw-low` metric), so ~17 dB is a real, meaningful correction
from a deliberately small, non-perceptual baseline. Full pipeline (data →
model → loss → checkpointing) is now proven end-to-end.

**Open risk, not yet acted on:** `crop_size=128` and `--no-amp` were the right
choices for *this* small baseline's iteration speed, not necessarily the right
defaults for the Restormer/SwinIR transformer in Phase 3 — that architecture
typically wants more spatial context (128–256 is standard for restoration
work), and DINOv2 (Phase 4) has its own patch-size conventions (14px patches,
often used near 224x224 inputs) that will partly dictate crop size anyway.
Re-test AMP if/when VRAM pressure reappears with the larger model; the
`diagnose_val_nan.py` bisection approach (raw data → single-sample → batch,
fp16 vs fp32) is directly reusable if precision-related NaNs resurface.

### Phase 3 — Transformer backbone (Restormer): COMPLETE
Goal: replace the Phase 2 U-Net with a transformer-based restoration backbone
and compare it directly against the baseline on the same data and metrics.

**Built:**
- `src/models/restormer.py` — `Restormer`: own implementation of the general
  architecture described in Zamir et al.'s Restormer paper, not a port of any
  specific codebase. Chosen over SwinIR specifically for VRAM reasons:
  SwinIR's window attention still scales with spatial window size, while
  Restormer's MDTA (Multi-DConv Head Transposed Attention) computes attention
  *across channels* rather than pixels, making its cost independent of image
  resolution — the more forgiving choice on a 4GB, non-Tensor-Core GTX 1650.
  Scaled down from the paper's reference config (dim=48, ~26M params) to
  `dim=24, num_blocks=(2,3,3,4), num_refinement_blocks=2,
  ffn_expansion_factor=2.0` (3,270,218 params — 1.7x the U-Net baseline's
  1,925,667). Same residual formulation as `UNetBaseline` (predicts a
  correction added to the input, unclamped in `forward()`) and the same
  H/W-divisible-by-8 requirement already anticipated by `PairedTransform`'s
  eval-mode center-crop back in Phase 1.
- `src/train_restormer.py` — training loop. Reuses `build_dataloaders` /
  `save_checkpoint` / `load_checkpoint` / `train_one_epoch` / `validate`
  unchanged from `src/train_baseline.py` (Phase 2 already proved these are
  model-agnostic — no `UNetBaseline`-specific logic anywhere in them); the
  only things that change are model construction and default hyperparameters:
  `batch_size=2, accum_steps=2` (effective batch 4, matching the baseline for
  a fair comparison), AMP off by default (same Turing/no-Tensor-Core
  reasoning as Phase 2, plus attention's softmax as an *additional* fp16
  overflow risk on top of the conv instability already diagnosed), and
  `ssim_weight=0.2` turned on in `ReconstructionLoss` (wired in during Phase 2
  specifically for this moment — Restormer's larger receptive field and
  attention mechanism should be able to exploit the structural signal more
  than the shallow baseline could).
- `scripts/verify_phase3.py` — smoke test on synthetic tensors, same
  philosophy as `verify_phase2.py`: forward/backward sanity, a short
  loss-decrease trend, checkpoint round-trip, parameter count against the
  baseline, and peak VRAM at the training defaults.
- `scripts/visualize_predictions.py` — qualitative check: runs both the
  Phase 2 and Phase 3 best checkpoints on real (not synthetic) `eval15` test
  images, prints per-image PSNR/SSIM for both, and saves a labeled
  before/after grid. Deliberately lightweight — not a replacement for the
  full Phase 6 benchmarking pass (PSNR/SSIM/LPIPS/NR-IQA across all three
  models on the whole test set), just a sanity check that the metric gains
  are visually real before building Phase 4 on top of this architecture.

**Bugs found and fixed, in order:**
1. None at the architecture/training-machinery level — the channel
   bookkeeping through the 4-level encoder/decoder (dim → 2dim → 4dim → 8dim,
   symmetric skip connections reduced back down via 1x1 convs) was traced by
   hand before running anything, given how easy off-by-one channel mismatches
   are in this class of architecture. `verify_phase3.py` passed all five
   checks on the first run, and the full 100-epoch training run completed
   with no NaNs anywhere.
2. **`--crop-size 256` was far too slow to be practical.** First run at 256
   took 743.6s/epoch (~20.7 hours for 100 epochs). Confirmed compute-bound,
   not VRAM-bound: only 0.89 GB of the 4 GB budget was used at crop=128 per
   `verify_phase3.py`'s VRAM report, so there was no memory pressure to
   relieve. Root cause was two-fold: validation runs on the *full native*
   400x600 image regardless of `--crop-size` (`PairedTransform`'s eval-mode
   center-crop only trims to a multiple of 8 — true since Phase 1, just
   invisible until now because the tiny U-Net made it cheap in absolute
   terms), and Restormer's 8x-widened bottleneck channels are genuinely more
   expensive per pixel than the baseline's plain convolutions. Fix: dropped
   to `--crop-size 128`, cutting the training portion ~4x (628.7s → ~85s)
   since compute scales with H×W; validation time was unaffected by the
   crop-size change (stays ~18.7s, since it always runs on the uncropped
   image). Net: ~104s/epoch, ~2.9 hours for the full 100-epoch run — a
   deliberate trade against the "more spatial context" Restormer typically
   wants, judged reasonable for a portfolio-project timeline.

**Verified working end-to-end** (`--crop-size 128 --batch-size 2
--accum-steps 2` (effective batch 4), AMP off, `--ssim-weight 0.2`,
100 epochs):
```
best val:      PSNR 20.44 dB / SSIM 0.839  (epoch 82, saved to best.pt)
baseline (Phase 2) best:  PSNR 16.96 dB / SSIM 0.765
improvement:   +3.48 dB PSNR, +0.074 SSIM over the U-Net baseline
epoch time:    ~104s (crop=128); ~2.9h wall clock for the full 100 epochs
```
`val_ssim` climbed steadily across all 100 epochs (0.724 → 0.84+) even after
`train_loss` plateaued around epoch 10 — the SSIM term in `ReconstructionLoss`
doing real work past the point the Charbonnier component stopped moving much.
`val_psnr` oscillated noticeably epoch-to-epoch (e.g. dipping to 18.79 at
epoch 59, recovering to 20.08 by epoch 61); expected given the validation set
is only 48 images at full native resolution, not a sign of instability — no
NaNs anywhere in the run.

Qualitative check via `scripts/visualize_predictions.py` on 6 held-out
`eval15` images confirmed the numeric gain is real, not a metric artifact: no
color casts, no blockiness/checkerboard patterns near the PixelShuffle
up/downsample layers, and Restormer's improvement over the baseline is
visually clear on the saved grid, not just in the PSNR/SSIM numbers.

**Open risk, not yet acted on:** training used a flat `lr=2e-4` for all 100
epochs; the late-epoch oscillation in `val_psnr` suggests an LR schedule
(cosine decay or `ReduceLROnPlateau`) could tighten convergence and possibly
recover another fraction of a dB, but wasn't worth a ~3-hour re-run given the
baseline was already cleared by a wide margin — revisit during Phase 6
tuning if the results table calls for it. `--crop-size 256` remains untested
beyond the too-slow first epoch; worth reconsidering only with a cloud GPU
fallback per the Phase 0 mitigation plan, not on local hardware. Full formal
benchmarking (LPIPS, NR-IQA, aggregate metrics across all three models on the
whole test set) is deliberately deferred to Phase 6, not duplicated here.

### Phase 4 — Perceptual loss (DINOv2): COMPLETE
Goal: add a frozen DINOv2 feature-space term to the reconstruction loss, to
reward perceptual/structural similarity that pixel-only losses (Charbonnier,
SSIM) can't capture, and compare against Phase 3.

**Built:**
- `src/perceptual_loss.py` — `DINOv2PerceptualLoss`: L1 distance between
  frozen `dinov2_vits14` (smallest DINOv2 release, chosen deliberately for
  VRAM headroom) patch-token embeddings of pred vs. target — not the CLS
  token, since patch tokens retain per-region structure, the relevant signal
  for a pixel-restoration task. Images resized to 224x224 (DINOv2's patch-14
  stride needs H, W divisible by 14; this project's 128x128 training crops
  don't satisfy that) and ImageNet-normalized before the backbone. Only
  `pred`'s branch keeps gradients; `target`'s features are computed under
  `torch.no_grad()`. Backbone loaded via `torch.hub.load(...)`, frozen
  (`requires_grad_(False)`), with a defensive `.train()` override keeping
  DINOv2 in `eval()` even if the parent loss module is put in train mode.
- `src/losses.py` — `ReconstructionLoss` extended with `perceptual_weight`.
  Unlike Charbonnier/SSIM, this branch holds real pretrained parameters and
  needs `.to(device)` on the loss module itself — a new failure mode vs.
  Phase 2/3, where the loss module was stateless.
- `src/train_phase4.py` — training loop, otherwise unchanged from
  `train_restormer.py`.
- `scripts/verify_phase4.py` — 6-check smoke test: forward/backward, raw
  loss-term-magnitude breakdown (the key check — see below), a short
  optimization trend, metrics sanity, checkpoint round-trip (correctly
  excludes the frozen DINOv2 weights, reconstructed from the hub cache
  rather than restored from the checkpoint), peak VRAM.
- `scripts/visualize_predictions_phase4.py` — three-way qualitative grid
  (Phase 3 / Run 1 / Run 2 / ground truth) on real eval15 images.

**Bugs found and fixed / tuning:**
1. **Run 1 (`perceptual_weight=0.1`) regressed vs. Phase 3.** DINOv2's raw
   loss term (~1.38) was ~4x SSIM's raw term (~0.36) at near-random init;
   at `perceptual_weight=0.1` vs. `ssim_weight=0.2`, the *weighted*
   perceptual contribution was still ~2x SSIM's, letting the perceptual
   gradient dominate/fight the structural-similarity signal instead of
   complementing it.
2. Fix: retuned to `perceptual_weight=0.05` (Run 2) after comparing raw
   per-term magnitudes via `verify_phase4.py`'s check [2/6]. Standing rule
   for any future loss-term addition in this project: always check raw
   magnitudes before picking a weight — the weight value alone doesn't tell
   you the actual gradient contribution.

**Verified working end-to-end** (Run 2, `perceptual_weight=0.05`, otherwise
same settings as Phase 3, 100 epochs):
```
best val:      PSNR 20.56 dB / SSIM 0.826  (epoch 96, saved to best.pt)
Phase 3 best:  PSNR 20.44 dB / SSIM 0.839
delta:         +0.12 dB PSNR, -0.013 SSIM
```
Small but real PSNR gain at a small SSIM cost — the expected signature of a
correctly-balanced perceptual loss, not a wash. Epoch time roughly tripled
from epoch 74 onward (~120s → 300-550s for the remainder of the run),
uncorrelated with the loss/metric curves; most likely external GPU/CPU
contention (WSL2/driver/thermal), not a training-dynamics issue — worth an
`nvidia-smi`/thermal check if it recurs on a longer Phase 6 run, since it
roughly tripled that portion of this run's wall-clock time.

**Open risk, not yet acted on:** same flat `lr=2e-4` schedule and LR-tuning
note carried over from Phase 3 — not revisited here either (see Phase 3's
open risk below). Formal LPIPS/NR-IQA benchmarking remains deliberately
deferred to Phase 6.

### Phase 5 — Quality-assessment head (NR-IQA): COMPLETE
Goal: integrate a quality-assessment metric per the project's lighter-vs-
fuller fork (see Architecture/Technique Plan below). Decided on the
**lighter option**: pretrained NR-IQA via `pyiqa`, eval-only, no new
training loop. Rationale: the fuller option (a trained KonIQ-10k regression
head) was explicitly optional/time-permitting in the original plan; MANIQA
integration gets a legitimate, human-judgment-calibrated NR-IQA number into
Phase 6's results table without a second, tangential training pipeline
competing for the same limited GTX 1650 budget and remaining project-
timeline days.

**Built:**
- `src/quality_metrics.py` — `NRIQAMetric`: thin wrapper around
  `pyiqa.create_metric("maniqa", device=...)`. Plain class, not `nn.Module`
  — eval-only, no gradient path, so none of `perceptual_loss.py`'s buffer/
  `.train()` machinery applies. Exposes `.lower_better` off the pyiqa
  instance rather than hardcoding a direction (MANIQA is higher-is-better;
  other NR-IQA metrics like NIQE/BRISQUE aren't).
- `scripts/verify_phase5.py` — 5-check smoke test: metric creation (weight
  fetch), output shape/finiteness, a blur-based direction sanity check
  (deliberately not `torch.rand()` noise — MANIQA is calibrated on real
  photographs, and uniform noise isn't a reliable direction signal for a
  learned aesthetic prior; heavy-vs-mild Gaussian blur is, since virtually
  every NR-IQA model penalizes blur regardless of training set), a
  same-input determinism check, and peak VRAM at eval15's native resolution.
- `scripts/evaluate_nriqa_phase5.py` — real eval15 run: scores raw
  low-light input / Phase 3 pred / Phase 4 pred / ground truth per image.

**Bugs found and fixed:**
1. **CUDA OOM on the first `verify_phase5.py` run**, batch=4 at 224x224
   (`"Tried to allocate 2.81 GiB"`) — self-inflicted: the smoke test used a
   batch size the real workload never needs (`evaluate_nriqa_phase5.py`
   always scores one image at a time; no-reference metrics don't need
   paired batching the way PSNR/SSIM's dataloaders do). Root cause was
   compounded by MANIQA's backbone (`vit_base` + multi-scale
   channel-attention) being substantially heavier than Phase 4's
   `dinov2_vits14`. `nvidia-smi` confirmed the GPU was otherwise clean
   (165 MiB / 4096 MiB in use) before the fix, ruling out a
   leftover-process explanation. Fix: rewrote the smoke test to batch=1
   throughout, matching real usage; added an OOM-specific error message
   pointing to a CPU fallback (`NRIQAMetric(device="cpu")`) for future
   reference, since GPU speed doesn't matter for eval-only scoring on a
   handful of images the way it does for training.

**Verified working end-to-end** (`verify_phase5.py`, batch=1, native
400x600): peak VRAM 2.65 GB — comfortably under the 4 GB budget on its own
(see open risk below re: running alongside a Restormer checkpoint).

**Real eval15 results** (`evaluate_nriqa_phase5.py`, MANIQA, higher-is-better):
```
mean scores:  input 0.298 | Phase 3 0.268 | Phase 4 0.293 | ground truth 0.587
```
Two findings:
1. **Phase 4 > Phase 3 in 14/15 images (tied on the 15th).** Sign test
   p≈0.0005 under a no-systematic-difference null — a highly consistent,
   not-outlier-driven result. Independent evidence (on a metric
   structurally different from PSNR/SSIM) that the Phase 4 perceptual loss
   achieved its actual goal — improving perceived quality in a way
   pixel-fidelity metrics can't detect — resolving the ambiguity left by
   Phase 4's near-flat PSNR/SSIM numbers.
2. **Both models score below the raw low-light input more often than not**
   (Phase 3: 11/15, Phase 4: 10/15 images). **Resolved** via
   `visualize_predictions_phase4.py`'s qualitative grid (6 evenly-spaced
   eval15 images, including `179.png` — the single largest input-vs-Phase3
   gap in the table above): in every row shown, the Phase 3/Run 2
   restorations look dramatically clearer and more correct than the raw
   input, several of which (`493.png`, `55.png`, `748.png`, `79.png`) are
   close to illegible, near-black frames. No visual evidence supports the
   over-smoothing/perception-distortion-tradeoff hypothesis (a) — nothing
   in the grid looks worse, restored, than the raw input. Conclusion:
   MANIQA's "input scores higher" result on this dataset is a **domain-shift
   artifact** (hypothesis b) — KonIQ-10k contains no near-black frames, so
   MANIQA has no reliable basis for scoring one the way it would a real
   in-distribution photo, and its score there shouldn't be read as a
   genuine quality judgment in either direction. Note: `146.png` and
   `669.png` (the other two largest-gap files) weren't in this particular
   6-image sample, so this isn't a complete image-by-image resolution — but
   the pattern was unanimous across all 6 shown, including the single
   worst-offending file, which is a reasonable basis to close this out.
   **Action for the README/Phase 6 writeup:** state this domain-shift
   caveat explicitly if the "MANIQA vs. raw input" comparison is reported
   at all, rather than presenting the raw numbers unqualified.

   Side finding from the same grid: Run 1's saved `best.pt` is from
   **epoch 24**, not the full 100 — its val_psnr peaked early and never
   recovered for the rest of the run. Concrete confirmation of the Run 1
   diagnosis above: the DINOv2 term's gradient was fighting SSIM's from
   early in training, not just showing up as a late-run wobble.

**Open risk, not yet acted on:** `verify_phase5.py`'s 2.65 GB VRAM figure is
for MANIQA alone; `evaluate_nriqa_phase5.py` also loads a Restormer
checkpoint (~0.89 GB per Phase 3) concurrently — combined footprint
(~3.5 GB estimated) wasn't independently VRAM-profiled and sits closer to
the 4 GB ceiling than either component alone, though the real run above
completed without incident.

### Phase 6 — Evaluation/benchmarking: COMPLETE
Goal: the formal three-way benchmark every earlier phase deferred here —
PSNR, SSIM, LPIPS, and NR-IQA across baseline (U-Net) vs. Restormer vs.
Restormer+perceptual, run on the untouched eval15 test set, plus a final
qualitative before/after grid. This closes out the MVP's core modeling work
(Phases 0–6); Phases 7–9 are deployment and documentation.

**Built:**
- `src/lpips_metric.py` — `LPIPSMetric`: full-reference wrapper around
  `lpips.LPIPS(net='alex')`. AlexNet over VGG because this is an eval-only
  metric, not a training loss — the LPIPS paper found AlexNet correlates at
  least as well with human judgments at a fraction of the compute. Always
  calls `normalize=True` internally: the `lpips` package defaults to
  expecting [-1,1] input, but this project's convention is [0,1] everywhere
  (same as `NRIQAMetric` and `DINOv2PerceptualLoss`) — measured a **65%
  relative discrepancy** on a synthetic pred/target pair between
  `normalize=True` and the package default, confirming this isn't a
  theoretical edge case worth skipping.
- `scripts/verify_phase6.py` — 5-check smoke test, same shape as
  `verify_phase5.py`: metric creation (weight fetch), output shape/
  finiteness, a combined normalize+direction check (an identical pred/
  target pair must score ~0 — the tell for the normalize convention above
  being wrong), determinism, peak VRAM at eval15's native 400x600.
- `scripts/benchmark_phase6.py` — the real eval15 run across all four
  metrics and all four sources (raw input, baseline, Restormer,
  Restormer+perceptual), plus ground truth (NR-IQA only — PSNR/SSIM/LPIPS
  against itself are trivial and uninformative, same convention
  `evaluate_nriqa_phase5.py` used). VRAM-conscious design: restoration
  checkpoints are loaded and freed **one at a time**; LPIPS and MANIQA stay
  resident throughout since every model's predictions need both. Phase 5
  already flagged MANIQA + one Restormer checkpoint at ~3.5GB estimated,
  uncomfortably close to the 4GB ceiling — three restoration checkpoints
  plus two metric networks were never held concurrently, by design.
  Outputs a per-image CSV, a summary CSV, and a ready-to-paste markdown
  results table.
- `scripts/visualize_predictions_phase6.py` — final qualitative grid
  (input → baseline U-Net → Restormer → Restormer+perceptual → ground
  truth), 6 evenly-spaced eval15 images. Extends
  `visualize_predictions_phase4.py` with the U-Net baseline swapped in as
  the leftmost model column and Run 1 dropped (already excluded from
  comparisons per the Phase 4 entry above).

**Verified working end-to-end** (`benchmark_phase6.py`, full 15-image eval15 set):
```
                                        PSNR (dB)   SSIM    LPIPS↓   MANIQA↑
input (low-light)                         7.77     0.196   0.5596   0.298
baseline (U-Net, Phase 2, ep 46)         18.28     0.738   0.3325   0.200
Restormer (Phase 3, ep 82)               20.05     0.796   0.2031   0.268
Restormer+perceptual (Phase 4, ep 96)    21.33     0.800   0.1857   0.293
ground truth                                —        —        —    0.587
```

**Headline finding: Phase 4 beats Phase 3 across all four metrics on
eval15** — PSNR +1.28 dB, SSIM +0.004, LPIPS −0.0174, MANIQA +0.025. This is
a materially different story than the near-flat val-set comparison in the
Phase 4 entry above (PSNR +0.12 dB, SSIM −0.013 on the 48-image val split
used for model selection). Four independent signals — two pixel-fidelity,
one human-judgment-calibrated perceptual metric, one no-reference aesthetic
metric — agreeing is strong evidence the perceptual loss genuinely
generalizes to the true held-out set, even though it looked close to a wash
on val. Worth stating explicitly in the README: val-set model selection and
eval15 test-set generalization told different stories here, and only the
full benchmark surfaced that gap.

**MANIQA's "beats every model" input score has two distinct explanations,
not one:**
1. Restormer and Restormer+perceptual scoring below the raw input's MANIQA
   (0.298 vs. 0.268/0.293) is the same **domain-shift artifact** identified
   in Phase 5 — KonIQ-10k contains no near-black frames, so MANIQA has no
   reliable basis for scoring those inputs as a real photo. `493.png`, one
   of the near-black files Phase 5 already flagged, reverses the usual PSNR
   ranking too (baseline 18.0 dB > Restormer 16.4 dB > Restormer+perceptual
   17.5 dB on that one image) — consistent with the same underlying cause,
   not a new problem.
2. The U-Net baseline's MANIQA score (0.200) is the **lowest of all four
   sources, including the raw input** — a different mechanism entirely. The
   qualitative grid shows why: `1.png`'s baseline output has a visible haze
   and cyan-blue color cast (the ground-truth-white cabinet door renders
   light blue) that Restormer's reconstruction doesn't share. MANIQA is
   known to reliably penalize blur regardless of training distribution (per
   Phase 5's own blur-based smoke-test design) — this looks like a genuine,
   correct quality judgment, not a domain-shift artifact.
   **Action for the README:** present these as two separate explanations for
   a below-raw-input MANIQA score, not one shared caveat.

**Bug caught before trusting any of the above:** the `lpips` package's
default `normalize=False` assumes [-1,1] input; this project's tensors are
[0,1] everywhere. `LPIPSMetric` always passes `normalize=True` internally,
and `verify_phase6.py` checks it explicitly so this can't silently regress
into wrong (not crashing, not NaN — just wrong) numbers later.

**Checkpoint provenance double-checked before finalizing the table above:**
confirmed directly from `runs/phase4_perceptual_run2/best.pt` (epoch 96,
`best_val_psnr=20.5619`) and cross-checked against `log.csv`'s epoch-96 row
(val_psnr 20.5619, val_ssim 0.8260, matching exactly) that the checkpoint
`benchmark_phase6.py` loaded is genuinely Run 2's current best, not a
mixed-up file from `runs/phase4_perceptual/` (Run 1, confirmed separately
to be epoch 24, val_psnr≈20.39 — matches Run 1's known "peaked early"
signature from the Phase 5 entry above, ruling out a path mix-up).

**Open items carried forward, not acted on in this phase:**
- DINOv2 (Phase 4's training loss) and LPIPS (this phase's eval metric) are
  architecturally distinct constructs — self-supervised ViT features with
  no human-judgment calibration vs. a CNN-based metric explicitly
  calibrated against human perceptual-similarity judgments (BAPPS).
  Document this distinction in the README rather than presenting "the
  perceptual loss went down" and "LPIPS improved" as the same claim twice.
- LR schedule tuning (flat `lr=2e-4` throughout, carried over from Phase
  3/4's open risk): given Phase 4's consistent, multi-metric win over
  Phase 3 on the actual test set, the case for spending more time here
  before the deployment/documentation phases looks weak — but this is a
  judgment call, not resolved here.
- `--crop-size 256` remains untested beyond Phase 3's too-slow first epoch;
  still only worth revisiting with a cloud GPU fallback.

### Phase 7 — Edge deployment: COMPLETE

**Built:**
- `src/onnx_export.py` — `DeployRestormer` (inference-only wrapper: reflect-pad to multiple of 8 → Restormer → crop → `clamp[0,1]`, baked into the graph so a deployment caller never needs to know about either convention), `export_onnx` (dynamic batch/height/width axes, opset 17), `quantize_dynamic_int8` (verifies the result is both runnable under `CPUExecutionProvider` *and* actually smaller before trusting it — see findings below).
- `scripts/verify_phase7.py` — 7-step smoke test on an untrained model: export, numerical equivalence vs. PyTorch, output clamp under out-of-range input, non-multiple-of-8 shape handling, dynamic axes on a second resolution, INT8 round-trip, size context. All passed.
- `scripts/benchmark_phase7.py` — real Phase 4 Run 2 checkpoint (epoch 96), eval15.

**Results:**
- Sizes: `.pt` checkpoint 37.68 MB, ONNX fp32 13.29 MB, ONNX INT8 N/A (not viable — see below).
- Latency (400×600, batch=1): PyTorch CPU 2473.05±147.02 ms | PyTorch CUDA 283.95±0.30 ms | ONNX Runtime CPU fp32 3026.92±20.71 ms | ONNX Runtime CUDA fp32 580.54±1.80 ms.
- Quality: PyTorch fp32 and ONNX fp32 both landed exactly on 21.33 dB / 0.800 SSIM — confirms export and the pad/crop/clamp wrapper didn't change predictions from Phase 6.

**INT8 quantization is not viable for this architecture, confirmed rather than assumed:** Restormer is entirely `Conv2d` — no `nn.Linear`/`MatMul`-with-weight ops — and many of those convs are grouped/depthwise (MDTA's `qkv_dwconv`, GDFN's `dwconv`). ONNX Runtime's CPU `ConvInteger` kernel doesn't support several of those configurations, surfacing only as `NOT_IMPLEMENTED` at session-load time, not at quantization time. The MatMul-only fallback was also rejected: it loads and runs, but has nothing real to quantize, producing a *larger* file with 0.000 numerical diff from fp32 — a false-positive "success" that `quantize_dynamic_int8` explicitly checks for (minimum 5% size reduction required) rather than reports.

**The `.pt`-vs-ONNX size gap (37.68 MB vs. 13.29 MB) is optimizer state, not model size:** `train_phase4.py`'s checkpoint carries AdamW's `exp_avg`/`exp_avg_sq` momentum buffers alongside the weights (2× the parameter count) plus scaler/epoch metadata; ONNX is inference-only. 13.29 × 3 ≈ 39.9 MB, consistent with the observed 37.68 MB.

**Two latency findings flagged as real but not fully closed out — carried forward, not resolved:**
- ONNX Runtime CUDA (580.54 ms) vs. PyTorch CUDA (283.95 ms): likely a benchmark methodology asymmetry, not a genuine ORT deficiency. `benchmark_phase7.py`'s ORT call passes a NumPy array through `session.run()` every timed iteration (H2D + D2H copy per call); the PyTorch CUDA call pre-transfers the tensor once outside the timed loop. Not re-benchmarked with `IOBinding` to confirm.
- ONNX Runtime CPU (3026.92 ms) vs. PyTorch CPU (2473.05 ms), a real ~18% gap with no transfer asymmetry to explain it: plausibly ORT's CPU EP being less tuned than PyTorch's own backend for this architecture's dwconv-heavy op mix (the same op class that broke INT8 above), but not root-caused at the per-op/profiling level.

**Open items carried forward:** static (calibrated, QDQ-format) quantization not attempted (the ORT-documented path for CNN-heavy models); GPU INT8 not benchmarked (ORT's CUDA EP doesn't run dynamically-quantized graphs — would need TensorRT).

### Phase 8 — Cloud deployment: COMPLETE

**Backend selection, decided on measured evidence rather than assumption:** built two parallel FastAPI serving wrappers — `src/serve_onnx.py` (ONNX Runtime CPU) and `src/serve_pytorch.py` (PyTorch CPU, CPU-only wheel) — identical `/health`/`/enhance` contracts, matching Dockerfiles and staging scripts. `scripts/benchmark_phase8.py` built, ran, and measured both containers locally:

| Metric | ONNX | PyTorch |
|---|---|---|
| Image size | 131.9 MB | 319.4 MB |
| Cold start | 2.03 s | 2.29 s |
| Server inference | 3000.2 ms | 2781.3 ms |
| Quality (PSNR/SSIM) | 25.64 / 0.88 | 25.64 / 0.88 |

**Decision: ONNX Runtime CPU selected.** A 2.4× smaller, torch-free image was judged more valuable than PyTorch's ~7.3% per-request latency edge, for a scale-to-zero serverless target where cold-start/pull cost matters more than shaving a few hundred ms. `scripts/export_for_serving_pytorch.py`'s optimizer-state stripping (37.68 MB → 12.57 MB, 67% reduction) confirmed Phase 7's "~3× momentum buffers" size prediction almost exactly, as a side validation.

**Live deployment:** `scripts/deploy_phase8.sh` — build → push to Artifact Registry (`us-central1`) → `gcloud run deploy`. Service: `low-light-enhancement`, project `low-light-enhancement-psb`, region `us-central1`. URL: `https://low-light-enhancement-il7b3gcc4a-uc.a.run.app`. Three revisions during setup, image unchanged throughout — only compute config changed:

- `-00001`: `--cpu=2 --memory=1Gi` (initial deploy)
- `-00002`: `--cpu=4 --memory=2Gi` (Cloud Run rejected `--cpu=4` at 1Gi outright — enforces a minimum 2Gi-per-4-vCPU ratio, caught immediately as a hard deploy error, not a silent misconfiguration)
- `-00003`: added `ORT_INTRA_OP_THREADS=2` env var on top of `-00002`'s allocation

**CPU/threading tuning: chased real-looking signal that turned out to be noise — documented as a finding, not hidden.** Local WSL2 container (`nproc`=12) benchmarked at 3000.2 ms mean server inference; first live deploy (`--cpu=2`) showed single-sample reads of 3709.3 ms and 3964.7 ms, prompting a `--cpu=4` bump on the theory that Cloud Run's CPU cap was the bottleneck. That single-sample comparison got worse (4430.8–4524.2 ms), prompting a second theory — ONNX Runtime auto-scaling thread count to visible CPUs, adding sync overhead without useful parallelism on this small, dwconv-heavy model — tested by pinning `ORT_INTRA_OP_THREADS=2`. That produced one lower single-sample read (4013.8 ms) and one similar to before (4650.1 ms), which is when the single-sample methodology itself was flagged as insufficient.

`scripts/benchmark_cloud_run.py` (new: repeated-sampling against a live URL, same mean±std discipline as `benchmark_phase7.py`/`benchmark_phase8.py`, applied to production rather than local) ran n=10 against the current (`-00003`) config: **server_inference_ms mean=3846.7, std=298.3, range 3369.6–4523.7 ms.** That range fully contains both of the earlier `--cpu=2` single-sample reads (3709.3, 3964.7 ms) — meaning **the config changes were almost certainly never distinguishable from request-to-request variance**, and the `--cpu=2`→`--cpu=4`→thread-pinning sequence was driven by underpowered single-sample comparisons, not real signal. Documented as a methodology lesson: single-request timing comparisons are not sufficient evidence for infra decisions, even when a plausible mechanism (thread-count scaling) makes the result feel explainable after the fact.

**What is real, from the n=10 sample:** mean server inference (3846.7 ms) sits ~28% above the local benchmark (3000.2 ms) by a margin (846 ms) larger than this config's own std (298 ms) — likely genuine cloud-vs-local overhead (virtualization, scheduling), not root-caused further. Round-trip-minus-inference gap (~1445 ms, one 7421.1 ms outlier) is consistent with real network variability (Belgium ↔ `us-central1`), not investigated further as out of scope for a portfolio demo. **Quality never wavered across any test, any config: PSNR 25.64 / SSIM 0.880, every single run since first deploy.**

**Final config, kept on the basis of "no evidence it's wrong" rather than "proven best":** `--cpu=4 --memory=2Gi`, `ORT_INTRA_OP_THREADS=2`. Reverting to `--cpu=2` was considered and rejected — no statistical basis to expect improvement, only the cost of another redeploy cycle.

**GCP account setup, worth a note for reproducibility:** original 2023 billing account (`01795D-87C15C-80F7E2`) was closed (trial period ended) and not reused — created a new billing account (`01383E-1E0074-6E51CB`) linked to a fresh project (`low-light-enhancement-psb`) rather than untangling the old one. $1 budget alert set as a tripwire. Expected cost: $0 under normal portfolio-demo traffic — Cloud Run's Always Free tier (2M requests, 180,000 vCPU-seconds/month) covers roughly 12,000 requests/month at the current `--cpu=4` allocation (~15 vCPU-sec/request); Artifact Registry's 0.5 GB free tier comfortably covers the 131.9 MB image.

**Two bugs caught and fixed during this phase:**
- `benchmark_phase8.py` originally swallowed `docker build` failures behind a bare exception repr (`check=True` without capturing/printing output) — fixed to print full stdout/stderr before raising, which is what surfaced the actual `Dockerfile.pytorch` build error.
- `src/models/__init__.py` was missing from the delivered scaffold; masked initially because `src/__init__.py` already existed in the repo from earlier phases while the `models/` package marker didn't exist anywhere yet — only surfaced when the PyTorch Dockerfile's `COPY` step failed on a missing file.

**Open items carried forward:** the ~28% local-vs-cloud inference gap is noted but not root-caused (plausible virtualization/scheduling explanation, not confirmed); structured JSON logging not implemented (`serve_onnx.py` logs plain text, so Cloud Logging indexes it as `textPayload`, not queryable `jsonPayload` fields) — fine for confirming the deploy works, would need revisiting for real production log queries; comparison throughout Phase 7 and 8 used a single eval15 image, not an eval15-wide aggregate.

### Next up: Phase 9 — Documentation/polish (not yet started)

## Goal
Build an ML portfolio project (for GitHub) demonstrating modern computer vision skills, aimed at Applied Scientist / ML Engineer job applications — specifically positioning toward industrial computer vision / inspection-flavored roles rather than generalist LLM/GenAI roles.

## Strategic Context (why this project, why this scope)
- Background: PhD in Computational Mechanics, 10+ years in FEA / nonlinear optimization / model order reduction, plus recent applied CNN work (print-quality automation pipeline built at current employer, ECO3).
- Positioning decision: rather than chasing generic LLM/RAG skills (a crowded, low-differentiation space), double down on computer vision specialization — a rarer combination given the numerical-methods + production-engineering background.
- Cloud/MLOps fundamentals: already covered via a Le Wagon ML Bootcamp (GCP-based) but rusty (no cloud use since, ~since 2023 while at ECO3). Decision: refresh GCP rather than switch to AWS, because (a) reactivating existing knowledge is faster than learning a new provider from scratch, (b) GCP is well-regarded specifically for ML/data-engineering roles, (c) Vertex AI has strong native tooling if GenAI/RAG work is ever needed later.
- Modern CV directions to incorporate, based on current industrial-inspection trends: Vision Transformers overtaking CNNs on complex inspection tasks; foundation vision models (SAM, DINOv2) for low-label/few-shot settings; synthetic data generation for scarce defect/degradation classes; edge deployment (quantization, ONNX) for real-time inference; and a narrow, deliberate touch of multimodal/VLM work framed as quality/degradation explainability rather than a generic chatbot/RAG project.

## Project Scope Decisions
- **Rejected: printing-behavior simulation.** Too similar to ECO3's proprietary work (IP concern), and no public dataset available for that specific application.
- **Rejected: training 3 separate models** (one each for denoising, super-resolution, low-light enhancement). This dilutes effort and mostly demonstrates repetition rather than depth. May revisit later as a unified "all-in-one restoration" model (shared backbone + conditioning mechanism) as a stretch goal, but not as the starting scope.
- **Chosen MVP: low-light image enhancement**, using the LOL dataset — clean, well-scoped, and produces visually compelling before/after results for a portfolio README.

## Architecture / Technique Plan
- **Backbone:** transformer-based restoration architecture — Restormer or SwinIR (current standard architectures for restoration/enhancement tasks). Built independently: own implementation, own data, not a reproduction of any employer method or code.
- **Perceptual loss:** frozen DINOv2 features added on top of standard reconstruction loss (L1/Charbonnier + SSIM).
- **Synthetic data (optional):** synthetic low-light degradation pipeline (gamma reduction + Poisson-Gaussian sensor noise) applied to clean images (e.g. DIV2K) to expand training pairs beyond LOL's ~500 real pairs.
- **Quality assessment:** start with an existing pretrained no-reference IQA metric (e.g. MANIQA) purely for evaluation; optionally train a small regression head on KonIQ-10k later if time allows.
- **Edge deployment:** export to ONNX, quantize, benchmark size/latency.
- **Cloud deployment:** FastAPI wrapper → Docker container → GCP Cloud Run or Vertex AI endpoint, with basic logging.
- **Optional stretch:** VLM-based degradation-type classification/captioning; generalize the backbone to a second degradation type (e.g. denoising) via a shared backbone + conditioning mechanism — only after the MVP is solid.

## Phased Implementation Plan
0. **Setup** — repo skeleton, environment (PyTorch, timm, kornia), GPU environment. *(~1-2 days)*
1. **Data** — LOL dataset loader, train/val/test split, augmentations; optional synthetic degradation pipeline. *(~2-3 days)*
2. **Baseline model** — simple U-Net-style baseline to validate the full pipeline (data loading, training loop, metrics, checkpointing) before the transformer. *(~2-3 days)*
3. **Transformer backbone** — Restormer/SwinIR adapted for low-light enhancement (illumination-map/residual formulation), compared against the baseline. *(~1-2 weeks — core phase)*
4. **Perceptual loss** — add frozen DINOv2 feature loss, retrain, compare. *(~2-3 days)*
5. **Quality-assessment head** — pretrained NR-IQA metric for eval (lighter option) or trained regression head (fuller option). *(~3-5 days)*
6. **Evaluation/benchmarking** — PSNR, SSIM, LPIPS, NR-IQA score across baseline vs. transformer vs. transformer+perceptual; qualitative before/after comparison grid. *(~2-3 days)*
7. **Edge deployment** — ONNX export, quantization, latency/size benchmarking. *(~2-3 days)*
8. **Cloud deployment** — FastAPI + Docker + Cloud Run/Vertex AI, basic logging. *(~3-5 days)*
9. **Documentation/polish** — README (architecture diagram, results table, before/after images, run instructions), optional Gradio/Streamlit demo, clean incremental commit history. *(~2-3 days)*

**Rough total for MVP (Phases 0-9): ~4-6 weeks** at a steady, part-time pace. Phases 7-9 are what turn "a model that works" into a real portfolio piece — don't compress these to rush toward stretch goals.

**Stretch (only after MVP is complete and solid):** VLM explainability layer; unified multi-degradation model.

## Development Environment
- **Use WSL2 (Ubuntu)**, not native Windows 11 — smoother CUDA/PyTorch support, matches the Linux-based deployment target (Docker/Cloud Run/Vertex AI), and better filesystem performance for data loading.
- Keep the project and datasets inside the WSL Linux filesystem (e.g. `~/projects/low-light-enhancement`), **not** under `/mnt/c/...` — meaningfully better I/O performance.
- Install CUDA-enabled PyTorch inside WSL per PyTorch's official WSL2 instructions. Only the Windows-side NVIDIA driver needs WSL support; no separate GPU driver is needed inside WSL itself.
- Use VS Code with the **WSL extension** to edit Linux-side files through the familiar Windows UI.
- Docker Desktop with the **WSL2 backend** for the containerization/deployment phases.

## IP / Independence Note
This project must remain clearly independent of ECO3's proprietary print-quality pipeline: own architecture, own (public/synthetic) data, own code — inspired only by the general class of problem (image quality/enhancement), never a reproduction of employer methods, code, or data.
