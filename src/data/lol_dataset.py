"""
LOL dataset loader for low-light image enhancement.

Directory layout expected (see scripts/download_lol.md):

    data/LOLdataset/
    ├── our485/{low,high}/*.png   -> split into train + val
    └── eval15/{low,high}/*.png  -> test

Usage:
    from src.data.lol_dataset import LOLDataset, make_splits

    train_ds, val_ds, test_ds = make_splits(
        root="data/LOLdataset", crop_size=256, val_fraction=0.1, seed=42
    )
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import AugmentConfig, PairedTransform


def _load_image(path: Path) -> torch.Tensor:
    """Loads an image as a CHW float tensor in [0, 1], RGB order."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).contiguous()


class LOLDataset(Dataset):
    def __init__(
        self,
        low_dir: Path,
        high_dir: Path,
        filenames: list[str],
        crop_size: int = 256,
        train: bool = True,
    ):
        self.low_dir = Path(low_dir)
        self.high_dir = Path(high_dir)
        self.filenames = filenames
        self.transform = PairedTransform(AugmentConfig(crop_size=crop_size, train=train))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname = self.filenames[idx]
        low = _load_image(self.low_dir / fname)
        high = _load_image(self.high_dir / fname)
        low, high = self.transform(low, high)
        return {"low": low, "high": high, "filename": fname}


def make_splits(
    root: str | Path,
    crop_size: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[LOLDataset, LOLDataset, LOLDataset]:
    """Builds train/val/test datasets from the LOL directory layout.

    our485/ is split into train + val (val_fraction held out, seeded for
    reproducibility); eval15/ is used as-is for the held-out test set.
    """
    root = Path(root)
    train_low_dir = root / "our485" / "low"
    train_high_dir = root / "our485" / "high"
    test_low_dir = root / "eval15" / "low"
    test_high_dir = root / "eval15" / "high"

    for d in [train_low_dir, train_high_dir, test_low_dir, test_high_dir]:
        if not d.is_dir():
            raise FileNotFoundError(
                f"Expected directory not found: {d}\n"
                f"See scripts/download_lol.md for the expected LOL dataset layout."
            )

    all_filenames = sorted(p.name for p in train_low_dir.glob("*"))
    if not all_filenames:
        raise RuntimeError(f"No images found in {train_low_dir}")

    rng = random.Random(seed)
    shuffled = all_filenames[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    test_files = sorted(p.name for p in test_low_dir.glob("*"))

    train_ds = LOLDataset(train_low_dir, train_high_dir, train_files, crop_size, train=True)
    val_ds = LOLDataset(train_low_dir, train_high_dir, val_files, crop_size, train=False)
    test_ds = LOLDataset(test_low_dir, test_high_dir, test_files, crop_size, train=False)

    return train_ds, val_ds, test_ds
