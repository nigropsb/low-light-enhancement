"""
Quick sanity check for Phase 1: confirms the LOL dataset loads, splits, and
augments correctly. Run from the project root:

    python scripts/verify_phase1.py --root data/LOLdataset
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from src.data.lol_dataset import make_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/LOLdataset")
    parser.add_argument("--crop_size", type=int, default=256)
    args = parser.parse_args()

    train_ds, val_ds, test_ds = make_splits(args.root, crop_size=args.crop_size)
    print(f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
    batch = next(iter(loader))

    print("low shape:", batch["low"].shape, "high shape:", batch["high"].shape)
    print("low range:", batch["low"].min().item(), "-", batch["low"].max().item())
    print("filenames in batch:", batch["filename"])
    print("\nPhase 1 data pipeline OK.")


if __name__ == "__main__":
    main()
