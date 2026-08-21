# Getting the LOL Dataset

The LOL (LOw-Light) dataset from Chen Wei et al., "Deep Retinex Decomposition for
Low-Light Enhancement" (BMVC 2018) is not distributed via a direct HTTP/pip source.
You need to fetch it manually from one of these official mirrors:

- Google Drive / Baidu links listed on the official project page:
  https://daooshee.github.io/BMVC2018website/
- Alternatively, a commonly used re-hosted copy is available via the
  "LOLdataset" Kaggle listing (search "LOL dataset low light Kaggle") —
  convenient if you'd rather use `kaggle` CLI / API from WSL.

## Expected structure after download

Download and unzip so you end up with this layout under your project's `data/` dir
(create it under the Linux filesystem path, NOT `/mnt/c/...`):

```
~/projects/low-light-enhancement/data/LOLdataset/
├── our485/
│   ├── low/        # 485 low-light training images
│   └── high/       # 485 corresponding normal-light images (same filenames)
└── eval15/
    ├── low/        # 15 low-light test images
    └── high/       # 15 corresponding normal-light images
```

Filenames in `low/` and `high/` match 1:1 (e.g. `low/1.png` <-> `high/1.png`).

## Quick sanity check after placing the data

```bash
cd ~/projects/low-light-enhancement
find data/LOLdataset -type f | wc -l   # should be 485*2 + 15*2 = 1000
ls data/LOLdataset/our485/low | head -5
ls data/LOLdataset/our485/high | head -5
```

## Train/val split

LOL only ships a train (`our485`) / test (`eval15`) split — no separate validation
set. The dataset loader below carves a validation split out of `our485` (default
90/10) so you get train/val/test without needing extra downloads.
