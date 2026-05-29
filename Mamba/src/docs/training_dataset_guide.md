# Training Dataset Guide

Hey, here's what the feature store compiler produces and how you can use it for training.

## What We Built

Run `python src/scripts/compile_features_store.py` and it gives us one file: `data/processed/hd5f/amr_features.h5`.

Inside we get:

| Dataset | Shape | Type | What it is |
|---------|-------|------|------------|
| `X` | (N, 30000) | float32 | Raw nanopore signal, MAD-normalized |
| `Y` | (N,) | int8 | 1 = AMR, 0 = not AMR |
| `read_id` | (N,) | string | Read UUID |
| `strain_id` | (N,) | string | Which strain (for LOSO validation) |
| `gene_target` | (N,) | string | Gene name — or "background" if unmapped |

## How Long Does It Take?

Roughly **2–5 minutes** total on a machine with an SSD:

- **Pass 1 (BAM scan):** ~30 seconds. We open each BAM, fetch reads by reference gene via the index (fast), and collect their metadata. ~2,400 reads across 4 strains.
- **Pass 2 (POD5 extraction + HDF5 write):** ~1.5–4 minutes. This is where we actually read the raw signal from the POD5 files, compute the geometry (Move Table → signal coordinates), crop, normalize, and write. The bottleneck is POD5 I/O.

If your data is on an HDD, expect closer to 5–8 minutes. If you add more strains, it scales roughly linearly.

## The Three Classes We're Working With

We designed the dataset around three sources of signal so the model learns the right thing:

| `label` | `gene_target` | What it is | Why we include it |
|---------|---------------|------------|-------------------|
| 1 | ENA\|... (APH, SHV, oqxA) | **AMR positive** | The target — we want to detect these |
| 0 | rpoB, gapA, mdh | **Hard negative** (MLST housekeeping) | Teaches the model that "gene-like signal != AMR" |
| 0 | background | **Background** (unmapped/junk DNA) | Teaches the model what random noise looks like |

Without background, the model only learns to tell apart AMR vs. MLST. Give it random junk DNA and it might flag everything gene-shaped as AMR.

## Loading It in PyTorch

Here's a simple Dataset class:

```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class NanoSquiggleDataset(Dataset):
    def __init__(self, h5_path="data/processed/hd5f/amr_features.h5", indices=None):
        self.h5 = h5py.File(h5_path, "r")
        self.indices = indices

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.h5["Y"])

    def __getitem__(self, i):
        idx = self.indices[i] if self.indices is not None else i
        x = torch.from_numpy(self.h5["X"][idx]).float()
        y = int(self.h5["Y"][idx])
        return x.unsqueeze(0), y  # shape (1, 30000) for 1D CNN
```

## Ways to Split the Data

### 1. Simple train/val (80/20)

```python
dataset = NanoSquiggleDataset()
n = len(dataset)
train_idx = np.random.choice(n, int(0.8 * n), replace=False)
val_idx = np.setdiff1d(np.arange(n), train_idx)

train_loader = DataLoader(NanoSquiggleDataset(indices=train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(NanoSquiggleDataset(indices=val_idx), batch_size=32, shuffle=False)
```

### 2. Balanced 50/50 (good idea)

The dataset is roughly balanced, but if you want to make sure:

```python
h5 = h5py.File("data/processed/hd5f/amr_features.h5", "r")
y = h5["Y"][:]
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
neg_idx = np.random.choice(neg_idx, len(pos_idx), replace=False)
balanced = np.concatenate([pos_idx, neg_idx])
np.random.shuffle(balanced)

loader = DataLoader(NanoSquiggleDataset(indices=balanced), batch_size=32, shuffle=True)
```

### 3. Leave-One-Strain-Out (LOSO) — the gold standard

This tells us if the model generalizes to unseen strains:

```python
h5 = h5py.File("data/processed/hd5f/amr_features.h5", "r")
strain_ids = h5["strain_id"][:]
all_strains = np.unique(strain_ids)

for test_strain in all_strains:
    test_idx = np.where(strain_ids == test_strain)[0]
    train_idx = np.where(strain_ids != test_strain)[0]

    train_loader = DataLoader(
        NanoSquiggleDataset(indices=train_idx), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        NanoSquiggleDataset(indices=test_idx), batch_size=32, shuffle=False
    )
```

## CNN Input Shape

| Property | Value |
|----------|-------|
| Input shape | `(batch, 1, 30000)` |
| Signal range | Roughly [-5, 5] — already normalized, don't re-scale |
| dtype | float32 |
| Pos/neg ratio | ~1:1 overall |

## A Few Things to Keep in Mind

- **Failed reads** — some reads can't be extracted (missing tags, corrupt POD5). We log them in `compilation_errors.log` and shrink the HDF5 automatically. Nothing for you to handle.
- **Background reads** — they're centered crops from the middle of the raw signal, not linked to any gene. Their job is to show the model what "ignore this" looks like.
- **Short signals** — if a read is shorter than 30,000 samples, we symmetrically zero-pad it. There aren't many of these.
- **The gene_target field** — useful if you want to do per-gene evaluation or filter out a specific category (e.g., train only on AMR vs. background).
