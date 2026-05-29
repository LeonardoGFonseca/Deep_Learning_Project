# Project: AMR Prediction from Raw Nanopore Squiggles

## Goal
Predict Antibiotic Resistance (AMR) directly from raw Oxford Nanopore Technologies (ONT) electrical signals ("squiggles") using a 1D Convolutional Neural Network (1D-CNN). This approach bypasses the traditional base-calling step, aiming for faster and more efficient diagnostics.

## Background & Rationale
Traditional AMR detection requires sequencing (base-calling), assembly/alignment, and gene identification. This is computationally expensive and introduces latency. By classifying raw signals directly, we can potentially identify resistant strains within seconds of the read beginning, enabling "Read-Until" sequencing where non-target or non-resistant reads are ejected from the pore.

## Directory Structure
```text
   /NanoSquiggle-AMR-CNN
   ├── configs/            # Hyperparameters (YAML/JSON). no code Python.
   ├── data/               # Local data storage (ignored by git).
   │   ├── raw/            # .pod5 files (Primary format).
   │   └── processed/      # Normalized tensors or windowed segments.
   ├── docs/               # Technical specs and literature summaries.
   ├── notebooks/          # Exploratory Data Analysis (EDA) and signal plotting.
   ├── scripts/            # Execution entry points.
   │   ├── train.py        # Main training script.
   │   ├── evaluate.py     # Rigorous testing on unseen datasets.
   │   └── preprocess.py   # Signal cleaning and MAD normalization (POD5 compatible).
   ├── src/                # The Core Engine (The "Library")
   │   ├── data/           # POD5 Dataset and DataLoader classes.
   │   ├── models/         # Modular architecture definitions (1D-CNN, ResNet).
   │   └── utils/          # Normalization, windowing, and logging utilities.   ├── tests/              # Unit tests for model shapes and data loading logic.
   ├── .gitignore          # Crucial: Exclude /data, .fast5, and /env.
   ├── requirements.txt    # Dependency list (torch, ont-fast5-api, etc.).
   └── README.md           # The Scientific Abstract and usage guide.
```

## Methodology (The Roadmap)
1.  **Data Acquisition:** Sourcing raw signal data (e.g., SquiggleNet dataset, PRJEB41335).
2.  **Preprocessing:** Normalization (MAD), windowing, and signal cleaning.
3.  **Architecture:** Implementation of a 1D-CNN (ResNet-like) using PyTorch.
4.  **Training:** Optimizing the model to recognize AMR-conferring patterns in signal motifs.
5.  **Validation:** Testing against base-called ground truth and clinical susceptibility labels.

---
*Note: This project is part of the 425282 - Deep Learning course (MSc). FCUL*
