# Squiggles-OxfordNanopore-Data-workflow

Pipeline for downloading Oxford Nanopore raw signal data (POD5), running targeted basecalling and alignment with Dorado, and preparing a feature store for training a MambaCNN model for antimicrobial resistance (AMR) prediction.

## Directory Structure

```
.
├── bin/              # Dorado binary + shared libraries
├── data/
│   ├── data_index/   # Sample/strain ID index
│   ├── raw/          # Downloaded POD5 files (per strain)
│   ├── processed/    # Alignment BAM files (per strain)
│   └── ref/          # Reference FASTA (AMR + MLST genes)
├── src/
│   ├── scripts/         # Pipeline scripts (Python)
│   │   ├── download_data.py           # Phase 1: download + verify
│   │   ├── run_dorado.py              # Phase 2: basecall + align
│   │   └── compile_features_store.py  # Phase 3: HDF5 feature store (WIP)
│   ├── training/        # ★ NEW — model training infrastructure
│   │   ├── sweep.py              # CLI entry: --model_type {gap,mamba}
│   │   ├── backbones.py          # CNNStem (shared backbone)
│   │   ├── classifiers.py        # GapBaseline, MambaClassifier
│   │   ├── dataset.py            # LOSO + balanced undersampling
│   │   └── trainer.py            # LOSO loop + metrics
│   ├── configs/         # ★ NEW
│   │   └── train_config.yaml
│   ├── utils/           # Shell utilities
│   │   └── setup_dependencies.sh  # Install Dorado + create dirs
│   ├── docs/            # Design docs, challenges, session summaries
│   └── notebooks/       # Colab-compatible full workflow
├── deep_env/         # Python virtual environment
├── tests/            # Integration tests
├── download.sh       # One-shot setup + download entry point
└── requirements.txt  # Python package dependencies
```

## Pipeline

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `src/scripts/download_data.py` | Multi-threaded POD5 download with HAC integrity verification |
| 2 | `src/scripts/run_dorado.py` | Targeted basecalling + alignment via Dorado, piped through `samtools sort` |
| 3 | `src/scripts/compile_features_store.py` | Coordinate-to-Signal mapping and HDF5 feature store compilation (WIP) |
| 4 | `src/training/sweep.py` | LOSO cross-validation: GAP vs Mamba binary classification |

## Quick Start

```bash
# 1. Install system-level dependency (samtools)
#    Linux (Debian/Ubuntu): sudo apt install samtools
#    macOS:                 brew install samtools
#    Linux (RHEL/Fedora):   sudo dnf install samtools

# 2. Set up Python environment
python3 -m venv deep_env
source deep_env/bin/activate
pip install -r requirements.txt

# 3. Download Dorado and prepare directories (auto-detects OS)
bash src/utils/setup_dependencies.sh

# 4. Download and verify raw data
python src/scripts/download_data.py

# 5. Run basecalling and alignment
python src/scripts/run_dorado.py

# 6. Compile feature store
python src/scripts/compile_features_store.py

# 7. Audit BAM quality
python src/scripts/audit_bam_quality.py

# 8. Run GAP baseline (bag of motifs — order destroyed)
python src/training/sweep.py \
    --h5_path data/processed/hd5f/amr_features.h5 \
    --model_type gap

# 9. Run Mamba classifier (sequential grammar — order preserved)
python src/training/sweep.py \
    --h5_path data/processed/hd5f/amr_features.h5 \
    --model_type mamba
```

To install the optional Mamba dependency: `pip install mamba-ssm`. If unavailable, it falls back to GRU automatically.

Or use the convenience entry point (Phase 1 only):

```bash
bash download.sh
```

## CLI Arguments

All pipeline scripts accept `--help` for available options:

| Script | Key arguments | Defaults |
|--------|--------------|----------|
| `download_data.py` | `--max-workers`, `--raw-dir`, `--index-file` | workers=4 |
| `run_dorado.py` | `--reference`, `--raw-dir`, `--processed-dir` | ref=`data/ref/resistance_genes.fasta` |
| `compile_features_store.py` | `--window-size`, `--coverage`, `--max-background`, `--index-file` | window=30000, cov=0.95, bg=500 |
| `audit_bam_quality.py` | `--coverage-threshold`, `--bam-dir` | cov=0.90 |
| `sweep.py` | `--model_type`, `--h5_path`, `--epochs`, `--batch_size`, `--lr`, `--dropout` | epochs=50, batch=32, lr=3e-4, dropout=0.2 |

## Tests

Tests are standalone scripts (NOT pytest). The test requires a completed Phase 3 run:

```bash
python tests/test_feature_store.py
```

This validates the HDF5 feature store structure, class balance, strain coverage, and signal quality.

## Training Architecture (Phase 4)

Three models share the **CNNStem** backbone (Conv1D layers [15,7,5,5] → BN → GELU → Dropout). Only the head differs:

| Model | Head | Params | What it tests |
|-------|------|--------|---------------|
| GapBaseline (primary) | GAP → Linear(256→1) | ~1.8M | Bag of motifs — ignores order, establishes lower bound |
| MambaClassifier (primary) | Mamba S6 → GAP → Linear(256→1) | ~1.93M | Grammar engine — preserves sequential order |
| FlattenBaseline (legacy) | Flatten → FC(64) → Linear(1) | ~32.5M | Reference — 17× param disparity, excluded from ablation |

### Evaluation: Leave-One-Strain-Out (LOSO)

4 folds — train on 3 strains, evaluate on the held-out 1. Balanced 50/50 undersample per epoch. Strains are read from `data/data_index/data_ids.txt`: add or remove a strain to automatically include or exclude it from the LOSO loop.

### Results

Per-fold metrics are saved to `training_output/ablation_results.csv`: accuracy, F1, precision, recall, AUPRC. Compare GAP vs Mamba across all 4 folds to determine whether sequential order matters for AMR detection.

## Dependencies

- **Dorado 0.5.0** — basecalling and alignment (installed by `setup_dependencies.sh`; supports Linux x64/ARM64 and macOS x64/ARM64)
- **samtools** — BAM sorting and indexing (system package)
- **Python 3.10+** — pipeline orchestration
- **PyTorch 2+** — model framework
- **mamba-ssm** — Selective SSM (optional; falls back to GRU if unavailable)
- See `requirements.txt` for Python package versions

## Docs

Design documents and technical notes are in `src/docs/`:

- `data_infrastructure_v1.md` — Full research roadmap (phases 1–3)
- `bam_audit_plan.md` — BAM quality filtering and audit plan
- `logical_challenges.md` — Floating Window and Move Table geometry
- `session_summary_20260506.md` — Architecture decisions and milestones

## License

MIT — see [LICENSE](LICENSE)
