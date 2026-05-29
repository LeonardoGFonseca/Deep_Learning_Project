# Agents.md — Squiggles-OxfordNanopore-Data-workflow

## Pipeline (3 sequential phases, each standalone)

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `src/scripts/download_data.py` | Download POD5 tar.gz from narodni-repozitar.cz, verify (HAC protocol), extract by strain |
| 2 | `src/scripts/run_dorado.py` | Dorado HAC basecaller → `samtools sort -m 2G` pipe, BAM + index per strain |
| 3 | `src/scripts/compile_features_store.py` | Map BAM alignments → POD5 signal windows → HDF5 (`amr_features.h5`) |

## Setup

```bash
# samtools: sudo apt install samtools (Linux) | brew install samtools (macOS)
python3 -m venv deep_env && source deep_env/bin/activate && pip install -r requirements.txt
bash src/utils/setup_dependencies.sh   # downloads Dorado 0.5.0 into bin/; auto-detects OS
bash download.sh                       # one-shot: setup + Phase 1
```

## Commands

| Script | Usage | Key args |
|--------|-------|----------|
| Phase 1 | `python src/scripts/download_data.py [--max-workers N]` | `--raw-dir`, `--index-file` |
| Phase 2 | `python src/scripts/run_dorado.py [--reference path]` | `--raw-dir`, `--processed-dir` |
| Phase 3 | `python src/scripts/compile_features_store.py [--window-size 30000]` | `--coverage`, `--max-background`, `--index-file` |
| Audit | `python src/scripts/audit_bam_quality.py [--coverage-threshold 0.90]` | `--bam-dir` |
| Test | `python tests/test_feature_store.py` | NOT pytest — standalone; requires Phase 3 output |
| Phase 4 | `python src/training/sweep.py --model_type gap --h5_path data/processed/hd5f/amr_features.h5` | GAP baseline |
| Phase 4 | `python src/training/sweep.py --model_type mamba --h5_path data/processed/hd5f/amr_features.h5` | Mamba classifier |
| Phase 4 | `python src/training/sweep.py --model_type flatten --h5_path ... --legacy` | Legacy flatten (non-primary) |

All scripts accept `--help`.

## Architecture notes

- **No package** — scripts are standalone, no `__init__.py`. Run from repo root.
- **State via sentinel files** — `.download_complete` in each `data/raw/<strain>/` directory gates Phase 1→2 progression. Idempotent.
- **Dorado** — binary at `bin/dorado`, libs at `bin/lib/`. `LD_LIBRARY_PATH` must include `bin/lib` (handled in script via `_get_env()`).
- **Data source** — `https://data.narodni-repozitar.cz/general/datasets/dj8ys-a4r49/files/<strain>_pod5.tar.gz`
- **Strain index** — `data/data_index/data_ids.txt`, one strain ID per line (currently: KP1779, KP1780, KP1526, KP1833)
- **Reference** — `data/ref/resistance_genes.fasta` (3 AMR + 3 MLST housekeeping genes)
- **Feature store output** — `data/processed/hd5f/amr_features.h5`. Datasets: `X` (float32 signal windows), `Y` (int8 labels), `read_id`, `strain_id`, `gene_target`.
- **Signal windows** — 30,000 samples, MAD-normalized, centered on alignment midpoint via Move Table (`mv` tag) + Trim Start (`ts` tag).

## Key conventions

- **Commit style prefix**: `(CREATE)`, `(FIX)`, `(update)` in subject line.
- **No CI/CD, no lint/typecheck config**.
- **No pytest** — test is `python tests/test_feature_store.py`.
- **Dorado version pinned**: 0.5.0 in `setup_dependencies.sh`.
- **samtools sort buffer**: 2G (hardcoded in `run_dorado.py`).
- **ThreadPoolExecutor**: `max_workers=4` in download script.
- Coverage threshold for positive reads: 0.95 (compile), 0.90 (audit default).
