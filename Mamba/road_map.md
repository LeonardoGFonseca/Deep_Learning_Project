# Roadmap: Training Infrastructure Integration

Integrate CNN + Mamba training into the Squiggles data pipeline.

---

## 1. Goal & Scope

**Primary question:** Does sequential modeling (Mamba) improve AMR detection over a motif-presence (GAP) baseline?

**Scope:**
- Single binary classification (AMR vs. not) — not the cascade from Deep_Learning_Project
- Two primary models sharing a `CNNStem` backbone, differing only in the head (GAP vs. Mamba)
- Flatten model kept as a legacy reference (non-primary — 17× parameter disparity invalidates direct comparison)
- Leave-One-Strain-Out (LOSO) cross-validation on 4 strains (KP1779, KP1780, KP1526, KP1833)
- Output: per-fold metrics, ablation comparison table, loss curves

---

## 2. Architecture: The Two-Model Ablation

```
Input (B, 1, 30000)
  │
  └── CNNStem (shared backbone)
        Conv1(15,2,32) → BN → GELU → Dropout(0.2)
        Conv2(7,2,64)  → BN → GELU → Dropout(0.2)
        Conv3(5,2,128) → BN → GELU → Dropout(0.2)
        Conv4(5,2,256) → BN → GELU
        Output: (B, 256, 1875)   ← 1,875 semantic tokens
  │
  ├── Model A: GapBaseline
  │     AdaptiveAvgPool1d(1) → (B, 256)
  │     Linear(256 → 1) + Sigmoid
  │     Params: ~1.8M (Backbone) + 257 (Head)
  │     Tests: "bag of motifs" — ignores order entirely
  │
  └── Model B: MambaClassifier
        Permute(1,2) → (B, 1875, 256)
        Mamba S6 (d_state=16, expand=2)
        Permute(2,1) → (B, 256, 1875)
        AdaptiveAvgPool1d(1) → (B, 256)
        Linear(256 → 1) + Sigmoid
        Params: ~1.8M (Backbone) + ~130k (Mamba) + 257 (Head)
        Tests: "grammar engine" — sequential order matters

---

## 3. Legacy Reference (Non-Primary)

**Model X: FlattenBaseline**
- Flatten → (B, 480000) → Linear(480000 → 64)
- Params: ~30M
- Purpose: Sanity check for "Rigid Template" performance. Not used for sequential ablation due to capacity bias.
```

### Expected results (per Plan_training_v2)

| Model | Recall | Precision | F1 | Interpretation |
|-------|--------|-----------|----|----------------|
| A. GAP | ~90% | ~40-50% | ~55-65% | Motifs present → high recall, but no order → false positives |
| B. Mamba | ~90% | ~70-80% | ~80-85% | Order + motifs → both high recall and precision |

> Legacy reference (Flatten, ~30M params): expected ~85-90% recall, ~50-60% precision. Excluded from primary ablation due to 17× parameter disparity over the Mamba model (~1.93M). Not comparable.

---

## 4. File Structure (to Create)

```
squiggles-oxfordnanopore-data-workflow/
├── data/                         (unchanged — Phase 1-3 output)
├── src/
│   ├── scripts/                  (unchanged — download, dorado, compile, audit)
│   ├── training/                 ★ NEW — training infrastructure
│   │   ├── __init__.py
│   │   ├── backbones.py          CNNStem (shared conv backbone)
│   │   ├── classifiers.py        GapBaseline, FlattenBaseline, MambaClassifier
│   │   ├── dataset.py            NanoSquiggleDataset + LOSO split + undersampling
│   │   ├── trainer.py            LOSO loop, metrics, checkpointing, logging
│   │   └── sweep.py              CLI entry: --model_type {gap,mamba} (+ --legacy for flatten)
│   └── configs/                  ★ NEW
│       └── train_config.yaml     Hyperparameters
├── tests/
│   ├── test_feature_store.py     (existing)
│   └── test_training.py          ★ NEW — forward pass, LOSO split sanity
├── requirements.txt              (updated — add mamba-ssm, pyyaml)
└── AGENTS.md                     (updated — add Phase 4: Training)
```

---

## 5. Phase-by-Phase Implementation

### Phase 1: CNNStem Refactor (~30 min)

**What:** Extract conv layers into a reusable `CNNStem` module.

**Changes from Deep_Learning_Project's `cnn_2.py`**:
- ReLU → GELU
- Add Dropout(0.2) after each conv block
- Remove `dummy_input` / `flattened_dim` — output is always `(B, 256, 1875)`
- Remove classifier heads (they move to `classifiers.py`)
- Add `@dataclass` config for kernel sizes, channels, dropout rate

**File:** `src/training/backbones.py`

### Phase 2: Two Primary + One Legacy Classifiers (~1 hr)

**What:** Implement the three heads (two primary, one legacy).

**Key details:**
- `GapBaseline` — attach to CNNStem output, `nn.AdaptiveAvgPool1d(1)` → `nn.Linear(256, 1)`
- `MambaClassifier` — install `mamba-ssm>=2,<3`, import `Mamba`, wire S6 block
- `FlattenBaseline` — port from `BinaryDetectorCNN`, keep Flatten → FC(64) → Linear(1) (legacy — not in ablation)

**Design constraint:** All three accept `(B, 1, 30000)` and return `(B, 1)` logits. The trainer treats them identically — only `--model_type` changes. Primary sweep runs `{gap, mamba}` only; `flatten` is a separate run outside the formal comparison.

**File:** `src/training/classifiers.py`

### Phase 3: LOSO Dataset (~30 min)

**What:** Modify `NanoSquiggleDatasetH5` for LOSO support.

**Changes:**
- Accept `exclude_strain: str | None` parameter
- When set, filter out all reads from that strain
- Add `balanced_undersample()` that creates a 50/50 split each epoch
- Keep `gene_target` for per-gene evaluation post-training

**File:** `src/training/dataset.py`

### Phase 4: LOSO Trainer (~1 hr)

**What:** 4-fold cross-validation loop.

```python
for test_strain in ["KP1779", "KP1780", "KP1526", "KP1833"]:
    train_loader = make_loader(exclude_strain=test_strain, balanced=True)
    test_loader  = make_loader(include_only=test_strain)
    model = create_model(model_type)  # fresh init per fold
    train(model, train_loader, val_split=0.15)
    metrics = evaluate(model, test_loader)
    log_to_csv(fold=test_strain, model=model_type, **metrics)
```

**Training config:**
| Param | Value |
|-------|-------|
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW (lr=3e-4, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Max epochs | 50 |
| Early stopping | patience=10, monitor=val_loss |
| Batch size | 32 |
| Balanced undersample | 50/50 positive/negative |

**File:** `src/training/trainer.py`

### Phase 5: CLI Sweep (~30 min)

**What:** Unified entry point.

```
python -m src.training.sweep \
    --h5_path data/processed/hd5f/amr_features.h5 \
    --model_type mamba \
    --epochs 50 \
    --batch_size 32
```

**Model types:** `gap` (primary), `mamba` (primary), `flatten` (legacy, flagged in output)

**Outputs:**
- `results/ablation_results.csv` — per-fold, per-model metrics
- `results/loss_curves/{model_type}_{fold}.png` — loss curves
- `results/cm/{model_type}_{fold}.png` — confusion matrices

**File:** `src/training/sweep.py`

### Phase 6: Integration (~30 min)

**What:** Wire into existing Squiggles repo.

- Update `requirements.txt`: add `mamba-ssm>=2,<3`, `pyyaml>=6`
- Create `src/configs/train_config.yaml`
- Update `AGENTS.md` with Phase 4 section (commands, conventions)
- Create `src/training/__init__.py`

### Phase 7: Tests (~30 min)

**What:** Sanity checks.

- `test_training.py`: forward pass for all 3 models, LOSO split sanity, balanced undersample shape
- Each test runs in <5s on CPU with a tiny batch

---

## 6. Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `mamba-ssm` fails to compile | Medium | Pin `mamba-ssm==2.0.0`, test import first; fallback: manual S6 implementation in PyTorch |
| Strain leakage inflates metrics | High | LOSO is the ONLY valid eval — never random_split for publication |
| Overfitting (4,391 samples) | High | Dropout, early stopping, balanced undersample, LOSO |
| Mamba doesn't improve over GAP | Medium | That is the result — publish "AMR detection is a motif-presence problem" |

---

## 7. Appendix: Remaining from Deep_Learning_Project

**Not ported** (redundant or out of scope):
- `master_server_pipeline.py` — Squiggles already has phase scripts + sentinel gating
- `GeneClassifierCNN` — multiclass cascade is a separate experiment; revisit after ablation
- `sweep_h5.py` — replaced by `src/training/sweep.py` with LOSO
- `flowchart.md` — Squiggles has its own README with equivalent info

**Ported and adapted:**
- Conv layer shapes (kernels [15,7,5,5], channels [32,64,128,256]) → `CNNStem`
- `NanoSquiggleDatasetH5` → `NanoSquiggleDataset` in `dataset.py`
- `train_model`, `evaluate_metrics` → `trainer.py` (with LOSO + early stopping)
