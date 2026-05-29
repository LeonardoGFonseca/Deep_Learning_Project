# Implementation Plan: HDF5 Feature Store Compilation

## Objective
Extract centered, normalized signal windows for the 2,392 audited reads and store them in a GPU-ready HDF5 file.
## Key Files & Context
- **Source Manifest:** The audit results/master.csv from Phase 2.2.1.
- **Raw Data:** `data/raw/` (POD5 files).
- **Output:** `data/processed/amr_features.h5`.

## Implementation Steps

### 1. Vectorized Geometry Resolution
Implement the optimized mapping using `numpy.searchsorted` on the cumulative sum of the `mv` tag. This resolves the Gene-to-Signal
mapping in $O(\log L)$ time instead of your previous linear loop.
### 2. Centered-Crop Extraction
For each read:
- Identify the signal center: $C = (S_{start} + S_{end}) // 2$.
- Extract a 30,000 sample window: $[C - 15000, C + 15000]$.
- **Padding:** If the raw read is shorter than 30k samples, apply symmetric zero-padding.

### 3. Per-Read MAD Normalization
Before writing to HDF5, apply the Median Absolute Deviation normalization to ensure all signals are on the same conductance scale
($[-1, 1]$ range).

### 4. Atomic HDF5 Write
- **Dataset 'X':** Shape `(N, 30000)`, Type `Float32`.
- **Dataset 'Y':** Shape `(N,)`, Type `Int8`.
- **Metadata:** Store `read_id`, `strain_id`, and `gene_target` for downstream Leave-One-Strain-Out validation.

## Verification & Testing
- **Visual Check:** Extract 1 sample and plot the signal to ensure it is not just noise/zeros.
- **Integrity Check:** Verify that the number of entries in the H5 file exactly matches the 1,086 + 1,306 count from the audit