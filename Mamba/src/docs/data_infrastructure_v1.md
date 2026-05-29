# NanoSquiggle-AMR: Comprehensive Research Roadmap

## Phase 1: The Verified Data Engine (Completed/In Production)
**Objective:** Establish a high-throughput, multi-threaded acquisition pipeline with structural integrity verification.

### Core Implementation
- **Script:** `src/scripts/download_data.py`
- **Integrity Protocol:** **Heuristic-Atomic-Commit (HAC)**.
    - *Temporal:* `Content-Length` vs. Local Byte Count.
    - *Formal:* GZIP magic byte verification (`1f 8b`).
    - *Logical:* `tarfile` header dry-run scan.
- **Hygiene Strategy:** **Promotional Move**.
    - Artifacts (`.tar.gz`) are destroyed post-extraction.
    - Signals are promoted from `pod5_pass` to the strain root.
    - Sentinel files (`.download_complete`) track atomic state.

---

## Phase 2: Spatial Anchoring & Feature Store (Current Priority)
**Objective:** Transform 20GB of sparse signal into a high-density, GPU-ready HDF5 Feature Store.

### 2.1: Dorado Alignment Wrapper (`src/utils/run_dorado.py`)
- **Strategy:** **Sequential GPU Execution**.
- **Task:** Execute `dorado basecaller` + `aligner` to generate `.bam`/`.paf` files.
- **Metric:** Capture translocation speed (bp/s) and GPU VRAM saturation.

### 2.2: Coordinate-to-Signal Mapping
- **The Challenge:** Non-linear translocation speed.
- **Mechanism:** Utilize Dorado "Move Tables" (SAM `ns` and `ts` tags) to map the genomic coordinates of `blaSHV` to specific sample indices in the POD5 bitstream.

### 2.3: The Feature Store (`data/processed/amr_features.h5`)
- **Strategy:** **Spatial Pruning**.
- **Schema:**
    - `/signals`: `[N, 30000]` Float32 tensor (MAD-Normalized).
    - `/labels`: `[N]` Binary (1=AMR+, 0=Wildtype/Noise).
    - `/metadata`: Read IDs and Strain IDs.
- **Class Balancing:** **Hard Negative Mining** (50/50 distribution) to prevent majority-class gradient dilution.

---

## Phase 3: Model Implementation (The MambaCNN)
**Objective:** Architectural comparison between 1D-CNN and Selective SSM for longjuange dependency detection.

- **Encoder:** 1D-Convolutional layers for local motif extraction.
- **Sequence Model:** Mamba blocks for $O(L)$ context handling.
- **Hypothesis:** Mamba will outperform CNN-only baselines on the non-stationary "tails" of long gene sequences.

---

## Technical Appendix: Amdahl's Law in Perspective
*Theoretical speedup calculated for the 20GB pipeline:*
- **Download (10MB/s):** ~2000s
- **Dorado (500k bp/s):** ~1000s
- **HDF5/Training:** Variable.
- **Python Overhead:** ~30s (1%).
- **C-Orchestration Gain:** <1% total pipeline time.
- **Conclusion:** Python orchestration is mathematically justified given the $O(1)$ complexity of logic vs. $O(N)$ complexity of data processing.
