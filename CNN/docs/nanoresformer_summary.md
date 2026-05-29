# Research Note: NanoResFormer (PMC12957220) Summary

## Core Innovation
**NanoResFormer** identifies Antibiotic Resistance Genes (ARGs) directly from raw Nanopore squiggles using a hybrid CNN-Transformer architecture.

## Methodology
1.  **Labeling:** Used **Dorado** for basecalling and **NANOSLAST** for signal-to-sequence alignment.
2.  **Architecture:** 
    *   **CNN Encoder:** Local feature extraction and downsampling.
    *   **Transformer Encoder:** 2 layers, 1 head for long-range dependency modeling.
3.  **Windowing:** 40,000-sample sliding window with 80% overlap.
4.  **Performance:** 92.6% sensitivity for 10 classes of ARGs in *K. pneumoniae*.

## Implications for our Project
- We should use **Mamba** instead of a Transformer to achieve $O(L)$ scaling instead of $O(L^2)$.
- We must adopt a similar **Signal-to-Sequence** mapping for our ground truth labeling.
- A **CNN Front-end** is mandatory to reduce the input resolution before the recurrent/state-space layers.
