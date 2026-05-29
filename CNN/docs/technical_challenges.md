# Technical Challenges in Nanopore Signal Processing

Predicting AMR from raw squiggles is a non-trivial task. This document outlines the primary engineering and scientific hurdles you must address.

## 1. Signal Variance & Normalization
Electrical current measurements in Nanopore sequencing are highly sensitive to experimental conditions (temperature, flow cell version, buffer concentration).
*   **The Problem:** The same DNA sequence will produce different raw current values across different sequencing runs.
*   **The Solution:** Use **Median Absolute Deviation (MAD)** normalization. Subtract the median of the signal and divide by the MAD to scale all signals to a standard range.

## 2. Temporal Variance (Base Translocation Speed)
DNA translocation through the pore is stochastic, not at a constant speed.
*   **The Problem:** The signal representing a specific motif might be "stretched" or "compressed" in time.
*   **The Solution:** 1D-CNNs with pooling layers or dilated convolutions are inherently more robust to these variations than rigid sequence-matching algorithms.

## 3. High Signal-to-Noise Ratio (SNR)
Raw squiggles are extremely noisy due to ionic flux and thermal motion.
*   **The Problem:** Identifying resistance-conferring single-nucleotide polymorphisms (SNPs) or small indels directly from the squiggle is significantly harder than from base-called sequences.
*   **The Solution:** Deep learning architectures like **ResNet** or **Inception** blocks can learn robust hierarchical features that ignore high-frequency noise.

## 4. Class Imbalance
Resistant strains are often rare in some clinical datasets compared to susceptible ones.
*   **The Problem:** The model may learn to always predict "Susceptible" to achieve high accuracy.
*   **The Solution:** Use weighted loss functions (e.g., Weighted Cross-Entropy) or oversampling techniques for the minority class.

## 5. Sequence Length & Memory
Nanopore reads can be very long (kilobases to megabases).
*   **The Problem:** Processing a full megabase-long signal at once will exceed GPU memory.
*   **The Solution:** Use **sliding window** techniques. Most researchers use the first 2000–4000 signal points (approx. 250–500 bases) to make the initial classification.

---
*Note:* We should not underestimate the difficulty of **Signal-to-Sequence mapping**. We will need base-called labels to identify which sections of the squiggle correspond to which AMR genes for your training set.
