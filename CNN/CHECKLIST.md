# AMR-CNN Project Checklist: Path to Scientific Validation

This checklist tracks the development of the 1D-CNN for direct Nanopore signal classification. **Adherence to mathematical rigour is mandatory.**

## Milestone 1: Data Sovereignty (The Foundation)
*Focus: Converting raw stochastic electrical signals into structured, normalized tensors.*

- [ ] **Raw Signal Access:** Implement a robust reader for `.pod5` using the `pod5` library.
  - *Note: We Prioritize `pod5.Reader` for efficient batch iteration.*
- [ ] **MAD Normalization:** Implement Median Absolute Deviation (MAD) normalization.
  - Formula: $\hat{x}_i = \frac{x_i - \text{median}(x)}{\text{median}(|x_i - \text{median}(x)|) \times 1.4826}$
- [ ] **Sliding Window Extraction:** Create a generator to extract fixed-length windows (e.g., 2000–4000 signal points) from the `pod5` signal array.
- [ ] **PyTorch Dataset/DataLoader:** Build a `SquiggleDataset` class that handles lazy loading of signals from POD5 files.
- [ ] **Data Verification:** Visualize normalized squiggles to ensure no signal clipping or scaling artifacts.

## Milestone 2: The Neural Architect (The Model)
*Focus: Designing a hierarchical feature detector capable of identifying AMR motifs.*

- [ ] **1D-CNN Backbone:** Implement a modular 1D-CNN using `nn.Module`.
  - [ ] Convolutional layers with appropriate kernel sizes (detecting 3-5 nucleotide "k-mers" in signal space).
  - [ ] Max-pooling or Strided Convolutions for temporal downsampling.
  - [ ] Batch Normalization to stabilize internal covariate shift.
- [ ] **Residual Connections:** Incorporate ResNet blocks to allow for deeper architectures without vanishing gradients.
- [ ] **Training Pipeline:** Write a manual training loop (no high-level wrappers like Lightning initially).
  - [ ] Loss Function: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
  - [ ] Optimizer: Adam or SGD with Momentum.
  - [ ] Learning Rate Scheduler: Implement a "ReduceLROnPlateau" strategy.
- [ ] **Overfitting Check:** Verify the model can "memorize" a small batch of 5-10 signals (confirming the backprop logic is sound).

## Milestone 3: Clinical Bench & Optimization (The Researcher)
*Focus: Quantitative validation and preparation for real-world diagnostic speed.*

- [ ] **Performance Metrics:** Implement AUC-ROC, Precision-Recall curves, and F1-Score calculation.
- [ ] **Confusion Matrix:** Analyze where the model fails (e.g., False Positives in specific bacterial species).
- [ ] **Inference Speed Benchmarking:** Measure the time taken to classify the first 2 seconds of a signal (must be < 100ms for "Read-Until" viability).
- [ ] **Ablation Study:** Test the impact of different window sizes (1000 vs 4000 points) on AMR prediction accuracy.
- [ ] **Final Report:** Document the relationship between signal motifs and the specific AMR genes identified.

---
*"The code is a reflection of your understanding of the physics of the pore. If the signal is noisy, your logic is noisy."*
