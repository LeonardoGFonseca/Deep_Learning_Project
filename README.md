# Oxford Nanopore AMR Detection: Deep Learning Project

This repository contains the code and documentation for our Oxford Nanopore Antimicrobial Resistance (AMR) detection project.

The project is organized into two main experimental directories:

## 📁 Repository Structure

### 1. [CNN](./CNN/)
Contains the baseline 1D-CNN implementation for targeted AMR detection directly from raw electrical signal squiggles.
* **Core components**: Data preprocessing scripts, 1D-CNN backbone architecture, and early hyperparameter tuning logs.

### 2. [Mamba](./Mamba/)
Contains the sequential model sweep, Leave-One-Strain-Out (LOSO) cross-validation, and the final sequential inference pipeline.
* **Core components**: Mamba and GRU architectures, RAM-cached HDF5 datasets, and Slurm scripts.
* **Sequential Inference**: The [sequential_inference.py](./Mamba/src/training/sequential_inference.py) script combines the Stage 1 Binary Detector with the Stage 2 Multiclass Classifier.
* **Reports**:
  * [AMR Performance Report](./Mamba/amr_models_report.md): Comparative analysis and final metric tables.
  * [Technical Methodology Guide](./Mamba/amr_technical_methodology.md): Technical breakdown of architectures, formulas, and intuitive analogies.
