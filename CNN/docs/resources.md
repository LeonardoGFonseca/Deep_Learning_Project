# AMR Prediction from Nanopore Signals: Key Resources

Mastery of this project requires an understanding of how raw electrical signals map to biological functionality without the intermediate (and lossy) step of base-calling.

## 1. Key Research Papers (The Foundations)

*   **SquiggleNet: Real-time, direct classification of Nanopore signals**
    *   *Authors:* Bao et al. (2021)
    *   *Journal:* Nature Communications
    *   *Significance:* The seminal paper on using 1D-CNNs for direct squiggle classification. It demonstrates identifying viral and bacterial species from the first 1-2 seconds of signal. This is your primary architectural reference.
*   **Real-time, selective sequencing on molecular devices**
    *   *Authors:* Loose et al. (2016)
    *   *Journal:* Nature Methods
    *   *Significance:* Introduced "Read-Until" or adaptive sampling. It explains the "Why" of direct signal analysis—the ability to eject non-target DNA in real-time.
*   **DeepSignal: Detecting DNA methylation from Nanopore sequencing reads**
    *   *Authors:* Ni et al. (2019)
    *   *Journal:* Bioinformatics
    *   *Significance:* While focused on methylation, it pioneered the use of Bi-LSTM and CNN architectures for raw signal feature extraction.
*   **Targeted Nanopore sequencing by real-time mapping of raw electrical signals with UNCALLED**
    *   *Authors:* Kovaka et al. (2021)
    *   *Journal:* Nature Biotechnology
    *   *Significance:* Uses signal matching (DTW or FM-index) for real-time mapping, providing a benchmark for your deep learning approach.

## 2. Datasets (The Raw Material)

*   **PRJEB41335 (SquiggleNet Data):** A large-scale repository of raw Nanopore signals (.fast5) used in the SquiggleNet study. Essential for pre-training.
*   **PRJNA494801 (ONT Bacterial AMR):** A dataset of multi-drug resistant bacterial genomes sequenced on MinION, providing the ground truth AMR labels you need.
*   **Zibra Project (Real-time Outbreak Surveillance):** High-quality datasets for testing real-time diagnostic models in clinical settings.
*   Sqidbase : https://squidbase.org/ a database that contains raw nanopore signals.

## 3. Technical Tools & Libraries (The Workbench)

*   **`ont-fast5-api`:** The official toolkit for handling the HDF5-based .fast5 format.
*   **`pod5`:** The high-performance successor to .fast5. Your pipeline must support both.
*   **`pyslow5`:** A high-speed alternative for accessing raw signal data, often 10x faster than standard HDF5 libraries.
*   **`pomegranate`:** A Python library for hidden Markov models (HMMs), useful if you want to hybridize your CNN with probabilistic sequential models.
*   **`ont-pyguppy-client-lib`:** For interfacing with the Guppy basecaller (to generate ground-truth labels for your training data).

## 4. Mathematics for Signal Processing

*   **The Fast Fourier Transform (FFT):** While 1D-CNNs operate in the time domain, frequency domain features (Power Spectral Density) can be critical for signal denoising.
*   **Dynamic Time Warping (DTW):** The classical method for aligning two signals of varying speeds.   Your CNN should ideally learn to be invariant to the speed fluctuations (400-450 bps) that DTW handles manually.
