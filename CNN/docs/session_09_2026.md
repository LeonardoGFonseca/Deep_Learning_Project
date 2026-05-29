# Session Summary: 09-04-2026

## Review of `dataset.py` and `normalization.py`

### Corrected Elements:
-   **Syntactic Fixes:** The basic syntax errors in both `src/data/dataset.py` (e.g., `OrderedDict`, `sys.path.append` usage, `unsqueeze`) and `src/utils/normalization.py` (variable names, test block structure) have been addressed. This is the absolute minimum standard.
-   **`Pod5ReaderCache`:** The LRU cache logic using `OrderedDict` is now functionally correct, although the off-by-one behavior for `max_size` (allowing `max_size + 1` entries temporarily) was noted as a minor inefficiency.

### Lingering Issues (To be mindful of):
-   **`sys.path.append`:** The use of `sys.path.append('../utils/')` remains a brittle dependency on execution context. In a production environment, this would be refactored for robustness.
-   **`normalization.py` Test Block:** While syntactically correct, the test block still uses `torch.tensor` with `numpy.median`, which works due to NumPy's `__array__` interface but represents a mixing of library conventions.

## Advanced Architectural Discussion: Mamba vs. Transformers

### Transformer Limitations (The $O(L^2)$ Problem):
-   **Self-Attention Bottleneck:** Transformers scale quadratically with sequence length ($L$) due to the $L 	imes L$ attention matrix, leading to prohibitive memory and computational costs for long sequences like Nanopore signals.
-   **KV Cache:** The KV cache further exacerbates memory issues for long sequence generation/processing.

### Mamba Advantages (The $O(L)$ Solution):
-   **State Space Model ($S_6$):** Mamba employs a Selective State Space Model, which compresses sequence history into a fixed-size hidden state, achieving linear scaling ($O(L)$).
-   **Input-Dependent Parameters:** The key innovation is making SSM parameters ($A, B, C, \Delta$) input-dependent, allowing selective processing of information, crucial for noisy biological signals.
-   **Infinite Context:** Retains the ability to capture long-range dependencies, overcoming a traditional RNN weakness, without the quadratic cost of Transformers.

## Rationale for Two Convolutional Layers

-   **Hierarchical Feature Extraction:** CNNs act as learned downsamplers. The initial layers (e.g., two layers with strides) extract progressively more complex local features (e.g., basic signal slopes, then k-mer motifs) while reducing the sequence length.
-   **Receptive Field:** Stacking convolutional layers with appropriate kernel sizes and strides rapidly increases the effective receptive field, allowing later layers to integrate information over broader signal regions.
-   **Efficiency for Mamba:** The CNN front-end efficiently reduces the high-resolution raw signal (e.g., 3000 samples) to a more manageable length (e.g., 750 tokens), preventing the Mamba block from being overwhelmed while retaining crucial signal information.

## Next Steps: Implementation of `MambaCNN`

The next immediate task is to implement the `MambaCNN` architecture in `src/models/mamba_cnn.py`, incorporating the discussed CNN front-end, Mamba block (or a suitable placeholder), and classification head.
