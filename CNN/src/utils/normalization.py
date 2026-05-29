
import numpy as np
import torch

def mad_normalization(signal_vector: np.ndarray, eps=1e-8)-> np.ndarray:
    """
    Normalizes a 1D Nanopore signal using Median Absolute Deviation.
    use vectorized C++ for maximum productivity.
    """
    median_val = np.median(signal_vector)

    # Calc the abs deviation from the median.
    mad_val = np.median(np.abs(signal_vector - median_val)) 
    # Normalize the original signal  
    normalize_signal = (signal_vector - median_val) / ((mad_val * 1.4826) + eps)
    
    return (normalize_signal)
"""
# Test
# proof of concept.

raw_read = torch.tensor([1.1, 1.2, 0.9, 1.1, 1.3, 10000.0, 1.1])

clean_read = mad_normalization(raw_read)
print(f"Clear Reads : {clean_read}")
"""
