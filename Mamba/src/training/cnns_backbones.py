import torch
import torch.nn as nn

class CNNStem(nn.Module):
    """
    Phase 1: CNNS backbone.
    This function will serve as a backbone to integrate with the CNN + GAP and the CNN + Mamba.
    In order to compare the performance of both approaches.
    """
    def __init__(self, in_channels=1, d_model=256, dropout=0.2):
        super().__init__()
        kernel_size = [15, 7, 5, 5]
        strides = [2, 2, 2, 2]
        channels = [32, 64, 128, d_model]

        layers = []
        current_in = in_channels
        for k, s, c in zip(kernel_size, strides, channels):
            layers.extend(
                [
                nn.Conv1d(current_in, c, kernel_size=k, stride=s, padding=k//2, bias=False),
                nn.BatchNorm1d(c),
                nn.GELU(),
                nn.Dropout(dropout)
                ])
            current_in = c

        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return (self.feature_extractor(x))

