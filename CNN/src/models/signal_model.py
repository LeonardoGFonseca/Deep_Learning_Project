import torch
import torch.nn as nn

class Simple1DCNN(nn.Module):
    def __init__(self, sequence_length: int):
        super(Simple1DCNN, self).__init__()

        # 1. Hierarchical Feature Extractor (Convolutional Backbone)
        self.feature_extractor = nn.Sequential(
            # Layer 1: Detects local slopes and edges
            # Input: (Batch, 1, sequence_length)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Downsample by 2

            # Layer 2: Detects complex motifs (Sine vs Square characteristics)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Downsample by 2
        )

        # Auto-dimension resolver for the flattening layer
        dummy_input = torch.zeros(1, 1, sequence_length)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        self.flattened_dim = dummy_output.numel()

        # 2. The Decision Module (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5), # REGULARIZATION: Prevent overfitting
            nn.Linear(64, 1)    # Output logit for Binary Classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fix the dimensionality check: dim() is a method
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dimension: (Batch, Length) -> (Batch, 1, Length)

        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
