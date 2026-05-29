import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from cnns_backbones import CNNStem


class GapBaseline(nn.Module):
    """CNNStem → GAP → Linear(256→num_classes). Bag-of-motifs baseline."""
    def __init__(self, backbone: CNNStem, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, 256, 1875) → (B, 256, 1)
            nn.Flatten(),               # (B, 256, 1) → (B, 256)
            nn.Linear(256, num_classes),  # (B, 256) → (B, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class MambaClassifier(nn.Module):
    """CNNStem → Mamba S6 → GAP → Linear(256→num_classes). Grammar engine."""
    def __init__(self, backbone: CNNStem, d_state=16, expand=2, num_classes=1):
        super().__init__()
        self.backbone = backbone
        if Mamba is None:
            print("Mamba unavailable — falling back to GRU")
            self.mamba = nn.GRU(input_size=256, hidden_size=256, batch_first=True)
        else:
            self.mamba = Mamba(d_model=256, d_state=d_state, expand_factor=expand)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)            # (B, 256, 1875)
        features = features.permute(0, 2, 1)   # (B, 1875, 256)
        mamba_out = self.mamba(features)        # (B, 1875, 256) or (output, h_n)
        if isinstance(mamba_out, tuple):
            features = mamba_out[0]             # GRU returns (output, h_n)
        else:
            features = mamba_out                # Mamba returns tensor directly
        features = features.permute(0, 2, 1)   # (B, 256, 1875)
        return self.head(features)

class FlattenBaseline(nn.Module):
    """CNNStem → Flatten → FC(64) → Linear(→num_classes). Legacy — not in ablation."""
    def __init__(self, backbone: CNNStem, num_classes=1):
        super().__init__()
        self.backbone = backbone
        # dummy to compute flattened dim 
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 1, 30000))
            flat_dim = dummy.numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)