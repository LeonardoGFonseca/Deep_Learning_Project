import torch
import torch.nn as nn

class BinaryDetectorCNN(nn.Module):
    """
    Fase 1: Classificação Binária.
    Identifica se a read contém algum gene de resistência ou se é apenas background/ruído.
    Classes: 0 -> Ruído/Background | 1 -> Presença de Gene
    """
    def __init__(self, sequence_length: int, in_channels: int = 1, d_model: int = 256, kernel_sizes=None, strides=None, channels=None):
        super(BinaryDetectorCNN, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [15, 7, 5, 5]
        if strides is None:
            strides = [2, 2, 2, 2]
        if channels is None:
            channels = [32, 64, 128, d_model]

        layers = []
        current_in_channels = in_channels
        
        for k, s, c in zip(kernel_sizes, strides, channels):
            padding = k // 2  # Padding dinâmico para manter as dimensões consistentes
            layers.extend([
                nn.Conv1d(current_in_channels, c, kernel_size=k, stride=s, padding=padding, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU()
            ])
            current_in_channels = c

        self.feature_extractor = nn.Sequential(*layers)

        dummy_input = torch.zeros(1, in_channels, sequence_length)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        self.flattened_dim = dummy_output.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2)  # Saída para 2 classes (Ruído vs Gene)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        return self.classifier(x)


class GeneClassifierCNN(nn.Module):
    """
    Fase 2: Classificação Multiclasse Exclusiva.
    Apenas recebe reads que foram identificadas como POSITIVAS pela Fase 1.
    Classes: 0 -> Gene A | 1 -> Gene B | 2 -> Gene C
    """
    def __init__(self, sequence_length: int, in_channels: int = 1, d_model: int = 256, kernel_sizes=None, strides=None, channels=None):
        super(GeneClassifierCNN, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = [15, 7, 5, 5]
        if strides is None:
            strides = [2, 2, 2, 2]
        if channels is None:
            channels = [32, 64, 128, d_model]

        layers = []
        current_in_channels = in_channels
        
        for k, s, c in zip(kernel_sizes, strides, channels):
            padding = k // 2
            layers.extend([
                nn.Conv1d(current_in_channels, c, kernel_size=k, stride=s, padding=padding, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU()
            ])
            current_in_channels = c

        self.feature_extractor = nn.Sequential(*layers)

        dummy_input = torch.zeros(1, in_channels, sequence_length)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        self.flattened_dim = dummy_output.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 3)  # Saída para 3 classes (Os 3 Genes AMR)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        return self.classifier(x)