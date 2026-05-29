import torch
import torch.nn as nn

# num_classes : is the number of classification that the model can do
# in this first step we can said that is present or not the antibiotic.

class MambaCNN(nn.Model):
  def __init(self, in_channels: int =1, d_model:int=128, num_classes:int =1):
    super(MambaCNN, self).__init__()
  # Self Encoder. 3000 -> 750
    self.encoder = nn.Sequential(
      # First Layer : 3000 -> 1500.
      nn.Conv1d(in_channels, 32, Kernel_size=3, padding=1),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(Kernel_size=2),

      # Layer 2 1500 -> 750
      nn.Conv1d(32, 64, kernel_size=3, padding=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),

      # Layer 3: Feature refinement.
      nn.Conv1d(64, d_model, kernel_size=3, padding=1),
      nn.BatchNorm1d(d_model),
      nn.ReLU()
  )
    #TODO Implement State Space Model.
    # TODO Final Classifier.