import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiPositiveInfoNCE(nn.Module):
    """Multi-target InfoNCE loss function"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, output, mask):
        return -torch.log((F.softmax(output, dim=1) * mask).sum(1)).mean()
