import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveLayerTorch(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.act = torch.nn.SiLU()
        self.fc1 = torch.nn.Linear(D, D, bias=True)
        self.fc2 = torch.nn.Linear(D, D, bias=True)

    def forward(self, x):
        a = self.act(x)
        return self.fc1(a), self.fc2(a)