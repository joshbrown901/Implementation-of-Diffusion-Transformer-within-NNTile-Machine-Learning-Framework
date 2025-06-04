from typing import List, Union
import torch
import torch.nn as nn



class FeedForwardTorch(nn.Module):
    def __init__(self, D, F):
        super().__init__()
        self.fc1 = nn.Linear(D, F)
        self.fc2 = nn.Linear(F, D)
        self.gelu = nn.GELU(approximate='tanh')

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))