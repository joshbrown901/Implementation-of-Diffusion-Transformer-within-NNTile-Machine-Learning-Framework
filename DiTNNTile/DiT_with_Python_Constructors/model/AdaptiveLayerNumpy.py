import numpy as np
from typing import List, Union


class AdaptiveLayerNumpy:
    def __init__(self, B, D, W1, b1, W2, b2):
        self.B, self.D = B, D
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2

    @staticmethod
    def silu(x):
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    @staticmethod
    def dsilu(x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)

    def forward(self, x):
        self.x = x
        self.a = self.silu(x)
        self.y1 = self.a @ self.W1.T + self.b1
        self.y2 = self.a @ self.W2.T + self.b2
        return self.y1, self.y2

    def backward(self, grad_y1, grad_y2):
        grad_a1 = grad_y1 @ self.W1
        grad_a2 = grad_y2 @ self.W2
        grad_a = grad_a1 + grad_a2
        grad_x = grad_a * self.dsilu(self.x)
        grad_W1 = grad_y1.T @ self.a
        grad_b1 = grad_y1.sum(axis=0)
        grad_W2 = grad_y2.T @ self.a
        grad_b2 = grad_y2.sum(axis=0)
        return grad_x, grad_W1, grad_b1, grad_W2, grad_b2