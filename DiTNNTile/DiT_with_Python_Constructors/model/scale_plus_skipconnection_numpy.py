import numpy as np


class ScalePlusSkipConnectionNumpy:
    @staticmethod
    def forward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray):
        # scale, shift are (B, D); broadcast to (B, N, D)
        scale_b = (1.0 + scale)[:, None, :]    # shape (B,1,D) → broadcast
        y = x * scale_b + shift
        return y

    @staticmethod
    def backward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray, grad_y: np.ndarray):
        s = (1.0 + scale)[:, None, :]           # shape (B,1,D)
        grad_x = grad_y * s
        # accumulate over N
        grad_scale = np.sum(grad_y * x, axis=1) # shape (B,D)
        #grad_shift = grad_y     # shape (B,D)
        return grad_x, grad_scale #grad_shift