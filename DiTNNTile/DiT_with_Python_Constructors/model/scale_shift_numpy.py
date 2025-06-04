import numpy as np
# NumPy ScaleShift class
class ScaleShiftNumpy:
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray):
        scale_b = (1.0 + scale)[:, None, :]    # shape (B,1,D) → broadcast
        y = x * scale_b + shift[:, None, :]
        return y

    @staticmethod
    def backward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray, grad_y: np.ndarray):
        # scale is (B,D)
        s = (1.0 + scale)[:, None, :]           # shape (B,1,D)
        grad_x = grad_y * s
        # grad_scale and grad_shift accumulate over N
        grad_scale = np.sum(grad_y * x, axis=1) # shape (B,D)
        grad_shift = np.sum(grad_y, axis=1)     # shape (B,D)
        return grad_x, grad_scale, grad_shift