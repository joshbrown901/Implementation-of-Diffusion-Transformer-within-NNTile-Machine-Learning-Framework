import numpy as np

class FeedForwardNumpy:
    def __init__(self, D, F):
        self.W1 = np.random.randn(F, D).astype(np.float32)
        self.b1 = np.random.randn(F).astype(np.float32)
        self.W2 = np.random.randn(D, F).astype(np.float32)
        self.b2 = np.random.randn(D).astype(np.float32)

    def gelu_tanh(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

    def gelu_tanh_grad(self, x):
        # More precise implementation
        sqrt_2_over_pi = np.sqrt(2/np.pi)
        x_cubed = x**3
        inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        tanh_inner = np.tanh(inner)
        derivative = 0.5 * (1 + tanh_inner) + \
                    0.5 * x * (1 - tanh_inner**2) * \
                    sqrt_2_over_pi * (1 + 3 * 0.044715 * x*x)
        return derivative

    def forward(self, x):
        self.x = x
        self.x1 = x @ self.W1.T + self.b1
        self.a1 = self.gelu_tanh(self.x1)
        self.out = self.a1 @ self.W2.T + self.b2
        return self.out

    def backward(self, grad_out):
        # Reshape all tensors to 2D (batch*sequence, features)
        B, N, D = grad_out.shape
        F = self.b1.shape[0]
        
        # Gradient for W2 and b2
        self.dW2 = grad_out.reshape(-1, D).T @ self.a1.reshape(-1, F)
        self.db2 = grad_out.sum(axis=(0, 1))
        
        # Gradient for a1
        da1 = grad_out @ self.W2
        
        # Gradient for x1 (before activation)
        dx1 = da1 * self.gelu_tanh_grad(self.x1)
        
        # Gradient for W1 and b1
        self.dW1 = dx1.reshape(-1, F).T @ self.x.reshape(-1, D)
        self.db1 = dx1.sum(axis=(0, 1))
        
        # Gradient for input x
        dx = dx1 @ self.W1
        return dx