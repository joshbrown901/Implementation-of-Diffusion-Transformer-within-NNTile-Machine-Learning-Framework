import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNumpyUnrolled:
    def __init__(self, embedding_dim, bias=True):
        self.D = embedding_dim
        self.bias_flag = bias
 
        self.W0 = np.random.randn(self.D, self.D).astype(np.float32)
        self.W1 = np.random.randn(self.D, self.D).astype(np.float32)
        self.W2 = np.random.randn(self.D, self.D).astype(np.float32)
        self.W3 = np.random.randn(self.D, self.D).astype(np.float32)
        self.W4 = np.random.randn(self.D, self.D).astype(np.float32)
        self.W5 = np.random.randn(self.D, self.D).astype(np.float32)

        if bias:
            self.b0 = np.random.randn(self.D).astype(np.float32)
            self.b1 = np.random.randn(self.D).astype(np.float32)
            self.b2 = np.random.randn(self.D).astype(np.float32)
            self.b3 = np.random.randn(self.D).astype(np.float32)
            self.b4 = np.random.randn(self.D).astype(np.float32)
            self.b5 = np.random.randn(self.D).astype(np.float32)
        else:
            self.b0 = self.b1 = self.b2 = self.b3 = self.b4 = self.b5 = None

    @staticmethod
    def silu(x):
        sig = 1 / (1 + np.exp(-x))
        return x * sig

    @staticmethod
    def dsilu(x):
        sig = 1 / (1 + np.exp(-x))
        return sig + x * sig * (1 - sig)

    def forward(self, emb):
        activated = MLPNumpyUnrolled.silu(emb)

        lin0 = activated @ self.W0.T
        if self.bias_flag: lin0 = lin0 + self.b0[None,:]
        lin1 = activated @ self.W1.T
        if self.bias_flag: lin1 = lin1 + self.b1[None,:]
        lin2 = activated @ self.W2.T
        if self.bias_flag: lin2 = lin2 + self.b2[None,:]
        lin3 = activated @ self.W3.T
        if self.bias_flag: lin3 = lin3 + self.b3[None,:]
        lin4 = activated @ self.W4.T
        if self.bias_flag: lin4 = lin4 + self.b4[None,:]
        lin5 = activated @ self.W5.T
        if self.bias_flag: lin5 = lin5 + self.b5[None,:]

        return lin0, lin1, lin2, lin3, lin4, lin5

    def backward(self, emb, grad0, grad1, grad2, grad3, grad4, grad5):
        activated = MLPNumpyUnrolled.silu(emb)
        ds = MLPNumpyUnrolled.dsilu(emb)       

        grad_activated = (
            grad0 @ self.W0 + grad1 @ self.W1 + grad2 @ self.W2 +
            grad3 @ self.W3 + grad4 @ self.W4 + grad5 @ self.W5
        ) 

        grad_emb = grad_activated * ds 
        dW0 = grad0.T @ activated
        dW1 = grad1.T @ activated
        dW2 = grad2.T @ activated
        dW3 = grad3.T @ activated
        dW4 = grad4.T @ activated
        dW5 = grad5.T @ activated

        if self.bias_flag:
            db0 = grad0.sum(axis=0)
            db1 = grad1.sum(axis=0)
            db2 = grad2.sum(axis=0)
            db3 = grad3.sum(axis=0)
            db4 = grad4.sum(axis=0)
            db5 = grad5.sum(axis=0)
        else:
            db0 = db1 = db2 = db3 = db4 = db5 = None

        return grad_emb, (dW0, dW1, dW2, dW3, dW4, dW5), (db0, db1, db2, db3, db4, db5)