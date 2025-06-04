import nntile
import numpy as np
nntile_config = nntile.starpu.Config(1,0,0,0)
nntile.starpu.init()
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.layer.base_layer import BaseLayer
from nntile.tensor import Tensor, TensorMoments, TensorTraits
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from AdaptiveLayerNNTile import AdaptiveLayerNNTile
from AdaptiveLayerNumpy import AdaptiveLayerNumpy
from AdaptiveLayerTorch import AdaptiveLayerTorch


def relative_error(a, b, eps=1e-6):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + eps)

@pytest.mark.parametrize("B,D", [(3, 5), (4, 7), (8, 8)])
def test_nntile_vs_numpy_and_torch(B, D):
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate random weights and biases
    W1 = np.random.randn(D, D).astype(np.float32)
    b1 = np.random.randn(D).astype(np.float32)
    W2 = np.random.randn(D, D).astype(np.float32)
    b2 = np.random.randn(D).astype(np.float32)

    # Initialize NNTile layer
    layer_nnt = AdaptiveLayerNNTile(B, D, W1, b1, W2, b2)

    # NumPy reference
    layer_np = AdaptiveLayerNumpy(B, D, W1, b1, W2, b2)

    # PyTorch reference
    layer_torch = AdaptiveLayerTorch(D)
    with torch.no_grad():
        layer_torch.fc1.weight.copy_(torch.from_numpy(W1))
        layer_torch.fc1.bias.copy_(torch.from_numpy(b1))
        layer_torch.fc2.weight.copy_(torch.from_numpy(W2))
        layer_torch.fc2.bias.copy_(torch.from_numpy(b2))

    # Random input
    x_np = np.random.randn(B, D).astype(np.float32)
    
    # Forward pass NNTile
    y1_nnt, y2_nnt = layer_nnt.forward(x_np)

    # Forward pass NumPy
    y1_np, y2_np = layer_np.forward(x_np)

    # Forward pass PyTorch
    x_torch = torch.tensor(x_np, requires_grad=True)
    y1_torch, y2_torch = layer_torch(x_torch)
    y1_torch_np = y1_torch.detach().numpy()
    y2_torch_np = y2_torch.detach().numpy()
    
    # Compare forwards
    err_y1_np = relative_error(y1_nnt, y1_np)
    err_y2_np = relative_error(y2_nnt, y2_np)
    err_y1_torch = relative_error(y1_nnt, y1_torch_np)
    err_y2_torch = relative_error(y2_nnt, y2_torch_np)
    
    print(f"Forward rel error vs NumPy: y1={err_y1_np:.2e}, y2={err_y2_np:.2e}")
    print(f"Forward rel error vs Torch: y1={err_y1_torch:.2e}, y2={err_y2_torch:.2e}")    

    # Compare forwards
    assert relative_error(y1_nnt, y1_np) < 1e-5
    assert relative_error(y2_nnt, y2_np) < 1e-5
    assert relative_error(y1_nnt, y1_torch_np) < 1e-5
    assert relative_error(y2_nnt, y2_torch_np) < 1e-5

    # Backward pass: random gradients
    grad_y1 = np.random.randn(B, D).astype(np.float32)
    grad_y2 = np.random.randn(B, D).astype(np.float32)

    # Compute gradients NNTile
    grads_nnt = layer_nnt.backward(grad_y1, grad_y2)
    dx_nnt, dW1_nnt, db1_nnt, dW2_nnt, db2_nnt = grads_nnt

    # Compute gradients NumPy
    grads_np = layer_np.backward(grad_y1, grad_y2)

    # Compute gradients PyTorch
    torch.autograd.backward((y1_torch, y2_torch), (torch.tensor(grad_y1), torch.tensor(grad_y2)))
    dx_torch = x_torch.grad.detach().numpy()
    dW1_torch = layer_torch.fc1.weight.grad.detach().numpy()
    db1_torch = layer_torch.fc1.bias.grad.detach().numpy()
    dW2_torch = layer_torch.fc2.weight.grad.detach().numpy()
    db2_torch = layer_torch.fc2.bias.grad.detach().numpy()
    

    # Compare backwards to NumPy
    err_dx_np = relative_error(dx_nnt, grads_np[0])
    err_dW1_np = relative_error(dW1_nnt, grads_np[1])
    err_db1_np = relative_error(db1_nnt, grads_np[2])
    err_dW2_np = relative_error(dW2_nnt, grads_np[3])
    err_db2_np = relative_error(db2_nnt, grads_np[4])
    
    print(f"Backward rel error vs NumPy: dx={err_dx_np:.2e}, dW1={err_dW1_np:.2e}, db1={err_db1_np:.2e}, dW2={err_dW2_np:.2e}, db2={err_db2_np:.2e}")
    

    # Compare backwards to NumPy
    assert relative_error(dx_nnt, grads_np[0]) < 1e-5
    assert relative_error(dW1_nnt, grads_np[1]) < 1e-5
    assert relative_error(db1_nnt, grads_np[2]) < 1e-5
    assert relative_error(dW2_nnt, grads_np[3]) < 1e-5
    assert relative_error(db2_nnt, grads_np[4]) < 1e-5

    # Compare backwards to PyTorch
    err_dx_torch = relative_error(dx_nnt, dx_torch)
    err_dW1_torch = relative_error(dW1_nnt, dW1_torch)
    err_db1_torch = relative_error(db1_nnt, db1_torch)
    err_dW2_torch = relative_error(dW2_nnt, dW2_torch)
    err_db2_torch = relative_error(db2_nnt, db2_torch)
    
    print(f"Backward rel error vs Torch: dx={err_dx_torch:.2e}, dW1={err_dW1_torch:.2e}, db1={err_db1_torch:.2e}, dW2={err_dW2_torch:.2e}, db2={err_db2_torch:.2e}")
    
    # Compare backwards to PyTorch
    assert relative_error(dx_nnt, dx_torch) < 1e-5
    assert relative_error(dW1_nnt, dW1_torch) < 1e-5
    assert relative_error(db1_nnt, db1_torch) < 1e-5
    assert relative_error(dW2_nnt, dW2_torch) < 1e-5
    assert relative_error(db2_nnt, db2_torch) < 1e-5

test_nntile_vs_numpy_and_torch(3, 5)

test_nntile_vs_numpy_and_torch(4, 7)