import numpy as np
import torch
import torch.nn as nn
import pytest

import nntile
import nntile.tensor
from nntile.tensor import TensorMoments, TensorTraits
import nntile.utils.constructors as nntc
import nntile.functions as nntf

from scale_shift_numpy import ScaleShiftNumpy
from scale_shift_torch import ScaleShiftTorch
from scale_shift_nntile import ScaleShiftNNTile
# import scale_shift

# Relative error function
def relative_error(a, b, eps=1e-8):
    return np.abs(a - b).sum() / (np.abs(a).sum() + eps)
        
@pytest.mark.parametrize("B, N, D", [(2, 3, 4), (1, 5, 5)])
def test_scale_shift_numpy_vs_torch_vs_nntile(B, N, D):
    np.random.seed(0)
    torch.manual_seed(0)

    x_np      = np.random.randn(B, N, D).astype(np.float32)
    scale_np  = np.random.randn(B, D).astype(np.float32)
    shift_np  = np.random.randn(B, D).astype(np.float32)
    grad_y_np = np.ones_like(x_np, dtype=np.float32)

    # NumPy forward/backward
    y_np = ScaleShiftNumpy.forward(x_np, scale_np, shift_np)
    grad_x_np, grad_scale_np, grad_shift_np = ScaleShiftNumpy.backward(
        x_np, scale_np, shift_np, grad_y_np
    )

    # PyTorch forward/backward
    x_torch     = torch.tensor(x_np,      requires_grad=True)
    scale_torch = torch.tensor(scale_np,  requires_grad=True)
    shift_torch = torch.tensor(shift_np,  requires_grad=True)

    model = ScaleShiftTorch()
    y_torch = model(x_torch, scale_torch, shift_torch)
    y_torch.sum().backward()

    # NNTile forward/backward
    nntile_scale_shift = ScaleShiftNNTile(B, N, D)
    y_nntile = nntile_scale_shift.forward(x_np, scale_np, shift_np)
    grad_x_nntile, grad_scale_nntile, grad_shift_nntile = nntile_scale_shift.backward(grad_y_np)

    # Compare raw outputs
    assert np.allclose(y_torch.detach().numpy(), y_np, atol=1e-5), "PyTorch vs NumPy forward mismatch"
    assert np.allclose(y_nntile, y_np, atol=1e-5), "NNTile vs NumPy forward mismatch"
    
    assert np.allclose(x_torch.grad.numpy(), grad_x_np, atol=1e-5), "PyTorch vs NumPy grad x mismatch"
    assert np.allclose(grad_x_nntile, grad_x_np, atol=1e-5), "NNTile vs NumPy grad x mismatch"
    
    assert np.allclose(scale_torch.grad.numpy(), grad_scale_np, atol=1e-5), "PyTorch vs NumPy grad scale mismatch"
    assert np.allclose(grad_scale_nntile, grad_scale_np, atol=1e-5), "NNTile vs NumPy grad scale mismatch"
    
    assert np.allclose(shift_torch.grad.numpy(), grad_shift_np, atol=1e-5), "PyTorch vs NumPy grad shift mismatch"
    assert np.allclose(grad_shift_nntile, grad_shift_np, atol=1e-5), "NNTile vs NumPy grad shift mismatch"

    # Compute and check relative errors
    print("\nPyTorch vs NumPy:")
    print("Forward Relative Error:", relative_error(y_torch.detach().numpy(), y_np))
    print("Grad X Relative Error:", relative_error(x_torch.grad.numpy(), grad_x_np))
    print("Grad Scale Relative Error:", relative_error(scale_torch.grad.numpy(), grad_scale_np))
    print("Grad Shift Relative Error:", relative_error(shift_torch.grad.numpy(), grad_shift_np))

    print("\nNNTile vs NumPy:")
    print("Forward Relative Error:", relative_error(y_nntile, y_np))
    print("Grad X Relative Error:", relative_error(grad_x_nntile, grad_x_np))
    print("Grad Scale Relative Error:", relative_error(grad_scale_nntile, grad_scale_np))
    print("Grad Shift Relative Error:", relative_error(grad_shift_nntile, grad_shift_np))

    # Assert relative errors
    assert relative_error(y_nntile, y_np) < 1e-5, "NNTile forward relative error too large"
    assert relative_error(grad_x_nntile, grad_x_np) < 1e-5, "NNTile grad x relative error too large"
    assert relative_error(grad_scale_nntile, grad_scale_np) < 1e-5, "NNTile grad scale relative error too large"
    assert relative_error(grad_shift_nntile, grad_shift_np) < 1e-5, "NNTile grad shift relative error too large"



test_scale_shift_numpy_vs_torch_vs_nntile(2, 3, 4)

test_scale_shift_numpy_vs_torch_vs_nntile(1, 5, 5)