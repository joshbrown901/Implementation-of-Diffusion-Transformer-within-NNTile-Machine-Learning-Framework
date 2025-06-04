import numpy as np
import torch
import torch.nn as nn
import pytest

import nntile
import nntile.tensor
from nntile.tensor import TensorMoments, TensorTraits
import nntile.utils.constructors as nntc
import nntile.functions as nntf

from scale_plus_skipconnection_nntile_withrandomgenerators import scale_plus_skipconnection

from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn as nn

import nntile
import nntile.tensor
import nntile.utils.constructors as nntc
from nntile.tensor import TensorMoments, TensorTraits
from nntile.utils.constructors import to_numpy

# NNTile dtype via corresponding Tensor type
dtype2nntile = {
    'fp32': nntile.tensor.Tensor_fp32,
}

dtype2tol = {
    'fp32': {'rtol': 1e-6},
}

@dataclass
class ScalePlusSkipConnectionTestParams:
    batch_size: int
    seq_length: int
    hidden_size: int
    batch_size_tile: int
    seq_length_tile: int
    hidden_size_tile: int

single_tile = ScalePlusSkipConnectionTestParams(
    batch_size=2,
    seq_length=3,
    hidden_size=4,
    batch_size_tile=2,
    seq_length_tile=3,
    hidden_size_tile=4,
)

multiple_tiles = ScalePlusSkipConnectionTestParams(
    batch_size=4,
    seq_length=6,
    hidden_size=8,
    batch_size_tile=2,
    seq_length_tile=3,
    hidden_size_tile=4,
)

def generate_inputs(dtype: str, params: ScalePlusSkipConnectionTestParams):
    rng = np.random.default_rng(42)

    # Generate random scale and shift parameters
    scale_np = rng.standard_normal((params.batch_size, params.hidden_size)).astype(np.float32)
    shift_np = rng.standard_normal((params.batch_size, params.hidden_size)).astype(np.float32)

    # Create PyTorch module
    class TorchScalePlusSkipConnection(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(scale_np))
            self.shift = nn.Parameter(torch.tensor(shift_np))

        def forward(self, x):
            scale_mod = self.scale + 1.0
            return x * scale_mod.unsqueeze(1) + self.shift.unsqueeze(1)

    torch_layer = TorchScalePlusSkipConnection()

    # Create input tensor
    x_shape = [params.batch_size, params.seq_length, params.hidden_size]
    x_basetile = [params.batch_size_tile, params.seq_length_tile, params.hidden_size_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = nntc.zeros_like(x_value)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    
    # Fill with random data
    x_random = rng.standard_normal(x_shape).astype(np.float32)
    x_value.from_array(x_random)
    x_torch = torch.tensor(x_random, requires_grad=True)

    # Create NNTile layer
    nntile_layer, next_tag = scale_plus_skipconnection.from_torch(torch_layer, X, 0)

    # Generate random gradient for backward pass
    y_grad_random = rng.standard_normal(x_shape).astype(np.float32)
    nntile_layer.y.grad.from_array(y_grad_random)
    y_grad_torch = torch.tensor(y_grad_random)

    # Clear gradients
    nntile.tensor.clear_async(nntile_layer.scale.grad)
    nntile.tensor.clear_async(nntile_layer.shift.grad)

    return torch_layer, nntile_layer, x_torch, y_grad_torch

@pytest.mark.parametrize('params', [
    pytest.param(single_tile, id='single_tile'),
    pytest.param(multiple_tiles, id='multiple_tiles'),
])
@pytest.mark.parametrize('dtype', ['fp32'])
class TestScalePlusSkipConnection:

    def test_torch_coercion(self, starpu_simple, torch_rng, dtype: str,
                           params: ScalePlusSkipConnectionTestParams):
        torch_layer, nntile_layer, *_ = generate_inputs(dtype, params)
        torch_layer_other = nntile_layer.to_torch()
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        rtol = dtype2tol[dtype]['rtol']
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                                      torch_layer_other.named_parameters()):
            assert n1 == n2
            assert torch.norm(p1 - p2) <= rtol * torch.norm(p1)
        print(f"Torch coercion test passed for {params.__dict__}")

    def test_forward(self, starpu_simple, torch_rng, dtype: str,
                    params: ScalePlusSkipConnectionTestParams):
        torch_layer, nntile_layer, x, *_ = generate_inputs(dtype, params)
        
        # PyTorch forward
        y_torch = torch_layer(x)
        
        # NNTile forward
        nntile_layer.forward_async()
        y_nntile = torch.tensor(to_numpy(nntile_layer.y.value))
        
        # Cleanup
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        # Compare results
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(y_torch - y_nntile) <= rtol * torch.norm(y_torch)
        print(f"Forward test passed for {params.__dict__}")

    def test_backward(self, starpu_simple, torch_rng, dtype: str,
                     params: ScalePlusSkipConnectionTestParams):
        torch_layer, nntile_layer, x, y_grad = generate_inputs(dtype, params)
        
        # PyTorch forward + backward
        y_torch = torch_layer(x)
        y_torch.backward(y_grad)
        
        # NNTile forward + backward
        nntile_layer.forward_async()
        nntile_layer.backward_async()
        
        # Get gradients
        torch_layer_other = nntile_layer.to_torch_with_grads()
        grad_x_nntile = torch.tensor(to_numpy(nntile_layer.x.grad))
        
        # Cleanup
        nntile_layer.unregister()
        nntile_layer.x.unregister()
        nntile_layer.y.unregister()

        # Compare gradients
        rtol = dtype2tol[dtype]['rtol']
        assert torch.norm(x.grad - grad_x_nntile) <= rtol * torch.norm(x.grad)
        
        # Compare parameter gradients
        for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(),
                                      torch_layer_other.named_parameters()):
            assert n1 == n2
            assert p1.requires_grad == p2.requires_grad
            if p1.requires_grad:
                g1, g2 = p1.grad, p2.grad
                assert torch.norm(g1 - g2) <= rtol * torch.norm(g1)
        print(f"Backward test passed for {params.__dict__}")

# Relative error function
def relative_error(a, b, eps=1e-8):
    return np.abs(a - b).sum() / (np.abs(a).sum() + eps)

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

class ScalePlusSkipConnectionTorch(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # scale, shift are (B,D); avoid any in-place ops
        scale_mod = scale + 1.0                   # out-of-place
        y = x * scale_mod.unsqueeze(1) + shift
        return y