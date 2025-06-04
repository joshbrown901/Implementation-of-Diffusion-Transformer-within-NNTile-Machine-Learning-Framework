import numpy as np
import torch
import torch.nn as nn
import pytest

import nntile
import nntile.tensor
from nntile.tensor import TensorMoments, TensorTraits
import nntile.utils.constructors as nntc
import nntile.functions as nntf


class ScaleShiftNumpy:
    @staticmethod
    def forward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray):
        # scale, shift are (B, D); broadcast to (B, N, D)
        scale_b = (1.0 + scale)[:, None, :]    # shape (B,1,D) → broadcast
        y = x * scale_b + shift[:, None, :]
        return y

    @staticmethod
    def backward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray, grad_y: np.ndarray):
        s = (1.0 + scale)[:, None, :]           # shape (B,1,D)
        grad_x = grad_y * s
        # accumulate over N
        grad_scale = np.sum(grad_y * x, axis=1) # shape (B,D)
        grad_shift = np.sum(grad_y, axis=1)     # shape (B,D)
        return grad_x, grad_scale, grad_shift

class ScaleShiftTorch(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # scale, shift are (B,D); avoid any in-place ops
        scale_mod = scale + 1.0                   # out-of-place
        y = x * scale_mod.unsqueeze(1) + shift.unsqueeze(1)
        return y


# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import nntile
import nntile.tensor
import nntile.utils.constructors as nntc
import nntile.functions as nntf
from nntile.tensor import Tensor, TensorMoments, TensorTraits
from nntile.layer.base_layer import BaseLayer
from dataclasses import dataclass

# Initialize StarPU with CPU-only configuration
nntile_config = nntile.starpu.Config(1, 0, 0, 0)
nntile.starpu.init()

# Define dtype mappings and tolerances
dtype2nntile = {'fp32': nntile.tensor.Tensor_fp32}
dtype2tol = {'fp32': {'rtol': 1e-6}}

# Define test parameters
@dataclass
class ScaleShiftTestParams:
    batch_size: int
    seq_length: int
    hidden_size: int
    batch_size_tile: int
    seq_length_tile: int
    hidden_size_tile: int

single_tile = ScaleShiftTestParams(
    batch_size=2, seq_length=3, hidden_size=4,
    batch_size_tile=2, seq_length_tile=3, hidden_size_tile=4
)

multiple_tiles = ScaleShiftTestParams(
    batch_size=4, seq_length=6, hidden_size=8,
    batch_size_tile=2, seq_length_tile=3, hidden_size_tile=4
)

# Define ScaleShift classes (from provided code)
class ScaleShiftNumpy:
    @staticmethod
    def forward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray):
        scale_b = (1.0 + scale)[:, None, :]  # shape (B,1,D)
        y = x * scale_b + shift[:, None, :]
        return y

    @staticmethod
    def backward(x: np.ndarray, scale: np.ndarray, shift: np.ndarray, grad_y: np.ndarray):
        s = (1.0 + scale)[:, None, :]  # shape (B,1,D)
        grad_x = grad_y * s
        grad_scale = np.sum(grad_y * x, axis=1)  # shape (B,D)
        grad_shift = np.sum(grad_y, axis=1)  # shape (B,D)
        return grad_x, grad_scale, grad_shift

class ScaleShiftTorch(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        scale_mod = scale + 1.0
        y = x * scale_mod.unsqueeze(1) + shift.unsqueeze(1)
        return y

class scale_shift(BaseLayer):
    def __init__(self, x: TensorMoments, scale: TensorMoments, shift: TensorMoments, 
                 y: TensorMoments, tmp: Tensor, scale_new: Tensor, 
                 grad_scale_applied: Tensor, tmp1: Tensor):
        self.x = x
        self.y = y
        self.scale = scale
        self.shift = shift
        self.tmp = tmp
        self.scale_new = scale_new
        self.grad_scale_applied = grad_scale_applied
        self.tmp1 = tmp1
        super().__init__([x, scale, shift], [y], [], [tmp, scale_new, tmp1])

    @staticmethod
    def generate_simple(x: TensorMoments, scale: TensorMoments, shift: TensorMoments, 
                        x_traits: TensorTraits, x_distr, next_tag: int):
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        x_distr = x.value.distribution
        scale_traits = TensorTraits(scale.value.shape, scale.value.basetile_shape)
        scale_distr = scale.value.distribution
        shift_traits = TensorTraits(shift.value.shape, shift.value.basetile_shape)
        shift_distr = shift.value.distribution
        y_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        y_distr = [0] * y_traits.grid.nelems
        y_value = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_value.next_tag
        y_grad = type(x.value)(y_traits, y_distr, next_tag)
        next_tag = y_grad.next_tag
        y = TensorMoments(y_value, y_grad, True)
        tmp = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp.next_tag
        scale_new = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = scale_new.next_tag
        grad_scale_applied = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = grad_scale_applied.next_tag
        tmp1 = type(x.value)(x_traits, x_distr, next_tag)
        next_tag = tmp1.next_tag
        return scale_shift(x, scale, shift, y, tmp, scale_new, grad_scale_applied, tmp1), next_tag

    def forward_async(self):
        nntf.fill_async(1.0, self.tmp)
        nntf.fill_async(0.0, self.scale_new)
        nntf.add_slice_async(1.0, self.scale.value, 1.0, self.tmp, self.scale_new, 1)
        self.tmp.wont_use()
        self.scale.value.wont_use()
        nntf.prod_async(self.x.value, self.scale_new, self.y.value)  # Fixed to avoid in-place
        self.x.value.wont_use()
        nntf.add_slice_async(1.0, self.shift.value, 1.0, self.y.value, self.y.value, 1)
        self.shift.value.wont_use()
        self.scale_new.wont_use()
        self.y.value.wont_use()

    def backward_async(self):
        nntf.fill_async(0.0, self.tmp1)
        nntf.add_slice_async(1.0, self.scale.value, 1.0, self.tmp, self.tmp1, 1)
        self.scale.value.wont_use()
        self.tmp.wont_use()
        nntf.prod_async(self.y.grad, self.tmp1, self.x.grad)
        self.tmp1.wont_use()
        self.x.grad.wont_use()
        nntf.prod_async(self.y.grad, self.x.value, self.grad_scale_applied)
        self.x.value.wont_use()
        nntf.sum_slice_async(1.0, self.grad_scale_applied, 0.0, self.scale.grad, 1)
        self.grad_scale_applied.wont_use()
        self.scale.grad.wont_use()
        nntf.sum_slice_async(1.0, self.y.grad, 0.0, self.shift.grad, 1)
        self.shift.grad.wont_use()
        self.y.grad.wont_use()

    @classmethod
    def from_torch(cls, torch_module: ScaleShiftTorch, x: TensorMoments, next_tag: int):
        scale_np = torch_module.scale.data.cpu().numpy()
        shift_np = torch_module.shift.data.cpu().numpy()
        x_traits = TensorTraits(x.value.shape, x.value.basetile_shape)
        x_distr = x.value.distribution
        scale_value = nntc.from_array(scale_np)
        scale_grad = nntc.zeros_like(scale_value)
        scale_tm = TensorMoments(scale_value, scale_grad, True)
        shift_value = nntc.from_array(shift_np)
        shift_grad = nntc.zeros_like(shift_value)
        shift_tm = TensorMoments(shift_value, shift_grad, True)
        layer, next_tag = cls.generate_simple(x, scale_tm, shift_tm, x_traits, x_distr, next_tag)
        return layer, next_tag

    def to_torch(self) -> ScaleShiftTorch:
        torch_mod = ScaleShiftTorch()
        scale_arr = nntc.to_numpy(self.scale.value)
        shift_arr = nntc.to_numpy(self.shift.value)
        torch_mod.scale = nn.Parameter(torch.tensor(scale_arr, dtype=torch.float32))
        torch_mod.shift = nn.Parameter(torch.tensor(shift_arr, dtype=torch.float32))
        return torch_mod

    def to_torch_with_grads(self) -> ScaleShiftTorch:
        torch_mod = self.to_torch()
        torch_mod.scale.grad = torch.tensor(nntc.to_numpy(self.scale.grad), dtype=torch.float32)
        torch_mod.shift.grad = torch.tensor(nntc.to_numpy(self.shift.grad), dtype=torch.float32)
        return torch_mod

# Relative error function
def relative_error(a, b, eps=1e-6):
    return np.max(np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + eps))

# Generate inputs with random data
def generate_inputs(dtype: str, params: ScaleShiftTestParams):
    rng = np.random.default_rng(42)
    x_shape = [params.batch_size, params.seq_length, params.hidden_size]
    x_basetile = [params.batch_size_tile, params.seq_length_tile, params.hidden_size_tile]
    x_traits = TensorTraits(x_shape, x_basetile)
    x_distr = [0] * x_traits.grid.nelems
    x_type = dtype2nntile[dtype]
    x_value = x_type(x_traits, x_distr, 0)
    x_grad = nntc.zeros_like(x_value)
    X = TensorMoments(x_value, x_grad, grad_required=True)
    x_random = rng.standard_normal(x_shape).astype(np.float32)
    x_value.from_array(x_random)
    x_torch = torch.tensor(x_random, requires_grad=True)
    
    scale_np = rng.standard_normal((params.batch_size, params.hidden_size)).astype(np.float32)
    shift_np = rng.standard_normal((params.batch_size, params.hidden_size)).astype(np.float32)
    
    torch_mod = ScaleShiftTorch()
    torch_mod.scale = nn.Parameter(torch.tensor(scale_np))
    torch_mod.shift = nn.Parameter(torch.tensor(shift_np))
    
    nntile_layer, next_tag = scale_shift.from_torch(torch_mod, X, 0)
    
    y_grad_random = rng.standard_normal(x_shape).astype(np.float32)
    nntile_layer.y.grad.from_array(y_grad_random)
    y_grad_torch = torch.tensor(y_grad_random)
    
    nntile.tensor.clear_async(nntile_layer.scale.grad)
    nntile.tensor.clear_async(nntile_layer.shift.grad)
    
    return torch_mod, nntile_layer, x_torch, y_grad_torch

# Test functions
def test_torch_coercion(dtype: str, params: ScaleShiftTestParams):
    torch_layer, nntile_layer, _, _ = generate_inputs(dtype, params)
    torch_layer_other = nntile_layer.to_torch()
    nntile_layer.unregister()
    nntile_layer.x.unregister()
    nntile_layer.y.unregister()
    rtol = dtype2tol[dtype]['rtol']
    for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(), torch_layer_other.named_parameters()):
        assert n1 == n2, f"Parameter names mismatch: {n1} != {n2}"
        assert torch.norm(p1 - p2) <= rtol * torch.norm(p1), f"Torch coercion failed for {n1}"
    print(f"Torch coercion test passed for {params.__dict__}")

def test_forward(dtype: str, params: ScaleShiftTestParams):
    torch_layer, nntile_layer, x, _ = generate_inputs(dtype, params)
    y_torch = torch_layer(x, torch_layer.scale, torch_layer.shift)
    nntile_layer.forward_async()
    y_nntile = torch.tensor(nntc.to_numpy(nntile_layer.y.value))
    nntile_layer.unregister()
    nntile_layer.x.unregister()
    nntile_layer.y.unregister()
    rtol = dtype2tol[dtype]['rtol']
    rel_err = relative_error(y_torch.detach().numpy(), y_nntile.detach().numpy())
    assert rel_err <= rtol, f"Forward pass mismatch: rel_err={rel_err}"
    print(f"Forward test passed for {params.__dict__}, rel_err={rel_err}")

def test_backward(dtype: str, params: ScaleShiftTestParams):
    torch_layer, nntile_layer, x, y_grad = generate_inputs(dtype, params)
    y_torch = torch_layer(x, torch_layer.scale, torch_layer.shift)
    y_torch.backward(y_grad)
    nntile_layer.forward_async()
    nntile_layer.backward_async()
    torch_layer_other = nntile_layer.to_torch_with_grads()
    grad_x_nntile = torch.tensor(nntc.to_numpy(nntile_layer.x.grad))
    nntile_layer.unregister()
    nntile_layer.x.unregister()
    nntile_layer.y.unregister()
    rtol = dtype2tol[dtype]['rtol']
    rel_err_x = relative_error(x.grad.numpy(), grad_x_nntile.numpy())
    assert rel_err_x <= rtol, f"Input gradient mismatch: rel_err={rel_err_x}"
    for (n1, p1), (n2, p2) in zip(torch_layer.named_parameters(), torch_layer_other.named_parameters()):
        assert n1 == n2, f"Parameter names mismatch: {n1} != {n2}"
        assert p1.requires_grad == p2.requires_grad, f"Grad requirement mismatch for {n1}"
        if p1.requires_grad:
            g1, g2 = p1.grad, p2.grad
            rel_err_g = relative_error(g1.numpy(), g2.numpy())
            assert rel_err_g <= rtol, f"Parameter gradient mismatch for {n1}: rel_err={rel_err_g}"
    print(f"Backward test passed for {params.__dict__}, grad_x_rel_err={rel_err_x}")