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
from nntile.layer.layer_norm import LayerNorm
import torch.nn.functional as F
from mlp_nntile import MLPNNTile
from scale_shift_nntile import ScaleShiftNNTile
from AdaptiveLayerNormZeroTorch import AdaptiveLayerNormZeroTorch
from AdaptiveLayerNormZeroNNTile import AdaptiveLayerNormZeroNNTile
from nntile.tensor import TensorMoments, to_numpy, from_array, fill_async, clear_async, sum_slice_async, norm_slice_async, hypot_scalar_inverse_async, prod_slice_async, add_slice_async, add_inplace_async, add_slice_inplace_async, sumprod_slice_async
from nntile.layer.layer_norm import LayerNorm
from layer_norm_noaffine_nntile import LayerNormNoAffine


# Set manual seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
B, N, D = 2, 3, 4

def rel_error(a, b, eps=1e-6):
    return np.max(np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + eps))

# Create random inputs
x_torch = torch.randn(B, N, D, requires_grad=True)
emb_torch = torch.randn(B, D, requires_grad=True)

# Initialize weights and bias
W_full = torch.randn(6 * D, D)
b_full = torch.randn(6 * D)

# Torch implementation
torch_mod = AdaptiveLayerNormZeroTorch(D, bias=True)
# Override parameters
with torch.no_grad():
    torch_mod.W.copy_(W_full)
    torch_mod.b.copy_(b_full)

# Zero gradients
if x_torch.grad is not None:
    x_torch.grad.zero_()
if emb_torch.grad is not None:
    emb_torch.grad.zero_()

# Forward pass (Torch)
out_torch, gate_msa_t, shift_mlp_t, scale_mlp_t, gate_mlp_t = torch_mod(x_torch, emb_torch)

# Convert to numpy for comparison
x_np = x_torch.detach().numpy()
emb_np = emb_torch.detach().numpy()
W_np = W_full.detach().numpy()
b_np = b_full.detach().numpy()
out_torch_np = out_torch.detach().numpy()
gate_msa_t_np = gate_msa_t.detach().numpy()
shift_mlp_t_np = shift_mlp_t.detach().numpy()
scale_mlp_t_np = scale_mlp_t.detach().numpy()
gate_mlp_t_np = gate_mlp_t.detach().numpy()

# Prepare NNTile parameters by splitting
Ws = np.split(W_np.astype(np.float32), 6, axis=0)
bs = np.split(b_np.astype(np.float32), 6)

# NNTile implementation
tile_mod = AdaptiveLayerNormZeroNNTile(
    B, N, D,
    Ws[0], Ws[1], Ws[2], Ws[3], Ws[4], Ws[5],
    bs[0], bs[1], bs[2], bs[3], bs[4], bs[5],
    bias=True
)

# Forward pass (NNTile)
out_nnt, gate_msa_n, shift_mlp_n, scale_mlp_n, gate_mlp_n = tile_mod.forward(x_np, emb_np)

# Generate random upstream gradients
d_out = np.random.randn(*out_torch.shape).astype(np.float32)
d_gate_msa = np.random.randn(*gate_msa_t.shape).astype(np.float32)
d_shift_mlp = np.random.randn(*shift_mlp_t.shape).astype(np.float32)
d_scale_mlp = np.random.randn(*scale_mlp_t.shape).astype(np.float32)
d_gate_mlp = np.random.randn(*gate_mlp_t.shape).astype(np.float32)

# Backward through Torch implementation
out_torch.backward(torch.from_numpy(d_out), retain_graph=True)
gate_msa_t.backward(torch.from_numpy(d_gate_msa), retain_graph=True)
shift_mlp_t.backward(torch.from_numpy(d_shift_mlp), retain_graph=True)
scale_mlp_t.backward(torch.from_numpy(d_scale_mlp), retain_graph=True)
gate_mlp_t.backward(torch.from_numpy(d_gate_mlp), retain_graph=True)

assert x_torch.grad is not None
d_x_torch = x_torch.grad.detach().numpy()
d_emb_torch = emb_torch.grad.detach().numpy()

# Backward pass (NNTile)
d_x_nnt, d_emb_nnt = tile_mod.backward(
    x_np,
    d_out, d_gate_msa, d_shift_mlp, d_scale_mlp, d_gate_mlp
)

# Compute relative errors
errors = {
    'x_out': rel_error(out_torch_np, out_nnt),
    'gate_msa': rel_error(gate_msa_t_np, gate_msa_n),
    'shift_mlp': rel_error(shift_mlp_t_np, shift_mlp_n),
    'scale_mlp': rel_error(scale_mlp_t_np, scale_mlp_n),
    'gate_mlp': rel_error(gate_mlp_t_np, gate_mlp_n),
    'd_x': rel_error(d_x_torch, d_x_nnt),
    'd_emb': rel_error(d_emb_torch, d_emb_nnt),
}

# Print results
print("Relative errors between Torch and NNTile implementations:")
for name, err in errors.items():
    print(f"  {name}: {err:.2e}")

# Assert that all errors are below tolerance
tolerance = 1e-5
assert all(err < tolerance for err in errors.values()), \
    f"Relative error exceeds tolerance of {tolerance}: {errors}"  

print("All outputs match within tolerance!")
