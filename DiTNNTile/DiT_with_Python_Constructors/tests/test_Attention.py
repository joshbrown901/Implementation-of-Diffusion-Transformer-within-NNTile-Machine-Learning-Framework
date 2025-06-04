import numpy as np
import torch
import torch.nn as nn

from AttentionNumpy import Attention_numpy
from AttentionTorch import Attention

# Assuming both Attention_numpy and Attention (from your snippet) are defined

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Config
config = {'num_heads': 4, 'hidden_size': 16, 'head_dim': 4}
B, N = 2, 5

# Init modules
attn_np = Attention_numpy(config)
attn_torch = Attention(config)

# Sync parameters
with torch.no_grad():
    attn_np.qkv_weight = attn_torch.qkv_proj.weight.cpu().numpy().astype(np.float64)
    attn_np.qkv_bias = attn_torch.qkv_proj.bias.cpu().numpy().astype(np.float64)
    attn_np.out_weight = attn_torch.output_proj.weight.cpu().numpy().astype(np.float64)
    attn_np.out_bias = attn_torch.output_proj.bias.cpu().numpy().astype(np.float64)

# Input
x_np = np.random.randn(B, N, config['hidden_size']).astype(np.float64)
x_torch = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)

# Forward
out_np = attn_np.forward(x_np)
out_torch = attn_torch(x_torch)
out_torch_np = out_torch.detach().cpu().numpy()

# Forward relative error
forward_rel_error = np.linalg.norm(out_np - out_torch_np) / np.linalg.norm(out_torch_np)
print(f"✅ Forward relative error: {forward_rel_error:.3e}")

# Backward
grad_out_np = np.random.randn(*out_np.shape).astype(np.float64)
grad_out_torch = torch.tensor(grad_out_np, dtype=torch.float32)

# NumPy backward
dx_np, grads_np = attn_np.backward(grad_out_np)

# PyTorch backward
out_torch.backward(grad_out_torch)

# Gradients for comparison
dx_torch = x_torch.grad.detach().cpu().numpy()
grad_qkv_weight_torch = attn_torch.qkv_proj.weight.grad.detach().cpu().numpy()
grad_qkv_bias_torch = attn_torch.qkv_proj.bias.grad.detach().cpu().numpy()
grad_out_weight_torch = attn_torch.output_proj.weight.grad.detach().cpu().numpy()
grad_out_bias_torch = attn_torch.output_proj.bias.grad.detach().cpu().numpy()

# Relative errors for backward gradients
def rel_error(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

print(f"✅ Backward relative error (input grad): {rel_error(dx_np, dx_torch):.3e}")
print(f"✅ Backward relative error (qkv weight): {rel_error(grads_np['qkv_weight'], grad_qkv_weight_torch):.3e}")
print(f"✅ Backward relative error (qkv bias): {rel_error(grads_np['qkv_bias'], grad_qkv_bias_torch):.3e}")
print(f"✅ Backward relative error (out weight): {rel_error(grads_np['out_weight'], grad_out_weight_torch):.3e}")
print(f"✅ Backward relative error (out bias): {rel_error(grads_np['out_bias'], grad_out_bias_torch):.3e}")