import torch
import torch.nn as nn
import numpy as np

def manual_layer_norm(x, eps=1e-6):
    """Reference implementation of LayerNorm without affine transformation"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)

def test_layernorm():
    # Configuration
    hidden_size = 512
    batch_size = 4
    eps = 1e-6
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create LayerNorm
    layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps).to(device)
    
    # Create random input
    torch.manual_seed(42)
    x = torch.randn(2, batch_size, hidden_size, dtype=dtype, device=device, requires_grad=True)
    
    # Forward pass test
    print("=== FORWARD PASS TEST ===")
    y_torch = layer_norm(x)
    y_manual = manual_layer_norm(x, eps=eps)
    
    # Check shape
    assert y_torch.shape == (2, batch_size, hidden_size), f"Shape mismatch! Got {y_torch.shape}"
    print("✓ Output shape correct")
    
    # Calculate relative error
    diff = torch.abs(y_torch - y_manual)
    rel_error = diff / (torch.abs(y_manual) + 1e-8)
    max_rel_error = rel_error.max().item()
    avg_rel_error = rel_error.mean().item()
    
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Avg relative error: {avg_rel_error:.2e}")
    assert avg_rel_error < 1e-6, "Forward pass relative error too large"
    print("✓ Forward pass matches reference implementation")
    
    # Backward pass test
    print("\n=== BACKWARD PASS TEST ===")
    # Create random gradient
    grad_output = torch.randn_like(y_torch)
    
    # Backprop through PyTorch implementation
    y_torch.backward(grad_output, retain_graph=True)
    grad_input_torch = x.grad.clone()
    x.grad.zero_()
    
    # Backprop through manual implementation
    y_manual.backward(grad_output, retain_graph=True)
    grad_input_manual = x.grad.clone()
    x.grad.zero_()
    
    # Check gradient shapes
    assert grad_input_torch.shape == x.shape, "Gradient shape mismatch"
    print("✓ Gradient shape correct")
    
    # Calculate gradient relative error
    grad_diff = torch.abs(grad_input_torch - grad_input_manual)
    grad_rel_error = grad_diff / (torch.abs(grad_input_manual) + 1e-8)
    max_grad_error = grad_rel_error.max().item()
    avg_grad_error = grad_rel_error.mean().item()
    
    print(f"Max gradient relative error: {max_grad_error:.2e}")
    print(f"Avg gradient relative error: {avg_grad_error:.2e}")
    assert avg_grad_error < 1e-6, "Backward pass relative error too large"
    print("✓ Backward pass matches reference implementation")
    
    # Statistics check
    print("\n=== STATISTICS CHECK ===")
    y_np = y_torch.detach().cpu().numpy()
    means = np.mean(y_np, axis=-1)
    stds = np.std(y_np, axis=-1)
    
    print("Per-sample means:", means)
    print("Per-sample stds:", stds)
    assert np.allclose(means, 0, atol=1e-4), "Means not close to 0"
    assert np.allclose(stds, 1, atol=1e-4), "Stds not close to 1"
    print("✓ Normalization statistics correct")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_layernorm()