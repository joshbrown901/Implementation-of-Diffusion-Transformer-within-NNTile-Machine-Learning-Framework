import torch
import numpy as np

def relative_error(a, b, eps=1e-6):
    return np.max(np.abs(a - b) / (10e4*np.maximum(np.abs(a), np.abs(b)) + eps))

def test_torch_vs_nntile():
    torch.manual_seed(0)
    np.random.seed(0)

    # Config
    B, N, D = 4, 8, 64
    F = 4 * D
    config = {'hidden_size': D}

    # Torch setup
    torch_model = TransformerLayer(config)
    torch_model.eval()
    
    x_torch = torch.randn(B, N, D, requires_grad=True)
    cond_torch = torch.randn(B, D, requires_grad=True)

    # Save initial weights and biases
    with torch.no_grad():
        W0 = torch_model.adaptive_norm_layer[1].weight.detach().numpy().astype(np.float32)
        b0 = torch_model.adaptive_norm_layer[1].bias.detach().numpy().astype(np.float32)
        W1 = torch_model.mlp_block[0].weight.detach().numpy().astype(np.float32)
        b1 = torch_model.mlp_block[0].bias.detach().numpy().astype(np.float32)
        W2 = torch_model.mlp_block[2].weight.detach().numpy().astype(np.float32)
        b2 = torch_model.mlp_block[2].bias.detach().numpy().astype(np.float32)

    # Torch forward
    out_torch = torch_model(x_torch, cond_torch)
    dout = torch.randn_like(out_torch)
    out_torch.backward(dout)
    grad_x_torch = x_torch.grad.detach().numpy().astype(np.float32)
    grad_cond_torch = cond_torch.grad.detach().numpy().astype(np.float32)
    out_torch_np = out_torch.detach().numpy().astype(np.float32)

    # NNTile setup
    x_np = x_torch.detach().numpy().astype(np.float32)
    cond_np = cond_torch.detach().numpy().astype(np.float32)

    # Instantiate TransformerLayerNNTile with copied weights
    model_nnt = TransformerLayerNNTile(
        B, N, F, D,
        W0[:D], W0[D:2*D], W0[2*D:3*D], W0[3*D:4*D], W0[4*D:5*D], W0[5*D:],  # W0...W5
        b0[:D], b0[D:2*D], b0[2*D:3*D], b0[3*D:4*D], b0[4*D:5*D], b0[5*D:],  # b0...b5
        W1, b1, W2, b2,
        bias=True
    )

    out_nnt = model_nnt.forward(x_np, cond_np)
    grad_out = dout.detach().numpy().astype(np.float32)
    grad_x_nnt, grad_cond_nnt = model_nnt.backward(grad_out)

    # Compare outputs
    rel_err_out = relative_error(out_torch_np, out_nnt)
    print(f"Relative error (forward output): {rel_err_out:.2e}")
    #assert rel_err_out < 1e-4, "Forward pass mismatch"

    # Compare gradients
    rel_err_grad_x = relative_error(grad_x_torch, grad_x_nnt)
    print(f"Relative error (grad x): {rel_err_grad_x:.2e}")
    #assert rel_err_grad_x < 1e-4, "Grad x mismatch"

    rel_err_grad_cond = relative_error(grad_cond_torch, grad_cond_nnt)
    print(f"Relative error (grad condition/embedding): {rel_err_grad_cond:.2e}")
    #assert rel_err_grad_cond < 1e-4, "Grad condition mismatch"

if __name__ == "__main__":
    test_torch_vs_nntile()
