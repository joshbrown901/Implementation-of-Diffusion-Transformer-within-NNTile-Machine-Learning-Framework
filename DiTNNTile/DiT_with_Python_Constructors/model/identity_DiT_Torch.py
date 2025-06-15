import torch
import torch.nn as nn

class IdentityAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # No parameters needed for identity operation
        self.hidden_size = config['hidden_size']

    def forward(self, x):
        return x  # Return input unchanged

class TransformerLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        ff_hidden_dim = 4 * self.hidden_size

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)

        # Replace Attention with Identity
        self.attn_block = IdentityAttention(config)

        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.hidden_size),
        )

        # Total 6 * hidden_size
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

    def forward(self, x, condition):
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim=1)
        (pre_attn_shift, pre_attn_scale, post_attn_scale,
         pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params
        out = x
        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1))
                            + pre_attn_shift.unsqueeze(1))
        out = out + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_output)
        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
                           pre_mlp_shift.unsqueeze(1))
        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)
        return out

# Test function to verify shapes and output forward tensor
def test_transformer_layer():
    # Configuration
    config = {'hidden_size': 64}
    batch_size = 32
    seq_len = 128

    # Initialize model
    model = TransformerLayer(config)
    model.eval()  # Disable any stochastic behavior

    # Create input tensors
    x = torch.randn(batch_size, seq_len, config['hidden_size'], requires_grad=True)
    condition = torch.randn(batch_size, config['hidden_size'])

    # Forward pass
    output = model(x, condition)

    # Verify forward shape
    assert output.shape == x.shape, f"Forward output shape {output.shape} does not match input shape {x.shape}"
    print("Forward output shape:", output.shape)
    # print("Forward output tensor:")
    # print(output.detach().numpy())

    # Backward pass
    output_grad = torch.randn_like(output)  # Dummy gradient
    output.backward(output_grad)

    # Verify input gradient shape
    assert x.grad.shape == x.shape, f"Input gradient shape {x.grad.shape} does not match input shape {x.shape}"
    print("Input gradient shape:", x.grad.shape)

if __name__ == "__main__":
    test_transformer_layer()