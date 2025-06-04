import torch.nn as nn
from AttentionTorch import Attention


class TransformerLayer(nn.Module):
    r"""
    Transformer block which is just doing the following based on VIT
        1. LayerNorm followed by Attention
        2. LayerNorm followed by Feed forward Block
        Both these also have residuals added to them

        For DiT we additionally have
        1. Layernorm mlp to predict layernorm affine parameters from
        2. Same Layernorm mlp to also predict scale parameters for outputs
            of both mlp/attention prior to residual connection.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']

        ff_hidden_dim = 4 * self.hidden_size

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.attn_block = Attention(config)

        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.hidden_size),
        )

        # Scale Shift Parameter predictions for this layer
        # 1. Scale and shift parameters for layernorm of attention (2 * hidden_size)
        # 2. Scale and shift parameters for layernorm of mlp (2 * hidden_size)
        # 3. Scale for output of attention prior to residual connection (hidden_size)
        # 4. Scale for output of mlp prior to residual connection (hidden_size)
        # Total 6 * hidden_size
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6*self.hidden_size, bias=True)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[2].weight)
        nn.init.constant_(self.mlp_block[2].bias, 0)

        nn.init.constant_(self.adaptive_norm_layer[1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[1].bias, 0)

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




import pytest
import torch
import torch.nn as nn
import numpy as np

def test_transformer_layer_forward_backward():
    # Test configuration
    config = {
            'hidden_size': 64,
            'num_heads': 8,  # Changed parameter name
            'head_dim': 8,   # Added this (64/8=8)
            'attention_dropout': 0.0,
            'residual_dropout': 0.0
        }
    B, T = 4, 16  # batch size, sequence length
    
    # Initialize the transformer layer
    layer = TransformerLayer(config)
    
    # Create test inputs
    x = torch.randn(B, T, config['hidden_size'], requires_grad=True)
    condition = torch.randn(B, 6 * config['hidden_size'])
    
    # Forward pass
    output = layer(x, condition)
    
    # Test output shape
    assert output.shape == (B, T, config['hidden_size']), \
        f"Output shape {output.shape} != expected {(B, T, config['hidden_size'])}"
    
    # Test forward pass numerical stability
    assert not torch.isnan(output).any(), "Forward pass produced NaNs"
    assert not torch.isinf(output).any(), "Forward pass produced infinite values"
    
    # Backward pass test
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    
    # Check gradients
    assert x.grad is not None, "No gradient computed for input x"
    assert x.grad.shape == x.shape, "Input gradient has wrong shape"
    
    # Check parameter gradients
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradients for {name}"
        assert not torch.isinf(param.grad).any(), f"Infinite values in gradients for {name}"
    
    # Test gradient shapes for all learnable parameters
    for name, param in layer.named_parameters():
        assert param.grad.shape == param.shape, \
            f"Gradient shape mismatch for {name}: {param.grad.shape} != {param.shape}"
    
    # Test specific components
    def test_component(component, input_shape):
        test_input = torch.randn(*input_shape, requires_grad=True)
        out = component(test_input)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        assert test_input.grad is not None
        assert test_input.grad.shape == test_input.shape
    
    # Test attention block
    test_component(layer.attn_block, (B, T, config['hidden_size']))
    
    # Test MLP block
    test_component(layer.mlp_block, (B, T, config['hidden_size']))
    
    # Test adaptive norm layer
    test_component(layer.adaptive_norm_layer, (B, 6 * config['hidden_size']))
    
    # Test with different input sizes
    for hidden_size in [32, 64, 128]:
        config['hidden_size'] = hidden_size
        layer = TransformerLayer(config)
        x = torch.randn(B, T, hidden_size, requires_grad=True)
        condition = torch.randn(B, 6 * hidden_size)
        output = layer(x, condition)
        assert output.shape == (B, T, hidden_size)
        
        # Backward pass
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        assert x.grad is not None

@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [8, 16, 32])
@pytest.mark.parametrize("hidden_size", [64, 128, 256])
def test_transformer_layer_shapes(batch_size, seq_len, hidden_size):
    config = {
        'hidden_size': hidden_size,
        'num_heads': 8,
        'head_dim': hidden_size // 8,
        'attention_dropout': 0.0,
        'residual_dropout': 0.0
    }
    
    layer = TransformerLayer(config)
    x = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)  # Added requires_grad=True
    condition = torch.randn(batch_size, 6 * hidden_size)
    
    output = layer(x, condition)
    
    # Test output shape
    assert output.shape == (batch_size, seq_len, hidden_size)
    
    # Test backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    
    # Check input gradient
    assert x.grad is not None, "Input gradient is None - did you forget requires_grad=True?"
    assert x.grad.shape == x.shape
    
    # Check all parameter gradients
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"No gradient for parameter {name}"
        assert param.grad.shape == param.shape, f"Gradient shape mismatch for {name}"

def test_transformer_layer_initialization():
    config = {
        'hidden_size': 64,
        'num_heads': 8,  # Changed parameter name
        'head_dim': 8,   # Added this
        'attention_dropout': 0.0,
        'residual_dropout': 0.0
    }
    
    layer = TransformerLayer(config)
    
    # Check MLP initialization
    assert torch.allclose(layer.mlp_block[-1].bias, torch.zeros_like(layer.mlp_block[-1].bias))
    
    # Check adaptive norm layer initialization
    assert torch.allclose(layer.adaptive_norm_layer[-1].weight, 
                         torch.zeros_like(layer.adaptive_norm_layer[-1].weight))
    assert torch.allclose(layer.adaptive_norm_layer[-1].bias, 
                         torch.zeros_like(layer.adaptive_norm_layer[-1].bias))
    
    # Check attention weights are initialized (assuming Attention class handles this)
    for name, param in layer.attn_block.named_parameters():
        assert param.data.abs().mean() > 0, f"Parameter {name} appears to be zero-initialized"

def test_transformer_layer_condition_handling():
    config = {
        'hidden_size': 64,
        'num_heads': 8,  # Changed parameter name
        'head_dim': 8,   # Added this
        'attention_dropout': 0.0,
        'residual_dropout': 0.0
    }
    B, T = 4, 16
    
    layer = TransformerLayer(config)
    
    # Test with zero condition
    x = torch.randn(B, T, config['hidden_size'], requires_grad=True)
    condition = torch.zeros(B, 6 * config['hidden_size'])
    output = layer(x, condition)
    
    # Should still produce valid output
    assert output.shape == (B, T, config['hidden_size'])
    
    # Test with large condition values
    condition = torch.randn(B, 6 * config['hidden_size']) * 100
    output = layer(x, condition)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

test_transformer_layer_shapes(4, 16, 128)
