import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from qan_transformers.mlx.attention import AnyonicBraidLinear

def test_initialization():
    input_dim = 64
    output_dim = 64
    depth = 8
    
    layer = AnyonicBraidLinear(input_dim=input_dim, output_dim=output_dim, depth=depth)
    
    assert layer.input_dim == input_dim
    assert layer.output_dim == output_dim
    assert layer.depth == depth
    assert layer.N == 64
    assert layer.num_pairs_odd == 32
    assert layer.num_pairs_even == 31
    
    assert isinstance(layer.weight_odd, mx.array)
    assert layer.weight_odd.shape == (32, 8)
    assert isinstance(layer.weight_even, mx.array)
    assert layer.weight_even.shape == (31, 8)
    assert isinstance(layer.theta, mx.array)

def test_orthogonality():
    # An orthogonal braid network should preserve L2 norm of input vectors
    dim = 128
    layer = AnyonicBraidLinear(input_dim=dim, output_dim=dim, depth=16)
    
    x = mx.random.normal((2, 10, dim))
    out = layer(x)
    
    # Calculate norms along the last dimension
    norm_in = mx.sqrt(mx.sum(x**2, axis=-1))
    norm_out = mx.sqrt(mx.sum(out**2, axis=-1))
    
    # Check that they are extremely close (within float32 precision limits)
    diff = mx.abs(norm_in - norm_out)
    assert mx.all(diff < 1e-4).item()

def test_dimensions():
    # Test mapping from small to large
    layer_up = AnyonicBraidLinear(input_dim=64, output_dim=128, depth=8)
    x_small = mx.random.normal((2, 5, 64))
    out_large = layer_up(x_small)
    assert out_large.shape == (2, 5, 128)
    
    # Test mapping from large to small
    layer_down = AnyonicBraidLinear(input_dim=128, output_dim=64, depth=8)
    x_large = mx.random.normal((2, 5, 128))
    out_small = layer_down(x_large)
    assert out_small.shape == (2, 5, 64)

def test_differentiability():
    # Verify that gradients propagate through to all parameters (logits and theta)
    layer = AnyonicBraidLinear(input_dim=64, output_dim=64, depth=8)
    
    x = mx.random.normal((2, 5, 64))
    
    target = mx.random.normal((2, 5, 64))
    
    def loss_fn(model, x):
        out = model(x)
        return mx.mean((out - target)**2)
        
    val_and_grad = nn.value_and_grad(layer, loss_fn)
    loss, grads = val_and_grad(layer, x)
    
    assert not mx.any(mx.isnan(loss)).item()
    
    # Ensure gradients exist and are non-zero for odd/even weights and theta
    assert "weight_odd" in grads
    assert "weight_even" in grads
    assert "theta" in grads
    
    assert mx.sum(mx.abs(grads["weight_odd"])).item() > 0.0
    assert mx.sum(mx.abs(grads["weight_even"])).item() > 0.0
    assert mx.abs(grads["theta"]).item() > 0.0

def test_straight_through_estimator():
    # Verify that the straight-through estimator works and allows updating the continuous logits
    # even though a thresholding/sign function is used in the forward pass.
    layer = AnyonicBraidLinear(input_dim=32, output_dim=32, depth=4)
    
    # Initialize logits to something very close to 0
    layer.weight_odd = mx.array(np.random.uniform(-0.01, 0.01, layer.weight_odd.shape), dtype=mx.float32)
    layer.weight_even = mx.array(np.random.uniform(-0.01, 0.01, layer.weight_even.shape), dtype=mx.float32)
    
    x = mx.random.normal((1, 1, 32))
    
    def loss_fn(model, x):
        out = model(x)
        # Choose a target that encourages positive crossings
        return mx.sum(out)
        
    val_and_grad = nn.value_and_grad(layer, loss_fn)
    loss, grads = val_and_grad(layer, x)
    
    # Check that gradients for the logits are non-zero
    assert mx.sum(mx.abs(grads["weight_odd"])).item() > 0.0
    assert mx.sum(mx.abs(grads["weight_even"])).item() > 0.0
