import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.attention import SymplecticHamiltonianAttention

def test_initialization():
    embed_dim = 128
    num_heads = 8
    num_steps = 3
    
    attn = SymplecticHamiltonianAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_steps=num_steps
    )
    
    assert attn.embed_dim == embed_dim
    assert attn.num_heads == num_heads
    assert attn.num_key_value_heads == num_heads
    assert attn.head_dim == 16
    assert attn.num_steps == num_steps
    assert attn.dt == 0.2
    
    # Check step-wise module list registration
    for k in range(num_steps):
        assert hasattr(attn, f"q_proj_{k}")
        assert hasattr(attn, f"k_proj_{k}")
        assert hasattr(attn, f"v_proj_{k}")

def test_shapes_and_outputs():
    embed_dim = 64
    num_heads = 4
    attn = SymplecticHamiltonianAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    B, S = 2, 64
    x = mx.random.normal((B, S, embed_dim))
    
    out = attn(x)
    assert out.shape == (B, S, embed_dim)
    assert not mx.any(mx.isnan(out)).item()

def test_causality():
    embed_dim = 64
    num_heads = 4
    attn = SymplecticHamiltonianAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    B, S = 1, 10
    x1 = mx.random.normal((B, S, embed_dim))
    
    # Create x2 identical to x1 up to index 5, but different at future steps
    x2 = mx.array(x1)
    x2[:, 6:, :] = mx.random.normal((B, 4, embed_dim))
    
    out1 = attn(x1)
    out2 = attn(x2)
    
    # Assert output up to index 5 is identical (causal)
    assert mx.allclose(out1[:, :6, :], out2[:, :6, :], rtol=1e-5, atol=1e-5).item()

def test_parallel_vs_cached():
    embed_dim = 64
    num_heads = 4
    num_steps = 3
    attn = SymplecticHamiltonianAttention(embed_dim=embed_dim, num_heads=num_heads, num_steps=num_steps)
    
    B, S = 2, 5
    x = mx.random.normal((B, S, embed_dim))
    
    # Parallel prefill mode
    kv_cache_par = {}
    out_par = attn(x, kv_cache=kv_cache_par)
    
    # Step-by-step cached generation
    kv_cache_seq = {}
    out_seq_list = []
    for i in range(S):
        x_step = x[:, i:i+1, :]
        out_step = attn(x_step, kv_cache=kv_cache_seq)
        out_seq_list.append(out_step)
        
    out_seq = mx.concatenate(out_seq_list, axis=1)
    
    # Verify outputs are mathematically equivalent
    assert mx.allclose(out_par, out_seq, rtol=1e-4, atol=1e-4).item()
    
    # Verify states stored for all integration steps are equivalent
    for k in range(num_steps):
        assert mx.allclose(kv_cache_par[f"K_state_{k}"], kv_cache_seq[f"K_state_{k}"], rtol=1e-4, atol=1e-4).item()
        assert mx.allclose(kv_cache_par[f"denom_state_{k}"], kv_cache_seq[f"denom_state_{k}"], rtol=1e-4, atol=1e-4).item()

def test_stability():
    embed_dim = 64
    num_heads = 4
    # Run with 10 integration steps to stress-test stability
    attn = SymplecticHamiltonianAttention(embed_dim=embed_dim, num_heads=num_heads, num_steps=10, dt=0.5)
    
    B, S = 1, 128
    # Input with large norm
    x = mx.random.normal((B, S, embed_dim)) * 10.0
    
    out = attn(x)
    # Check that outputs remain bounded (no NaNs or infinite scale)
    assert not mx.any(mx.isnan(out)).item()
    assert not mx.any(mx.isinf(out)).item()
    
    norm_in = mx.linalg.norm(x).item()
    norm_out = mx.linalg.norm(out).item()
    # Should not explode infinitely
    assert norm_out < norm_in * 10.0
