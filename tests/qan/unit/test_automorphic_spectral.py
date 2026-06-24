import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.attention import AutomorphicSpectralAttention

def test_initialization():
    embed_dim = 128
    num_heads = 8
    num_modes = 32
    
    attn = AutomorphicSpectralAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_modes=num_modes
    )
    
    assert attn.embed_dim == embed_dim
    assert attn.num_heads == num_heads
    assert attn.num_key_value_heads == num_heads
    assert attn.head_dim == 16
    assert attn.num_modes == num_modes
    assert attn.eigenvalues.shape == (num_modes // 2,)

def test_modular_reduction():
    embed_dim = 64
    num_heads = 4
    attn = AutomorphicSpectralAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Generate arbitrary points in upper half-plane (v > 0)
    # Some points inside unit circle, some far out, some with large shifts
    u = mx.array([0.1, 1.7, -3.2, 0.05, 0.4])
    v = mx.array([0.5, 2.0, 0.8, 0.2, 0.9])
    
    u_star, v_star = attn.modular_reduce(u, v, steps=5)
    
    # Check that they lie in the SL_2(Z) fundamental domain:
    # 1. |u*| <= 0.5 (with small tolerance)
    assert mx.all(mx.abs(u_star) <= 0.5 + 1e-5).item()
    # 2. u*^2 + v*^2 >= 1.0 (with small tolerance)
    norms_sq = u_star**2 + v_star**2
    assert mx.all(norms_sq >= 1.0 - 1e-5).item()
    # 3. v* must be positive and bounded
    assert mx.all(v_star > 0.0).item()

def test_modular_invariance():
    embed_dim = 64
    num_heads = 4
    attn = AutomorphicSpectralAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Choose a starting point tau = u + i*v
    u = mx.array([0.2])
    v = mx.array([0.5])
    
    # Reduce original point
    u_star, v_star = attn.modular_reduce(u, v, steps=8)
    
    # 1. Modular T-transformation: tau -> tau + 1
    u_t, v_t = attn.modular_reduce(u + 1.0, v, steps=8)
    assert abs(u_star.item() - u_t.item()) < 1e-4
    assert abs(v_star.item() - v_t.item()) < 1e-4
    
    # 2. Modular S-transformation: tau -> -1/tau
    denom = u**2 + v**2
    u_s = -u / denom
    v_s = v / denom
    u_reduced_s, v_reduced_s = attn.modular_reduce(u_s, v_s, steps=8)
    assert abs(u_star.item() - u_reduced_s.item()) < 1e-4
    assert abs(v_star.item() - v_reduced_s.item()) < 1e-4

def test_linear_scaling_and_shapes():
    embed_dim = 64
    num_heads = 4
    num_modes = 16
    attn = AutomorphicSpectralAttention(embed_dim=embed_dim, num_heads=num_heads, num_modes=num_modes)
    
    B, S = 2, 256
    x = mx.random.normal((B, S, embed_dim))
    
    out = attn(x)
    assert out.shape == (B, S, embed_dim)
    assert not mx.any(mx.isnan(out)).item()

def test_parallel_vs_cached():
    embed_dim = 64
    num_heads = 4
    num_modes = 16
    attn = AutomorphicSpectralAttention(embed_dim=embed_dim, num_heads=num_heads, num_modes=num_modes)
    
    B, S = 2, 5
    x = mx.random.normal((B, S, embed_dim))
    
    # Parallel prefill mode with cache initialization
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
    assert mx.allclose(out_par, out_seq, rtol=2e-3, atol=2e-3).item()
    
    # Verify final states in cache are equivalent
    assert mx.allclose(kv_cache_par["K_state"], kv_cache_seq["K_state"], rtol=2e-3, atol=2e-3).item()
    assert mx.allclose(kv_cache_par["denom_state"], kv_cache_seq["denom_state"], rtol=2e-3, atol=2e-3).item()
