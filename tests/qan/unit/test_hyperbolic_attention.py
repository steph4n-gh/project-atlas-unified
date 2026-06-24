import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.attention import HyperbolicAttention

def test_initialization():
    embed_dim = 128
    num_heads = 8
    sparse_ratio = 0.15
    depth = 5
    leaf_size = 128
    
    attn = HyperbolicAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=sparse_ratio,
        depth=depth,
        leaf_size=leaf_size
    )
    
    assert attn.embed_dim == embed_dim
    assert attn.num_heads == num_heads
    assert attn.sparse_ratio == sparse_ratio
    assert attn.depth == depth
    assert attn.leaf_size == leaf_size
    assert isinstance(attn.gamma, mx.array)
    assert attn.gamma.item() == 1.0

def test_poincare_projection():
    embed_dim = 64
    num_heads = 4
    attn = HyperbolicAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Generate some random inputs (some with very large scale to test tanh saturation)
    x = mx.random.normal((2, 10, embed_dim)) * 100.0
    z = attn.poincare_project(x)
    
    assert z.shape == (2, 10, 2)
    
    # Check that all projected coordinates lie strictly inside the open unit disk (|z| < 1.0)
    norms = mx.sqrt(mx.sum(z**2, axis=-1))
    assert mx.all(norms < 1.0).item()
    assert mx.all(norms >= 0.0).item()

def test_hyperbolic_distance_properties():
    embed_dim = 64
    num_heads = 4
    attn = HyperbolicAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Coordinates in Poincaré disk
    u = mx.array([[[[0.1, 0.2]]]]) # [1, 1, 1, 2]
    v = mx.array([[[[0.5, -0.1]]]]) # [1, 1, 1, 2]
    
    # 1. Non-negativity
    d1 = attn.hyperbolic_distance(u, v)
    assert d1.item() >= 0.0
    
    # 2. Distance to itself is zero
    d_self = attn.hyperbolic_distance(u, u)
    assert abs(d_self.item()) < 1e-5
    
    # 3. Symmetry
    d2 = attn.hyperbolic_distance(v, u)
    assert abs(d1.item() - d2.item()) < 1e-5

def test_fallback_dense_gemm():
    embed_dim = 64
    num_heads = 4
    attn = HyperbolicAttention(embed_dim=embed_dim, num_heads=num_heads, leaf_size=128)
    
    # S = 100 < 2048
    x = mx.random.normal((2, 100, embed_dim))
    kv_cache = {}
    out, kv_cache = attn(x, kv_cache=kv_cache)
    
    assert out.shape == (2, 100, embed_dim)
    assert kv_cache["K"].shape[2] == 100
    assert kv_cache["V"].shape[2] == 100
    assert kv_cache["coords"].shape[1] == 100
    
    # Concatenation test
    x_new = mx.random.normal((2, 50, embed_dim))
    out_new, kv_cache = attn(x_new, kv_cache=kv_cache)
    
    assert out_new.shape == (2, 50, embed_dim)
    assert kv_cache["K"].shape[2] == 150
    assert kv_cache["V"].shape[2] == 150
    assert kv_cache["coords"].shape[1] == 150

def test_fmm_loop_and_scaling():
    embed_dim = 64
    num_heads = 4
    leaf_size = 128
    attn = HyperbolicAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        leaf_size=leaf_size,
        depth=3
    )
    
    # S = 2048 >= 2048 (triggers FMM loop)
    x = mx.random.normal((2, 2048, embed_dim))
    out = attn(x)
    
    assert out.shape == (2, 2048, embed_dim)
    assert not mx.any(mx.isnan(out)).item()
    
    # Cached run
    kv_cache = {}
    out_cached, kv_cache = attn(x, kv_cache=kv_cache)
    assert out_cached.shape == (2, 2048, embed_dim)
    assert kv_cache["K"].shape[2] == 2048
    assert kv_cache["coords"].shape[1] == 2048
