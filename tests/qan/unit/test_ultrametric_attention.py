import pytest
import torch
import torch.nn as nn
import numpy as np
from qan_transformers.modeling.attention import UltrametricAttention

def test_initialization():
    embed_dim = 128
    num_heads = 8
    sparse_ratio = 0.15
    depth = 5
    leaf_size = 128
    
    attn = UltrametricAttention(
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
    assert isinstance(attn.gamma, nn.Parameter)
    assert attn.gamma.item() == 1.0

def test_digit_extraction_and_morton_packing():
    # Test poly-adelic digit extraction and Morton packing
    embed_dim = 64
    num_heads = 4
    attn = UltrametricAttention(embed_dim=embed_dim, num_heads=num_heads, depth=3)
    
    # We want to manually test the coordinate projection and Morton packing logic.
    x = torch.randn(2, 8, embed_dim) # [B, S, D]
    kv_cache = {}
    out, kv_cache = attn(x, kv_cache=kv_cache)
    
    assert "morton_codes" in kv_cache
    morton_codes = kv_cache["morton_codes"]
    assert morton_codes.shape == (2, 8)
    assert morton_codes.dtype == torch.long
    # Morton codes must be non-negative and less than 30**depth
    assert (morton_codes >= 0).all()
    assert (morton_codes < 30**3).all()

def test_fallback_dense_gemm():
    # S_total < 2048: fallback to standard scaled dot-product attention
    embed_dim = 64
    num_heads = 4
    attn = UltrametricAttention(embed_dim=embed_dim, num_heads=num_heads, leaf_size=128)
    
    # S = 100 < 2048
    x = torch.randn(2, 100, embed_dim)
    kv_cache = {}
    out, kv_cache = attn(x, kv_cache=kv_cache)
    
    assert out.shape == (2, 100, embed_dim)
    assert kv_cache["K"].shape[2] == 100
    assert kv_cache["V"].shape[2] == 100
    
    # Run again to test kv cache concatenation
    x_new = torch.randn(2, 50, embed_dim)
    out_new, kv_cache = attn(x_new, kv_cache=kv_cache)
    
    assert out_new.shape == (2, 50, embed_dim)
    assert kv_cache["K"].shape[2] == 150
    assert kv_cache["V"].shape[2] == 150

def test_fmm_loop_and_scaling():
    # Test FMM loop correctness and scaling behavior for S >= 2048
    embed_dim = 64
    num_heads = 4
    leaf_size = 128
    attn = UltrametricAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        leaf_size=leaf_size,
        depth=3
    )
    
    # S = 2048 >= 2048 (triggers FMM loop)
    x = torch.randn(2, 2048, embed_dim)
    out = attn(x)
    
    assert out.shape == (2, 2048, embed_dim)
    assert not torch.isnan(out).any()
    
    # Let's run with kv_cache to verify FMM handles it too
    kv_cache = {}
    out_cached, kv_cache = attn(x, kv_cache=kv_cache)
    assert out_cached.shape == (2, 2048, embed_dim)
    assert kv_cache["K"].shape[2] == 2048
    
    # Test that the output is close in norm or at least doesn't blow up
    # We can also verify autograd works through the FMM loop
    loss = out.sum()
    loss.backward()
    
    # Check that projections got gradients
    assert attn.q_proj.weight.grad is not None
    assert attn.coordinate_proj.weight.grad is not None
    assert attn.gamma.grad is not None

def test_initialization_mlx():
    embed_dim = 128
    num_heads = 8
    sparse_ratio = 0.15
    depth = 5
    leaf_size = 128
    
    import mlx.core as mx
    from qan_transformers.mlx.attention import UltrametricAttention as UltrametricAttentionMLX
    
    attn = UltrametricAttentionMLX(
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

def test_digit_extraction_and_morton_packing_mlx():
    embed_dim = 64
    num_heads = 4
    
    import mlx.core as mx
    from qan_transformers.mlx.attention import UltrametricAttention as UltrametricAttentionMLX
    
    attn = UltrametricAttentionMLX(embed_dim=embed_dim, num_heads=num_heads, depth=3)
    
    x = mx.random.normal((2, 8, embed_dim))
    kv_cache = {}
    out, kv_cache = attn(x, kv_cache=kv_cache)
    
    assert "morton_codes" in kv_cache
    morton_codes = kv_cache["morton_codes"]
    assert morton_codes.shape == (2, 8)
    assert morton_codes.dtype == mx.int64
    assert mx.all(morton_codes >= 0).item()
    assert mx.all(morton_codes < 30**3).item()

def test_fallback_dense_gemm_mlx():
    embed_dim = 64
    num_heads = 4
    
    import mlx.core as mx
    from qan_transformers.mlx.attention import UltrametricAttention as UltrametricAttentionMLX
    
    attn = UltrametricAttentionMLX(embed_dim=embed_dim, num_heads=num_heads, leaf_size=128)
    
    x = mx.random.normal((2, 100, embed_dim))
    kv_cache = {}
    out, kv_cache = attn(x, kv_cache=kv_cache)
    
    assert out.shape == (2, 100, embed_dim)
    assert kv_cache["K"].shape[2] == 100
    assert kv_cache["V"].shape[2] == 100
    
    x_new = mx.random.normal((2, 50, embed_dim))
    out_new, kv_cache = attn(x_new, kv_cache=kv_cache)
    
    assert out_new.shape == (2, 50, embed_dim)
    assert kv_cache["K"].shape[2] == 150
    assert kv_cache["V"].shape[2] == 150

def test_fmm_loop_and_scaling_mlx():
    embed_dim = 64
    num_heads = 4
    leaf_size = 128
    
    import mlx.core as mx
    from qan_transformers.mlx.attention import UltrametricAttention as UltrametricAttentionMLX
    
    attn = UltrametricAttentionMLX(
        embed_dim=embed_dim,
        num_heads=num_heads,
        leaf_size=leaf_size,
        depth=3
    )
    
    x = mx.random.normal((2, 2048, embed_dim))
    out = attn(x)
    
    assert out.shape == (2, 2048, embed_dim)
    assert not mx.any(mx.isnan(out)).item()
    
    kv_cache = {}
    out_cached, kv_cache = attn(x, kv_cache=kv_cache)
    assert out_cached.shape == (2, 2048, embed_dim)
    assert kv_cache["K"].shape[2] == 2048
