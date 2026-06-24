import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.attention import QuasicrystallineAttention

class MockCache:
    def __init__(self):
        self.offset = 0

def test_analytical_diagonalization():
    # Construct a random 2x2 symmetric matrix: [[a, b], [b, c]]
    # and verify that the analytical theta rotation yields the top eigenvector
    a = mx.array([3.0, 1.0, 5.0])
    b = mx.array([1.0, -0.5, 2.0])
    c = mx.array([2.0, 2.0, 1.0])
    
    # Calculate eigenvalues analytically to compare
    # lambda_1 = (a + c + sqrt((a-c)^2 + 4b^2)) / 2
    eig_val_top = 0.5 * (a + c + mx.sqrt((a - c)**2 + 4.0 * b**2))
    
    # Calculate theta and eigenvector psi = [alpha, beta]
    theta = 0.5 * mx.arctan2(2.0 * b, a - c)
    alpha = mx.cos(theta)
    beta = mx.sin(theta)
    
    # Matrix-vector multiplication rho @ psi
    # [a*alpha + b*beta, b*alpha + c*beta]
    rho_psi_x = a * alpha + b * beta
    rho_psi_y = b * alpha + c * beta
    
    # Check that rho @ psi is collinear with psi: rho @ psi = lambda * psi
    alpha_safe = mx.where(alpha == 0.0, 1e-9, alpha)
    beta_safe = mx.where(beta == 0.0, 1e-9, beta)
    
    lambda_x = rho_psi_x / alpha_safe
    lambda_y = rho_psi_y / beta_safe
    
    assert mx.allclose(lambda_x, eig_val_top, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(lambda_y, eig_val_top, rtol=1e-5, atol=1e-5).item()

def test_rg_flow_compression():
    embed_dim = 64
    num_heads = 4
    uv_window = 8
    rg_chunk_size = 4
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=1.0, # Keep all tokens in the cache
        rg_enabled=True,
        uv_window=uv_window,
        rg_chunk_size=rg_chunk_size
    )
    
    # Sequence length S = 20
    x = mx.random.normal((1, 20, embed_dim))
    
    # Run with MockCache
    cache = MockCache()
    attn(x, cache=cache, is_prefill=False)
    
    # Cache length after S=20:
    # S_old = 20 - uv_window = 12 tokens
    # Grouped into 6 blocks of 2, compressed to 6 tokens.
    # Total cache length should be S_recent (uv_window = 8) + 6 = 14 tokens
    assert cache.offset == 20
    assert attn.custom_kv_cache["K"].shape[2] == 14
    assert attn.custom_kv_cache["indices"].shape[1] == 14

def test_semantic_consistency():
    embed_dim = 64
    num_heads = 4
    uv_window = 8
    rg_chunk_size = 4
    
    attn_comp = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=1.0, # Keep all tokens in the cache
        rg_enabled=True,
        uv_window=uv_window,
        rg_chunk_size=rg_chunk_size
    )
    attn_uncomp = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=1.0, # Keep all tokens in the cache
        rg_enabled=False
    )
    
    # Share weights
    attn_uncomp.q_proj = attn_comp.q_proj
    attn_uncomp.k_proj = attn_comp.k_proj
    attn_uncomp.v_proj = attn_comp.v_proj
    attn_uncomp.o_proj = attn_comp.o_proj
    attn_uncomp.e8_proj = attn_comp.e8_proj
    
    x = mx.random.normal((1, 24, embed_dim))
    
    # Run uncompressed
    out_uncomp = attn_uncomp(x, is_prefill=False)
    
    # Run compressed
    out_comp = attn_comp(x, is_prefill=False)
    
    # Verify outputs are semantically aligned (high cosine similarity)
    cos_sim = mx.sum(out_comp * out_uncomp) / (mx.linalg.norm(out_comp) * mx.linalg.norm(out_uncomp) + 1e-9)
    assert cos_sim.item() > 0.95
