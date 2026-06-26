import torch
import pytest
import numpy as np
from qan_transformers.modeling.attention.base import QuasicrystallineAttention
from qan_transformers.modeling.rg_flow import KVRenormalizationFlow

def test_rg_flow_differentiable():
    flow = KVRenormalizationFlow(eta=0.1, alpha=0.05, max_steps=3)
    
    # [B, H, N, D]
    K = torch.randn(2, 4, 8, 16, requires_grad=True)
    V = torch.randn(2, 4, 8, 16, requires_grad=True)
    indices = torch.arange(8).unsqueeze(0).expand(2, -1)
    scores = torch.randn(2, 4, 8)
    
    # Run compression to K_total = 4
    K_comp, V_comp, ind_comp, scr_comp = flow.compress(
        K, V, indices, scores, K_total=4, compression_level=0.1
    )
    
    assert K_comp.shape == (2, 4, 4, 16)
    assert V_comp.shape == (2, 4, 4, 16)
    assert ind_comp.shape == (2, 4)
    
    # Backward pass
    loss = K_comp.sum() + V_comp.sum()
    loss.backward()
    
    assert K.grad is not None
    assert V.grad is not None
    assert (K.grad != 0.0).any()
    assert (V.grad != 0.0).any()


def test_rg_flow_compression_attention():
    embed_dim = 16
    num_heads = 2
    
    # Instantiate QC attention with RG flow cache compression enabled
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=0.5, # target size is 50%
        cache_compression='rg_flow',
        compression_level=0.1
    )
    
    # Sequence length 24
    x = torch.randn(2, 24, embed_dim)
    kv_cache = {}
    
    # Forward pass (prefill)
    out, kv_cache = attn(x, kv_cache=kv_cache)
    assert out.shape == (2, 24, embed_dim)
    
    # Verify cache was compressed to ~50%
    # 24 * 0.5 = 12 tokens target
    assert "K" in kv_cache
    assert kv_cache["K"].shape[2] == 12
    assert kv_cache["indices"].shape[1] == 12


def test_semantic_consistency():
    embed_dim = 16
    num_heads = 2
    
    attn_comp = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=0.5,
        cache_compression='rg_flow',
        compression_level=0.1
    )
    
    attn_morse = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=0.5,
        cache_compression='morse'
    )
    
    # Share projections and weights
    attn_morse.q_proj = attn_comp.q_proj
    attn_morse.k_proj = attn_comp.k_proj
    attn_morse.v_proj = attn_comp.v_proj
    attn_morse.out_proj = attn_comp.out_proj
    attn_morse.e8_proj = attn_comp.e8_proj
    attn_morse.e8_proj_momentum = attn_comp.e8_proj_momentum
    attn_morse.symplectic_attention = attn_comp.symplectic_attention
    
    x = torch.randn(2, 16, embed_dim)
    
    kv_cache_comp = {}
    out_comp, kv_cache_comp = attn_comp(x, kv_cache=kv_cache_comp)
    
    kv_cache_morse = {}
    out_morse, kv_cache_morse = attn_morse(x, kv_cache=kv_cache_morse)
    
    # Outputs should be very close semantically
    cosine_sim = torch.cosine_similarity(out_comp.flatten(), out_morse.flatten(), dim=0)
    assert cosine_sim.item() > 0.90
