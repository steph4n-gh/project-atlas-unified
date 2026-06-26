import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math

from qan_transformers.mlx.derived_composition import DerivedAttentionComposition
from qan_transformers.mlx.conformal import ConformalAttention, ConformalPositionalEncoding, ModularDuality
from qan_transformers.mlx.symplectic import SymplecticAttention
from qan_transformers.mlx.anyonic_braiding import QuantumGroupRMatrix, BraidedMultiHeadAttention
from qan_transformers.mlx.attention import QuasicrystallineAttention


def test_derived_composition_mlx():
    comp = DerivedAttentionComposition(init_alpha=0.1)
    
    # Create rank-deficient matrices in numpy, convert to MLX
    u = np.random.randn(8, 1)
    v = np.random.randn(8, 1)
    A_curr = mx.array(np.matmul(u, v.T))[None, None] # [1, 1, 8, 8]
    A_curr = mx.broadcast_to(A_curr, (2, 4, 8, 8)) # [2, 4, 8, 8]
    
    u2 = np.random.randn(8, 1)
    v2 = np.random.randn(8, 1)
    A_prev = mx.array(np.matmul(u2, v2.T))[None, None] # [1, 1, 8, 8]
    A_prev = mx.broadcast_to(A_prev, (2, 4, 8, 8))
    
    # Composed without Ext1 (should have rank 1)
    A_comp_base = mx.matmul(A_curr, A_prev)
    
    # Composed with Ext1
    A_composed = comp(A_curr, A_prev)
    assert A_composed.shape == (2, 4, 8, 8)
    
    cpu_stream = mx.default_stream(mx.cpu)
    _, S_base, _ = mx.linalg.svd(A_comp_base[0, 0], stream=cpu_stream)
    _, S_composed, _ = mx.linalg.svd(A_composed[0, 0], stream=cpu_stream)
    
    base_rank = int(mx.sum(S_base > 1e-4).item())
    composed_rank = int(mx.sum(S_composed > 1e-4).item())
    
    assert base_rank <= 1
    assert composed_rank > base_rank
    assert comp.last_ext1_norm > 0.0


def test_conformal_positional_encoding_mlx():
    cpe = ConformalPositionalEncoding(max_positions=64)
    x = mx.random.normal((2, 8, 16))
    
    # Check shape preservation
    x_enc = cpe(x, offset=0)
    assert x_enc.shape == x.shape
    
    # Check that it changes the input
    assert not mx.array_equal(x_enc, x).item()


def test_modular_duality_mlx():
    md = ModularDuality(reference_length=16.0)
    dist = mx.array([2.0, 4.0, 8.0])
    
    dist_8 = md(dist, 8.0)
    dist_32 = md(dist, 32.0)
    
    assert dist_8.shape == dist.shape
    # Check inverse scaling relation
    assert mx.all(dist_8 > dist).item() or mx.all(dist_8 < dist).item()


def test_conformal_attention_mlx():
    dim = 16
    attn = ConformalAttention(dim=dim, max_positions=64, reference_length=16.0)
    
    Q = mx.random.normal((2, 2, 8, 8))
    K = mx.random.normal((2, 2, 8, 8))
    V = mx.random.normal((2, 2, 8, 8))
    
    res = attn(Q, K, V)
    assert "out" in res
    assert "attn_weights" in res
    assert res["out"].shape == (2, 2, 8, 8)
    
    # OPE fusion check
    x = mx.random.normal((2, 8, dim))
    fused_x = attn.fuse_ope(x)
    assert fused_x.shape == x.shape


def test_symplectic_attention_mlx():
    sym = SymplecticAttention(num_steps=4, dt=0.05, sigma=0.5)
    
    q = mx.random.normal((2, 10, 8))
    p = mx.random.normal((2, 10, 8))
    
    q_new, p_new = sym(q, p)
    assert q_new.shape == q.shape
    assert p_new.shape == p.shape
    
    # Hamiltonian energy conservation
    h_init = sym.compute_hamiltonian(q, p)
    h_final = sym.compute_hamiltonian(q_new, p_new)
    
    mean_diff = mx.mean(mx.abs(h_init - h_final)).item()
    # Symplectic integrators preserve energy without drift (small difference allowed)
    assert mean_diff < 0.2


def test_yang_baxter_braid_relation_mlx():
    d_head = 4
    r_mat = QuantumGroupRMatrix(d_head=d_head)
    
    # Set custom raw_t
    r_mat.raw_t = mx.array(0.5)
    
    # Initial state
    h0 = mx.random.normal((1, 1, d_head))
    h1 = mx.random.normal((1, 1, d_head))
    h2 = mx.random.normal((1, 1, d_head))
    
    # LHS: s1 then s2 then s1
    a, b = r_mat(h0, h1)
    b, c = r_mat(b, h2)
    a, b = r_mat(a, b)
    lhs = (a, b, c)
    
    # RHS: s2 then s1 then s2
    y, z = r_mat(h1, h2)
    x, y = r_mat(h0, y)
    y, z = r_mat(y, z)
    rhs = (x, y, z)
    
    assert mx.allclose(lhs[0], rhs[0], rtol=1e-4, atol=1e-4).item()
    assert mx.allclose(lhs[1], rhs[1], rtol=1e-4, atol=1e-4).item()
    assert mx.allclose(lhs[2], rhs[2], rtol=1e-4, atol=1e-4).item()


def test_braided_attention_mlx():
    dim = 16
    num_heads = 4
    
    braid_attn = BraidedMultiHeadAttention(embed_dim=dim, num_heads=num_heads)
    assert len(braid_attn.r_matrices) == num_heads - 1
    
    head_outputs = mx.random.normal((2, 4, 8, 4))
    
    braid_attn.train()
    out = braid_attn(head_outputs)
    assert out.shape == head_outputs.shape
    
    assert len(braid_attn.tracker.history) == 1
    assert "trace" in braid_attn.tracker.history[0]


def test_qan_mlx_integration():
    embed_dim = 16
    num_heads = 2
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=0.5,
        attention_mode='octonionic',
        temperature_mode='tropical',
        use_derived_composition=True,
        use_braiding=True
    )
    
    # Configure mock config with prev attn weights
    class MockConfig:
        pass
    attn.config = MockConfig()
    attn.layer_idx = 1
    attn.config.shared_prev_attn_weights = mx.softmax(mx.random.normal((2, num_heads, 8, 4)), axis=-1)
    
    x = mx.random.normal((2, 8, embed_dim))
    
    out = attn(x)
    assert out.shape == (2, 8, embed_dim)
    assert not mx.any(mx.isnan(out)).item()
    assert not mx.any(mx.isinf(out)).item()
