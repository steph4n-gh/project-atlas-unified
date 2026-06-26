import torch
import pytest
from qan_transformers.modeling.anyonic_braiding import (
    QuantumGroupRMatrix,
    BraidedMultiHeadAttention
)
from qan_transformers.modeling.attention.base import QuasicrystallineAttention

def test_rmatrix_initialization():
    r_mat = QuantumGroupRMatrix(d_head=8)
    assert r_mat.d_head == 8
    assert torch.allclose(r_mat.t, torch.tensor(0.5)) # t = sigmoid(0) = 0.5


def test_yang_baxter_braid_relation():
    # Braid relation (Yang-Baxter equation):
    # s1 * s2 * s1 = s2 * s1 * s2
    # Let's test this with 3 heads: h0, h1, h2
    d_head = 4
    r_mat = QuantumGroupRMatrix(d_head=d_head)
    
    # Set a custom t value to make it non-trivial
    r_mat.raw_t.data.copy_(torch.tensor(0.5))
    
    # Initial state
    h0 = torch.randn(1, 1, d_head)
    h1 = torch.randn(1, 1, d_head)
    h2 = torch.randn(1, 1, d_head)
    
    # Left hand side: s1 then s2 then s1
    # 1. s1 on (h0, h1)
    a, b = r_mat(h0, h1)
    # 2. s2 on (b, h2)
    b, c = r_mat(b, h2)
    # 3. s1 on (a, b)
    a, b = r_mat(a, b)
    lhs = (a, b, c)
    
    # Right hand side: s2 then s1 then s2
    # 1. s2 on (h1, h2)
    y, z = r_mat(h1, h2)
    # 2. s1 on (h0, y)
    x, y = r_mat(h0, y)
    # 3. s2 on (y, z)
    y, z = r_mat(y, z)
    rhs = (x, y, z)
    
    # Check that LHS and RHS are equal
    assert torch.allclose(lhs[0], rhs[0], rtol=1e-4, atol=1e-4)
    assert torch.allclose(lhs[1], rhs[1], rtol=1e-4, atol=1e-4)
    assert torch.allclose(lhs[2], rhs[2], rtol=1e-4, atol=1e-4)


def test_braided_attention_forward():
    dim = 16
    num_heads = 4
    
    braid_attn = BraidedMultiHeadAttention(embed_dim=dim, num_heads=num_heads)
    assert len(braid_attn.r_matrices) == num_heads - 1
    
    # [B, H, S, d_head] = [2, 4, 8, 4]
    head_outputs = torch.randn(2, 4, 8, 4)
    
    # Set to training to trigger tracking
    braid_attn.train()
    out = braid_attn(head_outputs)
    assert out.shape == head_outputs.shape
    assert not torch.isnan(out).any()
    
    # Check that diagnostic was logged
    assert len(braid_attn.tracker.history) == 1
    assert "trace" in braid_attn.tracker.history[0]


def test_integration_and_gradients():
    embed_dim = 16
    num_heads = 4
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_braiding=True
    )
    
    # Run forward pass
    x = torch.randn(2, 8, embed_dim)
    out = attn(x)
    assert out.shape == (2, 8, embed_dim)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    # Verify gradients flow to R-matrix raw_t parameters
    for r_mat in attn.braid_attention.r_matrices:
        assert r_mat.raw_t.grad is not None
        assert not torch.isnan(r_mat.raw_t.grad).any()
