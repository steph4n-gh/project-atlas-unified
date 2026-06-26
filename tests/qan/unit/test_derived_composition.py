import torch
import pytest
from qan_transformers.modeling.attention.base import QuasicrystallineAttention
from qan_transformers.modeling.derived_composition import DerivedAttentionComposition

def test_derived_composition_forward():
    comp = DerivedAttentionComposition(init_alpha=0.05)
    
    # 2 batches, 4 heads, seq_len 8
    A_curr = torch.randn(2, 4, 8, 8)
    A_prev = torch.randn(2, 4, 8, 8)
    
    A_composed = comp(A_curr, A_prev)
    assert A_composed.shape == (2, 4, 8, 8)
    assert not torch.isnan(A_composed).any()
    
    assert comp.last_ext1_norm > 0.0


def test_derived_composition_differentiable():
    comp = DerivedAttentionComposition(init_alpha=0.05)
    
    A_curr = torch.randn(2, 2, 6, 6, requires_grad=True)
    A_prev = torch.randn(2, 2, 6, 6, requires_grad=True)
    
    A_composed = comp(A_curr, A_prev)
    loss = A_composed.sum()
    loss.backward()
    
    assert A_curr.grad is not None
    assert A_prev.grad is not None
    assert (A_curr.grad != 0.0).any()
    assert (A_prev.grad != 0.0).any()


def test_derived_composition_attention_integration():
    embed_dim = 16
    num_heads = 2
    
    class MockConfig:
        pass
        
    config = MockConfig()
    
    # Layer 0
    attn1 = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_derived_composition=True
    )
    attn1.config = config
    attn1.layer_idx = 0
    
    # Layer 1
    attn2 = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_derived_composition=True
    )
    attn2.config = config
    attn2.layer_idx = 1
    
    # Input
    x = torch.randn(2, 8, embed_dim)
    
    # Forward layer 1 (populates config)
    _ = attn1(x)
    assert hasattr(config, "shared_prev_attn_weights")
    assert config.shared_prev_attn_weights is not None
    
    # Forward layer 2 (retrieves and computes Ext1)
    _ = attn2(x)
    assert attn2.derived_composition.last_ext1_norm > 0.0


def test_rank_recovery():
    comp = DerivedAttentionComposition(init_alpha=0.1)
    
    # Create rank-deficient matrices
    # A_curr has rank 1
    u = torch.randn(8, 1)
    v = torch.randn(8, 1)
    A_curr = torch.matmul(u, v.t()).unsqueeze(0).unsqueeze(0) # [1, 1, 8, 8]
    
    # A_prev has rank 1
    u2 = torch.randn(8, 1)
    v2 = torch.randn(8, 1)
    A_prev = torch.matmul(u2, v2.t()).unsqueeze(0).unsqueeze(0) # [1, 1, 8, 8]
    
    # Composed without Ext1 (should have rank at most 1)
    A_comp_base = torch.matmul(A_curr, A_prev)
    
    # Composed with Ext1
    A_composed = comp(A_curr, A_prev)
    
    # Check effective rank (singular values)
    _, S_base, _ = torch.linalg.svd(A_comp_base[0, 0])
    _, S_composed, _ = torch.linalg.svd(A_composed[0, 0])
    
    # Base has only 1 non-zero singular value
    base_rank = (S_base > 1e-4).sum().item()
    assert base_rank <= 1
    
    # Composed with Ext1 should have recovered rank (more non-zero singular values)
    composed_rank = (S_composed > 1e-4).sum().item()
    assert composed_rank > base_rank
