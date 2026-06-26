import torch
import torch.nn.functional as F
import numpy as np
import pytest
from qan_transformers.math.tropical import TropicalSemiring, TropicalAttentionAnalyzer, AdaptiveTropicalTemperature
from qan_transformers.modeling.attention.base import QuasicrystallineAttention

def test_tropical_semiring_axioms():
    ts = TropicalSemiring()
    
    a = torch.tensor([2.0, 5.0, 1.0])
    b = torch.tensor([3.0, 2.0, 4.0])
    c = torch.tensor([1.0, 6.0, 0.0])
    
    # ⊕ (addition) associativity: (a ⊕ b) ⊕ c == a ⊕ (b ⊕ c)
    lhs_add = ts.tropical_add(ts.tropical_add(a, b), c)
    rhs_add = ts.tropical_add(a, ts.tropical_add(b, c))
    assert torch.allclose(lhs_add, rhs_add)

    # ⊗ (multiplication) distributivity: a ⊗ (b ⊕ c) == (a ⊗ b) ⊕ (a ⊗ c)
    lhs_dist = ts.tropical_mul(a, ts.tropical_add(b, c))
    rhs_dist = ts.tropical_add(ts.tropical_mul(a, b), ts.tropical_mul(a, c))
    assert torch.allclose(lhs_dist, rhs_dist)

    # Tropical matrix multiplication
    A = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    B = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    C = ts.tropical_matmul(A, B)
    expected = torch.tensor([[4.0, 5.0], [3.0, 4.0]])
    assert torch.allclose(C, expected)


def test_softmax_convergence():
    # As T -> 0, softmax(S/T) should converge to argmax/one-hot
    S = torch.tensor([[1.0, 3.0, 2.0]])
    
    # T = 1.0
    w_1 = F.softmax(S / 1.0, dim=-1)
    # T = 0.0001
    w_eps = F.softmax(S / 0.0001, dim=-1)
    
    # Argmax one-hot
    w_hard = torch.zeros_like(S)
    w_hard[0, S.argmax(dim=-1)] = 1.0
    
    assert not torch.allclose(w_1, w_hard, atol=1e-2)
    assert torch.allclose(w_eps, w_hard, atol=1e-4)


def test_tropical_analyzer():
    analyzer = TropicalAttentionAnalyzer()
    
    # Create scores with a clear max and a second-max
    # Row 1: gap is 2.0, Row 2: gap is 0.05 (near boundary)
    S = torch.tensor([
        [1.0, 4.0, 2.0],
        [2.0, 3.0, 2.95]
    ])
    
    variety = analyzer.compute_tropical_variety(S)
    assert variety['n_boundary_rows'] == 1  # only row 2 is near boundary
    assert variety['boundary_fraction'] == 0.5
    assert np.isclose(variety['mean_gap'], 1.025)  # mean of 2.0 and 0.05


def test_adaptive_temperature_forward():
    # Scores shape [B, H, S, S_seq]
    scores = torch.randn(2, 4, 8, 8)
    
    adapter = AdaptiveTropicalTemperature(init_temp=0.78, target_gap=1.0)
    adapter.eval()  # dynamic inference mode
    
    scaled_scores = adapter(scores)
    assert scaled_scores.shape == scores.shape
    
    # Verify that the average routing gap of scaled scores is close to target_gap (1.0)
    top2 = torch.topk(scaled_scores, 2, dim=-1).values
    gaps = top2[..., 0] - top2[..., 1]
    mean_gap = gaps.mean().item()
    assert np.isclose(mean_gap, 1.0, atol=1e-3)


def test_quasicrystalline_attention_tropical():
    embed_dim = 16
    num_heads = 2
    
    # Instantiate QC attention in tropical temperature mode
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        temperature_mode='tropical'
    )
    
    x = torch.randn(2, 4, embed_dim)
    
    # Forward pass should run successfully and apply tropical temp scaling
    out = attn(x)
    assert out.shape == (2, 4, embed_dim)
    assert not torch.isnan(out).any()
