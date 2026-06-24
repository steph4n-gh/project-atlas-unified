import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qan_transformers.modeling.attention import QuasicrystallineAttention
from qan_transformers.modeling import make_superposition_mlp_forward, make_superposition_sequential_mlp_forward

def test_taylor_softmax_correctness():
    # Setup dimension
    B, H, S, S_seq = 2, 4, 8, 8
    C = 3
    
    # Random reference scores and deviations
    A0 = torch.randn(B, H, S, S_seq)
    dA = torch.randn(B, C, H, S, S_seq) * 0.01  # Small deviation for Taylor validity
    
    # 1. Standard Softmax for each channel
    P0 = F.softmax(A0, dim=-1)
    
    # 2. Second-order Taylor Softmax
    mean_P0_dA = torch.sum(P0.unsqueeze(1) * dA, dim=-1, keepdim=True)
    dA_tilde = dA - mean_P0_dA
    mean_P0_dA_tilde_sq = torch.sum(P0.unsqueeze(1) * torch.square(dA_tilde), dim=-1, keepdim=True)
    dP = P0.unsqueeze(1) * dA_tilde + 0.5 * P0.unsqueeze(1) * (torch.square(dA_tilde) - mean_P0_dA_tilde_sq)
    
    P_taylor = P0.unsqueeze(1) + dP
    
    # 3. Exact channel-wise Softmax
    P_exact = F.softmax(A0.unsqueeze(1) + dA, dim=-1)
    
    # Check that Taylor approximation is close to exact softmax for small deviations
    assert torch.allclose(P_taylor, P_exact, atol=1e-4, rtol=1e-4)

def test_swiglu_linear_bypass():
    class MockFFN(nn.Module):
        def __init__(self, d_in, d_mlp):
            super().__init__()
            self.gate_proj = nn.Linear(d_in, d_mlp)
            self.up_proj = nn.Linear(d_in, d_mlp)
            self.down_proj = nn.Linear(d_mlp, d_in)
            
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
            
    B, C, S, D = 2, 4, 8, 16
    d_mlp = 32
    
    ffn = MockFFN(D, d_mlp)
    
    # Reference input and small deviation
    x0 = torch.randn(B, S, D)
    dx = torch.randn(B, C, S, D) * 0.001
    x = x0.unsqueeze(1) + dx
    
    # Channel-wise exact SwiGLU FFN output
    out_exact = []
    for c in range(C):
        out_exact.append(ffn(x[:, c]))
    out_exact = torch.stack(out_exact, dim=1)
    
    # Linearized SwiGLU FFN output
    patched_forward = make_superposition_mlp_forward(ffn.forward, ffn)
    out_patched = patched_forward(x)
    
    # For small dx, patched should be close to exact due to Taylor expansion accuracy
    assert torch.allclose(out_patched, out_exact, atol=1e-4, rtol=1e-4)

def test_superposition_dimensions():
    # Verify QuasicrystallineAttention supports shape [B, C, S, D]
    embed_dim = 32
    num_heads = 4
    sparse_ratio = 0.15
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=sparse_ratio
    )
    
    B, C, S, D = 2, 3, 5, embed_dim
    x = torch.randn(B, C, S, D)
    
    # Call forward pass in superposition mode
    out = attn(x, is_superposition=True)
    
    assert out.shape == (B, C, S, D)
    assert not torch.isnan(out).any()
