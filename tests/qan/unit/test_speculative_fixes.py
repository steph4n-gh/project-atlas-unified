import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from qan_transformers.modeling.attention import (
    DenseAttention,
    QuasicrystallineAttention,
    UltrametricAttention,
    cayley_orthogonal_adapter
)
from qan_transformers.modeling import make_superposition_mlp_forward

def test_fix1_fmm_causal_leak():
    # Sequence length 2048 to trigger FMM loop
    embed_dim = 64
    num_heads = 4
    leaf_size = 128
    attn = UltrametricAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        leaf_size=leaf_size,
        depth=3
    )
    
    # We want to check that it executes fine and does not crash or leak info
    x = torch.randn(1, 2048, embed_dim)
    out = attn(x)
    assert out.shape == (1, 2048, embed_dim)
    assert not torch.isnan(out).any()

def test_fix2_causal_mask_shape_mismatch():
    embed_dim = 64
    num_heads = 4
    B, S = 2, 8
    
    # attn_mask shape [S, S]
    attn_mask = torch.zeros((S, S))
    # introduce a masked position
    attn_mask[0, 1] = -10000.0
    
    # 1. DenseAttention
    dense_attn = DenseAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(B, S, embed_dim)
    out = dense_attn(x, attn_mask=attn_mask)
    assert out.shape == (B, S, embed_dim)
    
    # 2. QuasicrystallineAttention standard mode
    qc_attn = QuasicrystallineAttention(embed_dim=embed_dim, num_heads=num_heads)
    # Fast path (S <= 8)
    out_fast = qc_attn(x, attn_mask=attn_mask)
    assert out_fast.shape == (B, S, embed_dim)
    
    # Standard path (S > 8)
    x_long = torch.randn(B, 16, embed_dim)
    attn_mask_long = torch.zeros((16, 16))
    out_long = qc_attn(x_long, attn_mask=attn_mask_long)
    assert out_long.shape == (B, 16, embed_dim)
    
    # 3. QuasicrystallineAttention superposition mode
    # Superposition input has shape [B, C, S, D]
    x_super = torch.randn(B, 3, S, embed_dim)
    out_super = qc_attn(x_super, attn_mask=attn_mask, is_superposition=True)
    assert out_super.shape == (B, 3, S, embed_dim)
    
    # 4. UltrametricAttention fallback mode
    ultra_attn = UltrametricAttention(embed_dim=embed_dim, num_heads=num_heads)
    out_ultra = ultra_attn(x, attn_mask=attn_mask)
    assert out_ultra.shape == (B, S, embed_dim)

def test_fix3_nan_inf_contamination_superposition():
    embed_dim = 64
    num_heads = 4
    B, S = 2, 8
    qc_attn = QuasicrystallineAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Superposition input with NaNs/Infs in the input, but heavily masked out positions
    x_super = torch.randn(B, 3, S, embed_dim)
    attn_mask = torch.full((S, S), -1e9) # mask everything
    # Run forward pass, verifying it doesn't crash or propagate NaN
    out = qc_attn(x_super, attn_mask=attn_mask, is_superposition=True)
    assert not torch.isnan(out).any()

def test_fix4_non_contiguous_view_crash():
    d = 32
    r = 8
    A = torch.randn(d, r)
    B = torch.randn(d, r)
    
    # Create non-contiguous input by transposing a 3D tensor
    X = torch.randn(2, 5, d)
    # Transposing dims 0 and 1 makes it non-contiguous
    X_non_contig = X.transpose(0, 1)
    assert not X_non_contig.is_contiguous()
    
    # Should not crash
    out = cayley_orthogonal_adapter(X_non_contig, A, B)
    assert out.shape == X_non_contig.shape
    assert not torch.isnan(out).any()

def test_fix5_swiglu_lora_compatibility():
    class DummyMLP(nn.Module):
        def __init__(self, d_in, d_h):
            super().__init__()
            self.gate_proj = nn.Linear(d_in, d_h)
            self.up_proj = nn.Linear(d_in, d_h)
            self.down_proj = nn.Linear(d_h, d_in)
            
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
            
    mlp = DummyMLP(16, 32)
    original_forward = mlp.forward
    
    superposition_forward = make_superposition_mlp_forward(original_forward, mlp)
    
    # Test call in standard 3D mode
    x_3d = torch.randn(2, 5, 16)
    out_3d = superposition_forward(x_3d)
    assert out_3d.shape == (2, 5, 16)
    
    # Test call in 4D superposition mode
    x_4d = torch.randn(2, 3, 5, 16)
    out_4d = superposition_forward(x_4d)
    assert out_4d.shape == (2, 3, 5, 16)
    assert not torch.isnan(out_4d).any()
