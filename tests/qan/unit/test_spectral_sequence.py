import torch
import pytest
from qan_transformers.modeling.attention.base import QuasicrystallineAttention
from qan_transformers.modeling.spectral_sequence import SpectralSequenceAttention

def test_spectral_attention_forward():
    embed_dim = 16
    num_heads = 2
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_mode='spectral'
    )
    
    x = torch.randn(2, 4, embed_dim)
    
    # Forward pass
    out = attn(x)
    assert out.shape == (2, 4, embed_dim)
    assert not torch.isnan(out).any()
    
    # Check that pages were tracked
    assert hasattr(attn.spectral_attention, "last_computed_pages")
    assert attn.spectral_attention.last_computed_pages in [1, 2, 3]


def test_spectral_attention_early_exit():
    embed_dim = 16
    num_heads = 2
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_mode='spectral'
    )
    
    x = torch.randn(2, 4, embed_dim)
    
    # 1. Force immediate exit at page 2 by setting base_epsilon to very large value
    attn.spectral_attention.base_epsilon = 100.0
    _ = attn(x)
    assert attn.spectral_attention.last_computed_pages == 2
    
    # 2. Force evaluation of all 3 pages by setting base_epsilon to exactly 0.0
    attn.spectral_attention.base_epsilon = 0.0
    _ = attn(x)
    assert attn.spectral_attention.last_computed_pages == 3


def test_spectral_attention_gradients():
    embed_dim = 16
    num_heads = 2
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_mode='spectral'
    )
    
    x = torch.randn(2, 4, embed_dim, requires_grad=True)
    
    # Forward pass
    out = attn(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad != 0.0).any()
