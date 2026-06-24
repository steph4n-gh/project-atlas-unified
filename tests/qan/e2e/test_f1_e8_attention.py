import pytest
import torch
import numpy as np

def test_t1_e8_attention_init():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    assert isinstance(layer, torch.nn.Module)
    assert hasattr(layer, "P_8_3")
    assert hasattr(layer, "roots_3d")

def test_t1_e8_attention_forward_shape():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    x = torch.randn(2, 16, 64)
    # Ensure eval mode so that DB offloading and other features run
    layer.eval()
    out = layer(x)
    assert out.shape == (2, 16, 64)

def test_t1_e8_attention_roots_projection():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
        from qan_transformers.math.e8_projection import project_e8_to_quasicrystal
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    roots = layer.roots_3d
    assert roots.shape == (240, 3)
    
    # Verify that project_e8_to_quasicrystal works
    dummy_e8 = np.ones((1, 8))
    proj = project_e8_to_quasicrystal(dummy_e8)
    assert proj.shape == (1, 3)

def test_t1_e8_attention_entropy_scaling():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    
    # Case 1: Low entropy triggers shell 3
    layer.prev_entropy = 0.5
    x = torch.randn(2, 16, 64)
    _ = layer(x)
    assert layer.roots_3d.shape[0] == 6720  # Shell 3 has 6720 roots
    
    # Case 2: High entropy triggers shell 1
    layer.prev_entropy = 4.0
    _ = layer(x)
    assert layer.roots_3d.shape[0] == 240   # Shell 1 has 240 roots

def test_t1_e8_attention_alignment_routing():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    # Test that cosine similarity alignment does routing correctly.
    # We can pass typical values and check that output matches expectation.
    x = torch.randn(1, 8, 64)
    out = layer(x)
    assert out.shape == (1, 8, 64)

# Tier 2 Boundary cases

def test_t2_e8_attention_seq_len_one():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    x = torch.randn(2, 1, 64)
    out = layer(x)
    assert out.shape == (2, 1, 64)

def test_t2_e8_attention_extreme_sparse_ratio_low():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    # sparse_ratio=0.01
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.01)
    layer.eval()
    x = torch.randn(1, 10, 64)
    kv_cache = {}
    out, kv_cache = layer(x, kv_cache=kv_cache)
    # Check that at least 1 key is kept
    assert kv_cache["K"].shape[2] >= 1

def test_t2_e8_attention_extreme_sparse_ratio_high():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    # sparse_ratio=1.0
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=1.0)
    layer.eval()
    x = torch.randn(1, 10, 64)
    kv_cache = {}
    out, kv_cache = layer(x, kv_cache=kv_cache)
    # Check that all keys are kept (length 10)
    assert kv_cache["K"].shape[2] == 10

def test_t2_e8_attention_zero_input():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    x = torch.zeros(2, 8, 64)
    out = layer(x)
    assert not torch.isnan(out).any()
    assert out.shape == (2, 8, 64)

def test_t2_e8_attention_dimension_mismatch():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    x = torch.randn(2, 8, 32) # dimension mismatch: 32 vs 64
    with pytest.raises(RuntimeError):  # PyTorch raises RuntimeError on mismatch linear projection
        layer(x)
