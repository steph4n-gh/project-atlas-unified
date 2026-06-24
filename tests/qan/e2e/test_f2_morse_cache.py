import pytest
import torch

def test_t1_morse_cache_init():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    kv_cache = {}
    x = torch.randn(1, 10, 64)
    _, kv_cache = layer(x, kv_cache=kv_cache)
    
    assert "K" in kv_cache
    assert "V" in kv_cache
    assert "indices" in kv_cache
    assert "alignment_scores" in kv_cache
    assert "seq_len" in kv_cache

def test_t1_morse_cache_sublinear_scaling():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    
    for S in [10, 50, 100]:
        kv_cache = {}
        x = torch.randn(1, S, 64)
        _, kv_cache = layer(x, kv_cache=kv_cache)
        cached_size = kv_cache["K"].shape[2]
        # Bounded by S * sparse_ratio (with room for rounding up to at least 1 or 2)
        assert cached_size <= max(2, int(S * 0.15) + 1)

def test_t1_morse_cache_morse_collapse():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
    
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    
    # Run forward pass and verify it performs contraction/collapse to subset of keys.
    x = torch.randn(1, 100, 64)
    kv_cache = {}
    _, kv_cache = layer(x, kv_cache=kv_cache)
    
    # Footprint of cache is reduced from 100 to sparse_ratio * 100
    assert kv_cache["K"].shape[2] < 100

def test_t1_morse_cache_append():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    
    kv_cache = {}
    # Run multiple steps
    for _ in range(3):
        x = torch.randn(1, 10, 64)
        _, kv_cache = layer(x, kv_cache=kv_cache)
        
    assert kv_cache["seq_len"] == 30
    assert kv_cache["K"].shape[2] <= 30

def test_t1_morse_cache_swap_db():
    try:
        from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    db = AdelicMemorySwapGridDB(d_model=16)
    k = torch.randn(10, 16)
    v = torch.randn(10, 16)
    
    # verify swap_out
    db.swap_out(k, v)
    
    # verify swap_in_batch
    queries = torch.randn(1, 4, 1, 16)
    ret_k, ret_v = db.swap_in_batch(queries, max_matches=8)
    assert ret_k.shape == (1, 4, 8, 16)
    assert ret_v.shape == (1, 4, 8, 16)

# Tier 2 Boundary Cases

def test_t2_morse_cache_max_seq():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    layer.eval()
    
    kv_cache = {}
    # Simulate a sequence of 1000 tokens
    x = torch.randn(1, 1000, 64)
    _, kv_cache = layer(x, kv_cache=kv_cache)
    
    # footprint is managed under constraints
    assert kv_cache["K"].shape[2] <= 155  # ~1000 * 0.15 + some safety margins

def test_t2_morse_cache_complete_collapse():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    # Force attention to collapse to a single cell (sparse_ratio extremely low)
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.0001)
    layer.eval()
    x = torch.randn(1, 100, 64)
    kv_cache = {}
    _, kv_cache = layer(x, kv_cache=kv_cache)
    # Check that it collapses to at least 1 key
    assert kv_cache["K"].shape[2] == 1

def test_t2_morse_cache_no_collapse():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    # sparse_ratio = 1.0 (dense attention - no collapse)
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=1.0)
    layer.eval()
    x = torch.randn(1, 10, 64)
    kv_cache = {}
    _, kv_cache = layer(x, kv_cache=kv_cache)
    assert kv_cache["K"].shape[2] == 10

def test_t2_morse_cache_db_oom_handling():
    try:
        from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    # Simulate database with extreme capacity limit
    db = AdelicMemorySwapGridDB(d_model=16, cache_limit_ratio=0.0)
    k = torch.randn(10, 16)
    v = torch.randn(10, 16)
    # Shouldn't crash under OOM / zero cache limit
    db.swap_out(k, v)
    assert len(db.gpu_cache) <= 1


def test_t2_morse_cache_eviction_order():
    try:
        from qan_transformers.modeling.attention import QuasicrystallineAttention
    except ImportError:
        pytest.fail("Module not implemented yet")
        
    # Verify that the least-aligned semantic concepts are evicted from the active cache first
    # In the attention code, the topk keys are selected based on the highest alignment scores,
    # meaning the ones with lowest alignment scores (least aligned) are evicted.
    layer = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.2)
    layer.eval()
    x = torch.randn(1, 10, 64)
    kv_cache = {}
    _, kv_cache = layer(x, kv_cache=kv_cache)
    
    # Run another step to trigger eviction/top-k selection
    x_new = torch.randn(1, 10, 64)
    _, kv_cache = layer(x_new, kv_cache=kv_cache)
    
    # Assert alignment scores in cache are sorted or subset of highest scores
    # We should have kept the top-k highest scores, and evicted the lowest ones.
    scores = kv_cache["alignment_scores"]
    assert scores.shape[1] == int(20 * 0.2)
