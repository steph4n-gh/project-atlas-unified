import os
import tempfile
import time
import pytest
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ultrametric_ce.model import WeightManager, PagedEmbedding, UCEModel
from ultrametric_ce.tree import FiniteTree


def test_paging_correctness_and_performance():
    # 1. Setup virtual model file exceeding 20GB (21GB)
    dim = 4096
    dtype = mx.float16
    itemsize = 2  # float16 is 2 bytes
    embedding_size_bytes = dim * itemsize
    
    # 21 GB = 22,548,578,304 bytes
    logical_size = 21 * 1024 * 1024 * 1024
    num_balls = logical_size // embedding_size_bytes
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    
    try:
        # Create APFS sparse file (0 physical space, correct logical size)
        with open(temp_file_path, "wb") as f:
            f.truncate(logical_size)
            
        assert os.path.getsize(temp_file_path) == logical_size
        
        # 2. Write specific test values at target offsets
        idx_first = 0
        idx_mid = num_balls // 2
        idx_last = num_balls - 1
        
        val_first = np.random.randn(dim).astype(np.float16)
        val_mid = np.random.randn(dim).astype(np.float16)
        val_last = np.random.randn(dim).astype(np.float16)
        
        with open(temp_file_path, "r+b") as f:
            # First element
            f.seek(idx_first * embedding_size_bytes)
            f.write(val_first.tobytes())
            
            # Middle element
            f.seek(idx_mid * embedding_size_bytes)
            f.write(val_mid.tobytes())
            
            # Last element
            f.seek(idx_last * embedding_size_bytes)
            f.write(val_last.tobytes())
            
        # 3. Instantiate WeightManager and verify correctness of paged values
        # Test with mmap
        wm_mmap = WeightManager(
            weight_file_path=temp_file_path,
            num_balls=num_balls,
            dim=dim,
            dtype=dtype,
            max_vram_bytes=3 * 1024 * 1024 * 1024, # 3 GB VRAM ceiling (<4GB)
            use_mmap=True
        )
        
        # Verify correctness
        emb_first = np.array(wm_mmap.get_embedding(idx_first))
        emb_mid = np.array(wm_mmap.get_embedding(idx_mid))
        emb_last = np.array(wm_mmap.get_embedding(idx_last))
        
        assert np.allclose(emb_first, val_first), "First element mismatch"
        assert np.allclose(emb_mid, val_mid), "Middle element mismatch"
        assert np.allclose(emb_last, val_last), "Last element mismatch"
        
        # 4. Assert VRAM active parameter memory footprint is strictly below 4GB
        # Let's set a small VRAM ceiling to verify cache eviction works
        small_vram_bytes = 10 * 1024 * 1024 # 10 MB ceiling
        wm_evict = WeightManager(
            weight_file_path=temp_file_path,
            num_balls=num_balls,
            dim=dim,
            dtype=dtype,
            max_vram_bytes=small_vram_bytes,
            use_mmap=True
        )
        
        # Query enough indices to exceed 10MB
        # Each vector is 8KB, so 2000 indices is ~16MB
        query_indices = list(range(2000))
        wm_evict.prefetch(query_indices)
        
        # Verify footprint stays strictly below or equal to small_vram_bytes
        assert wm_evict.current_vram_bytes <= small_vram_bytes
        assert len(wm_evict.cache) * embedding_size_bytes == wm_evict.current_vram_bytes
        
        # 5. Benchmark dynamic branch loading latency (mmap)
        # Simulate active path loading: 150 random indices per step
        latencies = []
        for step in range(50):
            active_indices = np.random.randint(0, num_balls, size=150).tolist()
            
            start = time.perf_counter()
            wm_mmap.prefetch(active_indices)
            # Use mx.eval() to force actual read execution of lazy arrays
            mx.eval([wm_mmap.get_embedding(idx) for idx in active_indices])
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            # Assert footprint is strictly below 4GB at all times
            assert wm_mmap.current_vram_bytes < 4 * 1024 * 1024 * 1024
            
        avg_latency = np.mean(latencies)
        print(f"\nAverage branch-change load latency (mmap): {avg_latency:.3f} ms")
        assert avg_latency < 15.0, f"Mmap latency target violated: {avg_latency:.3f} ms >= 15 ms"
        
        wm_mmap.close()
        wm_evict.close()
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def test_paged_embedding_module_integration():
    # Verify PagedEmbedding behaves as a drop-in replacement for nn.Embedding
    dim = 16
    dtype = mx.float32
    num_balls = 100
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()
    
    try:
        # Create sparse file
        with open(temp_file_path, "wb") as f:
            f.truncate(num_balls * dim * 4)
            
        wm = WeightManager(
            weight_file_path=temp_file_path,
            num_balls=num_balls,
            dim=dim,
            dtype=dtype,
            max_vram_bytes=1 * 1024 * 1024,
            use_mmap=True
        )
        
        paged_emb = PagedEmbedding(wm, num_balls, dim)
        
        # Test scalar query
        scalar_idx = mx.array(10)
        res_scalar = paged_emb(scalar_idx)
        assert list(res_scalar.shape) == [dim]
        
        # Test batch query
        batch_idx = mx.array([[1, 2], [3, 4]])
        res_batch = paged_emb(batch_idx)
        assert list(res_batch.shape) == [2, 2, dim]
        
        # Test empty query
        empty_idx = mx.array([])
        res_empty = paged_emb(empty_idx)
        assert list(res_empty.shape) == [0, dim]
        
        wm.close()
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
