import os
import torch
import numpy as np
import pytest
import threading
from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB, CoWMemorySwapGridDB
from qan_transformers.math.rag import LatticeIndexer
from qan_transformers.firewall.cohomology import CohomologyFirewall

def test_mutex_safety():
    """
    Test that AdelicMemorySwapGridDB performs thread-safe concurrent operations
    without data corruption or race conditions.
    """
    db = AdelicMemorySwapGridDB(d_model=16, lock_path="/tmp/test_mutex.lock")
    db.clear()
    
    # Concurrent worker threads executing swap_out
    def worker(idx):
        keys = torch.randn(10, 16)
        values = torch.randn(10, 16)
        db.swap_out(keys, values)
        
    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Verify exact size matching number of written tokens
    assert db.cpu_k_target.shape[0] == 50
    assert db.grid_coords.shape[0] == 50

def test_cow_relocation():
    """
    Test that CoWMemorySwapGridDB isolates local writes, queries across combined parent/local
    contexts, and relocates conflicting E8 coordinates during merge to parent.
    """
    parent = AdelicMemorySwapGridDB(d_model=16, lock_path="/tmp/parent.lock")
    parent.clear()
    
    # Write reference coordinate to parent
    keys_p = torch.zeros(1, 16)
    parent.swap_out(keys_p, keys_p)
    parent_coord = parent.grid_coords[0].clone()
    
    # Branch Copy-on-Write database
    cow = CoWMemorySwapGridDB(parent, lock_path="/tmp/cow.lock")
    
    # Write conflicting coordinate to local CoW DB
    keys_c = torch.zeros(1, 16)
    cow.swap_out(keys_c, keys_c)
    
    # Query should return matches from both parent and local contexts
    q_res_k, _ = cow.swap_in(torch.zeros(1, 16))
    assert q_res_k.shape[0] == 2
    
    # Merge back to parent
    cow.merge_to_parent()
    
    # Relocation should map the conflict to the nearest unoccupied E8 shell 1 neighbor (distance squared = 2.0)
    assert parent.grid_coords.shape[0] == 2
    merged_coord = parent.grid_coords[1]
    
    diff = merged_coord - parent_coord
    dist2 = torch.sum(diff ** 2).item()
    assert abs(dist2 - 2.0) < 1e-4

def test_rag_cli_indexing(tmp_path):
    """
    Test that LatticeIndexer chunking and embedding works correctly over directory contents.
    """
    dir_path = tmp_path / "rag_test"
    dir_path.mkdir()
    file_path = dir_path / "sample.txt"
    file_path.write_text("Hello, this is a test for Lattice RAG indexer in Project Atlas. It chunks files and projects them.")
    
    indexer = LatticeIndexer(d_model=16)
    indexer.index_directory(str(dir_path))
    
    assert indexer.db.grid_coords is not None
    assert indexer.db.grid_coords.shape[0] > 0

def test_bisection_rollbacks():
    """
    Test that CohomologyFirewall graph Laplacian and Fiedler vector bisection correctly
    identifies the boundary of the topological split under low algebraic connectivity.
    """
    # Create firewall with a low algebraic connectivity threshold (tau)
    firewall = CohomologyFirewall(threshold=1.5, tau=0.2)
    
    # Construct a highly disconnected attention skeleton over summits [10, 20, 30, 40]
    attn = torch.zeros(1, 1, 45, 45)
    
    attn[0, 0, 10, 10] = 1.0
    attn[0, 0, 10, 20] = 0.9
    attn[0, 0, 20, 10] = 0.9
    attn[0, 0, 20, 20] = 1.0
    
    attn[0, 0, 30, 30] = 1.0
    attn[0, 0, 30, 40] = 0.9
    attn[0, 0, 40, 30] = 0.9
    attn[0, 0, 40, 40] = 1.0
    
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
    
    is_fractured, cfi, alt_idx = firewall.check_obstruction(attn)
    
    assert is_fractured
    boundary = firewall.split_boundary
    assert boundary in [10, 20, 30, 40]

def test_perplexity_canary_dense_fallback():
    """
    Test that the rolling perplexity canary fallback mechanism falls back to
    dense attention when rolling perplexity exceeds 2x the calibration baseline.
    """
    from qan_transformers.modeling.attention import QuasicrystallineAttention
    
    # Create QuasicrystallineAttention layer
    attn = QuasicrystallineAttention(embed_dim=64, num_heads=4, sparse_ratio=0.15)
    
    # Set calibration baseline to 10.0
    attn.calibration_baseline = 10.0
    # Pre-populate with baseline perplexities (normal regime)
    attn.ppl_canary_window = [10.0] * 10
    
    x = torch.randn(1, 10, 64)
    
    # First forward pass should use sparse attention, and populate "K" and "V" in kv_cache
    kv_cache_sparse = {}
    out_sparse, _ = attn(x, kv_cache=kv_cache_sparse)
    assert out_sparse.shape == (1, 10, 64)
    assert "K" in kv_cache_sparse
    assert "K_dense" not in kv_cache_sparse
    
    # Now set the rolling window to values exceeding 2x the baseline (20.0)
    attn.ppl_canary_window = [30.0] * 10
    
    # Second forward pass should fall back to dense attention, populating "K_dense" and "V_dense"
    kv_cache_dense = {}
    out_dense, _ = attn(x, kv_cache=kv_cache_dense)
    assert out_dense.shape == (1, 10, 64)
    assert "K_dense" in kv_cache_dense
    assert "V_dense" in kv_cache_dense

def test_rag_query():
    """
    Test that LatticeIndexer.query retrieves the correct indexed chunk.
    """
    indexer = LatticeIndexer(d_model=16)
    
    # We add two distinct text chunks
    content1 = "The quick brown fox jumps over the lazy dog."
    content2 = "Quantum computing relies on qubits to perform computations."
    
    indexer.chunks.append({"file": "fox.txt", "text": content1})
    indexer.chunks.append({"file": "quantum.txt", "text": content2})
    
    # Manually index them
    vec1 = indexer.embed_chunk(content1)
    indexer.db.swap_out(vec1.unsqueeze(0), vec1.unsqueeze(0))
    
    vec2 = indexer.embed_chunk(content2)
    indexer.db.swap_out(vec2.unsqueeze(0), vec2.unsqueeze(0))
    
    # Query matching content1
    matches = indexer.query("brown fox jumps")
    assert len(matches) > 0
    # The closest matching chunk should be content1
    found_text = [m["text"] for m in matches]
    assert content1 in found_text
    
    # Query matching content2
    matches2 = indexer.query("quantum computing qubit")
    assert len(matches2) > 0
    found_text2 = [m["text"] for m in matches2]
    assert content2 in found_text2


