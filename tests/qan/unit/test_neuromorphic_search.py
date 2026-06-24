import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.e8_swap import AdelicMemorySwapGridDB

def test_adjacency_matrix_construction():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Verify that the transition matrix A_E8 was constructed
    assert db.A_E8 is not None
    assert db.A_E8.shape == (240, 240)
    
    # Calling update adjacency matrix should be a safe no-op
    db._update_adjacency_matrix()

def test_neuromorphic_search_single():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Generate coordinates as a subset of E8 roots to guarantee uniqueness of root mapping
    coords = db.shell_1_roots[:50]
    db._grid_coords = coords
    db.grid_coords_len = 50
    
    # Query is exactly one of the coordinates
    query = coords[5:6]
    
    db_idx, is_neighbor = db.neuromorphic_search(query, k_val=5)
    
    # The exact matching coordinate should be returned in top-k
    db_idx_list = db_idx.tolist()
    assert 5 in db_idx_list
    assert is_neighbor.shape[0] == 5

def test_neuromorphic_search_batch():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    coords = db.shell_1_roots[:30]
    db._grid_coords = coords
    db.grid_coords_len = 30
    
    # Query shape [B, H, S, 8] -> [1, 2, 3, 8]
    queries = mx.random.normal((1, 2, 3, 8))
    
    db_idx, is_neighbor = db.neuromorphic_search_batch(queries, k_val=8, B=1, H=2, S=3)
    
    # Check shapes
    assert db_idx.shape == (1, 2, 8)
    assert is_neighbor.shape == (1, 2, 8)
    assert mx.min(db_idx).item() >= 0
    assert mx.max(db_idx).item() < 30
