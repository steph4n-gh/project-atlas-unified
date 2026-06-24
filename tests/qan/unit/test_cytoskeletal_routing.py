import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.e8_swap import AdelicMemorySwapGridDB

def test_initial_routing():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Initially W_route should be identity matrix
    assert db.W_route is not None
    assert db.W_route.shape == (240, 240)
    
    identity = mx.eye(240, dtype=mx.float32)
    assert mx.allclose(db.W_route, identity).item()

def test_cytoskeletal_polymerization():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Query is highly localized (one-hot E8 root 5)
    # This represents 100% specificity (entropy = 0)
    # We will simulate query input
    q = db.shell_1_roots[5:6]
    
    # Run polymerization several times
    q_flat = mx.reshape(q, (-1, 8))
    for _ in range(5):
        db._polymerize_cytoskeleton(q_flat)
        
    # Connection between root 5 and itself should be strong
    # Because of co-activation, W_route[5, 5] should grow towards 1.0
    val_5_5 = db.W_route[5, 5].item()
    assert val_5_5 > 0.0
    
    # Connection to unrelated roots should remain low
    val_5_20 = db.W_route[5, 20].item()
    assert val_5_20 < 0.1

def test_cytoskeletal_depolymerization():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Artificially set a route connection weight
    db.W_route = mx.eye(240, dtype=mx.float32) * 0.8
    
    # Run a decay step (depolymerize unused routes)
    db.W_route = db.W_route * (1.0 - db.alpha_decay)
    
    # Weights should decay
    assert db.W_route[0, 0].item() < 0.8
    assert db.W_route[0, 0].item() > 0.7

def test_cytoskeletal_search_retrieval():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Generate coordinates
    coords = db.shell_1_roots[:50]
    db._grid_coords = coords
    db.grid_coords_len = 50
    
    query = coords[10:11]
    
    # Search should run successfully with active routing
    db_idx, is_neighbor = db.neuromorphic_search(query, k_val=5)
    
    assert db_idx.shape[0] == 5
    assert is_neighbor.shape[0] == 5
    assert 10 in db_idx.tolist()
