import numpy as np
from qan_transformers.math.e8_projection import (
    generate_e8_coordinates,
    project_e8_to_quasicrystal,
    generate_e8_adjacency_matrix,
)

def test_e8_adjacency_matrix_properties():
    coords_8d = generate_e8_coordinates()
    adj = generate_e8_adjacency_matrix(coords_8d)
    
    # Verify shape
    assert adj.shape == (240, 240), f"Expected shape (240, 240), got {adj.shape}"
    
    # Verify it is symmetric
    assert np.all(adj == adj.T), "Adjacency matrix is not symmetric"
    
    # Verify no self-loops
    assert np.all(np.diag(adj) == 0), "Adjacency matrix contains self-loops"
    
    # Verify each node has exactly 56 active connections (nearest-neighbor kissing number for E8 roots)
    row_sums = np.sum(adj, axis=1)
    assert np.all(row_sums == 56), f"Expected 56 neighbors for each root, got row sums: {row_sums}"
    assert np.sum(adj) == 240 * 56, f"Expected {240*56} total connections, got {np.sum(adj)}"

def test_quasicrystal_adjacency_matrix_properties():
    coords_8d = generate_e8_coordinates()
    coords_3d = project_e8_to_quasicrystal(coords_8d, method="icosian")
    
    adj = generate_e8_adjacency_matrix(coords_3d)
    
    # Verify shape
    assert adj.shape == (240, 240), f"Expected shape (240, 240), got {adj.shape}"
    
    # Verify it is symmetric
    assert np.all(adj == adj.T), "Adjacency matrix in 3D is not symmetric"
    
    # Verify no self-loops
    assert np.all(np.diag(adj) == 0), "Adjacency matrix in 3D contains self-loops"
    
    # Verify active connections exist
    total_connections = np.sum(adj)
    assert total_connections > 0, "No active connections in the projected 3D graph"
    
    # Check that each node has a reasonable number of neighbors in 3D
    row_sums = np.sum(adj, axis=1)
    assert np.all(row_sums > 0), "Some nodes have zero neighbors in the projected 3D graph"
