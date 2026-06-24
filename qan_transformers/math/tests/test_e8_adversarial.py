import numpy as np
import pytest
from qan_transformers.math.e8_projection import (
    generate_e8_coordinates,
    project_e8_to_quasicrystal,
    verify_quasicrystalline_symmetries,
    generate_e8_adjacency_matrix
)

# =====================================================================
# ADVERSARIAL TEST GROUP 1: Symmetry Errors under Coordinate Scaling
# =====================================================================

def test_symmetry_false_negative_high_scaling():
    """
    Test that under high coordinate scaling (e.g., norm = 1e6),
    perfectly symmetric E8 projections correctly pass the symmetry verification
    thanks to the scale-invariant relative threshold.
    """
    # 1. Generate perfectly symmetric roots with a high norm scale
    high_norm = 1e6
    roots = generate_e8_coordinates(norm=high_norm)
    projected = project_e8_to_quasicrystal(roots, method="icosian")
    
    # 2. Run symmetry verification
    result = verify_quasicrystalline_symmetries(projected)
    
    print(f"\n[High Scaling] Norm: {high_norm:.1e}, Max Symmetry Error: {result['max_symmetry_error']:.2e}, Passes: {result['passes']}")
    
    assert result['passes'] is True, "Symmetry check should pass at high scale due to scale-invariant relative threshold"
    
    # Try a higher scale to confirm robust relative scale threshold
    super_norm = 1e9
    roots_super = generate_e8_coordinates(norm=super_norm)
    projected_super = project_e8_to_quasicrystal(roots_super, method="icosian")
    result_super = verify_quasicrystalline_symmetries(projected_super)
    print(f"[Super Scaling] Norm: {super_norm:.1e}, Max Symmetry Error: {result_super['max_symmetry_error']:.2e}, Passes: {result_super['passes']}")
    assert result_super['passes'] is True, "Symmetry check should pass at extremely high scale due to relative tolerance"

def test_symmetry_false_positive_small_scaling():
    """
    Test that under small coordinate scaling (e.g., norm = 1e-8),
    a symmetry-breaking perturbation is correctly detected and fails verification.
    """
    # 1. Generate roots with a small norm
    small_norm = 1e-8
    roots = generate_e8_coordinates(norm=small_norm)
    projected = project_e8_to_quasicrystal(roots, method="icosian")
    
    # 2. Add an extreme, symmetry-breaking perturbation
    perturbed = projected.copy()
    perturbed[0, 0] += 0.5 * small_norm  # 50% relative perturbation on a single coordinate
    
    # 3. Verify symmetry
    result = verify_quasicrystalline_symmetries(perturbed)
    
    print(f"\n[Small Scaling] Norm: {small_norm:.1e}, Perturbed Max Symmetry Error: {result['max_symmetry_error']:.2e}, Passes: {result['passes']}")
    
    assert result['passes'] is False, "Symmetry check should fail due to relative perturbation on small coordinates"

# =====================================================================
# ADVERSARIAL TEST GROUP 2: Double-Precision float64 Retention
# =====================================================================

def test_float64_precision_retention():
    """
    Verify that all operations retain float64 precision and do not implicitly
    downcast to float32 or lower, especially under large sequences.
    """
    # 1. Generate standard coordinates and verify type is float64
    roots = generate_e8_coordinates()
    assert roots.dtype == np.float64, f"E8 coordinates should be float64, got {roots.dtype}"
    
    # 2. Project coordinates and verify type remains float64
    projected = project_e8_to_quasicrystal(roots, method="icosian")
    assert projected.dtype == np.float64, f"Projected coordinates should be float64, got {projected.dtype}"
    
    projected_coxeter = project_e8_to_quasicrystal(roots, method="coxeter")
    assert projected_coxeter.dtype == np.float64, f"Coxeter projected coordinates should be float64, got {projected_coxeter.dtype}"
    
    # 3. Adjacency matrix generation
    adj = generate_e8_adjacency_matrix(projected)
    assert adj.dtype == np.float64, f"Adjacency matrix should be float64, got {adj.dtype}"
    
    # 4. Under a large sequence of inputs
    large_roots = np.tile(roots, (10, 1))  # 2400 coordinates (float64)
    assert large_roots.dtype == np.float64
    
    large_projected = project_e8_to_quasicrystal(large_roots, method="icosian")
    assert large_projected.dtype == np.float64, "Large projection downcast coordinates"
    
    large_adj = generate_e8_adjacency_matrix(large_projected)
    assert large_adj.dtype == np.float64, "Large adjacency matrix downcast coordinates"

# =====================================================================
# ADVERSARIAL TEST GROUP 3: KDTree Safety & Correctness under Duplicates
# =====================================================================

def test_kdtree_duplicates_empty_adjacency():
    """
    Verify that when input coordinates contain duplicate elements and trigger the KDTree path
    (len(coords) > 1000), the function correctly finds the same adjacencies as the non-KDTree path.
    """
    # 1. Generate standard E8 roots (norm = sqrt(2))
    roots = generate_e8_coordinates()
    
    # 2. Duplicate the roots 5 times to get 1200 coordinates, triggering the KDTree branch (>1000)
    large_coords_with_duplicates = np.tile(roots, (5, 1))
    assert len(large_coords_with_duplicates) == 1200
    
    # 3. Call generate_e8_adjacency_matrix (this will use the KDTree path)
    adj_kdtree = generate_e8_adjacency_matrix(large_coords_with_duplicates)
    
    # 4. Force the non-KDTree path by mocking len() or using a subset of the code
    # We can do this by executing the exact non-KDTree logic of the function on the same 1200 coordinates:
    dists = np.linalg.norm(large_coords_with_duplicates[:, None, :] - large_coords_with_duplicates[None, :, :], axis=2)
    non_zero_dists = dists[dists > 1e-5]
    assert len(non_zero_dists) > 0
    min_dist = np.min(non_zero_dists)
    threshold = min_dist + 1e-4
    adj_non_kdtree = ((dists > 1e-5) & (dists < threshold)).astype(np.float64)
    
    # 5. Show the correctness discrepancy
    kdtree_edges = np.sum(adj_kdtree)
    non_kdtree_edges = np.sum(adj_non_kdtree)
    
    print(f"\n[KDTree Duplicates Bug Resolved]")
    print(f"Total coordinates: {len(large_coords_with_duplicates)}")
    print(f"Edges found by KDTree branch: {kdtree_edges}")
    print(f"Edges found by correct non-KDTree formula: {non_kdtree_edges}")
    
    assert kdtree_edges > 0.0, "Expected KDTree path to return non-zero edges"
    assert np.allclose(adj_kdtree, adj_non_kdtree), "Expected KDTree path to match non-KDTree path exactly"

def test_kdtree_100_percent_duplicates_complexity():
    """
    Test the behavior under 100% duplicate inputs (1200 identical points).
    Verifies that the KDTree path does not crash but produces a valid all-zero matrix.
    Also highlights the potential O(N^2) complexity fallback in the Python loop.
    """
    # Create 1200 identical points
    single_point = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    duplicate_coords = np.tile(single_point, (1200, 1))
    
    # Call adjacency matrix generation (KDTree path)
    import time
    start_time = time.time()
    adj = generate_e8_adjacency_matrix(duplicate_coords)
    elapsed = time.time() - start_time
    
    print(f"\n[100% Duplicates] Elapsed time: {elapsed:.4f} seconds, Sum of Adjacency Matrix: {np.sum(adj)}")
    
    # Verify correctness: should be all zeros since all distances are 0 (< 1e-5)
    assert np.allclose(adj, 0.0)
    assert adj.shape == (1200, 1200)
