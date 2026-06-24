import numpy as np
from qan_transformers.math.e8_projection import generate_e8_coordinates

def test_e8_coordinates_count():
    coords = generate_e8_coordinates()
    assert coords.shape == (240, 8), f"Expected shape (240, 8), got {coords.shape}"

def test_e8_coordinates_uniqueness():
    coords = generate_e8_coordinates()
    # Check that all rows are unique
    unique_rows = np.unique(coords, axis=0)
    assert len(unique_rows) == 240, f"Expected 240 unique coordinates, got {len(unique_rows)}"

def test_e8_coordinates_norm_squared():
    coords = generate_e8_coordinates()
    norms_squared = np.sum(coords**2, axis=1)
    assert np.allclose(norms_squared, 2.0), "All E8 roots must have Euclidean norm squared equal to exactly 2.0"

def test_e8_coordinates_types_and_parity():
    coords = generate_e8_coordinates()
    
    type1_count = 0
    type2_count = 0
    
    for row in coords:
        # Check Type 1: Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0)
        # All elements should be either 0 or 1 or -1
        is_type1 = np.all(np.isin(row, [-1.0, 0.0, 1.0]))
        if is_type1:
            # Must have exactly two non-zero entries of magnitude 1
            assert np.sum(np.abs(row) == 1.0) == 2, "Type 1 root must have exactly two non-zero entries of magnitude 1"
            assert np.sum(row == 0.0) == 6, "Type 1 root must have exactly six zero entries"
            type1_count += 1
        else:
            # Check Type 2: (+-0.5, ..., +-0.5) with even parity
            is_type2 = np.all(np.isin(row, [-0.5, 0.5]))
            assert is_type2, f"Coordinate is neither Type 1 nor Type 2: {row}"
            # Even parity condition: sum of coordinates is an even integer
            # Sum of coordinates divided by 2 must be an even integer? No, sum of entries divided by 0.5 must have even number of negatives.
            # In particular, sum(row) is sum of 8 elements of +-0.5.
            # Sum is (8 - 2*k)*0.5 = 4 - k, where k is the number of negative signs.
            # For 4 - k to be even, k must be even. So sum(row) must be an even integer (since 4 - k is an integer, and even since k is even).
            s = np.sum(row)
            assert np.isclose(s % 2.0, 0.0), f"Type 2 root sum {s} is not an even integer"
            type2_count += 1
            
    assert type1_count == 112, f"Expected 112 Type 1 roots, got {type1_count}"
    assert type2_count == 128, f"Expected 128 Type 2 roots, got {type2_count}"
