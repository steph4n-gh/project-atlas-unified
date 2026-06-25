"""Tests for Leech lattice integration into QuasicrystallineAttention.

Covers:
- Golay code (foundation) correctness
- Leech lattice Shell 1 vector generation
- 24D → 3D projection methods
- LeechShellRouter shell detection and routing
"""
import pytest
import numpy as np


class TestGolay:
    """Verify the Golay code foundation."""
    def test_weight_distribution(self):
        from qan_transformers.math.leech_lattice import _generate_golay_code
        cw = _generate_golay_code()
        assert len(cw) == 4096
        weights = np.sum(cw, axis=1)
        wd = {}
        for w in weights:
            wd[int(w)] = wd.get(int(w), 0) + 1
        assert wd == {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}

    def test_octad_count(self):
        from qan_transformers.math.leech_lattice import _generate_golay_code, _get_golay_octads
        cw = _generate_golay_code()
        octads = _get_golay_octads(cw)
        assert len(octads) == 759

    def test_minimum_distance(self):
        from qan_transformers.math.leech_lattice import _generate_golay_code
        cw = _generate_golay_code()
        # Sample check (full check is O(n²))
        min_d = 24
        for i in range(100):
            for j in range(i + 1, 100):
                d = np.sum(cw[i] != cw[j])
                if d > 0 and d < min_d:
                    min_d = d
        assert min_d >= 8


class TestLeechLattice:
    """Verify lattice generation."""
    def test_vector_count(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates
        coords = generate_leech_coordinates(shell=1)
        assert len(coords) > 90000  # At least Type 1 + Type 2

    def test_norm_uniformity(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates
        coords = generate_leech_coordinates(shell=1)
        norms_sq = np.sum(coords**2, axis=1)
        assert np.allclose(norms_sq, 4.0, atol=0.01)

    def test_dimension(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates
        coords = generate_leech_coordinates(shell=1)
        assert coords.shape[1] == 24

    def test_no_duplicates(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates
        coords = generate_leech_coordinates(shell=1)
        rounded = np.round(coords * 1000).astype(np.int64)
        unique = np.unique(rounded, axis=0)
        assert len(unique) == len(coords)


class TestProjection:
    """Verify 3D projection."""
    def test_golden_cascade_shape(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates, project_leech_to_3d
        coords = generate_leech_coordinates(shell=1)
        coords_3d, info = project_leech_to_3d(coords, method='golden_cascade')
        assert coords_3d.shape == (len(coords), 3)

    def test_direct_shape(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates, project_leech_to_3d
        coords = generate_leech_coordinates(shell=1)
        coords_3d, info = project_leech_to_3d(coords, method='direct')
        assert coords_3d.shape == (len(coords), 3)

    def test_shells_detected(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates, project_leech_to_3d
        coords = generate_leech_coordinates(shell=1)
        _, info = project_leech_to_3d(coords, method='direct')
        assert info['n_shells'] >= 3

    def test_quality_positive(self):
        from qan_transformers.math.leech_lattice import generate_leech_coordinates, project_leech_to_3d
        coords = generate_leech_coordinates(shell=1)
        _, info = project_leech_to_3d(coords, method='direct')
        assert info['quality'] > 0


class TestLeechShellRouter:
    """Verify shell routing."""
    def test_initialize(self):
        from qan_transformers.math.leech_lattice import LeechShellRouter
        router = LeechShellRouter(method='golden_cascade')
        info = router.initialize()
        assert router.get_shell_count() > 0
        assert router.get_address_capacity() > 90000

    def test_shell_index_in_range(self):
        from qan_transformers.math.leech_lattice import LeechShellRouter, generate_leech_coordinates, project_leech_to_3d
        router = LeechShellRouter(method='direct')
        router.initialize()
        coords = generate_leech_coordinates(shell=1)
        coords_3d, _ = project_leech_to_3d(coords, method='direct')
        for i in range(min(100, len(coords_3d))):
            shell = router.get_shell_index(coords_3d[i])
            assert 0 <= shell < router.get_shell_count()

    def test_e8_capacity_ratio(self):
        from qan_transformers.math.leech_lattice import LeechShellRouter
        router = LeechShellRouter()
        router.initialize()
        # Must be at least 100x E8's 240 addresses
        assert router.get_address_capacity() > 240 * 100
