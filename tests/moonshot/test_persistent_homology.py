"""Tests for persistent homology training loss (Phase 4)."""
import pytest
import numpy as np
import mlx.core as mx

from qan_transformers.moonshot.persistent_homology import (
    RipsFilteredComplex,
    PersistenceWassersteinLoss,
    TopologicalRegularizer,
)


class TestRipsFilteredComplex:
    def test_distance_matrix_symmetry(self):
        rips = RipsFilteredComplex()
        skeleton = np.random.rand(8, 8).astype(np.float32)
        dist = rips.compute_distance_matrix(skeleton)
        np.testing.assert_allclose(dist, dist.T, atol=1e-6)
    
    def test_distance_matrix_diagonal_zero(self):
        rips = RipsFilteredComplex()
        skeleton = np.random.rand(8, 8).astype(np.float32)
        dist = rips.compute_distance_matrix(skeleton)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)
    
    def test_distance_high_attention_small_distance(self):
        rips = RipsFilteredComplex()
        skeleton = np.eye(4, dtype=np.float32)  # Identity = self-attention only
        dist = rips.compute_distance_matrix(skeleton)
        # Off-diagonal should be 1.0 (no connection = max distance)
        assert dist[0, 1] == pytest.approx(1.0, abs=0.01)
    
    def test_persistence_fully_connected(self):
        """A fully connected graph should have all components merging early."""
        rips = RipsFilteredComplex()
        skeleton = np.ones((4, 4), dtype=np.float32)
        diagram = rips.compute_persistence(skeleton)
        # All H_0 features should die at filtration 0 (distance 0)
        h0 = [(b, d) for dim, b, d in diagram if dim == 0]
        for birth, death in h0:
            assert death <= 0.01, f"H_0 feature should die early, got death={death}"
    
    def test_persistence_disconnected_graph(self):
        """A block-diagonal graph should show persistent disconnected components."""
        rips = RipsFilteredComplex()
        skeleton = np.zeros((6, 6), dtype=np.float32)
        skeleton[:3, :3] = 1.0  # Block 1
        skeleton[3:, 3:] = 1.0  # Block 2
        diagram = rips.compute_persistence(skeleton)
        # Should have at least one H_0 feature with high persistence
        # (a disconnected component that never merges)
        h0 = [(b, d) for dim, b, d in diagram if dim == 0]
        max_persistence = max(d - b for b, d in h0) if h0 else 0
        assert max_persistence >= 1.0, "Disconnected graph should have high H_0 persistence"
    
    def test_persistence_loop_detection(self):
        """A graph with cycles should produce H_1 features."""
        rips = RipsFilteredComplex()
        # Create a graph with a clear cycle: 0-1-2-0
        skeleton = np.zeros((4, 4), dtype=np.float32)
        skeleton[0, 1] = skeleton[1, 0] = 0.9
        skeleton[1, 2] = skeleton[2, 1] = 0.9
        skeleton[2, 0] = skeleton[0, 2] = 0.9
        skeleton[3, 0] = skeleton[0, 3] = 0.5  # Weak connection to node 3
        diagram = rips.compute_persistence(skeleton)
        h1 = [(b, d) for dim, b, d in diagram if dim == 1]
        assert len(h1) > 0, "Cyclic graph should produce H_1 features"
    
    def test_diagram_to_arrays_shapes(self):
        rips = RipsFilteredComplex()
        diagram = [(0, 0.0, 0.5), (0, 0.0, 1.0), (1, 0.3, 0.8)]
        h0, h1 = rips.diagram_to_arrays(diagram)
        assert h0.shape == (2, 2)
        assert h1.shape == (1, 2)
    
    def test_diagram_to_arrays_empty(self):
        rips = RipsFilteredComplex()
        h0, h1 = rips.diagram_to_arrays([])
        assert h0.shape == (0, 2)
        assert h1.shape == (0, 2)


class TestPersistenceWassersteinLoss:
    def test_identical_diagrams_zero_loss(self):
        loss_fn = PersistenceWassersteinLoss()
        skeleton = np.random.rand(8, 8).astype(np.float32)
        skeleton = 0.5 * (skeleton + skeleton.T)  # Symmetrize
        
        # Set reference from same skeleton
        loss_fn.set_reference_from_skeleton(skeleton)
        
        loss = loss_fn(mx.array(skeleton))
        assert float(loss) == pytest.approx(0.0, abs=0.01)
    
    def test_different_diagrams_positive_loss(self):
        loss_fn = PersistenceWassersteinLoss()
        
        # Reference: fully connected
        ref_skeleton = np.ones((6, 6), dtype=np.float32)
        loss_fn.set_reference_from_skeleton(ref_skeleton)
        
        # Current: disconnected
        curr_skeleton = np.zeros((6, 6), dtype=np.float32)
        curr_skeleton[:3, :3] = 1.0
        curr_skeleton[3:, 3:] = 1.0
        
        loss = loss_fn(mx.array(curr_skeleton))
        assert float(loss) > 0.0, "Different topologies should produce positive loss"
    
    def test_no_reference_penalizes_persistence(self):
        loss_fn = PersistenceWassersteinLoss()  # No reference set
        
        # Disconnected graph should get penalized
        skeleton = np.zeros((6, 6), dtype=np.float32)
        skeleton[:3, :3] = 1.0
        skeleton[3:, 3:] = 1.0
        
        loss = loss_fn(mx.array(skeleton))
        assert float(loss) > 0.0
    
    def test_output_is_mx_array(self):
        loss_fn = PersistenceWassersteinLoss()
        skeleton = np.random.rand(4, 4).astype(np.float32)
        loss = loss_fn(mx.array(skeleton))
        assert isinstance(loss, mx.array)


class TestTopologicalRegularizer:
    def test_regularizer_non_negative(self):
        reg = TopologicalRegularizer()
        skeleton = np.random.rand(8, 8).astype(np.float32)
        loss = reg(mx.array(skeleton))
        assert float(loss) >= 0.0
    
    def test_regularizer_penalizes_loops(self):
        reg = TopologicalRegularizer(betti_weight=1.0)
        
        # Graph with cycle
        skeleton_cycle = np.zeros((4, 4), dtype=np.float32)
        skeleton_cycle[0, 1] = skeleton_cycle[1, 0] = 0.9
        skeleton_cycle[1, 2] = skeleton_cycle[2, 1] = 0.9
        skeleton_cycle[2, 0] = skeleton_cycle[0, 2] = 0.9
        
        # Graph without cycle
        skeleton_tree = np.zeros((4, 4), dtype=np.float32)
        skeleton_tree[0, 1] = skeleton_tree[1, 0] = 0.9
        skeleton_tree[1, 2] = skeleton_tree[2, 1] = 0.9
        skeleton_tree[2, 3] = skeleton_tree[3, 2] = 0.9
        
        loss_cycle = float(reg(mx.array(skeleton_cycle)))
        loss_tree = float(reg(mx.array(skeleton_tree)))
        
        assert loss_cycle > loss_tree, "Cyclic graph should have higher regularization loss"
