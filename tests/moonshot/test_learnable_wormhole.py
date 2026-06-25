"""Tests for learnable wormhole gate and integration with diffusion."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from ultrametric_ce.wormhole_gate import LearnableWormholeGate, WormholeRegularizer
from ultrametric_ce.diffusion import UltrametricDiffusion


class TestLearnableWormholeGate:
    """Tests for the LearnableWormholeGate MLP."""

    def test_output_shape_single_pair(self):
        """Gate output should be a scalar for a single pair of vectors."""
        gate = LearnableWormholeGate(dim=8)
        h_i = mx.random.normal((8,))
        h_j = mx.random.normal((8,))
        out = gate(h_i, h_j)
        mx.eval(out)
        assert out.shape == (), f"Expected scalar, got shape {out.shape}"

    def test_output_shape_batch(self):
        """Gate should handle batched inputs."""
        gate = LearnableWormholeGate(dim=8)
        h_i = mx.random.normal((5, 8))
        h_j = mx.random.normal((5, 8))
        out = gate(h_i, h_j)
        mx.eval(out)
        assert out.shape == (5,), f"Expected (5,), got shape {out.shape}"

    def test_output_range(self):
        """Gate output should be in [0, 1] (sigmoid output)."""
        gate = LearnableWormholeGate(dim=16)
        h_i = mx.random.normal((20, 16))
        h_j = mx.random.normal((20, 16))
        out = gate(h_i, h_j)
        mx.eval(out)
        vals = out.tolist()
        for v in vals:
            assert 0.0 <= v <= 1.0, f"Gate value {v} out of [0, 1]"


class TestComputeGateMatrix:
    """Tests for pairwise gate matrix computation."""

    def test_gate_matrix_shape(self):
        """Gate matrix should be (N, N) for N input vectors."""
        gate = LearnableWormholeGate(dim=8)
        vecs = mx.random.normal((4, 8))
        mat = gate.compute_gate_matrix(vecs)
        mx.eval(mat)
        assert mat.shape == (4, 4), f"Expected (4, 4), got {mat.shape}"

    def test_gate_matrix_range(self):
        """All gate matrix entries should be in [0, 1]."""
        gate = LearnableWormholeGate(dim=8)
        vecs = mx.random.normal((4, 8))
        mat = gate.compute_gate_matrix(vecs)
        mx.eval(mat)
        assert float(mx.min(mat)) >= 0.0 - 1e-6
        assert float(mx.max(mat)) <= 1.0 + 1e-6

    def test_gate_matrix_diagonal_consistent(self):
        """Diagonal entries (self-connections) should all produce valid [0,1] values."""
        gate = LearnableWormholeGate(dim=8)
        vecs = mx.random.normal((3, 8))
        mat = gate.compute_gate_matrix(vecs)
        mx.eval(mat)
        for i in range(3):
            val = float(mat[i, i])
            assert 0.0 <= val <= 1.0


class TestWormholeRegularizer:
    """Tests for WormholeRegularizer."""

    def test_non_negative_loss(self):
        """Regularization loss should be non-negative."""
        reg = WormholeRegularizer(sparsity_weight=0.01, symmetry_weight=0.1)
        mat = mx.random.uniform(shape=(5, 5))
        loss = reg(mat)
        mx.eval(loss)
        assert float(loss) >= 0.0

    def test_symmetric_matrix_has_zero_symmetry_loss(self):
        """A perfectly symmetric matrix should have zero symmetry loss."""
        reg = WormholeRegularizer(sparsity_weight=0.0, symmetry_weight=0.1)
        mat = mx.random.uniform(shape=(5, 5))
        sym_mat = (mat + mat.T) / 2.0
        loss = reg(sym_mat)
        mx.eval(loss)
        assert float(loss) < 1e-6

    def test_zero_matrix_has_zero_loss(self):
        """All-zero gate matrix should give zero loss."""
        reg = WormholeRegularizer(sparsity_weight=0.01, symmetry_weight=0.1)
        mat = mx.zeros((4, 4))
        loss = reg(mat)
        mx.eval(loss)
        assert float(loss) < 1e-8

    def test_asymmetric_matrix_penalized(self):
        """An asymmetric matrix should incur symmetry loss."""
        reg = WormholeRegularizer(sparsity_weight=0.0, symmetry_weight=0.1)
        mat = mx.array([[0.0, 1.0], [0.0, 0.0]])
        loss = reg(mat)
        mx.eval(loss)
        assert float(loss) > 0.0


class TestGateDifferentiability:
    """Test that the gate is differentiable through mx.grad."""

    def test_gate_gradient_flows(self):
        """Gradients should flow through the LearnableWormholeGate."""
        gate = LearnableWormholeGate(dim=4, hidden_dim=8)

        def loss_fn(h_i, h_j):
            out = gate(h_i, h_j)
            return mx.mean(out)

        h_i = mx.random.normal((3, 4))
        h_j = mx.random.normal((3, 4))

        loss_and_grad = nn.value_and_grad(gate, loss_fn)
        loss_val, grads = loss_and_grad(h_i, h_j)
        mx.eval(loss_val, grads)

        # Check that at least some gradient is nonzero
        has_nonzero = False
        def check_grads(g):
            nonlocal has_nonzero
            if isinstance(g, dict):
                for v in g.values():
                    check_grads(v)
            elif isinstance(g, list):
                for v in g:
                    check_grads(v)
            elif isinstance(g, mx.array):
                if float(mx.sum(mx.abs(g))) > 1e-12:
                    has_nonzero = True
        
        check_grads(grads)
        assert has_nonzero, "Expected nonzero gradients through the gate"


class TestDiffusionIntegration:
    """Integration tests: UltrametricDiffusion with wormhole_gate=True."""

    def _make_states(self, p=3, depth=2, dim=8):
        """Create sample states for a p=3, depth=2 tree."""
        states = {}
        states[(0, 0)] = mx.random.normal((dim,))
        for pref in range(p):
            states[(1, pref)] = mx.random.normal((dim,))
        for pref in range(p * p):
            states[(2, pref)] = mx.random.normal((dim,))
        # Evaluate all initial states
        mx.eval(*states.values())
        return states

    def test_forward_with_wormhole_gate(self):
        """Diffusion with wormhole_gate=True should produce valid outputs."""
        model = UltrametricDiffusion(
            p=3, depth=2, dim=8, num_layers=1,
            wormhole_gate=True, epsilon=0.1
        )
        states = self._make_states(p=3, depth=2, dim=8)
        result = model(states)
        mx.eval(*result.values())

        # Should return same keys
        assert set(result.keys()) == set(states.keys())
        # All values should be finite
        for key, val in result.items():
            assert val.shape == (8,), f"Bad shape for key {key}: {val.shape}"
            assert mx.all(mx.isfinite(val)), f"Non-finite value at key {key}"

    def test_forward_without_wormhole_gate(self):
        """Diffusion with wormhole_gate=False should still work (no floquet/gate)."""
        model = UltrametricDiffusion(
            p=3, depth=2, dim=8, num_layers=1,
            wormhole_gate=False, epsilon=0.1
        )
        states = self._make_states(p=3, depth=2, dim=8)
        result = model(states)
        mx.eval(*result.values())
        assert set(result.keys()) == set(states.keys())

    def test_output_changes_with_layers(self):
        """More layers should produce different (further mixed) output."""
        states = self._make_states(p=3, depth=2, dim=8)
        model1 = UltrametricDiffusion(p=3, depth=2, dim=8, num_layers=1, wormhole_gate=True)
        model2 = UltrametricDiffusion(p=3, depth=2, dim=8, num_layers=3, wormhole_gate=True)
        r1 = model1(states)
        r2 = model2(states)
        mx.eval(*r1.values(), *r2.values())
        # Outputs should differ with different layer counts
        any_diff = False
        for key in r1:
            if float(mx.sum(mx.abs(r1[key] - r2[key]))) > 1e-6:
                any_diff = True
                break
        assert any_diff, "Expected different outputs for 1 vs 3 layers"
