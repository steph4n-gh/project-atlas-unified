"""Tests for cross-model wormhole bridge (Phase 5).

Note: Tests that require a live Gemini API key are marked with
@pytest.mark.skipif and will only run when GEMINI_API_KEY is set.
All other tests use mocked API responses.
"""
import pytest
import numpy as np
import mlx.core as mx
import os

from qan_transformers.moonshot.cross_model_bridge import (
    CayleyPrivacyAdapter,
    GeminiWormholeBridge,
    WormholeConfig,
)


class TestCayleyPrivacyAdapter:
    def test_rotate_preserves_norm(self):
        """Orthogonal rotation should preserve vector norms."""
        adapter = CayleyPrivacyAdapter(dim=64, rank=8)
        h = mx.random.normal((64,))
        h_rotated = adapter.rotate(h[None, :]).squeeze(0)
        
        norm_original = float(mx.linalg.norm(h))
        norm_rotated = float(mx.linalg.norm(h_rotated))
        assert norm_rotated == pytest.approx(norm_original, rel=0.01), \
            f"Norm changed: {norm_original:.4f} -> {norm_rotated:.4f}"
    
    def test_rotate_changes_values(self):
        """Rotation should actually change the hidden state values."""
        adapter = CayleyPrivacyAdapter(dim=64, rank=8)
        h = mx.random.normal((64,))
        h_rotated = adapter.rotate(h[None, :]).squeeze(0)
        
        # Should not be identical to input
        diff = float(mx.mean(mx.abs(h - h_rotated)))
        assert diff > 0.001, "Rotation should change values"
    
    def test_inverse_recovers_original(self):
        """rotate then inverse_rotate should recover the original vector."""
        adapter = CayleyPrivacyAdapter(dim=64, rank=8)
        h = mx.random.normal((1, 64))
        
        h_rotated = adapter.rotate(h)
        h_recovered = adapter.inverse_rotate(h_rotated)
        
        diff = float(mx.max(mx.abs(h - h_recovered)))
        assert diff < 0.01, f"Inverse rotation error too large: {diff:.6f}"
    
    def test_batch_rotation(self):
        """Should handle batched inputs."""
        adapter = CayleyPrivacyAdapter(dim=32, rank=4)
        h = mx.random.normal((5, 32))
        h_rotated = adapter.rotate(h)
        assert h_rotated.shape == (5, 32)
    
    def test_regenerate_changes_adapter(self):
        """Regenerating should produce different rotation parameters."""
        adapter = CayleyPrivacyAdapter(dim=32, rank=4)
        h = mx.random.normal((1, 32))
        
        h_rot1 = adapter.rotate(h)
        adapter.regenerate()
        h_rot2 = adapter.rotate(h)
        
        diff = float(mx.mean(mx.abs(h_rot1 - h_rot2)))
        assert diff > 0.001, "Regenerated adapter should produce different rotations"
    
    def test_preserves_pairwise_distances(self):
        """Orthogonal rotation should preserve pairwise distances."""
        adapter = CayleyPrivacyAdapter(dim=64, rank=8)
        h1 = mx.random.normal((1, 64))
        h2 = mx.random.normal((1, 64))
        
        # Original distance
        d_orig = float(mx.linalg.norm(h1 - h2))
        
        # Rotated distance
        h1_rot = adapter.rotate(h1)
        h2_rot = adapter.rotate(h2)
        d_rot = float(mx.linalg.norm(h1_rot - h2_rot))
        
        assert d_rot == pytest.approx(d_orig, rel=0.02), \
            f"Distance changed: {d_orig:.4f} -> {d_rot:.4f}"


class TestWormholeConfig:
    def test_default_values(self):
        config = WormholeConfig()
        assert config.model_name == "gemini-embedding-001"
        assert config.cayley_rank == 16
        assert config.soft_threshold < config.hard_threshold
    
    def test_custom_values(self):
        config = WormholeConfig(
            model_name="gemini-2.5-pro",
            cayley_rank=32,
            soft_threshold=0.5,
        )
        assert config.model_name == "gemini-2.5-pro"
        assert config.cayley_rank == 32


class TestGeminiWormholeBridge:
    def test_not_available_without_key(self):
        config = WormholeConfig(api_key=None)
        # Clear env var if set
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            bridge = GeminiWormholeBridge(config)
            assert not bridge.is_available
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
    
    def test_should_open_wormhole_in_range(self):
        config = WormholeConfig(
            api_key="test-key",
            soft_threshold=0.8,
            hard_threshold=1.5,
        )
        bridge = GeminiWormholeBridge(config)
        bridge._calibrated = True  # Mock calibration
        
        # Below soft threshold — don't open
        assert not bridge.should_open_wormhole(cfi=0.5)
        
        # In the uncertain zone — open
        assert bridge.should_open_wormhole(cfi=1.0)
        
        # Above hard threshold — don't open (firewall handles this)
        assert not bridge.should_open_wormhole(cfi=2.0)
    
    def test_should_open_wormhole_connectivity_dropping(self):
        config = WormholeConfig(
            api_key="test-key",
            soft_threshold=0.8,
            lambda_2_threshold=0.05,
        )
        bridge = GeminiWormholeBridge(config)
        bridge._calibrated = True
        
        # Low lambda_2 with moderate CFI — open
        assert bridge.should_open_wormhole(cfi=0.5, lambda_2=0.03)
    
    def test_query_returns_none_when_unavailable(self):
        config = WormholeConfig(api_key=None)
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            bridge = GeminiWormholeBridge(config)
            result = bridge.query(mx.random.normal((64,)))
            assert result is None
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
    
    def test_stats_tracking(self):
        config = WormholeConfig(api_key=None)
        old_key = os.environ.pop("GEMINI_API_KEY", None)
        try:
            bridge = GeminiWormholeBridge(config)
            stats = bridge.get_stats()
            assert "total_queries" in stats
            assert "successful_queries" in stats
            assert "timeouts" in stats
            assert "avg_latency_ms" in stats
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
    
    def test_regenerate_adapter(self):
        config = WormholeConfig(api_key="test-key")
        bridge = GeminiWormholeBridge(config)
        
        h = mx.random.normal((1, config.local_dim))
        rot1 = bridge._adapter.rotate(h)
        
        bridge.regenerate_adapter()
        rot2 = bridge._adapter.rotate(h)
        
        diff = float(mx.mean(mx.abs(rot1 - rot2)))
        assert diff > 0.001, "Regenerated adapter should produce different rotations"


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY environment variable"
)
class TestGeminiWormholeBridgeLive:
    """Live integration tests that require a real Gemini API key."""
    
    def test_embed_single(self):
        config = WormholeConfig()
        bridge = GeminiWormholeBridge(config)
        
        embedding = bridge._embed_single("Hello, world!")
        assert embedding is not None
        assert embedding.ndim == 1
        assert embedding.shape[0] > 0
    
    def test_batch_embed(self):
        config = WormholeConfig()
        bridge = GeminiWormholeBridge(config)
        
        texts = ["Hello", "World", "Test"]
        embeddings = bridge._batch_embed(texts)
        assert embeddings is not None
        assert embeddings.shape[0] == 3
    
    def test_calibrate_and_query(self):
        config = WormholeConfig(local_dim=64, cayley_rank=4)
        bridge = GeminiWormholeBridge(config)
        
        # Generate fake local hidden states
        local_states = mx.random.normal((5, 64))
        texts = [
            "The quick brown fox",
            "jumps over the lazy dog",
            "Machine learning is fascinating",
            "Neural networks process information",
            "Attention mechanisms are powerful",
        ]
        
        quality = bridge.calibrate(local_states, texts)
        assert 0.0 <= quality <= 1.0
        assert bridge.is_available
        
        # Query
        h = mx.random.normal((64,))
        result = bridge.query(h, context_text="Test query for wormhole")
        assert result is not None
        assert result.shape == (64,)
