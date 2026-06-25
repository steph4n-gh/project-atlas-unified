"""Tests for geometric draft filtering using E8 lattice distance."""
import pytest
import mlx.core as mx
import numpy as np

from qan_transformers.moonshot.geometric_filter import GeometricDraftFilter


# ---------------------------------------------------------------------------
# Helpers — build the standard icosian 8x3 projection matrix used by
# QuasicrystallineAttention (P_8_3 from attention.py lines 190-209)
# ---------------------------------------------------------------------------

def _make_projection_matrix() -> mx.array:
    """Reproduce the P_8_3 projection matrix from QuasicrystallineAttention."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    scale = 1.0 / np.sqrt(1.0 + phi ** 2)

    P_8_4 = np.zeros((8, 4))
    P_8_4[0, 0] = phi * scale
    P_8_4[4, 0] = 1.0 * scale
    P_8_4[1, 1] = phi * scale
    P_8_4[5, 1] = 1.0 * scale
    P_8_4[2, 2] = phi * scale
    P_8_4[6, 2] = 1.0 * scale
    P_8_4[3, 3] = phi * scale
    P_8_4[7, 3] = 1.0 * scale

    P_4_3 = np.zeros((4, 3))
    P_4_3[1, 0] = 1.0
    P_4_3[2, 1] = 1.0
    P_4_3[3, 2] = 1.0

    return mx.array(P_8_4 @ P_4_3, dtype=mx.float32)


# ---------------------------------------------------------------------------
# get_shell_index
# ---------------------------------------------------------------------------

class TestGetShellIndex:
    def test_origin_maps_to_shell_0(self):
        gf = GeometricDraftFilter()
        coord = mx.zeros(3)
        assert gf.get_shell_index(coord) == 0

    def test_radii_map_to_correct_shells(self):
        """Each standard shell radius should map to its own shell index."""
        gf = GeometricDraftFilter()
        for shell_idx, radius in enumerate(GeometricDraftFilter.STANDARD_SHELL_RADII):
            # Build a coordinate at exactly the expected radius along x-axis
            coord = mx.array([radius, 0.0, 0.0])
            assert gf.get_shell_index(coord) == shell_idx, (
                f"radius {radius} should map to shell {shell_idx}"
            )

    def test_intermediate_radius_snaps_to_nearest(self):
        gf = GeometricDraftFilter()
        # 0.75 is between shell 1 (0.5878) and shell 2 (0.8660)
        # Distance to shell 1: 0.75 - 0.5878 = 0.1622
        # Distance to shell 2: 0.8660 - 0.75  = 0.1160
        coord = mx.array([0.75, 0.0, 0.0])
        assert gf.get_shell_index(coord) == 2


# ---------------------------------------------------------------------------
# get_shell_radius
# ---------------------------------------------------------------------------

class TestGetShellRadius:
    def test_increasing_radii_for_higher_shells(self):
        gf = GeometricDraftFilter(r_base=0.6)
        radii = [gf.get_shell_radius(i) for i in range(5)]
        for i in range(len(radii) - 1):
            assert radii[i] < radii[i + 1], (
                f"shell {i} radius {radii[i]} should be < shell {i+1} radius {radii[i+1]}"
            )

    def test_shell_0_equals_r_base(self):
        gf = GeometricDraftFilter(r_base=0.5)
        assert gf.get_shell_radius(0) == pytest.approx(0.5)

    def test_shell_4_equals_r_base_times_1_8(self):
        gf = GeometricDraftFilter(r_base=0.5)
        # 0.5 * (1.0 + 0.2 * 4) = 0.5 * 1.8 = 0.9
        assert gf.get_shell_radius(4) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# update_trajectory
# ---------------------------------------------------------------------------

class TestUpdateTrajectory:
    def test_first_update_sets_coordinate(self):
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P)

        h = mx.ones(16)
        gf.update_trajectory(h)
        assert gf._trajectory_coord is not None
        assert gf._trajectory_coord.shape == (3,)

    def test_ema_tracks_moving_average(self):
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, ema_decay=0.5)

        h1 = mx.zeros(16)
        h2 = mx.ones(16)
        gf.update_trajectory(h1)
        coord_after_first = np.array(gf._trajectory_coord.tolist())

        gf.update_trajectory(h2)
        coord_after_second = np.array(gf._trajectory_coord.tolist())

        # With decay=0.5, second update = 0.5 * coord1 + 0.5 * project(h2)
        h2_8 = mx.ones(8)
        expected_new = np.array(mx.matmul(h2_8, P).tolist())
        expected = 0.5 * coord_after_first + 0.5 * expected_new
        np.testing.assert_allclose(coord_after_second, expected, atol=1e-5)

    def test_squeezes_batch_dim(self):
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P)

        h = mx.ones((1, 16))  # shape (1, D)
        gf.update_trajectory(h)
        assert gf._trajectory_coord.shape == (3,)

    def test_without_projection_uses_first_3_dims(self):
        gf = GeometricDraftFilter(projection_matrix=None)

        h = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gf.update_trajectory(h)
        np.testing.assert_allclose(
            gf._trajectory_coord.tolist(), [1.0, 2.0, 3.0], atol=1e-6
        )


# ---------------------------------------------------------------------------
# filter_candidates
# ---------------------------------------------------------------------------

class TestFilterCandidates:
    def test_nearby_tokens_accepted(self):
        """Candidates near the trajectory should be accepted."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, r_base=10.0)

        # Establish trajectory
        h = mx.ones(16)
        gf.update_trajectory(h)

        # Draft candidates very close to the trajectory hidden state
        drafts = mx.broadcast_to(mx.ones(16), (4, 16))
        mask, first_rej = gf.filter_candidates(drafts, 4)

        assert all(mask.tolist()), "All nearby candidates should be accepted"
        assert first_rej == 4

    def test_distant_tokens_rejected(self):
        """Candidates far from the trajectory should be rejected."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, r_base=0.001)

        # Establish trajectory at origin-ish
        h = mx.zeros(16)
        gf.update_trajectory(h)

        # Draft candidates far away
        drafts = mx.ones((4, 16)) * 100.0
        mask, first_rej = gf.filter_candidates(drafts, 4)

        assert not any(mask.tolist()), "All distant candidates should be rejected"
        assert first_rej == 0

    def test_first_rejection_index_correct(self):
        """first_rejection should point to the first rejected candidate."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, r_base=0.5)

        h = mx.zeros(16)
        gf.update_trajectory(h)

        # Build drafts: first 2 near (zeros), then 2 far (large values)
        near = mx.zeros((2, 16))
        far = mx.ones((2, 16)) * 1000.0
        drafts = mx.concatenate([near, far], axis=0)

        mask, first_rej = gf.filter_candidates(drafts, 4)
        mask_list = mask.tolist()

        # First two should be accepted, next two rejected
        assert mask_list[0] is True
        assert mask_list[1] is True
        assert first_rej == 2

    def test_mixed_accept_reject(self):
        """Verify mask is element-wise correct with mixed candidates."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, r_base=0.5)

        h = mx.zeros(16)
        gf.update_trajectory(h)

        # Interleave near and far candidates
        near = mx.zeros((1, 16))
        far = mx.ones((1, 16)) * 1000.0
        drafts = mx.concatenate([near, far, near], axis=0)

        mask, first_rej = gf.filter_candidates(drafts, 3)
        mask_list = mask.tolist()

        assert mask_list[0] is True
        assert mask_list[1] is False
        # first_rejection should be at index 1 (first rejected)
        assert first_rej == 1


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_trajectory(self):
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P)

        gf.update_trajectory(mx.ones(16))
        assert gf._trajectory_coord is not None

        gf.reset()
        assert gf._trajectory_coord is None
        assert gf._trajectory_shell == 2

    def test_after_reset_filter_accepts_everything(self):
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, r_base=0.001)

        gf.update_trajectory(mx.ones(16))
        gf.reset()

        drafts = mx.ones((3, 16)) * 9999.0
        mask, first_rej = gf.filter_candidates(drafts, 3)
        assert all(mask.tolist()), "After reset, should accept everything"
        assert first_rej == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_trajectory_accepts_everything(self):
        """Filter with no trajectory established should accept all candidates."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P)

        drafts = mx.ones((5, 16)) * 9999.0
        mask, first_rej = gf.filter_candidates(drafts, 5)
        assert all(mask.tolist())
        assert first_rej == 5

    def test_no_projection_matrix_accepts_everything(self):
        """Filter with no projection matrix should accept all candidates."""
        gf = GeometricDraftFilter(projection_matrix=None)
        gf.update_trajectory(mx.ones(16))  # Has trajectory but no projection

        drafts = mx.ones((5, 16)) * 9999.0
        mask, first_rej = gf.filter_candidates(drafts, 5)
        assert all(mask.tolist())
        assert first_rej == 5

    def test_small_hidden_state_padded(self):
        """Hidden states smaller than 8 dims should be zero-padded."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P)

        h = mx.array([1.0, 2.0, 3.0])  # Only 3 dims
        gf.update_trajectory(h)
        assert gf._trajectory_coord is not None
        assert gf._trajectory_coord.shape == (3,)

    def test_single_candidate(self):
        """Filter should work with a single candidate."""
        P = _make_projection_matrix()
        gf = GeometricDraftFilter(projection_matrix=P, r_base=10.0)

        gf.update_trajectory(mx.ones(16))
        drafts = mx.ones((1, 16))
        mask, first_rej = gf.filter_candidates(drafts, 1)
        assert mask.tolist() == [True]
        assert first_rej == 1

    def test_custom_shell_radii(self):
        """Custom shell radii should be used instead of defaults."""
        custom_radii = [0.0, 0.25, 0.5, 0.75, 1.0]
        gf = GeometricDraftFilter(shell_radii=custom_radii)
        assert gf.shell_radii == custom_radii

        # 0.3 is between 0.25 (shell 1) and 0.5 (shell 2)
        # Distance to shell 1: 0.05, Distance to shell 2: 0.20
        coord = mx.array([0.3, 0.0, 0.0])
        assert gf.get_shell_index(coord) == 1
