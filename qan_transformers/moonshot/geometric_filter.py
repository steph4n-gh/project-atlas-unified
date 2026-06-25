"""Geometric draft filtering for speculative decoding.

Uses E8 lattice distance to pre-filter speculative draft candidates
before the expensive target model verification pass. Tokens whose
E8 coordinates are far from the current generation trajectory are
rejected early, improving acceptance rates and throughput.

The filter operates on the 5 concentric icosahedral shells of the
projected E8 lattice, with adaptive acceptance radii per shell:
  - Shell 0 (hub, r=0.0): tightest radius (high-information tokens)
  - Shell 4 (boundary, r=1.0): loosest radius (peripheral tokens)
"""
import mlx.core as mx
import numpy as np
from typing import Optional, Tuple


class GeometricDraftFilter:
    """Filters speculative draft tokens using E8 lattice distance.
    
    During speculative decoding, draft tokens are proposed by a smaller
    model and verified by the larger target model. This filter adds a
    pre-verification step: if a draft token's E8 coordinate is too far
    from the current generation trajectory in lattice space, it's rejected
    without needing the expensive target model forward pass.
    
    Args:
        projection_matrix: The 8x3 E8-to-3D projection matrix (already loaded
            in QuasicrystallineAttention). Shape [8, 3].
        shell_radii: The 5 shell radii from the E8 projection. If None,
            uses the standard values [0.0, 0.5878, 0.8660, 0.9511, 1.0].
        r_base: Base acceptance radius. Tokens with E8 distance > r_base
            from the trajectory are rejected. Adapted per shell.
        ema_decay: Exponential moving average decay for trajectory tracking.
    """
    # Standard shell radii from the E8 golden ratio projection
    STANDARD_SHELL_RADII = [0.0, 0.5878, 0.8660, 0.9511, 1.0]
    
    def __init__(
        self,
        projection_matrix: Optional[mx.array] = None,
        shell_radii: Optional[list] = None,
        r_base: float = 0.6,
        ema_decay: float = 0.9,
    ):
        self.projection_matrix = projection_matrix
        self.shell_radii = shell_radii or self.STANDARD_SHELL_RADII
        self.r_base = r_base
        self.ema_decay = ema_decay
        self._trajectory_coord = None  # EMA of recent token E8 coordinates
        self._trajectory_shell = 2  # Default to middle shell
    
    def get_shell_index(self, coord_3d: mx.array) -> int:
        """Determine which concentric shell a 3D coordinate belongs to.
        
        Args:
            coord_3d: 3D projected coordinate, shape (3,) or (N, 3)
        Returns:
            Shell index (0-4)
        """
        r = float(mx.linalg.norm(coord_3d))
        # Find nearest shell
        min_dist = float('inf')
        best_shell = 0
        for i, shell_r in enumerate(self.shell_radii):
            dist = abs(r - shell_r)
            if dist < min_dist:
                min_dist = dist
                best_shell = i
        return best_shell
    
    def get_shell_radius(self, shell_index: int) -> float:
        """Get adaptive acceptance radius for a given shell.
        
        Tighter on hub shells (high-information tokens),
        looser on boundary shells (peripheral tokens).
        
        Args:
            shell_index: Shell index (0-4)
        Returns:
            Acceptance radius for this shell
        """
        return self.r_base * (1.0 + 0.2 * shell_index)
    
    def update_trajectory(self, hidden_state: mx.array):
        """Update the trajectory coordinate with a new token's hidden state.
        
        Uses EMA to track the moving generation trajectory in E8 space.
        
        Args:
            hidden_state: Hidden state of the most recently accepted token.
                Shape (D,) or (1, D).
        """
        if hidden_state.ndim > 1:
            hidden_state = hidden_state.squeeze(0)
        
        # Project to 3D coordinate space if projection matrix available
        if self.projection_matrix is not None:
            # Determine projection dimension (8 for E8, 24 for Leech)
            proj_dim = self.projection_matrix.shape[0]
            h_proj = hidden_state[:proj_dim] if hidden_state.shape[0] >= proj_dim else mx.pad(hidden_state, [(0, proj_dim - hidden_state.shape[0])])
            coord_3d = mx.matmul(h_proj, self.projection_matrix)
        else:
            coord_3d = hidden_state[:3]
        
        if self._trajectory_coord is None:
            self._trajectory_coord = coord_3d
        else:
            self._trajectory_coord = (
                self.ema_decay * self._trajectory_coord +
                (1.0 - self.ema_decay) * coord_3d
            )
        self._trajectory_shell = self.get_shell_index(self._trajectory_coord)
    
    def filter_candidates(
        self,
        draft_hidden_states: mx.array,
        num_candidates: int,
    ) -> Tuple[mx.array, int]:
        """Filter draft token candidates by E8 lattice distance.
        
        Args:
            draft_hidden_states: Hidden states of draft tokens.
                Shape (num_candidates, D).
            num_candidates: Number of draft candidates to evaluate.
        Returns:
            mask: Boolean acceptance mask, shape (num_candidates,)
            first_rejection: Index of first rejected token (or num_candidates
                if all accepted). Draft sequence should be truncated here.
        """
        if self._trajectory_coord is None or self.projection_matrix is None:
            # No trajectory established yet — accept everything
            return mx.ones(num_candidates, dtype=mx.bool_), num_candidates
        
        threshold = self.get_shell_radius(self._trajectory_shell)
        
        # Project each draft candidate to 3D
        # Determine projection dimension (8 for E8, 24 for Leech)
        proj_dim = self.projection_matrix.shape[0]
        D = draft_hidden_states.shape[-1]
        if D >= proj_dim:
            h_proj = draft_hidden_states[:, :proj_dim]
        else:
            h_proj = mx.pad(draft_hidden_states, [(0, 0), (0, proj_dim - D)])
        
        coords_3d = mx.matmul(h_proj, self.projection_matrix)  # (N, 3)
        
        # Compute distances from trajectory
        diffs = coords_3d - self._trajectory_coord[None, :]  # (N, 3)
        distances = mx.linalg.norm(diffs, axis=-1)  # (N,)
        
        # Accept if distance < adaptive threshold
        mask = distances < threshold
        
        # Find first rejection point — truncate draft sequence there
        mask_list = mask.tolist()
        first_rejection = num_candidates
        for i, accepted in enumerate(mask_list):
            if not accepted:
                first_rejection = i
                break
        
        return mask, first_rejection
    
    def reset(self):
        """Reset trajectory state for a new conversation/generation."""
        self._trajectory_coord = None
        self._trajectory_shell = 2
