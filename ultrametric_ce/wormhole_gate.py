"""Learnable wormhole gate for ultrametric diffusion.

Replaces the hard cosine similarity threshold with a differentiable
MLP that learns when to open wormhole shortcuts between distant
tree branches based on their hidden states.
"""
import mlx.core as mx
import mlx.nn as nn


class LearnableWormholeGate(nn.Module):
    """Differentiable gate that decides when to open wormhole connections.
    
    Takes pairs of branch states and outputs a continuous gate weight in [0, 1].
    Input features: concatenated states, element-wise product, cosine similarity.
    
    Args:
        dim: Hidden state dimension of branch vectors.
        hidden_dim: Internal MLP hidden dimension.
        temperature: Sigmoid temperature for sharpening/softening gate decisions.
    """
    def __init__(self, dim: int, hidden_dim: int = 32, temperature: float = 1.0):
        super().__init__()
        # Input: [h_i || h_j || h_i * h_j || cos_sim] -> dim*2 + dim + 1 = dim*3 + 1
        input_dim = dim * 3 + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temperature = temperature
    
    def __call__(self, h_i: mx.array, h_j: mx.array) -> mx.array:
        """Compute gate weight for a pair of branch states.
        
        Args:
            h_i: First branch state, shape (..., dim)
            h_j: Second branch state, shape (..., dim)
        Returns:
            gate: Scalar gate weight in [0, 1], shape (...,)
        """
        # Element-wise product captures interaction
        h_prod = h_i * h_j
        # Cosine similarity preserves the existing signal
        cos_sim = mx.sum(h_i * h_j, axis=-1, keepdims=True) / (
            mx.linalg.norm(h_i, axis=-1, keepdims=True) *
            mx.linalg.norm(h_j, axis=-1, keepdims=True) + 1e-6
        )
        # Concatenate all features
        features = mx.concatenate([h_i, h_j, h_prod, cos_sim], axis=-1)
        logit = self.net(features).squeeze(-1)
        return mx.sigmoid(logit / self.temperature)
    
    def compute_gate_matrix(self, vecs: mx.array) -> mx.array:
        """Compute pairwise gate weights for all pairs of vectors.
        
        Args:
            vecs: Branch state vectors, shape (N, dim)
        Returns:
            gate_matrix: Pairwise gate weights, shape (N, N)
        """
        N = vecs.shape[0]
        # Expand for pairwise computation
        h_i = mx.broadcast_to(vecs[:, None, :], (N, N, vecs.shape[-1]))  # (N, N, dim)
        h_j = mx.broadcast_to(vecs[None, :, :], (N, N, vecs.shape[-1]))  # (N, N, dim)
        # Reshape for batch MLP evaluation
        h_i_flat = h_i.reshape(N * N, -1)
        h_j_flat = h_j.reshape(N * N, -1)
        gates_flat = self(h_i_flat, h_j_flat)
        return gates_flat.reshape(N, N)


class WormholeRegularizer:
    """Regularization losses for learned wormhole gates.
    
    Encourages sparsity (wormholes should be rare, opened only when needed)
    and symmetry (wormholes should be bidirectional).
    
    Args:
        sparsity_weight: Weight for L1 sparsity penalty on gate activations.
        symmetry_weight: Weight for symmetry penalty.
    """
    def __init__(self, sparsity_weight: float = 0.01, symmetry_weight: float = 0.1):
        self.sparsity_weight = sparsity_weight
        self.symmetry_weight = symmetry_weight
    
    def __call__(self, gate_matrix: mx.array) -> mx.array:
        """Compute regularization loss.
        
        Args:
            gate_matrix: Pairwise gate weights, shape (N, N)
        Returns:
            loss: Scalar regularization loss
        """
        # Sparsity: L1 penalty encourages most gates to be near zero
        sparsity_loss = self.sparsity_weight * mx.mean(mx.abs(gate_matrix))
        # Symmetry: wormholes should be bidirectional
        symmetry_loss = self.symmetry_weight * mx.mean((gate_matrix - gate_matrix.T) ** 2)
        return sparsity_loss + symmetry_loss
