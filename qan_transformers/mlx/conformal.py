import mlx.core as mx
import mlx.nn as nn
import numpy as np

class ModularDuality(nn.Module):
    """
    Modular S-matrix transformation relating short and long context behaviors.
    Maps context length L to effective dual length L_dual = L_ref^2 / L.
    """
    def __init__(self, reference_length: float = 2048.0):
        super().__init__()
        self.reference_length = reference_length

    def __call__(self, dist: mx.array, L: float) -> mx.array:
        L = max(L, 1.0)
        # S-matrix dual length
        L_dual = (self.reference_length ** 2) / L
        # Map dist under dual scaling
        dist_dual = dist * (L_dual / L)
        return dist_dual


class ConformalPositionalEncoding(nn.Module):
    """
    Applies conformal scaling to hidden states: A(λx) = λ^-Δ A(x)
    """
    def __init__(self, max_positions: int = 8192):
        super().__init__()
        self.max_positions = max_positions
        self.delta = mx.array(np.ones(max_positions) * 0.1, dtype=mx.float32) # small initial scaling

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        B, S, D = x.shape
        pos = mx.arange(offset, offset + S, dtype=mx.float32)
        pos_clamped = mx.clip(pos.astype(mx.int32), 0, self.max_positions - 1)
        
        # delta shape: [S]
        delta_p = self.delta[pos_clamped]
        
        # Scaling factor: λ^-Δ where λ = pos + 1
        scale = (pos + 1.0) ** (-delta_p) # [S]
        scale = mx.expand_dims(scale, (0, 2))
        
        return x * scale


class ConformalAttention(nn.Module):
    """
    CFT two-point function attention with length generalization.
    Scale covariance ensures attention patterns trained at length L
    generalize to length λL via conformal symmetry.
    """
    def __init__(self, dim: int, max_positions: int = 8192, use_modular_duality: bool = True, reference_length: float = 2048.0):
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.use_modular_duality = use_modular_duality
        self.reference_length = reference_length
        
        self.delta = mx.array(np.ones(max_positions) * 0.5, dtype=mx.float32)  # Conformal dimensions
        self.ope_net = nn.Linear(dim * 3, dim)  # OPE fusion
        self.modular_duality = ModularDuality(reference_length)

    def fuse_ope(self, x: mx.array) -> mx.array:
        """
        Operator Product Expansion (OPE) fusion for nearby tokens:
        O_i * O_j ~ C_ij^k O_k
        """
        B, S, D = x.shape
        if S < 2:
            return x
        # Concatenate consecutive tokens and their product
        x_left = x[:, :-1]
        x_right = x[:, 1:]
        x_prod = x_left * x_right
        
        # Fuse using the OPE network
        fused = self.ope_net(mx.concatenate([x_left, x_right, x_prod], axis=-1)) # [B, S-1, D]
        
        # Pad or blend back to original shape [B, S, D]
        out = mx.concatenate([fused, x[:, -1:]], axis=1)
        return out

    def __call__(self, Q: mx.array, K: mx.array, V: mx.array, 
                 q_pos: mx.array = None, k_pos: mx.array = None, 
                 L: float = None) -> dict:
        """
        Q: [B, H, S, d_head]
        K: [B, H, S_keys, d_head]
        V: [B, H, S_keys, d_head]
        q_pos: Query positions [B, S] or [1, S]
        k_pos: Key positions [B, S_keys] or [1, S_keys]
        L: Current sequence/context length
        """
        B, H, S, d_head = Q.shape
        S_keys = K.shape[2]
        
        # 1. Resolve positions if not provided
        if q_pos is None:
            q_pos = mx.broadcast_to(mx.expand_dims(mx.arange(S), 0), (B, S))
        if k_pos is None:
            k_pos = mx.broadcast_to(mx.expand_dims(mx.arange(S_keys), 0), (B, S_keys))
            
        if L is None:
            L = float(max(S, S_keys))
            
        # 2. Compute conformal distances |i-j|
        q_pos_expanded = mx.reshape(q_pos, (B, 1, S, 1))
        k_pos_expanded = mx.reshape(k_pos, (B, 1, 1, S_keys))
        
        dist = mx.abs(q_pos_expanded - k_pos_expanded)
        dist = mx.maximum(dist, 1.0) # Regularize short-distance cutoff
        
        # Apply Modular Duality S-transformation
        if self.use_modular_duality:
            # S-dual distance
            dist_dual = self.modular_duality(dist, L)
            # Blend based on how much L exceeds reference length
            alpha = mx.sigmoid(mx.array((L - self.reference_length) / 512.0))
            dist_eff = (1.0 - alpha) * dist + alpha * dist_dual
        else:
            dist_eff = dist
            
        # 3. Retrieve conformal dimensions Delta_i + Delta_j
        q_pos_clamped = mx.clip(q_pos.astype(mx.int32), 0, self.max_positions - 1)
        k_pos_clamped = mx.clip(k_pos.astype(mx.int32), 0, self.max_positions - 1)
        
        delta_q = mx.reshape(self.delta[q_pos_clamped], (B, 1, S, 1))
        delta_k = mx.reshape(self.delta[k_pos_clamped], (B, 1, 1, S_keys))
        
        exponents = delta_q + delta_k
        
        # 4. Compute two-point function denominator: |i-j|^(Delta_i + Delta_j)
        denominator = dist_eff ** exponents
        
        # 5. Compute coupling C_ij as Q K^T scaled dot product
        coupling = mx.matmul(Q, mx.transpose(K, (0, 1, 3, 2))) # [B, H, S, S_keys]
        
        # Attention scores
        attn_scores = coupling / (denominator + 1e-6)
        
        # Softmax and compute output
        attn_weights = mx.softmax(attn_scores, axis=-1)
        out = mx.matmul(attn_weights, V)
        
        return {
            "out": out,
            "attn_weights": attn_weights,
            "attn_scores": attn_scores
        }
