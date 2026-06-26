import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ModularDuality(nn.Module):
    """
    Modular S-matrix transformation relating short and long context behaviors.
    Maps context length L to effective dual length L_dual = L_ref^2 / L.
    """
    def __init__(self, reference_length: float = 2048.0):
        super().__init__()
        self.reference_length = reference_length

    def forward(self, dist: torch.Tensor, L: float) -> torch.Tensor:
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
        self.delta = nn.Parameter(torch.ones(max_positions) * 0.1) # small initial scaling

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, S, D = x.shape
        pos = torch.arange(offset, offset + S, device=x.device, dtype=torch.float32)
        pos_clamped = torch.clamp(pos.long(), 0, self.max_positions - 1)
        
        # delta shape: [S]
        delta_p = self.delta[pos_clamped]
        
        # Scaling factor: λ^-Δ where λ = pos + 1
        scale = (pos + 1.0) ** (-delta_p) # [S]
        scale = scale.view(1, S, 1).to(x.dtype)
        
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
        
        self.delta = nn.Parameter(torch.ones(max_positions) * 0.5)  # Conformal dimensions
        self.ope_net = nn.Linear(dim * 3, dim)  # OPE fusion
        self.modular_duality = ModularDuality(reference_length)

    def fuse_ope(self, x: torch.Tensor) -> torch.Tensor:
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
        fused = self.ope_net(torch.cat([x_left, x_right, x_prod], dim=-1)) # [B, S-1, D]
        
        # Pad or blend back to original shape [B, S, D]
        out = torch.zeros_like(x)
        out[:, :-1] = fused
        out[:, -1] = x[:, -1]
        return out

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                q_pos: torch.Tensor = None, k_pos: torch.Tensor = None, 
                L: float = None) -> torch.Tensor:
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
        device = Q.device
        dtype = Q.dtype
        
        # 1. Resolve positions if not provided
        if q_pos is None:
            q_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        if k_pos is None:
            k_pos = torch.arange(S_keys, device=device).unsqueeze(0).expand(B, -1)
            
        if L is None:
            L = float(max(S, S_keys))
            
        # 2. Compute conformal distances |i-j|
        q_pos_expanded = q_pos.view(B, 1, S, 1)
        k_pos_expanded = k_pos.view(B, 1, 1, S_keys)
        
        dist = torch.abs(q_pos_expanded - k_pos_expanded).to(dtype=dtype)
        dist = torch.clamp(dist, min=1.0) # Regularize short-distance cutoff
        
        # Apply Modular Duality S-transformation
        if self.use_modular_duality:
            # S-dual distance
            dist_dual = self.modular_duality(dist, L)
            # Blend based on how much L exceeds reference length
            alpha = torch.sigmoid(torch.tensor((L - self.reference_length) / 512.0, device=device, dtype=dtype))
            dist_eff = (1.0 - alpha) * dist + alpha * dist_dual
        else:
            dist_eff = dist
            
        # 3. Retrieve conformal dimensions Delta_i + Delta_j
        q_pos_clamped = torch.clamp(q_pos.long(), 0, self.max_positions - 1)
        k_pos_clamped = torch.clamp(k_pos.long(), 0, self.max_positions - 1)
        
        delta_q = self.delta[q_pos_clamped].view(B, 1, S, 1)
        delta_k = self.delta[k_pos_clamped].view(B, 1, 1, S_keys)
        
        exponents = delta_q + delta_k
        
        # 4. Compute two-point function denominator: |i-j|^(Delta_i + Delta_j)
        denominator = dist_eff ** exponents
        
        # 5. Compute coupling C_ij as Q K^T scaled dot product
        coupling = torch.matmul(Q, K.transpose(-2, -1)) # [B, H, S, S_keys]
        
        # Attention scores
        attn_scores = coupling / (denominator + 1e-6)
        
        # Softmax and compute output
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        return {
            "out": out,
            "attn_weights": attn_weights,
            "attn_scores": attn_scores
        }
