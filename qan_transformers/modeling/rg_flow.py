import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class KVRenormalizationFlow(nn.Module):
    """
    Wilsonian Renormalization Group (RG) Flow for continuous KV cache compression.
    Treats context length as energy scale, diffusing redundant keys/values using
    the attention graph Laplacian and amplifying relevant operators (high-attention tokens).
    """
    def __init__(self, eta: float = 0.1, alpha: float = 0.05, max_steps: int = 5):
        super().__init__()
        self.eta = eta          # Diffusion coefficient (Laplacian coupling)
        self.alpha = alpha      # Relevant operator scaling coefficient
        self.max_steps = max_steps

    def beta_function(self, K: torch.Tensor, W: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Computes the beta function: β(K) = eta * L_neg * K + alpha * diag(scores) * K
        K: [B, H, N, D_head]
        W: [B, H, N, N] symmetric similarity matrix
        scores: [B, N] or [B, H, N] alignment scores
        """
        B, H, N, D = K.shape
        
        # Compute degree matrix and Laplacian L = D - W
        D_diag = W.sum(dim=-1) # [B, H, N]
        
        # -L * K = W * K - D * K
        LK_neg = torch.matmul(W, K) - D_diag.unsqueeze(-1) * K
        
        # diag(scores) * K
        if scores.ndim == 2:
            scores_expanded = scores.unsqueeze(1).unsqueeze(-1) # [B, 1, N, 1]
        elif scores.ndim == 3:
            scores_expanded = scores.unsqueeze(-1) # [B, H, N, 1]
        else:
            scores_expanded = scores.view(B, H, N, 1)
            
        relevant_ops = scores_expanded * K
        
        beta = self.eta * LK_neg + self.alpha * relevant_ops
        return beta

    def flow_step(self, K: torch.Tensor, V: torch.Tensor, W: torch.Tensor, scores: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        beta_K = self.beta_function(K, W, scores)
        beta_V = self.beta_function(V, W, scores)
        return K + dt * beta_K, V + dt * beta_V

    def compress(
        self,
        K_combined: torch.Tensor,
        V_combined: torch.Tensor,
        indices_combined: torch.Tensor,
        scores_combined: torch.Tensor,
        K_total: int,
        compression_level: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evolves the KV cache under RG flow and returns the compressed cache.
        """
        B, H, N, D = K_combined.shape
        device = K_combined.device
        dtype = K_combined.dtype
        
        if N <= K_total:
            return K_combined, V_combined, indices_combined, scores_combined
            
        # 1. Compute similarity matrix W from keys
        K_norm = F.normalize(K_combined, p=2, dim=-1, eps=1e-6)
        W = torch.matmul(K_norm, K_norm.transpose(-2, -1)) # [B, H, N, N]
        # Zero out diagonal
        diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        W = W.masked_fill(diag_mask, 0.0)
        
        # 2. Run RG flow
        K_flow = K_combined.clone()
        V_flow = V_combined.clone()
        dt = compression_level
        
        for step in range(self.max_steps):
            K_flow, V_flow = self.flow_step(K_flow, V_flow, W, scores_combined, dt)
            
        # 3. Coarse-graining / Selection: select the top-K_total Morse critical cells (fixed points)
        mean_scores = scores_combined.mean(dim=1) if scores_combined.ndim == 3 else scores_combined # [B, N]
        
        topk_res = torch.topk(mean_scores, K_total, dim=-1, sorted=True)
        topk_indices = topk_res.indices # [B, K_total]
        topk_indices, sort_idx = torch.sort(topk_indices, dim=-1)
        
        # Gather keys, values, indices, and scores
        gather_indices_kv = topk_indices.view(B, 1, K_total, 1).expand(-1, H, -1, D)
        K_compressed = torch.gather(K_flow, 2, gather_indices_kv)
        V_compressed = torch.gather(V_flow, 2, gather_indices_kv)
        
        indices_compressed = torch.gather(indices_combined, 1, topk_indices)
        
        if scores_combined.ndim == 3:
            gather_indices_scores = topk_indices.unsqueeze(1).expand(-1, H, -1)
            scores_compressed = torch.gather(scores_combined, 2, gather_indices_scores)
        else:
            scores_compressed = torch.gather(scores_combined, 1, topk_indices)
            
        return K_compressed, V_compressed, indices_compressed, scores_compressed
