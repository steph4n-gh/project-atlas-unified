import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Union

class MotivicCohomologyFirewall(nn.Module):
    """
    Bi-graded motivic cohomology firewall H^{p,q} for attention matrix skeletons.
    Decomposes the attention skeleton into weak and strong weight filtrations,
    computes connected components (H^0) and loops (H^1) at each scale,
    and dispatches specific remedial actions (rollback, sharpen, halt).
    """
    def __init__(
        self,
        weak_threshold: float = 0.15,
        strong_threshold: float = 0.45,
        tau_sig: float = 0.02,
        c_eigen: float = 5.0,
        halt_threshold_h11: float = 1.0,
        rollback_threshold_h10: float = 1.5,
        sharpen_threshold_h01: float = 3.0
    ):
        super().__init__()
        self.weak_threshold = weak_threshold
        self.strong_threshold = strong_threshold
        self.tau_sig = tau_sig
        self.c_eigen = c_eigen
        
        # Action thresholds
        self.halt_threshold_h11 = halt_threshold_h11
        self.rollback_threshold_h10 = rollback_threshold_h10
        self.sharpen_threshold_h01 = sharpen_threshold_h01

    def compute_betti_numbers(self, W: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes hard and soft H0 and H1 Betti numbers for a given filtration threshold.
        W: [B_total, K, K] symmetric attention weights
        Returns:
            hard_h0: [B_total]
            hard_h1: [B_total]
            soft_h0: [B_total]
            soft_h1: [B_total]
        """
        B_total, K, _ = W.shape
        device = W.device
        dtype = W.dtype
        
        # 1. Hard graph metrics
        # Adjacency matrix
        M_hard = (W > threshold).to(dtype)
        # Mask out diagonal to avoid self-loops
        diag_mask = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
        M_hard = M_hard.masked_fill(diag_mask, 0.0)
        # Degree matrix diagonal
        D_hard_diag = M_hard.sum(dim=-1)
        # Laplacian
        L_hard = torch.diag_embed(D_hard_diag) - M_hard
        
        # Eigenvalues
        try:
            # We compute eigenvalues of the symmetric Laplacian
            eigvals_hard = torch.linalg.eigvalsh(L_hard) # [B_total, K]
            # H0 is the count of eigenvalues close to 0 (isolated components)
            hard_h0 = (eigvals_hard < 1e-4).sum(dim=-1).to(dtype)
        except Exception:
            # Fallback for numerical instability
            hard_h0 = torch.ones(B_total, device=device, dtype=dtype)
            
        # Edges (undirected)
        E_hard = 0.5 * M_hard.sum(dim=(-2, -1))
        # H1 = E - V + H0
        hard_h1 = torch.clamp(E_hard - K + hard_h0, min=0.0)
        
        # 2. Soft/differentiable graph metrics
        # Sigmoid relaxation for differentiable adjacency
        M_soft = torch.sigmoid((W - threshold) / self.tau_sig)
        # Mask out diagonal to avoid self-loops
        diag_mask = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0)
        M_soft = M_soft.masked_fill(diag_mask, 0.0)
        
        D_soft_diag = M_soft.sum(dim=-1)
        L_soft = torch.diag_embed(D_soft_diag) - M_soft
        
        try:
            eigvals_soft = torch.linalg.eigvalsh(L_soft) # [B_total, K]
            # Soft H0 approximation: sum(exp(-c * lambda))
            soft_h0 = torch.exp(-self.c_eigen * eigvals_soft).sum(dim=-1)
        except Exception:
            soft_h0 = torch.ones(B_total, device=device, dtype=dtype)
            
        E_soft = 0.5 * M_soft.sum(dim=(-2, -1))
        soft_h1 = torch.clamp(E_soft - K + soft_h0, min=0.0)
        
        return hard_h0, hard_h1, soft_h0, soft_h1

    def forward(self, skeleton: torch.Tensor) -> Dict[str, Any]:
        """
        skeleton: [B, H, K, K] or [B, K, K] or [K, K] attention skeleton
        Returns:
            diagnostics: Dict containing h00, h10, h01, h11 and action decisions.
        """
        # Save original shape info
        orig_shape = skeleton.shape
        device = skeleton.device
        dtype = skeleton.dtype
        
        # Normalize to [B_total, K, K]
        if skeleton.ndim == 2:
            W = skeleton.unsqueeze(0)
        elif skeleton.ndim == 3:
            W = skeleton
        elif skeleton.ndim == 4:
            B, H, K, _ = skeleton.shape
            W = skeleton.view(B * H, K, K)
        else:
            raise ValueError(f"Unsupported skeleton dimensions: {skeleton.ndim}")
            
        B_total, K, _ = W.shape
        
        # Symmetrize
        W_sym = 0.5 * (W + W.transpose(-1, -2))
        
        # Compute motivic cohomology at weak filtration (q=0)
        h00_hard, h10_hard, h00_soft, h10_soft = self.compute_betti_numbers(W_sym, self.weak_threshold)
        
        # Compute motivic cohomology at strong filtration (q=1)
        h01_hard, h11_hard, h01_soft, h11_soft = self.compute_betti_numbers(W_sym, self.strong_threshold)
        
        # Action decisions based on hard Betti numbers (per element in batch)
        actions = []
        for i in range(B_total):
            h11 = h11_hard[i].item()
            h10 = h10_hard[i].item()
            h01 = h01_hard[i].item()
            
            if h11 >= self.halt_threshold_h11:
                actions.append("halt")
            elif h10 >= self.rollback_threshold_h10:
                actions.append("rollback")
            elif h01 >= self.sharpen_threshold_h01:
                actions.append("sharpen")
            else:
                actions.append("none")
                
        # Reshape metrics back to original batch size structure if needed
        if skeleton.ndim == 4:
            h00_hard = h00_hard.view(B, H)
            h10_hard = h10_hard.view(B, H)
            h01_hard = h01_hard.view(B, H)
            h11_hard = h11_hard.view(B, H)
            
            h00_soft = h00_soft.view(B, H)
            h10_soft = h10_soft.view(B, H)
            h01_soft = h01_soft.view(B, H)
            h11_soft = h11_soft.view(B, H)
            
            # Map actions back to [B, H]
            action_grid = []
            for b in range(B):
                action_grid.append(actions[b * H : (b + 1) * H])
            actions = action_grid
            
        elif skeleton.ndim == 2:
            h00_hard = h00_hard[0]
            h10_hard = h10_hard[0]
            h01_hard = h01_hard[0]
            h11_hard = h11_hard[0]
            
            h00_soft = h00_soft[0]
            h10_soft = h10_soft[0]
            h01_soft = h01_soft[0]
            h11_soft = h11_soft[0]
            actions = actions[0]
            
        return {
            "h00": h00_hard,
            "h10": h10_hard,
            "h01": h01_hard,
            "h11": h11_hard,
            
            "soft_h00": h00_soft,
            "soft_h10": h10_soft,
            "soft_h01": h01_soft,
            "soft_h11": h11_soft,
            
            "action": actions
        }
