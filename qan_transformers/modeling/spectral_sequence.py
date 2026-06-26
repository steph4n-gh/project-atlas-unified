import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Any

class SpectralSequenceAttention(nn.Module):
    """
    Multi-page attention refinement using E8 lattice shells of increasing resolution.
    Sequentially computes attention pages E1, E2, E3 and stops early if the differential norm
    falls below a dynamic convergence threshold epsilon derived from tropical temperature.
    """
    def __init__(self, base_epsilon: float = 0.05, beta_c: float = 1.0):
        super().__init__()
        self.base_epsilon = base_epsilon
        self.beta_c = beta_c

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        base_attn: Any,
        kv_cache: Optional[dict] = None,
        attn_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Computes multi-page spectral sequence attention.
        q, k, v: projected query, key, value tensors
        x: input hidden states [B, S, D]
        base_attn: the parent QuasicrystallineAttention instance
        Returns:
            dict containing accumulated out, final attn_weights, K_sparse, V_sparse, indices_sparse, topk_scores
        """
        device = q.device
        dtype = q.dtype
        
        # 1. Determine dynamic convergence threshold epsilon
        T = 1.0
        if getattr(base_attn, "temperature_mode", "fixed") == "tropical":
            T = getattr(base_attn.adaptive_tropical_temp, "temperature", torch.tensor(0.78, device=device)).item()
            
        epsilon = self.base_epsilon * np.exp(-self.beta_c * T)
        
        accumulated_out = None
        current_out = None
        final_page_dict = None
        
        self.last_computed_pages = 0
        
        # Loop through E8 shells/pages
        for page in [1, 2, 3]:
            self.last_computed_pages = page
            # Compute attention for this page using base_attn helper
            page_dict = base_attn._compute_page_attention(
                q, k, v, x, page, kv_cache, attn_mask
            )
            
            out_page = page_dict["out"]
            
            if accumulated_out is None:
                accumulated_out = out_page
                current_out = out_page
                final_page_dict = page_dict
            else:
                # Differential d_r is the correction
                d_r = out_page - current_out
                accumulated_out = accumulated_out + d_r
                current_out = out_page
                final_page_dict = page_dict
                
                # Check convergence via Frobenius norm of d_r
                d_r_norm = (d_r.norm() / np.sqrt(d_r.numel())).item()
                if d_r_norm < epsilon:
                    break
                    
        final_page_dict["out"] = accumulated_out
        return final_page_dict
