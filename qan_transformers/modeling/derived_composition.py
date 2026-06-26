import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DerivedAttentionComposition(nn.Module):
    """
    Computes the derived tensor product of attention layers:
    A_{composed} = A_{l+1} * A_l + alpha * Ext^1(A_l, A_{l+1})
    Uses SVD null-space analysis to recover information lost in deep layer stacks.
    """
    def __init__(self, init_alpha: float = 0.01):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.last_ext1_norm = 0.0

    def forward(self, A_curr: torch.Tensor, A_prev: torch.Tensor, indices: torch.Tensor = None, S_total: int = None) -> torch.Tensor:
        """
        A_curr: attention weights of current layer [B, H, S, K_len]
        A_prev: attention weights of previous layer [B, H, S, K_len]
        indices: indices of selected keys in the full sequence [B, K_total]
        S_total: total sequence length of keys
        Returns:
            A_composed: [B, H, S, K_len]
        """
        B, H, S, K_len = A_curr.shape
        device = A_curr.device
        dtype = A_curr.dtype
        
        # If sparse/gather indices are provided, we scatter to full space, compose, and gather back
        if indices is not None:
            K_total = indices.shape[-1]
            if S_total is None:
                S_total = int(indices.max().item()) + 1
                
            # If K_len > K_total (e.g. swapped historical keys appended), add dummy indices
            if K_len > K_total:
                num_extra = K_len - K_total
                extra_indices = torch.arange(S_total, S_total + num_extra, device=device).unsqueeze(0).expand(B, -1)
                indices = torch.cat([indices, extra_indices], dim=-1)
                S_scatter = S_total + num_extra
            else:
                S_scatter = S_total
                
            # If the scattered sequence length doesn't match the query sequence length,
            # we cannot perform layer multiplication/SVD composition. Fallback to A_curr.
            if S_scatter != S:
                return A_curr
                
            A_curr_full = torch.zeros(B, H, S, S_scatter, device=device, dtype=dtype)
            A_prev_full = torch.zeros(B, H, S, S_scatter, device=device, dtype=dtype)
            
            index_expanded = indices.view(B, 1, 1, K_len).expand(B, H, S, K_len)
            A_curr_full.scatter_(dim=-1, index=index_expanded, src=A_curr)
            A_prev_full.scatter_(dim=-1, index=index_expanded, src=A_prev)
            
            A_composed_full = self._compose_full(A_curr_full, A_prev_full)
            A_composed = torch.gather(A_composed_full, dim=-1, index=index_expanded)
            return A_composed
            
        else:
            # Dense case: shape is [B, H, S, K_len]
            if K_len != S:
                return A_curr
            return self._compose_full(A_curr, A_prev)

    def _compose_full(self, A_curr: torch.Tensor, A_prev: torch.Tensor) -> torch.Tensor:
        B, H, S, _ = A_curr.shape
        device = A_curr.device
        dtype = A_curr.dtype
        
        # Base composition
        A_comp = torch.matmul(A_curr, A_prev)
        
        if S < 4:
            # SVD is not stable or useful on very small sequence lengths
            return A_comp
            
        A_curr_flat = A_curr.view(B * H, S, S)
        A_prev_flat = A_prev.view(B * H, S, S)
        
        Ext1_list = []
        for i in range(B * H):
            ac = A_curr_flat[i] # [S, S]
            ap = A_prev_flat[i] # [S, S]
            
            try:
                # SVD of ac (current layer): ac = U_c * S_c * V_c^T
                Uc, Sc, Vhc = torch.linalg.svd(ac, full_matrices=False)
                Vc = Vhc.transpose(-2, -1)
                
                # SVD of ap (previous layer): ap = U_p * S_p * V_p^T
                Up, Sp, Vhp = torch.linalg.svd(ap, full_matrices=False)
                
                # Null space of ac: columns of Vc where Sc is small
                max_sc = torch.max(Sc)
                null_mask = Sc < 0.15 * max_sc
                # Range of ap: columns of Up where Sp is large
                max_sp = torch.max(Sp)
                range_mask = Sp >= 0.15 * max_sp
                
                Vc_null = Vc[:, null_mask]
                Up_range = Up[:, range_mask]
                
                if Vc_null.shape[1] > 0 and Up_range.shape[1] > 0:
                    # Ext^1 projection: Up_range * (Up_range^T * Vc_null) * Vc_null^T
                    proj = torch.matmul(Up_range.transpose(-2, -1), Vc_null) # [k_range, k_null]
                    ext1 = torch.matmul(torch.matmul(Up_range, proj), Vc_null.transpose(-2, -1)) # [S, S]
                else:
                    ext1 = torch.zeros(S, S, device=device, dtype=dtype)
            except Exception:
                ext1 = torch.zeros(S, S, device=device, dtype=dtype)
                
            Ext1_list.append(ext1)
            
        Ext1 = torch.stack(Ext1_list).view(B, H, S, S)
        
        # Log/monitor Ext^1 norm
        ext1_norm = Ext1.norm().item()
        self.last_ext1_norm = ext1_norm
        
        # Apply learnable correction
        alpha_clamped = torch.clamp(self.alpha, min=-0.2, max=0.2)
        A_composed = A_comp + alpha_clamped * Ext1
        
        return A_composed
