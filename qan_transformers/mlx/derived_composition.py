import mlx.core as mx
import mlx.nn as nn
import numpy as np

class DerivedAttentionComposition(nn.Module):
    """
    MLX version of Derived Category Attention Composition:
    A_{composed} = A_{l+1} * A_l + alpha * Ext^1(A_l, A_{l+1})
    Uses SVD null-space analysis to recover information lost in deep layer stacks.
    """
    def __init__(self, init_alpha: float = 0.01):
        super().__init__()
        self.alpha = mx.array(init_alpha)
        self.last_ext1_norm = 0.0

    def __call__(self, A_curr: mx.array, A_prev: mx.array, indices: mx.array = None, S_total: int = None) -> mx.array:
        """
        A_curr: attention weights of current layer [B, H, S, K_len]
        A_prev: attention weights of previous layer [B, H, S, K_len]
        indices: indices of selected keys in the full sequence [B, K_total]
        S_total: total sequence length of keys
        Returns:
        A_composed: [B, H, S, K_len]
        """
        B, H, S, K_len = A_curr.shape
        
        # If sparse/gather indices are provided, we scatter to full space, compose, and gather back
        if indices is not None:
            K_total = indices.shape[-1]
            if S_total is None:
                S_total = int(mx.max(indices).item()) + 1
                
            # If K_len > K_total (e.g. swapped historical keys appended), add dummy indices
            if K_len > K_total:
                num_extra = K_len - K_total
                extra_indices = mx.expand_dims(mx.arange(S_total, S_total + num_extra), 0)
                extra_indices = mx.broadcast_to(extra_indices, (B, num_extra))
                indices = mx.concatenate([indices, extra_indices], axis=-1)
                S_scatter = S_total + num_extra
            else:
                S_scatter = S_total
                
            # If the scattered sequence length doesn't match the query sequence length,
            # we cannot perform layer multiplication/SVD composition. Fallback to A_curr.
            if S_scatter != S:
                return A_curr
                
            # Create zeros for scatter
            A_curr_full = mx.zeros((B, H, S, S_scatter), dtype=A_curr.dtype)
            A_prev_full = mx.zeros((B, H, S, S_scatter), dtype=A_prev.dtype)
            
            # Simple scatter loop per batch
            for b in range(B):
                idx_b = indices[b] # [K_len]
                for h in range(H):
                    for s in range(S):
                        A_curr_full[b, h, s, idx_b] = A_curr[b, h, s]
                        A_prev_full[b, h, s, idx_b] = A_prev[b, h, s]
            
            A_composed_full = self._compose_full(A_curr_full, A_prev_full)
            
            # Gather back
            A_composed = mx.zeros_like(A_curr)
            for b in range(B):
                idx_b = indices[b]
                for h in range(H):
                    for s in range(S):
                        A_composed[b, h, s] = A_composed_full[b, h, s, idx_b]
            return A_composed
            
        else:
            # Dense case: shape is [B, H, S, K_len]
            if K_len != S:
                return A_curr
            return self._compose_full(A_curr, A_prev)

    def _compose_full(self, A_curr: mx.array, A_prev: mx.array) -> mx.array:
        B, H, S, _ = A_curr.shape
        
        # Base composition
        A_comp = mx.matmul(A_curr, A_prev)
        
        if S < 4:
            # SVD is not stable or useful on very small sequence lengths
            return A_comp
            
        A_curr_flat = mx.reshape(A_curr, (B * H, S, S))
        A_prev_flat = mx.reshape(A_prev, (B * H, S, S))
        
        cpu_stream = mx.default_stream(mx.cpu)
        
        Ext1_list = []
        for i in range(B * H):
            ac = A_curr_flat[i] # [S, S]
            ap = A_prev_flat[i] # [S, S]
            
            try:
                # SVD of ac (current layer): ac = U_c * S_c * V_c^T
                # In MLX, mx.linalg.svd returns (U, S, Vt)
                Uc, Sc, Vtc = mx.linalg.svd(ac, stream=cpu_stream)
                Vc = mx.transpose(Vtc, (1, 0))
                
                # SVD of ap (previous layer)
                Up, Sp, Vtp = mx.linalg.svd(ap, stream=cpu_stream)
                
                max_sc = mx.max(Sc)
                null_mask = Sc < 0.15 * max_sc
                max_sp = mx.max(Sp)
                range_mask = Sp >= 0.15 * max_sp
                
                null_indices = [idx for idx, val in enumerate(null_mask.tolist()) if val]
                range_indices = [idx for idx, val in enumerate(range_mask.tolist()) if val]
                
                if len(null_indices) > 0 and len(range_indices) > 0:
                    Vc_null = Vc[:, mx.array(null_indices)]
                    Up_range = Up[:, mx.array(range_indices)]
                    
                    proj = mx.matmul(mx.transpose(Up_range, (1, 0)), Vc_null)
                    ext1 = mx.matmul(mx.matmul(Up_range, proj), mx.transpose(Vc_null, (1, 0)))
                else:
                    ext1 = mx.zeros((S, S), dtype=A_curr.dtype)
            except Exception as e:
                ext1 = mx.zeros((S, S), dtype=A_curr.dtype)
                
            Ext1_list.append(ext1)
            
        Ext1 = mx.stack(Ext1_list)
        Ext1 = mx.reshape(Ext1, (B, H, S, S))
        
        # Log/monitor Ext^1 norm
        ext1_norm = mx.sqrt(mx.sum(Ext1 ** 2)).item()
        self.last_ext1_norm = ext1_norm
        
        # Apply learnable correction
        alpha_clamped = mx.clip(self.alpha, -0.2, 0.2)
        A_composed = A_comp + alpha_clamped * Ext1
        
        return A_composed
