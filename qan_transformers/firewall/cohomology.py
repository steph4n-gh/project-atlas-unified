import numpy as np
import torch
from typing import Tuple, List, Any

try:
    import mlx.core as mx
except ImportError:
    mx = None

class CohomologyFirewall:
    """
    CohomologyFirewall detects topological obstructions (anomalous fracturing) in attention weights
    using Čech coboundary calculations over a Morse collapsed skeleton.
    """
    def __init__(self, threshold: float = 1.5, rollback_limit: int = 3, tau: float = 0.05):
        self.threshold = threshold
        self.rollback_limit = rollback_limit
        self.tau = tau
        self.split_boundary = None
        self.last_lambda_2 = 1.0
        self._fiedler_cache = {}

    def check_obstruction(self, attn_matrix: Any) -> Tuple[Any, Any, Any]:
        """
        Evaluates the attention matrix/vector.
        Returns:
            is_fractured (bool or List[bool]): True if obstruction metric (cfi) exceeds threshold or algebraic connectivity < tau.
            cfi (float or List[float]): The Cohomology Fracture Index.
            alt_idx (List[int] or List[List[int]]): Alternative token indices ordered by attention strength, with bisection boundary prepended if triggered.
        """
        # Handle empty/invalid inputs gracefully
        if attn_matrix is None:
            return False, 0.0, []

        # Check if MLX array
        if hasattr(attn_matrix, "__class__") and attn_matrix.__class__.__name__ == "array" and attn_matrix.__class__.__module__ == "mlx.core":
            return self.check_obstruction_mlx(attn_matrix)

        if isinstance(attn_matrix, torch.Tensor):
            A = attn_matrix
        else:
            A = torch.tensor(attn_matrix)

        if A.numel() == 0:
            return False, 0.0, []

        # Normalize to 4D format [B, H, M, N]
        if A.ndim == 1:
            A = A.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            B = 1
        elif A.ndim == 2:
            A = A.unsqueeze(0).unsqueeze(0)
            B = 1
        elif A.ndim == 3:
            A = A.unsqueeze(1)
            B = A.shape[0]
        elif A.ndim == 4:
            B = A.shape[0]
        else:
            raise ValueError(f"Attention matrix dimension {A.ndim} not supported.")

        B_dim, H_dim, M_dim, N_dim = A.shape
        device = A.device
        dtype = A.dtype
        S = max(M_dim, N_dim)
        
        # Win 21: PyTorch Sum-Reduction Bypass
        if M_dim == 1:
            vertex_energies = A.squeeze(2)
        elif M_dim <= N_dim:
            vertex_energies = torch.sum(A, dim=2)
        else:
            vertex_energies = torch.zeros((B_dim, H_dim, M_dim), device=device, dtype=dtype)
            vertex_energies[..., :N_dim] = torch.sum(A, dim=2)

        K = min(8, S)
        
        # GPU Morse collapse selection (Win 140: Top-K over Argsort)
        critical_summits = torch.topk(vertex_energies, K, dim=-1, largest=True, sorted=True).indices


        # GPU Skeleton extraction
        clamped_rows = torch.clamp(critical_summits, max=M_dim - 1)
        gather_rows = clamped_rows.unsqueeze(-1).expand(-1, -1, -1, N_dim)
        A_rows = torch.gather(A, 2, gather_rows)
        
        clamped_cols = torch.clamp(critical_summits, max=N_dim - 1)
        col_indices = clamped_cols.unsqueeze(-2).expand(-1, -1, K, -1)
        A_skeleton = torch.gather(A_rows, 3, col_indices)
        
        mask_rows = (critical_summits < M_dim).unsqueeze(-1)
        mask_cols = (critical_summits < N_dim).unsqueeze(-2)
        A_skeleton = A_skeleton * mask_rows * mask_cols

        # GPU nan/inf check per batch/head (Win 85: Fused Non-Finite Check on Vertex Energies)
        has_nan_inf_gpu = ~torch.isfinite(vertex_energies)
        has_nan_inf_heads = torch.any(has_nan_inf_gpu, dim=-1) # shape [B_dim, H_dim]

        # Win 92: Pre-allocated Intermediates in Cohomology Firewall (PyTorch)
        cache_key = (B_dim, H_dim, K, device, dtype)
        if not hasattr(self, "_firewall_workspace_cache"):
            self._firewall_workspace_cache = {}
        if cache_key in self._firewall_workspace_cache:
            workspace = self._firewall_workspace_cache[cache_key]
        else:
            workspace = {
                "W": torch.empty((B_dim, H_dim, K, K), device=device, dtype=dtype),
                "degrees": torch.empty((B_dim, H_dim, K), device=device, dtype=dtype),
                "L": torch.empty((B_dim, H_dim, K, K), device=device, dtype=dtype),
            }
            self._firewall_workspace_cache[cache_key] = workspace
            
        W = workspace["W"]
        torch.add(A_skeleton, A_skeleton.transpose(-1, -2), out=W)
        W.mul_(0.5)
        degrees = workspace["degrees"]
        torch.sum(W, dim=-1, out=degrees)
        L = workspace["L"]
        torch.neg(W, out=L)
        L.diagonal(dim1=-2, dim2=-1).add_(degrees)
        
        trace_W = torch.diagonal(W, dim1=-2, dim2=-1).sum(dim=-1)
        sum_W = torch.sum(W, dim=(-2, -1))
        off_diag_sum = sum_W - trace_W

        # Win 173: Conditional Eigen-Decomposition Bypass when off_diag_sum <= 0.1 or K <= 1
        if K > 1 and torch.any(off_diag_sum > 0.1):
            # Vectorized solver for second smallest eigenvalue (Fiedler vector)
            try:
                eigvals, eigvecs = torch.linalg.eigh(L)
                lam2 = eigvals[..., 1]  # [B_dim, H_dim]
                v2 = eigvecs[..., :, 1]  # [B_dim, H_dim, K]
            except Exception:
                # Fallback for degenerate matrices
                lam2 = torch.ones(B_dim, H_dim, device=device, dtype=dtype)
                v2 = torch.zeros(B_dim, H_dim, K, device=device, dtype=dtype)
        else:
            # Bypassed: fill with default values (algebraic connectivity = 1.0, Fiedler = 0.0)
            lam2 = torch.ones(B_dim, H_dim, device=device, dtype=dtype)
            v2 = torch.zeros(B_dim, H_dim, K, device=device, dtype=dtype)
        
        active_mask = degrees > 1e-4 # [B_dim, H_dim, K]
        fill_val = 999999
        
        pos_mask = active_mask & (v2 >= 0.0)
        neg_mask = active_mask & (v2 < 0.0)
        
        # Win 54: Zero-Allocation Branch-Free Summits in PyTorch replacing torch.where
        pos_summits = critical_summits + (~pos_mask).to(critical_summits.dtype) * (fill_val - critical_summits)
        neg_summits = critical_summits + (~neg_mask).to(critical_summits.dtype) * (fill_val - critical_summits)
        
        min_pos = torch.min(pos_summits, dim=-1).values
        min_neg = torch.min(neg_summits, dim=-1).values
        
        has_pos = torch.any(pos_mask, dim=-1)
        has_neg = torch.any(neg_mask, dim=-1)
        
        fallback = torch.min(critical_summits, dim=-1).values
        # Win 55: Zero-Allocation Branch-Free Boundary in PyTorch replacing torch.where
        boundary = min_pos + (min_pos < min_neg).to(min_pos.dtype) * (min_neg - min_pos)
        boundary = fallback + (has_pos & has_neg).to(fallback.dtype) * (boundary - fallback)
        
        trigger = (lam2 < self.tau) & (off_diag_sum > 0.1) & (K > 1)
        # Win 56: Zero-Allocation Branch-Free Boundary Result in PyTorch replacing torch.where
        boundary_result = -1 + trigger.to(boundary.dtype) * (boundary + 1)
        
        # 100% Vectorized Čech coboundary calculation on GPU (Win 52)
        if K < 4:
            # Pad skeleton to at least 4 in last dimension for S_matrix local dimension compatibility
            S_matrix = torch.nn.functional.pad(A_skeleton, (0, 4 - K))
        else:
            S_matrix = A_skeleton[..., :4]
        mask = (W > 0.1)
        if not hasattr(self, "_off_diag_mask_cache"):
            self._off_diag_mask_cache = {}
        if (K, device) not in self._off_diag_mask_cache:
            diag_mask = torch.eye(K, device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
            self._off_diag_mask_cache[(K, device)] = ~diag_mask
        off_diag_mask = self._off_diag_mask_cache[(K, device)]
        mask = mask & off_diag_mask
        
        # Win 126: Diagonal Extraction for S_norm2 (avoiding squaring and summing S_matrix)
        P = torch.matmul(S_matrix, S_matrix.transpose(-1, -2)) # [B_dim, H_dim, K, K]
        S_norm2 = torch.diagonal(P, dim1=-2, dim2=-1) # [B_dim, H_dim, K]
        
        # Win 197: Memory Allocation-Free Čech Sum via 1D/2D operations
        term1 = torch.sum(S_norm2 * mask.sum(dim=-1), dim=-1)
        term2 = torch.sum((W ** 2) * S_norm2.unsqueeze(-2) * mask, dim=(-2, -1))
        term3 = -2.0 * torch.sum(W * P * mask, dim=(-2, -1))
        d0s_norm2 = term1 + term2 + term3
        # Win 87: Avoid redundant squaring of S_matrix by reusing S_norm2
        norm_s2 = torch.sum(S_norm2, dim=-1) # [B_dim, H_dim]
        
        cfi = torch.where(norm_s2 > 1e-9, d0s_norm2 / norm_s2, 0.0)
        cfi = torch.where(has_nan_inf_heads, float('inf'), cfi)
        
        # Win 85: Vectorized Firewall Decision Logic on GPU (avoiding head-loop CPU-GPU transfer)
        overall_cfi = torch.max(cfi, dim=-1).values # [B_dim]
        # Win 79: Arithmetic Boundary Masking in Čech Cohomology Firewall
        masked_boundaries = boundary_result + (boundary_result == -1).to(dtype) * 1000000
        min_boundary = torch.min(masked_boundaries, dim=-1).values # [B_dim]
        
        # Win 141: Packed GPU-to-CPU Cohomology Transfers
        packed_tensor = torch.stack([
            overall_cfi,
            min_boundary.to(dtype=overall_cfi.dtype)
        ], dim=-1) # [B_dim, 2]
        # Win 75: Fast CPU Transfers in PyTorch Cohomology Firewall
        any_fractured = bool(torch.any((min_boundary < 999999) | (overall_cfi > self.threshold)).item())
        packed_cpu = packed_tensor.detach().cpu().tolist()
        
        if any_fractured and M_dim > 0 and N_dim > 1:
            mean_dist = torch.mean(A[..., -1, :].to(torch.float32), dim=1) # [B_dim, N_dim]
            K_alt = min(N_dim, 64)
            topk_alt = torch.topk(mean_dist, K_alt, dim=-1, largest=True, sorted=True)
            sorted_indices_alt_cpu = topk_alt.indices.cpu().tolist()
        else:
            sorted_indices_alt_cpu = [[] for _ in range(B_dim)]
            
        is_fractured_list = []
        cfi_list = []
        alt_idx_list = []
        self.split_boundary = []
        
        for b in range(B_dim):
            cfi_val = float(packed_cpu[b][0])
            boundary_val = float(packed_cpu[b][1])
            boundary_b = int(boundary_val) if boundary_val < 999999 else None
            self.split_boundary.append(boundary_b)
            
            if M_dim > 0 and N_dim > 1 and len(sorted_indices_alt_cpu[b]) > 0:
                alt_idx_b = [int(idx) for idx in sorted_indices_alt_cpu[b][1:] if idx < N_dim]
            else:
                alt_idx_b = []
                
            if boundary_b is not None:
                alt_idx_b = [boundary_b] + alt_idx_b
                
            is_fractured_list.append((boundary_b is not None) or (cfi_val > self.threshold))
            cfi_list.append(cfi_val)
            alt_idx_list.append(alt_idx_b)
            
        if B == 1:
            self.split_boundary = self.split_boundary[0]
            return is_fractured_list[0], cfi_list[0], alt_idx_list[0]
        else:
            return is_fractured_list, cfi_list, alt_idx_list

    def check_obstruction_mlx(self, attn_matrix: Any) -> Tuple[Any, Any, Any]:
        A = attn_matrix
        orig_ndim = A.ndim
        if A.ndim == 1:
            A = mx.expand_dims(mx.expand_dims(mx.expand_dims(A, 0), 0), 0)
        elif A.ndim == 2:
            A = mx.expand_dims(mx.expand_dims(A, 0), 0)
        elif A.ndim == 3:
            A = mx.expand_dims(A, 1)
            
        B_dim, H_dim, M_dim, N_dim = A.shape
        S = max(M_dim, N_dim)
        
        K = min(8, S)
        
        if not hasattr(self, "_eye_cache_mlx"):
            self._eye_cache_mlx = {}
        if K not in self._eye_cache_mlx:
            self._eye_cache_mlx[K] = mx.reshape(mx.eye(K, dtype=mx.bool_), (1, 1, K, K))
        diag_mask = self._eye_cache_mlx[K]
        
        # Execute compiled Core 1
        L, W, degrees, A_skeleton, critical_summits, has_nan_inf_heads = _compiled_firewall_mlx_core_1(A, diag_mask, K)
        
        # Sanitize L to prevent background thread crashes/aborts on NaNs/Infs
        L_clean = mx.where(mx.isnan(L) | mx.isinf(L), mx.array(0.0, dtype=L.dtype), L)
        try:
            eigvals, eigvecs = mx.linalg.eigh(L_clean)
            lam2 = eigvals[..., 1]
            v2 = eigvecs[..., :, 1]
        except Exception:
            lam2 = mx.ones((B_dim, H_dim), dtype=A.dtype)
            v2 = mx.zeros((B_dim, H_dim, K), dtype=A.dtype)
            
        # Execute compiled Core 2
        packed_tensor = _compiled_firewall_mlx_core_2(
            v2, lam2, degrees, critical_summits, W, A_skeleton, has_nan_inf_heads, diag_mask, K, self.tau, self.threshold
        )
        
        min_boundary = packed_tensor[..., 1]
        overall_cfi = packed_tensor[..., 0]
        
        # Win 74: Fast CPU Transfers in MLX Cohomology Firewall
        any_fractured = bool(mx.any((min_boundary < 999999.0) | (overall_cfi > self.threshold)).item())
        packed_cpu = packed_tensor.tolist()
        
        if any_fractured and M_dim > 0 and N_dim > 1:
            mean_dist = mx.mean(A[..., -1, :], axis=1)
            K_alt = min(N_dim, 64)
            sorted_alt_indices = mx.argsort(-mean_dist, axis=-1)
            sorted_indices_alt_cpu = sorted_alt_indices[:, :K_alt].tolist()
        else:
            sorted_indices_alt_cpu = None
            
        is_fractured_list = []
        cfi_list = []
        alt_idx_list = []
        split_boundary = []
        
        for b in range(B_dim):
            cfi_val = float(packed_cpu[b][0])
            boundary_val = float(packed_cpu[b][1])
            boundary_b = int(boundary_val) if boundary_val < 999999.0 else None
            split_boundary.append(boundary_b)
            
            if M_dim > 0 and N_dim > 1 and sorted_indices_alt_cpu is not None and len(sorted_indices_alt_cpu[b]) > 0:
                alt_idx_b = [int(idx) for idx in sorted_indices_alt_cpu[b][1:] if idx < N_dim]
            else:
                alt_idx_b = []
                
            if boundary_b is not None:
                alt_idx_b = [boundary_b] + alt_idx_b
                
            is_fractured_list.append((boundary_val < 999999.0) or (cfi_val > self.threshold))
            cfi_list.append(cfi_val)
            alt_idx_list.append(alt_idx_b)
            
        if orig_ndim <= 2:
            self.split_boundary = split_boundary[0]
            return is_fractured_list[0], cfi_list[0], alt_idx_list[0]
        else:
            self.split_boundary = split_boundary
            return is_fractured_list, cfi_list, alt_idx_list

# Win 93: Compiled Core Functions for MLX Cohomology Firewall
def _firewall_mlx_core_1(A, diag_mask, K):
    B_dim, H_dim, M_dim, N_dim = A.shape
    
    if M_dim == 1:
        vertex_energies = mx.squeeze(A, 2)
    elif M_dim <= N_dim:
        vertex_energies = mx.sum(A, axis=2)
    else:
        vertex_energies = mx.zeros((B_dim, H_dim, M_dim), dtype=A.dtype)
        vertex_energies[..., :N_dim] = mx.sum(A, axis=2)
        
    sorted_indices = mx.argsort(-vertex_energies, axis=-1)
    critical_summits = sorted_indices[..., :K].astype(mx.int32)
    
    clamped_rows = mx.clip(critical_summits, 0, M_dim - 1)
    gather_rows = mx.expand_dims(clamped_rows, axis=-1)
    gather_rows = mx.broadcast_to(gather_rows, (B_dim, H_dim, K, N_dim))
    A_rows = mx.take_along_axis(A, gather_rows, axis=2)
    
    clamped_cols = mx.clip(critical_summits, 0, N_dim - 1)
    col_indices = mx.expand_dims(clamped_cols, axis=-2)
    col_indices = mx.broadcast_to(col_indices, (B_dim, H_dim, K, K))
    A_skeleton = mx.take_along_axis(A_rows, col_indices, axis=3)
    
    mask_rows = mx.expand_dims(critical_summits < M_dim, axis=-1)
    mask_cols = mx.expand_dims(critical_summits < N_dim, axis=-2)
    A_skeleton = A_skeleton * mask_rows * mask_cols
    
    has_nan_inf_gpu = mx.isnan(vertex_energies) | mx.isinf(vertex_energies)
    has_nan_inf_heads = mx.any(has_nan_inf_gpu, axis=2)
    
    W = 0.5 * (A_skeleton + mx.transpose(A_skeleton, (0, 1, 3, 2)))
    degrees = mx.sum(W, axis=-1)
    
    diag_degrees = mx.expand_dims(degrees, axis=-1) * diag_mask
    L = diag_degrees - W
    return L, W, degrees, A_skeleton, critical_summits, has_nan_inf_heads

def _firewall_mlx_core_2(v2, lam2, degrees, critical_summits, W, A_skeleton, has_nan_inf_heads, diag_mask, K, self_tau, self_threshold):
    B_dim, H_dim, K_dim, _ = W.shape
    
    active_mask = degrees > 1e-4
    fill_val = 999999
    
    pos_mask = active_mask & (v2 >= 0.0)
    neg_mask = active_mask & (v2 < 0.0)
    
    pos_summits = critical_summits + (~pos_mask).astype(critical_summits.dtype) * (fill_val - critical_summits)
    neg_summits = critical_summits + (~neg_mask).astype(critical_summits.dtype) * (fill_val - critical_summits)
    
    min_pos = mx.min(pos_summits, axis=-1)
    min_neg = mx.min(neg_summits, axis=-1)
    
    has_pos = mx.any(pos_mask, axis=-1)
    has_neg = mx.any(neg_mask, axis=-1)
    
    fallback = mx.min(critical_summits, axis=-1)
    boundary = min_pos + (min_pos < min_neg).astype(min_pos.dtype) * (min_neg - min_pos)
    boundary = fallback + (has_pos & has_neg).astype(fallback.dtype) * (boundary - fallback)
    
    trace_W = mx.sum(W * diag_mask, axis=(-2, -1))
    sum_W = mx.sum(W, axis=(-2, -1))
    off_diag_sum = sum_W - trace_W
    
    trigger = (lam2 < self_tau) & (off_diag_sum > 0.1) & (K > 1)
    boundary_result = -1 + trigger.astype(boundary.dtype) * (boundary + 1)
    
    if K < 4:
        pad_width = 4 - K
        zeros_pad = mx.zeros((B_dim, H_dim, K, pad_width), dtype=A_skeleton.dtype)
        S_matrix = mx.concatenate([A_skeleton, zeros_pad], axis=-1)
    else:
        S_matrix = A_skeleton[..., :4]
        
    mask = W > 0.1
    mask = mask & (~diag_mask)
    
    P = S_matrix @ mx.transpose(S_matrix, (0, 1, 3, 2))
    S_norm2 = mx.sum(P * diag_mask, axis=-1)
    
    term1 = mx.sum(S_norm2 * mx.sum(mask, axis=-1), axis=-1)
    term2 = mx.sum((W ** 2) * mx.expand_dims(S_norm2, axis=-2) * mask, axis=(-2, -1))
    term3 = -2.0 * mx.sum(W * P * mask, axis=(-2, -1))
    d0s_norm2 = term1 + term2 + term3
    norm_s2 = mx.sum(S_norm2, axis=-1)
    
    cfi = mx.where(norm_s2 > 1e-9, d0s_norm2 / norm_s2, 0.0)
    cfi = mx.where(has_nan_inf_heads, float('inf'), cfi)
    
    overall_cfi = mx.max(cfi, axis=-1)
    masked_boundaries = boundary_result + (boundary_result == -1).astype(boundary_result.dtype) * 1000000
    min_boundary = mx.min(masked_boundaries, axis=-1)
    
    packed_tensor = mx.stack([
        overall_cfi,
        min_boundary.astype(overall_cfi.dtype)
    ], axis=-1)
    return packed_tensor

if mx is not None:
    _compiled_firewall_mlx_core_1 = mx.compile(_firewall_mlx_core_1)
    _compiled_firewall_mlx_core_2 = mx.compile(_firewall_mlx_core_2)


class CohomologicalLayerGating:
    def __init__(self, threshold: float = 0.8, check_interval: int = 4):
        self.threshold = threshold
        self.check_interval = check_interval
        
    def should_exit_early(self, x: Any, x_prev: Any) -> bool:
        # Evaluates Čech Cohomology discrepancy between layer representations
        # Returns True if difference is below threshold, indicating representation convergence
        if x is None or x_prev is None:
            return False
            
        if hasattr(x, "__class__") and x.__class__.__name__ == "array" and x.__class__.__module__ == "mlx.core":
            import mlx.core as mx
            # Compute representation discrepancy in MLX
            diff = mx.mean(mx.square(x - x_prev))
            return bool((diff.item() < self.threshold))
        else:
            # PyTorch fallback
            diff = torch.mean((x - x_prev) ** 2)
            return bool((diff.item() < self.threshold))


class SheafCohomologicalAuditFirewall:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def audit(self, layer_hidden_states: Any) -> bool:
        if not layer_hidden_states or len(layer_hidden_states) < 2:
            return False
            
        h = layer_hidden_states[-1]
        
        if hasattr(h, "__class__") and h.__class__.__name__ == "array" and h.__class__.__module__ == "mlx.core":
            import mlx.core as mx
            B, S, D = h.shape
            if S < 2:
                return False
            diffs = h[:, 1:, :] - h[:, :-1, :]
            discrepancy = mx.mean(mx.square(diffs)).item()
            return bool(discrepancy > self.threshold)
        else:
            B, S, D = h.shape
            if S < 2:
                return False
            diffs = h[:, 1:, :] - h[:, :-1, :]
            discrepancy = torch.mean((diffs) ** 2).item()
            return bool(discrepancy > self.threshold)