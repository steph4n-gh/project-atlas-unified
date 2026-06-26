import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

class TropicalSemiring:
    """
    Implements the tropical (max-plus) semiring:
      - Tropical addition: a ⊕ b = max(a, b)
      - Tropical multiplication: a ⊗ b = a + b
      - Tropical matrix multiplication: (A ⊗ B)_{ij} = max_k (A_{ik} + B_{kj})
      - Tropical determinant: tdet(A) = max_σ Σ_i A_{i,σ(i)}
    """
    @staticmethod
    def tropical_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    @staticmethod
    def tropical_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    @staticmethod
    def tropical_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_exp = A.unsqueeze(-1)      # [..., m, k, 1]
        B_exp = B.unsqueeze(-3)      # [..., 1, k, n]
        products = A_exp + B_exp     # [..., m, k, n]
        result = products.max(dim=-2).values  # [..., m, n]
        return result

    @staticmethod
    def tropical_determinant(A: torch.Tensor) -> torch.Tensor:
        n = A.shape[-1]
        assert A.shape[-2] == n, "Matrix must be square"
        if n <= 8:
            perms = list(itertools.permutations(range(n)))
            row_indices = torch.arange(n, device=A.device)
            max_val = torch.tensor(float('-inf'), device=A.device)
            for perm in perms:
                col_indices = torch.tensor(perm, device=A.device)
                val = A[row_indices, col_indices].sum()
                max_val = torch.maximum(max_val, val)
            return max_val
        else:
            remaining_cols = list(range(n))
            total = torch.tensor(0.0, device=A.device)
            for i in range(n):
                col_vals = A[i, remaining_cols]
                best_idx = col_vals.argmax().item()
                total = total + col_vals[best_idx]
                remaining_cols.pop(best_idx)
            return total

    @staticmethod
    def tropical_rank(A: torch.Tensor) -> int:
        m, n = A.shape
        A_np = A.detach().cpu().numpy()
        rank = 0
        approx = np.full_like(A_np, -1e30)
        tolerance = np.max(np.abs(A_np)) * 0.01
        
        for _ in range(min(m, n)):
            residual = A_np - approx
            residual = np.where(approx > -1e20, residual, A_np)
            best_err = float('inf')
            best_rank1 = None
            
            for i in range(m):
                u = A_np[i, :]
                v = np.max(A_np - u[None, :], axis=1)
                rank1 = v[:, None] + u[None, :]
                err = np.mean(np.abs(A_np - np.maximum(approx, rank1)))
                if err < best_err:
                    best_err = err
                    best_rank1 = rank1
            
            if best_rank1 is not None:
                approx = np.maximum(approx, best_rank1)
                rank += 1
                max_residual = np.max(np.abs(A_np - approx))
                if max_residual < tolerance:
                    break
        return rank


class TropicalAttentionAnalyzer:
    """
    Analyzes attention routing matrices from a tropical geometric perspective.
    """
    def compute_tropical_variety(self, S: torch.Tensor) -> dict:
        n = S.shape[0]
        S_np = S.detach().cpu().numpy()
        variety_points = []
        boundary_pairs = []
        boundary_gaps = []
        
        for i in range(n):
            row = S_np[i]
            sorted_vals = np.sort(row)[::-1]
            max_val = sorted_vals[0]
            second_val = sorted_vals[1] if n > 1 else float('-inf')
            gap = max_val - second_val
            boundary_gaps.append(gap)
            
            score_range = max(max_val - sorted_vals[-1], 1e-6)
            tol = max(score_range * 0.05, 1e-6)
            max_positions = np.where(np.abs(row - max_val) < tol)[0]
            
            if len(max_positions) >= 2:
                variety_points.append(i)
                for p1, p2 in itertools.combinations(max_positions, 2):
                    boundary_pairs.append((i, p1, p2))
                    
        boundary_fraction = len(variety_points) / n if n > 0 else 0
        mean_gap = float(np.mean(boundary_gaps)) if boundary_gaps else 0.0
        
        return {
            'boundary_fraction': boundary_fraction,
            'mean_gap': mean_gap,
            'n_boundary_rows': len(variety_points),
            'n_boundary_pairs': len(boundary_pairs)
        }

    def compute_newton_polytope_analysis(self, S: torch.Tensor) -> dict:
        n = S.shape[0]
        S_np = S.detach().cpu().numpy()
        sparsity_per_row = []
        
        for i in range(n):
            row = S_np[i]
            sorted_vals = np.sort(row)[::-1]
            max_val = sorted_vals[0]
            gap_threshold = max_val - (max_val - sorted_vals[-1]) * 0.1
            support_size = np.sum(row >= gap_threshold)
            sparsity_per_row.append(1.0 - support_size / n)
            
        return {
            'mean_newton_sparsity': float(np.mean(sparsity_per_row))
        }


class AdaptiveTropicalTemperature(nn.Module):
    """
    Dynamically adapts attention scaling temperature T to keep the average
    routing gap at the tropical critical point (target_gap = 1.0).
    """
    def __init__(self, init_temp=0.78, target_gap=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.target_gap = target_gap

    def _routing_gap(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.shape[-1] < 2:
            return torch.ones_like(scores[..., 0])
        top2 = torch.topk(scores, 2, dim=-1).values
        gap = top2[..., 0] - top2[..., 1]
        return gap

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # During inference, use dynamic routing gap to adapt scale
            gap = self._routing_gap(scores)  # [B, H, S]
            mean_gap = gap.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, H, 1, 1]
            mean_gap = torch.clamp(mean_gap, min=1e-6)
            T = mean_gap / self.target_gap
            T = torch.clamp(T, min=0.1, max=5.0)
        else:
            # During training, learn T via backpropagation
            T = torch.clamp(self.temperature, min=0.1, max=5.0)
            
        return scores / T
