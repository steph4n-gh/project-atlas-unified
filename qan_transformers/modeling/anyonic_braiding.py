import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict

class QuantumGroupRMatrix(nn.Module):
    """
    Parametric R-matrix from the Burau representation of the braid group.
    Satisfies the Yang-Baxter relation exactly by construction.
    """
    def __init__(self, d_head: int):
        super().__init__()
        self.d_head = d_head
        # Parameterize t in (0, 1) using sigmoid
        self.raw_t = nn.Parameter(torch.tensor(0.0))  # t = sigmoid(0.0) = 0.5

    @property
    def t(self):
        return torch.sigmoid(self.raw_t)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.t
        h_i_new = (1.0 - t) * h_i + t * h_j
        h_j_new = h_i
        return h_i_new, h_j_new


class BraidGroupTracker:
    """
    Logs and tracks the braid word representations and computes a trace-based diagnostic.
    """
    def __init__(self):
        self.history = []

    def log_braid(self, step: int, t_val: float):
        # Trace of [[1-t, t], [1, 0]] is 1 - t
        trace_val = 1.0 - t_val
        self.history.append({
            "step": step,
            "t": t_val,
            "trace": trace_val
        })
        return trace_val


class BraidedMultiHeadAttention(nn.Module):
    """
    Replaces standard head concatenation with topological braiding using U_q(sl_2) R-matrices.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Learnable R-matrix per adjacent head pair
        self.r_matrices = nn.ModuleList([
            QuantumGroupRMatrix(self.head_dim) for _ in range(num_heads - 1)
        ])
        
        self.tracker = BraidGroupTracker()
        self.step_count = 0

    def forward(self, head_outputs: torch.Tensor) -> torch.Tensor:
        """
        head_outputs: [B, H, S, d_head]
        Returns:
            braided_outputs: [B, H, S, d_head]
        """
        B, H, S, d_head = head_outputs.shape
        assert H == self.num_heads
        
        # Clone to avoid in-place modification issues
        h = [head_outputs[:, i].clone() for i in range(H)]
        
        # Layer 1: Odd pairs (0, 1), (2, 3), ...
        for i in range(0, H - 1, 2):
            r_mat = self.r_matrices[i]
            h[i], h[i+1] = r_mat(h[i], h[i+1])
            
        # Layer 2: Even pairs (1, 2), (3, 4), ...
        for i in range(1, H - 1, 2):
            r_mat = self.r_matrices[i]
            h[i], h[i+1] = r_mat(h[i], h[i+1])
            
        # Stack back to [B, H, S, d_head]
        braided = torch.stack(h, dim=1)
        
        if self.training:
            self.step_count += 1
            mean_t = torch.stack([r.t for r in self.r_matrices]).mean().item()
            self.tracker.log_braid(self.step_count, mean_t)
            
        return braided
