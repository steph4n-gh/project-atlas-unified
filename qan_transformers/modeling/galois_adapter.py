import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaloisAdapterLinear(nn.Module):
    """
    Galois-theoretic adapter layer using GF(2^8) arithmetic.
    Decomposes adapter weights via finite fields and Lagrange interpolation.
    """
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, num_tasks: int = 4, scale: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.num_tasks = num_tasks
        self.scale = scale
        
        # 1. Precompute GF(2^8) tables
        exp_table_np = np.zeros(512, dtype=np.int32)
        log_table_np = np.zeros(256, dtype=np.int32)
        val = 1
        for i in range(255):
            exp_table_np[i] = val
            log_table_np[val] = i
            val <<= 1
            if val & 256:
                val ^= 285
        for i in range(255, 512):
            exp_table_np[i] = exp_table_np[i - 255]
            
        self.register_buffer("exp_table", torch.tensor(exp_table_np, dtype=torch.uint8))
        self.register_buffer("log_table", torch.tensor(log_table_np, dtype=torch.uint8))
        
        # 2. Setup A_proj
        self.A_proj = nn.Linear(input_dim, rank, bias=False)
        
        # 3. Setup points and Lagrange weights
        x_pts = list(range(1, num_tasks + 1))
        y_pts = list(range(num_tasks + 1, 2 * num_tasks + 1))
        
        exp_py = exp_table_np.tolist()
        log_py = log_table_np.tolist()
        
        def gf_add_py(u, v):
            return u ^ v
        def gf_mul_py(u, v):
            if u == 0 or v == 0:
                return 0
            return exp_py[(log_py[u] + log_py[v]) % 255]
        def gf_div_py(u, v):
            if u == 0:
                return 0
            return exp_py[(log_py[u] - log_py[v] + 255) % 255]
            
        lagrange_weights_np = np.zeros((num_tasks, num_tasks), dtype=np.uint8)
        for c in range(num_tasks):
            for i in range(num_tasks):
                w_val = 1
                for j in range(num_tasks):
                    if j != i:
                        num = gf_add_py(y_pts[c], x_pts[j])
                        den = gf_add_py(x_pts[i], x_pts[j])
                        term = gf_div_py(num, den)
                        w_val = gf_mul_py(w_val, term)
                lagrange_weights_np[c, i] = w_val
                
        self.register_buffer("lagrange_weights", torch.tensor(lagrange_weights_np, dtype=torch.uint8))
        
        # Precompute encoding weights matrix v_np
        v_np = np.zeros((num_tasks, num_tasks), dtype=np.uint8)
        for i in range(num_tasks):
            for c in range(num_tasks):
                v_val = 1
                for d in range(num_tasks):
                    if d != c:
                        num = gf_add_py(x_pts[i], y_pts[d])
                        den = gf_add_py(y_pts[c], y_pts[d])
                        term = gf_div_py(num, den)
                        v_val = gf_mul_py(v_val, term)
                v_np[i, c] = v_val
                
        # Initialize original W to 128 (corresponding to float 0.0)
        W_int = np.ones((num_tasks, rank, output_dim), dtype=np.uint8) * 128
        
        # Encode W_int to M
        M_np = np.zeros((num_tasks, rank, output_dim), dtype=np.uint8)
        for i in range(num_tasks):
            val_mat = np.zeros((rank, output_dim), dtype=np.uint8)
            for c in range(num_tasks):
                v_expanded = v_np[i, c]
                for r in range(rank):
                    for d_idx in range(output_dim):
                        term = gf_mul_py(v_expanded, W_int[c, r, d_idx])
                        val_mat[r, d_idx] = gf_add_py(val_mat[r, d_idx], term)
            M_np[i] = val_mat
            
        self.M = nn.Parameter(torch.tensor(M_np, dtype=torch.uint8), requires_grad=False)
        
    def gf_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return u ^ v
        
    def gf_mul(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        log_u = self.log_table[u.long()].long()
        log_v = self.log_table[v.long()].long()
        exp_idx = (log_u + log_v) % 255
        prod = self.exp_table[exp_idx]
        return torch.where((u == 0) | (v == 0), torch.zeros_like(prod), prod)
        
    def decode_adapter(self, task_idx: int) -> torch.Tensor:
        w_target = self.lagrange_weights[task_idx]
        w_expanded = w_target.view(self.num_tasks, 1, 1)
        
        term = self.gf_mul(w_expanded, self.M)
        
        decoded = term[0]
        for idx in range(1, self.num_tasks):
            decoded = decoded ^ term[idx]
            
        # Standard floating-point dequantization: (x - 128) * scale
        W_float = (decoded.float() - 128.0) * self.scale
        return W_float
        
    def forward(self, x: torch.Tensor, task_idx: int = 0) -> torch.Tensor:
        r = self.A_proj(x)
        W_B = self.decode_adapter(task_idx).to(device=x.device, dtype=x.dtype)
        return torch.matmul(r, W_B)
