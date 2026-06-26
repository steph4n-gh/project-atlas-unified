import torch
import numpy as np
import pytest
from qan_transformers.modeling.galois_adapter import GaloisAdapterLinear
from qan_transformers.lora.pipeline import GaloisLoRA, inject_lora
from qan_transformers.modeling.attention.base import QuasicrystallineAttention

def test_gf256_arithmetic_pytorch():
    adapter = GaloisAdapterLinear(input_dim=8, output_dim=8, rank=4, num_tasks=4)
    
    # 1. Addition (XOR)
    assert adapter.gf_add(torch.tensor(10), torch.tensor(20)).item() == 30
    assert adapter.gf_add(torch.tensor(0), torch.tensor(255)).item() == 255
    
    # 2. Multiplication properties
    for u in [0, 1, 10, 255]:
        assert adapter.gf_mul(torch.tensor(u), torch.tensor(1)).item() == u
        assert adapter.gf_mul(torch.tensor(1), torch.tensor(u)).item() == u
        
    for u in [0, 1, 10, 255]:
        assert adapter.gf_mul(torch.tensor(u), torch.tensor(0)).item() == 0
        assert adapter.gf_mul(torch.tensor(0), torch.tensor(u)).item() == 0
        
    # Commutativity
    u = torch.tensor([5, 12, 100, 250], dtype=torch.uint8)
    v = torch.tensor([7, 30, 80, 2], dtype=torch.uint8)
    mul_uv = adapter.gf_mul(u, v)
    mul_vu = adapter.gf_mul(v, u)
    assert torch.equal(mul_uv, mul_vu)


def test_lagrange_interpolation_pytorch():
    num_tasks = 4
    adapter = GaloisAdapterLinear(input_dim=8, output_dim=8, rank=4, num_tasks=num_tasks)
    
    # Check that decoding each task reconstructs a weight matrix that corresponds to
    # dequantized version of W_int (which was initialized to 128, meaning 0.0)
    for c in range(num_tasks):
        W_B = adapter.decode_adapter(c)
        assert W_B.shape == (4, 8)
        assert torch.all(torch.abs(W_B) < 1e-5)


def test_galois_adapter_linear_forward_pytorch():
    input_dim = 16
    output_dim = 32
    rank = 4
    num_tasks = 4
    scale = 0.05
    
    adapter = GaloisAdapterLinear(
        input_dim=input_dim, output_dim=output_dim, rank=rank, num_tasks=num_tasks, scale=scale
    )
    
    # Manually configure M to represent custom task weights
    # Set task 2 to 150 (meaning (150-128)*0.05 = 1.1) and task 0 to 100
    W_int = np.ones((num_tasks, rank, output_dim), dtype=np.uint8) * 128
    W_int[2, :, :] = 150
    W_int[0, :, :] = 100
    
    x_pts = list(range(1, num_tasks + 1))
    y_pts = list(range(num_tasks + 1, 2 * num_tasks + 1))
    
    exp_py = adapter.exp_table.tolist()
    log_py = adapter.log_table.tolist()
    
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
            
    adapter.M.copy_(torch.tensor(M_np, dtype=torch.uint8))
    
    # Run forward pass for task 2
    x = torch.randn(2, 10, input_dim)
    out2 = adapter(x, task_idx=2)
    assert out2.shape == (2, 10, output_dim)
    
    # Run forward pass for task 0
    out0 = adapter(x, task_idx=0)
    assert out0.shape == (2, 10, output_dim)
    
    assert not torch.equal(out2, out0)
    
    # Verify exact math for task 2: (150 - 128) * 0.05 = 1.1
    W_B_2 = adapter.decode_adapter(2)
    assert torch.all(torch.abs(W_B_2 - 1.1) < 1e-4)
    
    # Verify exact math for task 0: (100 - 128) * 0.05 = -1.4
    W_B_0 = adapter.decode_adapter(0)
    assert torch.all(torch.abs(W_B_0 - (-1.4)) < 1e-4)


def test_galois_lora_inject_and_train():
    embed_dim = 16
    num_heads = 2
    
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=1.0
    )
    
    # Inject Galois adapters
    inject_lora(attn, r=4, adapter_type="galois", num_tasks=3)
    
    assert isinstance(attn.q_proj, GaloisLoRA)
    assert isinstance(attn.k_proj, GaloisLoRA)
    assert isinstance(attn.v_proj, GaloisLoRA)
    assert isinstance(attn.out_proj, GaloisLoRA)
    
    # Set M to non-128 values so that decoded weights W_B are non-zero,
    # which allows gradients to flow back to A_proj
    with torch.no_grad():
        attn.q_proj.galois_adapter.M.fill_(130)
        attn.k_proj.galois_adapter.M.fill_(130)
        attn.v_proj.galois_adapter.M.fill_(130)
        attn.out_proj.galois_adapter.M.fill_(130)
        
    # Run forward pass
    x = torch.randn(2, 8, embed_dim)
    out = attn(x)
    assert out.shape == (2, 8, embed_dim)
    
    # Check gradient flow
    loss = out.sum()
    loss.backward()
    
    assert attn.q_proj.galois_adapter.A_proj.weight.grad is not None
    assert (attn.q_proj.galois_adapter.A_proj.weight.grad != 0.0).any()
