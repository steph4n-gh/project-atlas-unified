import mlx.core as mx
import numpy as np
import pytest
from qan_transformers.mlx.attention import GaloisAdapterLinear

def test_gf256_arithmetic():
    # Instantiate a small GaloisAdapterLinear to test its GF methods
    adapter = GaloisAdapterLinear(input_dim=8, output_dim=8, rank=4, num_tasks=4)
    
    # 1. Addition (XOR)
    assert adapter.gf_add(mx.array(10), mx.array(20)).item() == 30
    assert adapter.gf_add(mx.array(0), mx.array(255)).item() == 255
    
    # 2. Multiplication properties
    # u * 1 = u
    for u in [0, 1, 10, 255]:
        assert adapter.gf_mul(mx.array(u), mx.array(1)).item() == u
        assert adapter.gf_mul(mx.array(1), mx.array(u)).item() == u
        
    # u * 0 = 0
    for u in [0, 1, 10, 255]:
        assert adapter.gf_mul(mx.array(u), mx.array(0)).item() == 0
        assert adapter.gf_mul(mx.array(0), mx.array(u)).item() == 0
        
    # Commutativity: u * v = v * u
    u = mx.array([5, 12, 100, 250], dtype=mx.int32)
    v = mx.array([7, 30, 80, 2], dtype=mx.int32)
    mul_uv = adapter.gf_mul(u, v)
    mul_vu = adapter.gf_mul(v, u)
    assert mx.array_equal(mul_uv, mul_vu).item()

def test_lagrange_interpolation():
    num_tasks = 4
    adapter = GaloisAdapterLinear(input_dim=8, output_dim=8, rank=4, num_tasks=num_tasks)
    
    # Check that decoding each task reconstructs a weight matrix that corresponds to the
    # dequantized version of W_int (which was initialized to 128, meaning float 0.0)
    for c in range(num_tasks):
        W_B = adapter.decode_adapter(c)
        assert W_B.shape == (4, 8)
        # Since it was initialized to 128 (meaning 0.0 float), dequantized should be zero
        assert mx.all(mx.abs(W_B) < 1e-5).item()

def test_galois_adapter_linear_forward():
    # Setup adapter
    input_dim = 16
    output_dim = 32
    rank = 4
    num_tasks = 4
    scale = 0.05
    
    adapter = GaloisAdapterLinear(
        input_dim=input_dim, output_dim=output_dim, rank=rank, num_tasks=num_tasks, scale=scale
    )
    
    # Modify M to represent custom task weights
    # Let's set task 2's decoded weight matrix to be non-zero
    # To do this, let's manually write W_int and encode it to self.M
    # Let's set task 2's weights to 150 (which is (150-128)*0.05 = 1.1) and task 0's weights to 100
    W_int = np.ones((num_tasks, rank, output_dim), dtype=np.int32) * 128
    W_int[2, :, :] = 150 # task 2
    W_int[0, :, :] = 100 # task 0
    
    # Encode W_int using the Vandermonde-Lagrange encoding
    # We can fetch encoding matrix from adapter's precomputed points
    x_pts = list(range(1, num_tasks + 1))
    y_pts = list(range(num_tasks + 1, 2 * num_tasks + 1))
    
    # Helper python arithmetic using adapter's exp/log tables
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
        
    v_np = np.zeros((num_tasks, num_tasks), dtype=np.int32)
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
            
    M_np = np.zeros((num_tasks, rank, output_dim), dtype=np.int32)
    for i in range(num_tasks):
        val_mat = np.zeros((rank, output_dim), dtype=np.int32)
        for c in range(num_tasks):
            v_expanded = v_np[i, c]
            for r in range(rank):
                for d_idx in range(output_dim):
                    term = gf_mul_py(v_expanded, W_int[c, r, d_idx])
                    val_mat[r, d_idx] = gf_add_py(val_mat[r, d_idx], term)
            M_np[i] = val_mat
            
    adapter.M = mx.array(M_np, dtype=mx.int32)
    
    # Run forward pass for task 2
    x = mx.random.normal((2, 10, input_dim))
    out2 = adapter(x, task_idx=2)
    assert out2.shape == (2, 10, output_dim)
    
    # Run forward pass for task 0
    out0 = adapter(x, task_idx=0)
    assert out0.shape == (2, 10, output_dim)
    
    # Verify they are different (zero-crosstalk reconstruction works)
    assert not mx.array_equal(out2, out0).item()
    
    # Verify exact math for task 2
    # W_B dequantized should be 1.1 (since 150 - 128 = 22, 22 * 0.05 = 1.1)
    W_B_2 = adapter.decode_adapter(2)
    assert mx.all(mx.abs(W_B_2 - 1.1) < 1e-4).item()
    
    # W_B dequantized for task 0 should be -1.4 (since 100 - 128 = -28, -28 * 0.05 = -1.4)
    W_B_0 = adapter.decode_adapter(0)
    assert mx.all(mx.abs(W_B_0 - (-1.4)) < 1e-4).item()
