import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import pytest

from qan_transformers.math.e8_projection import ConwaySloaneE8Decoder
from qan_transformers.mlx.e8_swap import ConwaySloaneE8DecoderMLX, AdelicMemorySwapGridDB
from qan_transformers.mlx.attention import QuasicrystallineAttention
from qan_transformers.mlx.modeling import graft_mlx_model

def test_mlx_e8_decoder_equivalence():
    """
    Verifies that ConwaySloaneE8DecoderMLX behaves identically to ConwaySloaneE8Decoder.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate random input
    x_np = np.random.normal(size=(5, 10, 8)).astype(np.float32)
    
    # Run PyTorch decoder
    x_torch = torch.from_numpy(x_np)
    torch_decoder = ConwaySloaneE8Decoder()
    y_torch = torch_decoder.decode(x_torch).numpy()
    
    # Run MLX decoder
    x_mlx = mx.array(x_np)
    mlx_decoder = ConwaySloaneE8DecoderMLX()
    y_mlx = np.array(mlx_decoder.decode(x_mlx))
    
    # Check max absolute difference
    diff = np.max(np.abs(y_torch - y_mlx))
    assert diff < 1e-4, f"E8 decoders are not mathematically equivalent. Max diff: {diff}"

def test_mlx_swap_grid_database():
    """
    Verifies swap out and swap in operations on the MLX Adelic memory swap database.
    """
    db = AdelicMemorySwapGridDB(d_model=16, cache_limit_ratio=0.15)
    
    # Create matching coordinates and key-value vectors
    # Let's generate Shell 1 roots (240 vectors in E8 lattice)
    from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates
    shell_1_roots_np = generate_dynamic_e8_coordinates(1).astype(np.float32)
    
    # Map back to feature dimension 16 using a pseudo-inverse/projection
    W_p = np.ones((16, 8), dtype=np.float32)
    W_p = W_p / (np.linalg.norm(W_p, axis=0, keepdims=True) + 1e-6)
    
    # Keys/values corresponding exactly to E8 roots
    # Since keys @ W_p gives the projected coordinates, we can construct keys
    # that map directly to Shell 1 roots:
    # Keys shape: [240, 16]
    keys_np = shell_1_roots_np @ W_p.T
    values_np = keys_np * 2.0
    
    keys = mx.array(keys_np)
    values = mx.array(values_np)
    
    # Swap out to unified database
    db.swap_out(keys, values)
    assert db.grid_coords.shape == (240, 8)
    assert db.cpu_k.shape == (240, 16)
    assert db.cpu_v.shape == (240, 16)
    
    # Swap in a matching neighborhood query
    # A query exactly at root 0 should match root 0 and its E8 neighbors (distance squared <= 2.05)
    queries = mx.array(keys_np[0:1])  # [1, 16]
    retrieved_k, retrieved_v = db.swap_in(queries)
    
    assert retrieved_k.shape[0] > 0
    assert retrieved_k.shape[1] == 16
    assert retrieved_v.shape[0] == retrieved_k.shape[0]
    
    # Batch swap in check
    batched_queries = mx.reshape(queries, (1, 1, 1, 16)) # [B, H, S, D]
    ret_k, ret_v = db.swap_in_batch(batched_queries, max_matches=8)
    assert ret_k.shape == (1, 1, 8, 16)
    assert ret_v.shape == (1, 1, 8, 16)

class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 4
        self.n_kv_heads = 4
        self.head_dim = 16
        self.scale = 16**-0.5
        
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        
        # Simple RoPE wrapper
        self.rope = lambda x, offset=0: x

    def __call__(self, x, mask=None, cache=None):
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return self.o_proj(queries)

class MockTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MockAttention()

def test_mlx_model_grafting():
    """
    Verifies that graft_mlx_model correctly replaces attention layers with QuasicrystallineAttention
    and runs a successful forward pass.
    """
    model = MockTransformer()
    
    # Verify pre-graft layer
    assert isinstance(model.attn, MockAttention)
    
    # Graft
    model = graft_mlx_model(model, sparse_ratio=0.15)
    
    # Verify post-graft layer
    assert isinstance(model.attn, QuasicrystallineAttention)
    assert model.attn.q_proj.weight.shape == (64, 64)
    assert model.attn.e8_proj.weight.shape == (8, 64)
    
    # Run a forward pass
    x = mx.random.normal((1, 10, 64))
    out = model.attn(x)
    assert out.shape == (1, 10, 64)
