import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.attention import OctonionicHyperbolicAttention

def test_initialization():
    embed_dim = 64
    num_heads = 4
    attn = OctonionicHyperbolicAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    assert attn.embed_dim == embed_dim
    assert attn.num_heads == num_heads
    assert attn.head_dim == 16
    assert attn.C_oct == 2 # 16 // 8
    
    # Check that error is raised if head_dim is not divisible by 8
    with pytest.raises(ValueError):
        OctonionicHyperbolicAttention(embed_dim=12, num_heads=4) # head_dim = 3

def test_octonionic_hyperbolic_distance_properties():
    # Let's test properties of the custom hyperbolic distance
    embed_dim = 32
    num_heads = 2
    attn = OctonionicHyperbolicAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Generate identical query and key
    x = mx.random.normal((1, 5, embed_dim))
    Q = attn.q_proj(x)
    
    Q = mx.transpose(mx.reshape(Q, (1, 5, 2, 2, 8)), (0, 2, 1, 3, 4))
    
    # Check that distance from a point to itself is exactly 0 (within epsilon clipping tolerance)
    t_Q = mx.sqrt(1.0 + mx.sum(Q**2, axis=-1))
    t_prod = mx.expand_dims(t_Q, 3) * mx.expand_dims(t_Q, 2)
    
    # algebraic product components
    Q_t = mx.transpose(Q, (0, 1, 3, 2, 4))
    real_part = Q_t @ mx.transpose(Q_t, (0, 1, 2, 4, 3))
    real_part = mx.transpose(real_part, (0, 1, 3, 4, 2))
    
    Q_imag = Q[..., 1:]
    A = mx.sum(Q_imag * attn.gamma, axis=-1)
    q_0 = Q[..., 0]
    
    A_exp = mx.expand_dims(A, 3)
    q0_exp = mx.expand_dims(q_0, 3)
    k0_exp = mx.expand_dims(q_0, 2)
    linear_part = k0_exp * A_exp - q0_exp * mx.expand_dims(A, 2)
    
    G = mx.sum(attn.C_str * mx.reshape(attn.gamma, (1, 1, 7)), axis=-1)
    Q_imag_t = mx.transpose(Q_imag, (0, 1, 3, 2, 4))
    Q_g = Q_imag_t @ G
    cross_part = Q_g @ mx.transpose(Q_imag_t, (0, 1, 2, 4, 3))
    cross_part = mx.transpose(cross_part, (0, 1, 3, 4, 2))
    
    lorentz_inner = -t_prod + real_part + linear_part - cross_part
    
    for i in range(5):
        diag_val = lorentz_inner[0, 0, i, i, 0].item()
        assert abs(diag_val + 1.0) < 1e-4
        
        y = mx.maximum(1.0 + 1e-6, -lorentz_inner[0, 0, i, i, 0])
        dist = mx.log(y + mx.sqrt(y**2 - 1.0)).item()
        assert dist < 3e-3

def test_forward_pass_and_caching():
    embed_dim = 64
    num_heads = 4
    attn = OctonionicHyperbolicAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Prefill forward pass
    x = mx.random.normal((2, 10, embed_dim))
    out = attn(x)
    assert out.shape == (2, 10, embed_dim)
    
    # Caching forward pass
    kv_cache = {}
    out_cached = attn(x, kv_cache=kv_cache)
    assert out_cached.shape == (2, 10, embed_dim)
    assert kv_cache["K"].shape == (2, 4, 10, 2, 8)
    assert kv_cache["V"].shape == (2, 4, 10, 16)
    
    # Decode next token
    x_next = mx.random.normal((2, 1, embed_dim))
    out_next = attn(x_next, kv_cache=kv_cache)
    assert out_next.shape == (2, 1, embed_dim)
    assert kv_cache["K"].shape == (2, 4, 11, 2, 8)
    assert kv_cache["V"].shape == (2, 4, 11, 16)
