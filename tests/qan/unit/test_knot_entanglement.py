import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from qan_transformers.mlx.attention import KnotEntanglementAttention

def test_knot_entanglement_shapes():
    B = 2
    S = 8
    D = 16
    H = 4
    
    layer = KnotEntanglementAttention(embed_dim=D, num_heads=H)
    x = mx.random.normal((B, S, D))
    
    # Prefill mode
    out = layer(x)
    assert out.shape == (B, S, D)

def test_knot_entanglement_equivalence():
    B = 1
    S = 6
    D = 8
    H = 2
    
    layer = KnotEntanglementAttention(embed_dim=D, num_heads=H)
    x = mx.random.normal((B, S, D))
    
    # 1. Forward pass to get projected values
    t = mx.transpose(mx.sigmoid(layer.t_proj(x)) + 1e-4, (0, 2, 1)) # (B, H, S)
    P = mx.cumprod(t, axis=-1)
    S_sum = mx.cumsum(P, axis=-1)
    
    t_np = np.array(t[0, 0])      # (S,)
    P_np = np.array(P[0, 0])      # (S,)
    S_sum_np = np.array(S_sum[0, 0]) # (S,)
    
    def get_P(idx):
        return P_np[idx] if idx >= 0 else 1.0
    def get_S(idx):
        return S_sum_np[idx] if idx >= 0 else 0.0
        
    manual_dets = np.zeros((S, S))
    for i in range(S):
        for j in range(S):
            if j <= i:
                if j == i:
                    manual_dets[i, j] = 1.0
                else:
                    manual_dets[i, j] = 1.0 + (get_S(i-1) - get_S(j-1)) / get_P(j-1)
                    
    # Now run the vectorized version in layer
    S_pad = mx.concatenate([mx.zeros((B, H, 1)), S_sum], axis=-1)
    P_pad = mx.concatenate([mx.ones((B, H, 1)), P], axis=-1)
    
    S_i = S_pad[..., :S]
    S_j = S_pad[..., :S]
    P_j = P_pad[..., :S]
    
    S_i_exp = mx.expand_dims(S_i, axis=-1)
    S_j_exp = mx.expand_dims(S_j, axis=-2)
    P_j_exp = mx.expand_dims(P_j, axis=-2)
    
    dets = 1.0 + (S_i_exp - S_j_exp) / mx.clip(P_j_exp, 1e-12, None)
    dets_np = np.array(dets[0, 0]) # (S, S)
    
    # Verify causality lower-triangular equivalence
    for i in range(S):
        for j in range(i + 1):
            assert abs(dets_np[i, j] - manual_dets[i, j]) < 1e-5

def test_knot_entanglement_cache():
    B = 2
    S_prefill = 8
    S_decode = 3
    D = 16
    H = 4
    
    layer = KnotEntanglementAttention(embed_dim=D, num_heads=H)
    
    # 1. Prefill
    x_prefill = mx.random.normal((B, S_prefill, D))
    cache = {}
    prefill_mask = nn.MultiHeadAttention.create_additive_causal_mask(S_prefill)
    out_prefill = layer(x_prefill, mask=prefill_mask, kv_cache=cache)
    
    assert "K" in cache
    assert "V" in cache
    assert "t" in cache
    assert cache["t"].shape == (B, H, S_prefill)
    assert cache["P"].shape == (B, H, S_prefill)
    assert cache["S_sum"].shape == (B, H, S_prefill)
    
    # 2. Sequential Decoding Step by Step
    x_seq = mx.random.normal((B, S_decode, D))
    
    # Run sequential updates
    outputs_seq = []
    for i in range(S_decode):
        x_step = x_seq[:, i:i+1, :]
        out_step = layer(x_step, kv_cache=cache)
        outputs_seq.append(out_step)
    
    out_seq_concat = mx.concatenate(outputs_seq, axis=1)
    
    # Run full input at once with cache to compare
    cache_full = {}
    prefill_mask_full = nn.MultiHeadAttention.create_additive_causal_mask(S_prefill)
    layer(x_prefill, mask=prefill_mask_full, kv_cache=cache_full)
    
    # Construct extended causal mask for multi-token decoding step
    zeros_mask = mx.zeros((S_decode, S_prefill))
    decode_mask_suffix = nn.MultiHeadAttention.create_additive_causal_mask(S_decode)
    extended_mask = mx.concatenate([zeros_mask, decode_mask_suffix], axis=-1)
    
    out_full = layer(x_seq, mask=extended_mask, kv_cache=cache_full)
    
    # They should produce equivalent outputs
    assert mx.allclose(out_seq_concat, out_full, atol=1e-5)
    assert cache["t"].shape == (B, H, S_prefill + S_decode)
