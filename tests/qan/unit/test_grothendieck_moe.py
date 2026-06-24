import mlx.core as mx
import numpy as np
import pytest
from qan_transformers.mlx.attention import GrothendieckTopologyMoE

def test_grothendieck_moe_shapes():
    B = 2
    S = 8
    D = 16
    E = 4
    
    layer = GrothendieckTopologyMoE(embed_dim=D, num_experts=E, hidden_dim=32, overlap_dim=8)
    x = mx.random.normal((B, S, D))
    
    out = layer(x)
    assert out.shape == (B, S, D)

def test_grothendieck_moe_gluing_consistency():
    B = 1
    S = 2
    D = 8
    E = 3
    
    # Initialize the layer
    layer = GrothendieckTopologyMoE(embed_dim=D, num_experts=E, hidden_dim=16, overlap_dim=4)
    x = mx.random.normal((B, S, D))
    
    # 1. Capture intermediate components manually to verify correct mathematical execution
    router_scores = layer.router(x)
    w = mx.softmax(router_scores / layer.temperature, axis=-1)
    
    Y_list = []
    O_list = []
    for k in range(E):
        y_k = getattr(layer, f"expert_{k}")(x)
        o_k = getattr(layer, f"restriction_{k}")(y_k)
        Y_list.append(mx.expand_dims(y_k, axis=2))
        O_list.append(mx.expand_dims(o_k, axis=2))
        
    Y = mx.concatenate(Y_list, axis=2) # (B, S, E, D)
    O = mx.concatenate(O_list, axis=2) # (B, S, E, d_overlap)
    
    # Manual loop calculation for gluing correction
    manual_corrections = []
    for k in range(E):
        sum_disp = mx.zeros((B, S, layer.overlap_dim))
        for j in range(E):
            disp = O[:, :, j] - O[:, :, k]
            sum_disp = sum_disp + mx.expand_dims(w[:, :, j], axis=-1) * disp
        corr_k = layer.glue_proj(sum_disp)
        manual_corrections.append(mx.expand_dims(corr_k, axis=2))
        
    manual_corrections = mx.concatenate(manual_corrections, axis=2)
    Y_corrected_manual = Y + manual_corrections
    
    w_k = mx.expand_dims(w, axis=-1)
    out_manual = mx.sum(w_k * Y_corrected_manual, axis=2)
    
    # Call layer forward
    out_layer = layer(x)
    
    # Verify correctness of vectorized implementation
    assert mx.allclose(out_layer, out_manual, atol=1e-5)
