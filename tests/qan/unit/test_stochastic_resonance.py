import pytest
import mlx.core as mx
import numpy as np
from qan_transformers.mlx.e8_swap import AdelicMemorySwapGridDB

def test_noise_doping_active():
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    db.noise_scale = 0.05
    
    # Coordinates and buffers
    db._grid_coords = db.shell_1_roots[:10]
    db.grid_coords_len = 10
    db.target_len = 10
    db._cpu_k_target_bufs[16] = mx.zeros((10, 16))
    db._cpu_v_target_bufs[16] = mx.zeros((10, 16))
    
    # Query input
    q = mx.random.normal((1, 16))
    
    # Running swap_in target should complete successfully with noise scale > 0
    k_out, v_out = db.swap_in_target(q)
    assert k_out.shape == (8, 16)

def test_sub_threshold_signal_recovery():
    db = AdelicMemorySwapGridDB(d_model=8, d_model_draft=8)
    
    # We will test the _quantize behavior under noise
    # We feed a sub-threshold coordinate vector (all elements = 0.25)
    # Standard rounding to nearest integer maps 0.25 directly to 0.0
    x_sub = mx.full((1, 8), 0.25)
    
    # Undoped quantization
    quant_undoped = db._quantize(x_sub)
    assert mx.all(quant_undoped == 0.0).item()
    
    # Doped quantization over multiple trials to measure average recovery
    trials = 1000
    db.noise_scale = 0.3 # use larger noise to increase crossings
    
    sum_quant = mx.zeros((1, 8))
    for _ in range(trials):
        # Manually dope the query vector
        std = 1.0 # fixed std for test
        noise = mx.random.normal(x_sub.shape)
        x_doped = x_sub + db.noise_scale * std * noise
        sum_quant = sum_quant + db._quantize(x_doped)
        
    avg_quant = sum_quant / trials
    
    # Due to stochastic resonance, average quantized values should be positive
    # representing recovery of the sub-threshold 0.25 signal!
    avg_vals = avg_quant.tolist()[0]
    print("Stochastic Resonance Average Quantized Vals:", avg_vals)
    for val in avg_vals:
        assert val > 0.0
        assert val < 1.0
