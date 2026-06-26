import mlx.core as mx
import numpy as np

def test_metal_kernels_equivalence():
    B = 2
    H = 64
    D = 128
    num_blocks = D // 32
    
    np.random.seed(42)
    
    # Test with Float16 inputs/outputs
    inp = mx.array(np.random.normal(size=(B, D)).astype(np.float16))
    indices_np = np.random.randint(0, 2**32 - 1, size=(H, num_blocks, 4), dtype=np.uint32)
    indices = mx.array(indices_np)
    scales_np = np.random.uniform(0.1, 1.0, size=(H, num_blocks)).astype(np.float16)
    scales = mx.array(scales_np)
    
    from qan_transformers.kernels.elq_dequantize_metal import elq_dequantize_weights
    dequantized_weights = elq_dequantize_weights(indices, scales)
    
    from qan_transformers.kernels.elq_metal import elq_fused_matmul
    actual_out = elq_fused_matmul(inp, indices, scales)
    
    W_T = mx.transpose(dequantized_weights)
    expected_out = mx.matmul(inp, W_T)
    
    max_diff = mx.max(mx.abs(actual_out - expected_out)).item()
    # atol=0.08 is required due to accumulated float16 precision limits
    assert max_diff < 0.08, f"Fused matmul result differs from reference by {max_diff}"

def test_fused_geglu_equivalence():
    B = 1
    H = 64
    D = 128
    num_blocks = D // 32
    
    np.random.seed(42)
    
    inp = mx.array(np.random.normal(size=(B, D)).astype(np.float16))
    
    # Gate weights
    gate_indices = mx.array(np.random.randint(0, 2**32 - 1, size=(H, num_blocks, 4), dtype=np.uint32))
    gate_scales = mx.array(np.random.uniform(0.1, 1.0, size=(H, num_blocks)).astype(np.float16))
    
    # Up weights
    up_indices = mx.array(np.random.randint(0, 2**32 - 1, size=(H, num_blocks, 4), dtype=np.uint32))
    up_scales = mx.array(np.random.uniform(0.1, 1.0, size=(H, num_blocks)).astype(np.float16))
    
    from qan_transformers.kernels.elq_dequantize_metal import elq_dequantize_weights
    dequantized_gate = elq_dequantize_weights(gate_indices, gate_scales)
    dequantized_up = elq_dequantize_weights(up_indices, up_scales)
    
    # Expected output
    gate_expected = mx.matmul(inp, dequantized_gate.T)
    up_expected = mx.matmul(inp, dequantized_up.T)
    
    # Fused output
    from qan_transformers.kernels.elq_metal import elq_fused_gate_up
    actual_gate_up = elq_fused_gate_up(inp, gate_indices, gate_scales, up_indices, up_scales)
    
    actual_gate = actual_gate_up[..., :H]
    actual_up = actual_gate_up[..., H:]
    
    max_diff_gate = mx.max(mx.abs(actual_gate - gate_expected)).item()
    max_diff_up = mx.max(mx.abs(actual_up - up_expected)).item()
    
    assert max_diff_gate < 0.08, f"Fused gate result differs by {max_diff_gate}"
    assert max_diff_up < 0.08, f"Fused up result differs by {max_diff_up}"
