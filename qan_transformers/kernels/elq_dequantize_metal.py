import os
import mlx.core as mx

# Read the Metal source code
METAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elq_dequantize.metal")
with open(METAL_FILE, "r") as f:
    METAL_SOURCE = f.read()

# Build the MLX dequantization kernel
_elq_dequant_kernel = mx.fast.metal_kernel(
    name="elq_dequantize_kernel",
    input_names=["indices", "scales"],
    output_names=["out"],
    source=METAL_SOURCE,
    header=""
)

def elq_dequantize_weights(indices: mx.array, scales: mx.array) -> mx.array:
    """
    Dequantizes E8-lattice indices and scales into a Float16 weight matrix of shape [H, D].
    Args:
        indices: packed E8 indices array of shape [H, num_blocks, 4]
        scales: scale factors array of shape [H, num_blocks]
    Returns:
        out: dequantized weight matrix of shape [H, D]
    """
    H = indices.shape[0]
    num_blocks = indices.shape[1]
    D = num_blocks * 32
    
    prev_dev = mx.default_device()
    mx.set_default_device(mx.Device(mx.DeviceType.gpu))
    try:
        out = _elq_dequant_kernel(
            inputs=[indices, scales],
            output_shapes=[[H, D]],
            output_dtypes=[mx.float16],
            template=[
                ("H", H),
                ("D", D),
                ("num_blocks", num_blocks)
            ],
            grid=(num_blocks * 4, H, 1),
            threadgroup=(32, 8, 1)
        )[0]
    finally:
        mx.set_default_device(prev_dev)
    
    return out
