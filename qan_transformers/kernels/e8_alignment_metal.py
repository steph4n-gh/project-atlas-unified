import os
import mlx.core as mx

# Load the Metal shader source
METAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e8_alignment.metal")
with open(METAL_FILE, "r") as f:
    METAL_SOURCE = f.read()

# Define the custom MLX Metal kernel
_e8_alignment_kernel = mx.fast.metal_kernel(
    name="e8_alignment_kernel",
    input_names=["inp_8d", "P", "roots"],
    output_names=["alignment_scores", "winning_indices"],
    source=METAL_SOURCE,
    header=""
)

def e8_alignment(inp_8d: mx.array, P: mx.array, roots: mx.array) -> tuple[mx.array, mx.array]:
    """
    Fuses E8 projection, L2 normalization, and cosine similarity checking against roots.
    
    Args:
        inp_8d: Input tensor of shape (..., 8).
        P: Projection matrix of shape (8, 3).
        roots: Dynamic buffer of E8 roots of shape (N, 3).
        
    Returns:
        alignment_scores: Maximum similarity scores of shape (...).
        winning_indices: Index of the winning E8 root of shape (...).
    """
    orig_shape = inp_8d.shape
    orig_dtype = inp_8d.dtype
    
    # Flatten the batch/sequence dimensions for the Metal kernel
    inp_flat = inp_8d.reshape(-1, 8)
    TotalTokens = inp_flat.shape[0]
    N = roots.shape[0]
    
    # Determine the tile size for cooperative caching in threadgroup shared memory.
    # Since E8 root buffers are always multiples of 240, 240 is the optimal tile size.
    TILE_SIZE = 240
    
    # Choose threadgroup size
    tg_x = min(TotalTokens, 256)
    if tg_x <= 0:
        tg_x = 1
        
    # Grid specifies total thread counts in MLX.
    grid_x = TotalTokens
    
    # Ensure default device is GPU for Metal kernel execution
    prev_dev = mx.default_device()
    mx.set_default_device(mx.Device(mx.DeviceType.gpu))
    
    try:
        # Run the kernel
        alignment_scores, winning_indices = _e8_alignment_kernel(
            inputs=[inp_flat, P, roots],
            output_shapes=[[TotalTokens], [TotalTokens]],
            output_dtypes=[orig_dtype, mx.uint32],
            template=[
                ("T", orig_dtype),
                ("N", N),
                ("TotalTokens", TotalTokens),
                ("TILE_SIZE", TILE_SIZE),
                ("TG_SIZE", tg_x)
            ],
            grid=(grid_x, 1, 1),
            threadgroup=(tg_x, 1, 1)
        )
        
        # Reshape back to the original batch/sequence dimensions
        out_shape = orig_shape[:-1]
        if len(out_shape) == 0:
            # Scalar/1D case
            alignment_scores = alignment_scores[0]
            winning_indices = winning_indices[0]
        else:
            alignment_scores = alignment_scores.reshape(out_shape)
            winning_indices = winning_indices.reshape(out_shape)
            
    finally:
        mx.set_default_device(prev_dev)
        
    return alignment_scores, winning_indices
