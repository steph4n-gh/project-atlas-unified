"""
ELQ (E8 Lattice Quantization) fused decode + matmul — MLX Metal kernel launcher.

Computes Y = X @ W^T where W is stored in E8-lattice quantized form (indices + scales).
Optimized using Apple Silicon GPU Tensor Cores (simdgroup_matrix).
"""

import os
import mlx.core as mx

# ── Load Metal shader source ─────────────────────────────────────────────────
METAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elq_decode.metal")
with open(METAL_FILE, "r") as f:
    METAL_SOURCE = f.read()

# ── JIT-compiled Metal kernel handle ─────────────────────────────────────────
_elq_kernel = mx.fast.metal_kernel(
    name="elq_fused_matmul_kernel",
    input_names=["inp", "indices", "scales"],
    output_names=["out"],
    source=METAL_SOURCE,
    header="",
)


def elq_fused_matmul(inp: mx.array, indices: mx.array, scales: mx.array) -> mx.array:
    """Fused E8 lattice dequantization + matrix-vector multiply on Metal.

    Computes ``out = inp @ W^T`` where ``W`` is stored as packed E8 lattice
    indices and per-block scale factors. The computation is fused and hardware-accelerated
    using GPU SIMD-group matrix operations.

    Args:
        inp: Input activations of shape ``(..., D)`` where ``D`` is the model
            dimension. Must be a multiple of 32. dtype can be float16 or float32.
        indices: Packed E8 lattice indices of shape ``(H, num_blocks, 4)``
            stored as uint32. Each uint32 encodes one D8+ lattice sub-block
            of 8 coordinates.
        scales: Per-block scale factors of shape ``(H, num_blocks)`` stored
            as float16. One scale per 32-element block.

    Returns:
        Output activations of shape ``(..., H)``. dtype matches ``inp.dtype``.
    """
    orig_shape = inp.shape
    orig_dtype = inp.dtype
    
    # Force float16 for custom Metal kernel execution to stay within threadgroup memory limit (32KB)
    # and utilize native half-precision GPU acceleration on Apple Silicon.
    if orig_dtype != mx.float16:
        inp = inp.astype(mx.float16)

    D = orig_shape[-1]
    H = indices.shape[0]

    # ── Flatten batch dimensions to 2D ────────────────────────────────────────
    if len(orig_shape) == 1:
        inp_2d = inp[None, :]
    else:
        inp_2d = inp.reshape(-1, D)

    B, D_2d = inp_2d.shape
    assert D == D_2d, "Dimension mismatch between inp and indices"

    # ── Compute grid dimensions ───────────────────────────────────────────────
    # We tile H and B in blocks of 8.
    grid_x = ((H + 7) // 8) * 32
    grid_y = (B + 7) // 8

    # ── Launch the Metal kernel ───────────────────────────────────────────────
    out_2d = _elq_kernel(
        inputs=[inp_2d, indices, scales],
        output_shapes=[[B, H]],
        output_dtypes=[inp_2d.dtype],
        template=[
            ("T", inp_2d.dtype),      # Element type (float16 / float32)
            ("H", H),                 # Output dimension (weight rows)
            ("D", D),                 # Input dimension (weight columns)
            ("B", B),                 # Batch size
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(32, 1, 1),       # 32 threads cooperatively (1 SIMD-group)
    )[0]

    # ── Restore original batch shape and precision ────────────────────────────
    if len(orig_shape) == 1:
        res = out_2d[0]
    else:
        res = out_2d.reshape(orig_shape[:-1] + (H,))
        
    if res.dtype != orig_dtype:
        res = res.astype(orig_dtype)
    return res


# ── Load geglu Metal shader source ───────────────────────────────────────────
METAL_GEGLU_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elq_geglu.metal")
with open(METAL_GEGLU_FILE, "r") as f:
    METAL_GEGLU_SOURCE = f.read()

# ── JIT-compiled Fused GeGLU Metal kernel handle ──────────────────────────────
_elq_geglu_kernel = mx.fast.metal_kernel(
    name="elq_fused_geglu_kernel",
    input_names=["inp", "gate_indices", "gate_scales", "up_indices", "up_scales"],
    output_names=["out"],
    source=METAL_GEGLU_SOURCE,
    header="",
)


def elq_fused_gate_up(
    inp: mx.array,
    gate_indices: mx.array,
    gate_scales: mx.array,
    up_indices: mx.array,
    up_scales: mx.array,
) -> mx.array:
    """Fused E8 lattice dequantization and matmuls for both gate and up projections.

    Computes a concatenated tensor [B, 2 * H] where the first half contains
    inp @ W_gate^T and the second half contains inp @ W_up^T.
    """
    orig_shape = inp.shape
    orig_dtype = inp.dtype
    
    if orig_dtype != mx.float16:
        inp = inp.astype(mx.float16)

    D = orig_shape[-1]
    H = gate_indices.shape[0]

    if len(orig_shape) == 1:
        inp_2d = inp[None, :]
    else:
        inp_2d = inp.reshape(-1, D)

    B, D_2d = inp_2d.shape
    assert D == D_2d, "Dimension mismatch between inp and gate_indices"

    grid_x = ((H + 7) // 8) * 32
    grid_y = (B + 7) // 8

    out_2d = _elq_geglu_kernel(
        inputs=[inp_2d, gate_indices, gate_scales, up_indices, up_scales],
        output_shapes=[[B, 2 * H]],
        output_dtypes=[inp_2d.dtype],
        template=[
            ("T", inp_2d.dtype),
            ("H", H),
            ("D", D),
            ("B", B),
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(32, 1, 1),
    )[0]

    if len(orig_shape) == 1:
        res = out_2d[0]
    else:
        res = out_2d.reshape(orig_shape[:-1] + (2 * H,))
        
    if res.dtype != orig_dtype:
        res = res.astype(orig_dtype)
    return res
