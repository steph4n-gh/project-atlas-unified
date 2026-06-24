"""
ELQ (E8 Lattice Quantization) fused decode + matmul — MLX Metal kernel launcher.

Computes Y = X @ W^T where W is stored in E8-lattice quantized form (indices + scales).
The Metal kernel uses SIMD-tiled parallelism for high throughput on Apple Silicon:
  - 32 SIMD lanes stripe the D-dimension blocks for parallel dot-product accumulation
  - TILE_H output rows per threadgroup share the input vector via shared memory
  - simd_sum() butterfly-reduces partial products across lanes

Grid layout:
  Total threads: (ceil(H/TILE_H) * 32,  B * TILE_H,  1)
  Threadgroup:   (32,                    TILE_H,       1)
  
  threadgroup_position_in_grid.x  = which H-tile
  threadgroup_position_in_grid.y  = which batch element
  thread_position_in_threadgroup.x = SIMD lane (0..31)
  thread_position_in_threadgroup.y = row within tile (0..TILE_H-1)
"""

import os
import mlx.core as mx

# ── Load Metal shader source ─────────────────────────────────────────────────
METAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elq_decode.metal")
with open(METAL_FILE, "r") as f:
    METAL_SOURCE = f.read()

# ── Tile height: output rows computed per threadgroup ─────────────────────────
# Each threadgroup is (32, TILE_H, 1) = 32 SIMD lanes × TILE_H rows.
#
# TILE_H=4 → 128 threads/threadgroup. Rationale:
#   - M4 Pro has 1024 threads/EU max occupancy. 128 threads = 8 concurrent
#     threadgroups per EU, which is near-optimal for latency hiding.
#   - TILE_H=4 means 4 rows share the input vector load — 4× bandwidth savings.
#   - Higher TILE_H (e.g. 8) would use more shared memory (D*4 bytes per group),
#     potentially reducing occupancy for large D. TILE_H=4 keeps shared memory
#     at 16KB for D=4096, well within the 32KB threadgroup memory limit.
TILE_H = 4

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
    indices and per-block scale factors. The computation is fused: lattice
    points are decoded on-the-fly and immediately dot-producted with the input,
    avoiding materialization of the full float weight matrix.

    Args:
        inp: Input activations of shape ``(..., D)`` where ``D`` is the model
            dimension. Must be a multiple of 32. dtype can be float16 or float32.
        indices: Packed E8 lattice indices of shape ``(H, num_blocks, 4)``
            stored as uint32. Each uint32 encodes one D8+ lattice sub-block
            of 8 coordinates (signs, magnitudes, shift flag, parity bit).
        scales: Per-block scale factors of shape ``(H, num_blocks)`` stored
            as float16. One scale per 32-element block.

    Returns:
        Output activations of shape ``(..., H)``. dtype matches ``inp.dtype``.

    Performance notes:
        For D=4096, H=4096, B=1 on M4 Pro:
        - Old kernel: 1 thread/row × 128 sequential blocks = severe ALU underutilization
        - New kernel: 32 lanes/row × 4 blocks/lane + shared input + simd_sum()
          Expected ~20-30× throughput improvement from parallelism + bandwidth savings.
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
    # Grid specifies TOTAL thread counts (not threadgroup counts) in MLX.
    #
    #   grid.x = ceil(H / tile_h) * 32
    #            ~~~~~~~~~~~~~~~~~~~  ~~
    #            number of row tiles  SIMD lanes per tile
    #
    #   grid.y = tile_h
    #            ~~~~~~~
    #            rows per tile (NOT scaled by B — batching is handled
    #            inside the kernel via the template B loop)
    #
    # The ceiling division ensures we launch enough tiles even when H is not
    # divisible by tile_h. Excess threads (global_row >= H) exit early in the
    # kernel — no correctness impact.
    tile_h = 8 if B == 1 else 4
    tile_count = (H + tile_h - 1) // tile_h   # ceil(H / tile_h)
    grid_x = tile_count * 32                    # total threads in x
    grid_y = tile_h                             # rows per tile (batch is handled by kernel's B loop)

    # ── Launch the Metal kernel ───────────────────────────────────────────────
    out_2d = _elq_kernel(
        inputs=[inp_2d, indices, scales],
        output_shapes=[[B, H]],
        output_dtypes=[inp_2d.dtype],
        template=[
            ("T", inp_2d.dtype),      # Element type (float16 / float32)
            ("H", H),                 # Output dimension (weight rows)
            ("D", D),                 # Input dimension (weight columns)
            ("TILE_H", tile_h),       # Rows per threadgroup tile
            ("B", B),                 # Batch size
        ],
        grid=(grid_x, grid_y, 1),
        threadgroup=(32, tile_h, 1),  # 32 SIMD lanes × tile_h rows
    )[0]

    # ── Restore original batch shape and precision ────────────────────────────
    if len(orig_shape) == 1:
        res = out_2d[0]
    else:
        res = out_2d.reshape(orig_shape[:-1] + (H,))
        
    if res.dtype != orig_dtype:
        res = res.astype(orig_dtype)
    return res
