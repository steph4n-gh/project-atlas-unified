// ═══════════════════════════════════════════════════════════════════════════════
// MSL shader body: E8 Lattice Vector Quantization (ELQ) fused decode + matmul
// Batched implementation for Apple Silicon M4 Pro
// ═══════════════════════════════════════════════════════════════════════════════
//
// PARALLELISM STRATEGY
// ────────────────────
// New kernel:  32 SIMD lanes × TILE_H rows per threadgroup.
//              - D-dimension blocks are striped across 32 SIMD lanes.
//              - TILE_H output rows per threadgroup.
//              - For any batch size B, weights are loaded and decoded exactly ONCE.
//              - Loop over B batch elements computes dot products in registers.
//              - simd_sum() butterfly-reduces partial dot products across lanes.
//              - Coalesced index loading using uint4 vectorized loads.
//
// GRID LAYOUT
// ───────────
// Grid:        (ceil(H/TILE_H) * 32,  TILE_H,  1)   — total threads
// Threadgroup: (32,                    TILE_H,  1)   — 32 lanes × TILE_H rows
//
// THREAD INDEXING
// ───────────────
// thread_position_in_threadgroup.x  = lane_id    (0..31)         SIMD lane
// thread_position_in_threadgroup.y  = local_row  (0..TILE_H-1)   row within tile
// threadgroup_position_in_grid.x    = tile_group                  which H-tile
//
// TEMPLATE PARAMETERS (from mx.fast.metal_kernel)
// ────────────────────
// T       — element type (float16 or float32)
// H       — total output rows (weight matrix height)
// D       — input dimension (must be multiple of 32)
// TILE_H  — output rows per threadgroup tile (typically 4)
// B       — batch size (number of input vectors to multiply in parallel)
//
// INPUTS (auto-declared by MLX template)
// ──────
// device const T*        inp      — input activations  [B, D]
// device const uint32_t* indices  — packed E8 indices   [H, D/32, 4]
// device const half*     scales   — per-block scales    [H, D/32]
// device T*              out      — output activations  [B, H]
// ═══════════════════════════════════════════════════════════════════════════════

// ── Thread and threadgroup identification ────────────────────────────────────
uint lane_id    = thread_position_in_threadgroup.x;   // SIMD lane index (0..31)
uint local_row  = thread_position_in_threadgroup.y;   // Row within this tile (0..TILE_H-1)
uint tile_group = threadgroup_position_in_grid.x;     // Which tile of output rows

uint global_row = tile_group * TILE_H + local_row;    // Absolute output row

// ── Guard: skip computation for out-of-bounds row tiles ─────────────────────
if (global_row >= H) {
    return;
}

// ── Accumulators for all B batch elements ────────────────────────────────────
float sum[B];
#pragma unroll
for (uint b_idx = 0; b_idx < B; ++b_idx) {
    sum[b_idx] = 0.0f;
}

uint num_blocks = D / 32;

// Precompute row-specific offsets
uint scales_row_offset  = global_row * num_blocks;

for (uint b = lane_id; b < num_blocks; b += 32) {
    // Load the per-block scale factor (only once!)
    T scale = (T)scales[scales_row_offset + b];

    // Vectorized read of 4 indices (16 bytes) at once, perfectly coalesced across lanes
    uint4 index_vec = ((device const uint4*)indices)[global_row * num_blocks + b];

    #pragma unroll
    for (uint sb = 0; sb < 4; ++sb) {
        uint32_t index = index_vec[sb];

        // Extract shift flag and upper 2 bits of m7
        uint32_t shift_flag = (index >> 31) & 1;
        uint32_t m7_upper   = (index >> 29) & 3;

        // Extract magnitudes and signs directly using branchless bitwise operations (saves ternary/select instructions)
        int m0 = (index >> 8) & 7;
        int s0 = -((index >> 0) & 1);
        int i0 = (m0 ^ s0) - s0;

        int m1 = (index >> 11) & 7;
        int s1 = -((index >> 1) & 1);
        int i1 = (m1 ^ s1) - s1;

        int m2 = (index >> 14) & 7;
        int s2 = -((index >> 2) & 1);
        int i2 = (m2 ^ s2) - s2;

        int m3 = (index >> 17) & 7;
        int s3 = -((index >> 3) & 1);
        int i3 = (m3 ^ s3) - s3;

        int m4 = (index >> 20) & 7;
        int s4 = -((index >> 4) & 1);
        int i4 = (m4 ^ s4) - s4;

        int m5 = (index >> 23) & 7;
        int s5 = -((index >> 5) & 1);
        int i5 = (m5 ^ s5) - s5;

        int m6 = (index >> 26) & 7;
        int s6 = -((index >> 6) & 1);
        int i6 = (m6 ^ s6) - s6;

        // Parity reconstruction of c7/m7 entirely in integer math
        int sum_c7 = i0 + i1 + i2 + i3 + i4 + i5 + i6;
        int m7_lsb = sum_c7 & 1;  // parity is sign-invariant in two's complement
        int m7 = (m7_upper << 1) | m7_lsb;
        int s7 = -((index >> 7) & 1);
        int i7 = (m7 ^ s7) - s7;

        T c0 = (T)i0;
        T c1 = (T)i1;
        T c2 = (T)i2;
        T c3 = (T)i3;
        T c4 = (T)i4;
        T c5 = (T)i5;
        T c6 = (T)i6;
        T c7 = (T)i7;

        // Apply shift
        T shift_val = shift_flag ? (T)0.5f : (T)0.0f;

        // Vectorized read of 8 input values from global memory for all B batch elements
        #pragma unroll
        for (uint b_idx = 0; b_idx < B; ++b_idx) {
            uint inp_offset = b_idx * D + b * 32 + sb * 8;
            device const metal::vec<T, 4>* inp_inner_vec = (device const metal::vec<T, 4>*)(inp + inp_offset);
            metal::vec<T, 4> inp_lo = inp_inner_vec[0];
            metal::vec<T, 4> inp_hi = inp_inner_vec[1];

            // Factored scale multiplication (saves 7 float multiplications per sub-block)
            float term = (float)(c0 + shift_val) * (float)inp_lo.x +
                         (float)(c1 + shift_val) * (float)inp_lo.y +
                         (float)(c2 + shift_val) * (float)inp_lo.z +
                         (float)(c3 + shift_val) * (float)inp_lo.w +
                         (float)(c4 + shift_val) * (float)inp_hi.x +
                         (float)(c5 + shift_val) * (float)inp_hi.y +
                         (float)(c6 + shift_val) * (float)inp_hi.z +
                         (float)(c7 + shift_val) * (float)inp_hi.w;
                     
            sum[b_idx] += term * (float)scale;
        }
    }
}

// ── SIMD-level reduction across lanes for all B batch elements ────────────────
#pragma unroll
for (uint b_idx = 0; b_idx < B; ++b_idx) {
    sum[b_idx] = simd_sum(sum[b_idx]);
}

// ── Write output: only SIMD lane 0 stores the final result ───────────────────
if (lane_id == 0) {
    #pragma unroll
    for (uint b_idx = 0; b_idx < B; ++b_idx) {
        out[b_idx * H + global_row] = (T)sum[b_idx];
    }
}
