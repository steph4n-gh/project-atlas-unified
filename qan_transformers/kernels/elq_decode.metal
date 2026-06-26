// ═══════════════════════════════════════════════════════════════════════════════
// MSL shader body: E8 Lattice Vector Quantization (ELQ) fused decode + matmul
// Optimized using Apple Silicon GPU Tensor Cores (simdgroup_matrix)
// ═══════════════════════════════════════════════════════════════════════════════

uint lane_id = thread_position_in_threadgroup.x;
uint h_tile  = threadgroup_position_in_grid.x;
uint b_tile  = threadgroup_position_in_grid.y;
uint thread_idx = lane_id;

// D: input dimension (channels)
// H: output dimension (weight rows)
// B: batch size

// Threadgroup shared memory for cooperative loading and dequantization
threadgroup half tile_inputs[8 * 8];
threadgroup half tile_weights[8 * 8];
threadgroup float tile_outputs[8 * 8];

// Accumulator matrix: 8 batch rows x 8 output columns (FP32 for precision)
simdgroup_matrix<float, 8, 8> accum(0.0f);

uint num_blocks = D / 32;

// Loop over channels in steps of 8
for (uint d_tile = 0; d_tile < D / 8; ++d_tile) {
    // 1. Cooperatively load inputs from global memory into tile_inputs
    // We load 8 batch rows x 8 channel columns = 64 elements.
    // 32 threads in the SIMD-group load 2 elements each.
    uint idx1 = 2 * thread_idx;
    uint idx2 = 2 * thread_idx + 1;

    uint row1 = idx1 / 8;
    uint col1 = idx1 % 8;
    uint row2 = idx2 / 8;
    uint col2 = idx2 % 8;

    uint global_b1 = b_tile * 8 + row1;
    uint global_b2 = b_tile * 8 + row2;
    uint global_d1 = d_tile * 8 + col1;
    uint global_d2 = d_tile * 8 + col2;

    tile_inputs[idx1] = (global_b1 < B && global_d1 < D) ? (half)inp[global_b1 * D + global_d1] : 0.0h;
    tile_inputs[idx2] = (global_b2 < B && global_d2 < D) ? (half)inp[global_b2 * D + global_d2] : 0.0h;

    // 2. Cooperatively decode weights from E8-lattice into tile_weights
    // 8 weight rows x 8 columns = 64 weights.
    uint block_idx = d_tile / 4;
    uint sub_block_idx = d_tile % 4;

    // The first 8 threads each decode 1 entire row (8 elements) of the tile.
    if (thread_idx < 8) {
        uint r = h_tile * 8 + thread_idx;
        if (r < H) {
            float scale = (float)scales[r * num_blocks + block_idx];
            uint32_t index = indices[r * num_blocks * 4 + block_idx * 4 + sub_block_idx];

            uint32_t shift_flag = (index >> 31) & 1;
            uint32_t m7_upper = (index >> 29) & 3;

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

            int sum_c7 = i0 + i1 + i2 + i3 + i4 + i5 + i6;
            int m7_lsb = sum_c7 & 1;
            int m7 = (m7_upper << 1) | m7_lsb;
            int s7 = -((index >> 7) & 1);
            int i7 = (m7 ^ s7) - s7;

            float shift_val = shift_flag ? 0.5f : 0.0f;

            // Store weights in transposed form (tile_weights[c * 8 + r] = W[r, c])
            // to represent W^T in row-major order.
            tile_weights[0 * 8 + thread_idx] = (half)((float)i0 + shift_val) * scale;
            tile_weights[1 * 8 + thread_idx] = (half)((float)i1 + shift_val) * scale;
            tile_weights[2 * 8 + thread_idx] = (half)((float)i2 + shift_val) * scale;
            tile_weights[3 * 8 + thread_idx] = (half)((float)i3 + shift_val) * scale;
            tile_weights[4 * 8 + thread_idx] = (half)((float)i4 + shift_val) * scale;
            tile_weights[5 * 8 + thread_idx] = (half)((float)i5 + shift_val) * scale;
            tile_weights[6 * 8 + thread_idx] = (half)((float)i6 + shift_val) * scale;
            tile_weights[7 * 8 + thread_idx] = (half)((float)i7 + shift_val) * scale;
        } else {
            // Pad out-of-bounds weight rows with 0
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                tile_weights[c * 8 + thread_idx] = 0.0h;
            }
        }
    }

    // Barrier to synchronize writes to threadgroup memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Load from threadgroup memory into simdgroup_matrix and multiply-accumulate
    simdgroup_matrix<half, 8, 8> sg_inp;
    simdgroup_matrix<half, 8, 8> sg_W;
    simdgroup_load(sg_inp, tile_inputs, 8);
    simdgroup_load(sg_W, tile_weights, 8);

    simdgroup_multiply_accumulate(accum, sg_inp, sg_W, accum);

    // Barrier before next iteration to prevent threadgroup memory overwrite
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// 4. Store accumulator back to threadgroup memory and write to global out
simdgroup_store(accum, tile_outputs, 8);

threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint idx = thread_idx; idx < 64; idx += 32) {
    uint b = idx / 8;
    uint r = idx % 8;
    uint global_b = b_tile * 8 + b;
    uint global_r = h_tile * 8 + r;
    if (global_b < B && global_r < H) {
        out[global_b * H + global_r] = (T)tile_outputs[b * 8 + r];
    }
}
