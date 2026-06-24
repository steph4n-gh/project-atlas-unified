// MSL shader body: E8 alignment check
// Fuses E8 projection (8D to 3D matrix multiplication), 3D L2 normalization,
// and cosine similarity checking against a dynamic buffer of E8 roots.
// Caches root coordinates in threadgroup shared memory.
//
// TEMPLATE PARAMETERS (from mx.fast.metal_kernel):
//   T           - Element type (float16 / float32 -> half / float)
//   N           - Number of roots (e.g., 240, 2160, 6720)
//   TotalTokens - Total tokens in the batch (B * S)
//   TILE_SIZE   - Tile size for shared memory caching of roots (e.g. 240)
//
// INPUTS:
//   device const T* inp_8d            - Input coordinates of shape [TotalTokens, 8]
//   device const T* P                 - 8to3 projection matrix of shape [8, 3] (24 elements)
//   device const T* roots             - Roots buffer of shape [N, 3] (3*N elements)
//
// OUTPUTS:
//   device T* alignment_scores        - Output max similarities [TotalTokens]
//   device uint32_t* winning_indices   - Output winning root index [TotalTokens]

uint token_idx = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint tg_size = TG_SIZE;

// Declare threadgroup shared memory for caching roots.
// We use float3 for high precision calculation.
threadgroup float3 shared_roots[TILE_SIZE];

float max_sim = -2.0f;
uint winning_idx = 0;

// Project coordinate of current token to 3D and L2 normalize it.
float3 u_norm_vec = float3(0.0f);
if (token_idx < TotalTokens) {
    float u_x = 0.0f;
    float u_y = 0.0f;
    float u_z = 0.0f;

    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        float x_d = (float)inp_8d[token_idx * 8 + d];
        u_x += x_d * (float)P[d * 3 + 0];
        u_y += x_d * (float)P[d * 3 + 1];
        u_z += x_d * (float)P[d * 3 + 2];
    }

    float len = sqrt(u_x * u_x + u_y * u_y + u_z * u_z) + 1e-6f;
    u_norm_vec = float3(u_x, u_y, u_z) / len;
}

// Loop over all roots in tiles of TILE_SIZE
for (uint tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
    uint num_roots_in_tile = N - tile_start;
    if (num_roots_in_tile > TILE_SIZE) {
        num_roots_in_tile = TILE_SIZE;
    }

    // Cooperatively load this tile of roots into threadgroup shared memory.
    // Every thread in the threadgroup participates to maximize throughput.
    for (uint i = tid; i < num_roots_in_tile; i += tg_size) {
        uint root_idx = tile_start + i;
        float3 r = float3((float)roots[root_idx * 3 + 0],
                          (float)roots[root_idx * 3 + 1],
                          (float)roots[root_idx * 3 + 2]);
        // L2 normalize the root vector to ensure dot product is exactly cosine similarity
        float r_len = sqrt(r.x * r.x + r.y * r.y + r.z * r.z) + 1e-12f;
        shared_roots[i] = r / r_len;
    }

    // Synchronize to guarantee that the whole tile is loaded into shared memory
    // before any thread attempts to read it.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // If this thread represents a valid token, compute similarity against the tile
    if (token_idx < TotalTokens) {
        for (uint i = 0; i < num_roots_in_tile; ++i) {
            float3 r = shared_roots[i];
            float sim = u_norm_vec.x * r.x + u_norm_vec.y * r.y + u_norm_vec.z * r.z;
            if (sim > max_sim) {
                max_sim = sim;
                winning_idx = tile_start + i;
            }
        }
    }

    // Synchronize again to ensure no thread starts overwriting shared_roots in the
    // next iteration before all threads have finished reading the current tile.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Write the final outputs for valid tokens
if (token_idx < TotalTokens) {
    alignment_scores[token_idx] = (T)max_sim;
    winning_indices[token_idx] = (uint32_t)winning_idx;
}
