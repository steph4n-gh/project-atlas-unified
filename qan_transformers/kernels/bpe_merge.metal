// BPE Parallel Reaction Kinetics merge body with Boltzmann Sampling & E8 Lattice Projection.
// Grid variables automatically provided by MLX:
//   threadgroup_position_in_grid
//   thread_position_in_threadgroup

inline uint32_t rand_hash(uint32_t x, uint32_t y, uint32_t z) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    
    y = ((y >> 16) ^ y) * 0x45d9f3b;
    y = ((y >> 16) ^ y) * 0x45d9f3b;
    y = (y >> 16) ^ y;
    
    z = ((z >> 16) ^ z) * 0x45d9f3b;
    z = ((z >> 16) ^ z) * 0x45d9f3b;
    z = (z >> 16) ^ z;
    
    return x ^ y ^ z;
}

uint block_id = threadgroup_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;

// shared memory for local symbols (BLOCK_SIZE = 64)
threadgroup uint32_t shared_syms[64];
shared_syms[tid] = symbols[block_id * 64 + tid];

threadgroup_barrier(mem_flags::mem_threadgroup);

float temp_val = temperature[0];

for (uint pass = 0; pass < 32; ++pass) {
    uint32_t left = shared_syms[tid];
    uint32_t right = (tid + 1 < 64) ? shared_syms[tid + 1] : 0xFFFFFFFF;
    
    uint64_t rate = 0;
    uint32_t new_id = 0xFFFFFFFF;
    
    if (left != 0xFFFFFFFF && right != 0xFFFFFFFF) {
        uint64_t pair_key = ((uint64_t)left << 32) | right;
        
        // Binary search lookup in merge_keys (NUM_MERGES is a template constant)
        int low = 0;
        int high = NUM_MERGES - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            uint64_t mid_key = merge_keys[mid];
            if (mid_key == pair_key) {
                uint64_t val = merge_vals[mid];
                new_id = (uint32_t)(val >> 32);
                uint32_t rank = (uint32_t)(val & 0xFFFFFFFF);
                
                rate = (1000000 - rank) * 100;
                if (catalysts[left] || catalysts[right]) {
                    rate += 50;
                }
                break;
            } else if (mid_key < pair_key) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
    
    // 0021 Boltzmann sampling perturbation
    float final_rate = (float)rate;
    if (rate > 0 && temp_val > 0.0f) {
        uint32_t h = rand_hash(block_id, tid, pass);
        float r = (float)(h % 1000) / 1000.0f;
        // Perturb the rate based on temperature and random value
        final_rate += temp_val * 5000.0f * (r - 0.5f);
    }
    
    threadgroup float shared_rates[64];
    threadgroup uint32_t shared_new_ids[64];
    shared_rates[tid] = final_rate;
    shared_new_ids[tid] = new_id;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Local maximum selection to prevent overlap conflicts
    bool should_merge = false;
    if (rate > 0) {
        float rate_left = (tid > 0) ? shared_rates[tid - 1] : 0.0f;
        float rate_right = (tid + 1 < 64) ? shared_rates[tid + 1] : 0.0f;
        
        if (final_rate > rate_left && final_rate >= rate_right) {
            should_merge = true;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (should_merge) {
        shared_syms[tid] = new_id;
        shared_syms[tid + 1] = 0xFFFFFFFF;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Cumsum/Prefix scan for compaction via Metal SIMD-Butterfly
    uint32_t active = (shared_syms[tid] != 0xFFFFFFFF) ? 1 : 0;
    
    uint lane_id = tid % 32;
    uint simd_id = tid / 32;
    uint32_t simd_sum = simd_prefix_inclusive_sum(active);
    
    threadgroup uint32_t group_sums[2];
    if (lane_id == 31) {
        group_sums[simd_id] = simd_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint32_t global_prefix = simd_sum;
    if (simd_id == 1) {
        global_prefix += group_sums[0];
    }
    
    threadgroup uint32_t total_active_shared;
    if (tid == 63) {
        total_active_shared = global_prefix;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint32_t total_active = total_active_shared;
    
    if (total_active == 64) {
        break; // No merges happened
    }
    
    uint32_t sym_val = shared_syms[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sym_val != 0xFFFFFFFF) {
        uint32_t new_idx = global_prefix - 1;
        shared_syms[new_idx] = sym_val;
    }
    
    if (tid >= total_active) {
        shared_syms[tid] = 0xFFFFFFFF;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Output writing
uint32_t final_length = 0;
while (final_length < 64 && shared_syms[final_length] != 0xFFFFFFFF) {
    final_length++;
}

if (tid < final_length) {
    uint32_t out_sym = shared_syms[tid];
    output_symbols[block_id * 64 + tid] = out_sym;
    
    // 0019 E8 Lattice Projection in Metal
    for (int c = 0; c < 8; ++c) {
        output_coords[(block_id * 64 + tid) * 8 + c] = vocab_e8_coords[out_sym * 8 + c];
    }
} else {
    // Write padding values
    output_symbols[block_id * 64 + tid] = 0xFFFFFFFF;
    for (int c = 0; c < 8; ++c) {
        output_coords[(block_id * 64 + tid) * 8 + c] = 0.0f;
    }
}

if (tid == 0) {
    output_lengths[block_id] = final_length;
}
