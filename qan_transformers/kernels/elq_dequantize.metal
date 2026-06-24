// MSL shader body for E8 Lattice dequantization
// Optimized: uses half4 vectorized stores for 2x better write coalescing
uint sb_idx = thread_position_in_grid.x; // Sub-block index (0..num_blocks*4 - 1)
uint row = thread_position_in_grid.y;    // Output row (0..H-1)

if (row >= H || sb_idx >= num_blocks * 4) {
    return;
}

uint b = sb_idx / 4;
uint sb = sb_idx % 4;

float scale = (float)scales[row * num_blocks + b];
uint32_t index = indices[row * num_blocks * 4 + b * 4 + sb];

// Extract shift flag and upper bits of m7
uint32_t shift_flag = (index >> 31) & 1;
uint32_t m7_upper = (index >> 29) & 3;

// Extract magnitudes and signs directly using branchless bitwise operations (saves ternary/select instructions and float multiplies)
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

// Reconstruct LSB of c7/m7 to preserve even sum parity of D8
int sum_c7 = i0 + i1 + i2 + i3 + i4 + i5 + i6;
uint32_t m7_lsb = sum_c7 & 1;  // parity is sign-invariant in two's complement
int m7 = (m7_upper << 1) | m7_lsb;
int s7 = -((index >> 7) & 1);
int i7 = (m7 ^ s7) - s7;

float4 c_lo = float4((float)i0, (float)i1, (float)i2, (float)i3);
float4 c_hi = float4((float)i4, (float)i5, (float)i6, (float)i7);

float shift_val = shift_flag ? 0.5f : 0.0f;
c_lo += shift_val;
c_hi += shift_val;

// Vectorized half4 stores: 2 stores instead of 8 scalar stores
// This doubles write coalescing bandwidth on Apple Silicon GPU
uint out_offset = row * D + b * 32 + sb * 8;
half4 vec_lo = half4(c_lo * scale);
half4 vec_hi = half4(c_hi * scale);

// Cast output pointer to half4* for aligned vector store
// out_offset is always 8-element aligned (sb * 8), so half4 alignment is guaranteed
device half4* out_vec = (device half4*)(out + out_offset);
out_vec[0] = vec_lo;
out_vec[1] = vec_hi;
