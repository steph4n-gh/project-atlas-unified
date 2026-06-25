import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Any
from qan_transformers.mlx.e8_swap import AdelicMemorySwapGridDB
from qan_transformers.firewall.cohomology import CohomologyFirewall

_PROJECTED_ROOTS_CACHE = {}


def _get_dequantized_weight(proj: nn.Module, dtype: mx.Dtype) -> Optional[mx.array]:
    if proj is None:
        return None
    from qan_transformers.mlx.modeling import ELQLinear
    if isinstance(proj, ELQLinear):
        from qan_transformers.mlx.modeling import ELQSlidingCache
        cache = proj.cache if proj.cache is not None else ELQSlidingCache.get()
        if proj._layer_id in cache._cache:
            W_T, _ = cache._cache[proj._layer_id]
            return W_T.T.astype(dtype)
        dequant_fn = ELQLinear._get_dequantize_fn()
        W_dequant = dequant_fn(proj._indices, proj._scales)
        if proj._outliers.size > 0:
            W_dequant[:, proj._outlier_indices] = proj._outliers
        return W_dequant.astype(dtype)
    if hasattr(proj, "scales"):
        w = mx.dequantize(
            proj.weight,
            scales=proj.scales,
            biases=getattr(proj, "biases", None),
            group_size=proj.group_size,
            bits=proj.bits,
            mode=getattr(proj, "mode", "affine")
        )
        return w.astype(dtype)
    if hasattr(proj, "weight") and proj.weight is not None:
        return proj.weight.astype(dtype)
    return None


def _jit_slice_assignment(T: mx.array, S: mx.array, idx: mx.array, L: int) -> mx.array:
    M = T.shape[2]
    p = mx.arange(M)
    p_rel = (p - idx) % M
    mask = p_rel < L
    src_idx_clamped = mx.minimum(p_rel, L - 1)
    S_aligned = mx.take(S, src_idx_clamped, axis=2)
    return mx.where(mask[None, None, :, None], S_aligned, T)


@mx.compile
def _compile_attn_projections(
    x: mx.array,
    w_q: mx.array,
    b_q: Optional[mx.array],
    w_k: mx.array,
    b_k: Optional[mx.array],
    w_v: mx.array,
    b_v: Optional[mx.array],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int
):
    B, S, _ = x.shape
    
    # Q projection
    q = x @ w_q.T
    if b_q is not None:
        q = q + b_q
    q = mx.transpose(mx.reshape(q, (B, S, num_heads, head_dim)), (0, 2, 1, 3))
    
    # K projection
    k = x @ w_k.T
    if b_k is not None:
        k = k + b_k
    k = mx.transpose(mx.reshape(k, (B, S, num_kv_heads, head_dim)), (0, 2, 1, 3))
    
    # V projection
    v = x @ w_v.T
    if b_v is not None:
        v = v + b_v
    v = mx.transpose(mx.reshape(v, (B, S, num_kv_heads, head_dim)), (0, 2, 1, 3))
    
    return q, k, v


@mx.compile
def _compile_q_projection(
    x: mx.array,
    w_q: mx.array,
    b_q: Optional[mx.array],
    num_heads: int,
    head_dim: int
):
    B, S, _ = x.shape
    q = x @ w_q.T
    if b_q is not None:
        q = q + b_q
    q = mx.transpose(mx.reshape(q, (B, S, num_heads, head_dim)), (0, 2, 1, 3))
    return q


def apply_rope_with_indices(x, indices, rope_module):
    B, H, K, d = x.shape
    dims = getattr(rope_module, "dims", d)
    base = getattr(rope_module, "base", 10000.0)
    scale = getattr(rope_module, "scale", 1.0)
    traditional = getattr(rope_module, "traditional", False)
    
    half_dims = dims // 2
    freqs = base ** (-mx.arange(0, dims, 2, dtype=mx.float32) / dims) * scale
    
    indices_expanded = mx.expand_dims(mx.expand_dims(indices, 1), -1).astype(mx.float32)
    freqs_expanded = mx.reshape(freqs, (1, 1, 1, -1))
    
    angles = indices_expanded * freqs_expanded
    cos = mx.cos(angles).astype(x.dtype)
    sin = mx.sin(angles).astype(x.dtype)
    
    if traditional:
        x_rot_part = x[..., :dims]
        x_reshaped = mx.reshape(x_rot_part, (B, H, K, half_dims, 2))
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]
        
        x0_rot = x0 * cos - x1 * sin
        x1_rot = x1 * cos + x0 * sin
        
        x_reshaped_rot = mx.stack([x0_rot, x1_rot], axis=-1)
        x_rot_part = mx.reshape(x_reshaped_rot, (B, H, K, dims))
        if d > dims:
            return mx.concatenate([x_rot_part, x[..., dims:]], axis=-1)
        else:
            return x_rot_part
    else:
        x1 = x[..., :half_dims]
        x2 = x[..., half_dims:dims]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x2 * cos + x1 * sin
        if d > dims:
            return mx.concatenate([x1_rot, x2_rot, x[..., dims:]], axis=-1)
        else:
            return mx.concatenate([x1_rot, x2_rot], axis=-1)


class QuasicrystallineAttention(nn.Module):
    _shared_prev_entropy = None
    _pending_entropy = None
    _shared_entropy_history = []
    _shared_firewall_interval = 16

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sparse_ratio: float = 0.15,
        firewall: Optional[CohomologyFirewall] = None,
        num_key_value_heads: Optional[int] = None,
        is_draft: bool = False,
        head_dim: Optional[int] = None,
        rg_enabled: bool = False,
        uv_window: int = 64,
        rg_chunk_size: int = 32,
        lattice: str = 'e8'
    ):
        """
        MLX-native Quasicrystalline Attention Layer.
        Uses Coxeter/Icosian projections to align query/key vectors to E8 lattices
        in a zero-copy Apple Silicon Unified Memory space.
        """
        super().__init__()
        self.rg_enabled = rg_enabled
        self.uv_window = uv_window
        self.rg_chunk_size = rg_chunk_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparse_ratio = sparse_ratio
        self.head_dim = head_dim if head_dim is not None else (embed_dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.min_keep = 0
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        
        # Projection layers (placeholder structure; populated during grafting)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        # 8D mapping layer
        self.e8_proj = nn.Linear(embed_dim, 8)
        
        # Icosian 3D projection matrix P_8_3
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        scale = 1.0 / np.sqrt(1.0 + phi**2)
        P_8_4 = np.zeros((8, 4))
        P_8_4[0, 0] = phi * scale
        P_8_4[4, 0] = 1.0 * scale
        P_8_4[1, 1] = phi * scale
        P_8_4[5, 1] = 1.0 * scale
        P_8_4[2, 2] = phi * scale
        P_8_4[6, 2] = 1.0 * scale
        P_8_4[3, 3] = phi * scale
        P_8_4[7, 3] = 1.0 * scale
        
        P_4_3 = np.zeros((4, 3))
        P_4_3[1, 0] = 1.0
        P_4_3[2, 1] = 1.0
        P_4_3[3, 2] = 1.0
        
        self.P_8_3 = mx.array(P_8_4 @ P_4_3, dtype=mx.float32)
        
        # MLX Swap Grid DB
        self.swap_db = AdelicMemorySwapGridDB(d_model=self.head_dim)
        self.is_draft = is_draft
        
        # Pre-cache Shell 1, 2, 3 projected roots globally per device type to avoid redundant math per layer
        # and prevent device leakage between CPU and GPU contexts.
        device_type = mx.default_device().type
        global _PROJECTED_ROOTS_CACHE
        if (device_type, 1) not in _PROJECTED_ROOTS_CACHE:
            from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates, project_e8_to_quasicrystal
            prev_dev = mx.default_device()
            mx.set_default_device(mx.Device(device_type))
            try:
                for lvl in [1, 2, 3]:
                    roots_8d = generate_dynamic_e8_coordinates(lvl)
                    roots_3d = mx.array(project_e8_to_quasicrystal(roots_8d), dtype=mx.float32)
                    roots_3d_norm = roots_3d / (mx.linalg.norm(roots_3d, axis=-1, keepdims=True) + 1e-6)
                    mx.eval(roots_3d, roots_3d_norm)
                    _PROJECTED_ROOTS_CACHE[(device_type, lvl)] = (roots_3d, roots_3d_norm)
            finally:
                mx.set_default_device(prev_dev)
                
        self.cached_roots = {lvl: _PROJECTED_ROOTS_CACHE[(device_type, lvl)][0] for lvl in [1, 2, 3]}
        for lvl in [1, 2, 3]:
            setattr(self, f"roots_3d_norm_lvl_{lvl}", _PROJECTED_ROOTS_CACHE[(device_type, lvl)][1])
            
        self.roots_3d = self.cached_roots[1]
        self.roots_3d_norm = getattr(self, "roots_3d_norm_lvl_1")
        
        # Moonshot: Leech lattice mode (196,560 addresses vs E8's 240)
        self._lattice_mode = lattice
        if lattice == 'leech':
            leech_cache_key = (device_type, 'leech')
            if leech_cache_key not in _PROJECTED_ROOTS_CACHE:
                try:
                    from qan_transformers.math.leech_lattice import generate_leech_coordinates, project_leech_to_3d, LeechShellRouter
                    leech_coords = generate_leech_coordinates(shell=1)
                    leech_3d, leech_info = project_leech_to_3d(leech_coords, method='direct')
                    leech_3d_mx = mx.array(leech_3d, dtype=mx.float32)
                    leech_3d_norm = leech_3d_mx / (mx.linalg.norm(leech_3d_mx, axis=-1, keepdims=True) + 1e-6)
                    mx.eval(leech_3d_mx, leech_3d_norm)
                    _PROJECTED_ROOTS_CACHE[leech_cache_key] = (leech_3d_mx, leech_3d_norm, leech_info, leech_coords)
                    print(f"[QCA] Leech lattice initialized: {len(leech_coords)} addresses, {leech_info['n_shells']} shells, quality={leech_info['quality']:.2f}", flush=True)
                except Exception as e:
                    print(f"[QCA] Leech lattice init failed, falling back to E8: {e}", flush=True)
                    self._lattice_mode = 'e8'

            if leech_cache_key in _PROJECTED_ROOTS_CACHE:
                cached = _PROJECTED_ROOTS_CACHE[leech_cache_key]
                self._leech_roots_3d = cached[0]
                self._leech_roots_3d_norm = cached[1]
                self._leech_info = cached[2]
                leech_coords_np = cached[3]
                # Build 24D projection matrix for the lattice bias
                # Use SVD to find the optimal 24D -> 3D projection
                centered = leech_coords_np - np.mean(leech_coords_np, axis=0)
                _, _, Vt = np.linalg.svd(centered[:10000], full_matrices=False)
                self.P_24_3 = mx.array(Vt[:3].T, dtype=mx.float32)  # (24, 3)
                # Also need a 24D mapping layer instead of 8D
                self.e8_proj_leech = nn.Linear(embed_dim, 24)
            else:
                self._lattice_mode = 'e8'
        else:
            self._lattice_mode = 'e8'

        # Expose for moonshot geometric filter
        QuasicrystallineAttention._projection_matrix_3d = self.P_8_3
        if hasattr(self, 'P_24_3'):
            QuasicrystallineAttention._projection_matrix_3d = self.P_24_3

        self.firewall = firewall if firewall is not None else CohomologyFirewall()
        self.prev_entropy = None
        self._token_count = 0          # Token counter for gating expensive ops
        self._firewall_interval = 16   # Run firewall every N tokens (not every token)
        self.positions_buffer = mx.arange(8192, dtype=mx.int32)
        self._cpu_cache_len_history = {}
        
    def _apply_e8_lattice_bias(self, attn_scores, S, S_total, rope_offset, dtype, is_prefill=False):
        if getattr(self.__class__, "in_jit", False):
            return attn_scores

        precomputed_coords = getattr(self.__class__, "precomputed_e8_coords", None)
        if precomputed_coords is not None:
            # Check cached projected 3D coordinates to save matmul overhead during decoding
            if not hasattr(self, "_precomputed_coords_3d") or getattr(self, "_precomputed_coords_id", None) != id(precomputed_coords):
                self._precomputed_coords_3d = mx.matmul(precomputed_coords, self.P_8_3)
                self._precomputed_coords_id = id(precomputed_coords)

        # Moonshot: Use Leech lattice coordinates when available
        if getattr(self, '_lattice_mode', 'e8') == 'leech' and hasattr(self, 'P_24_3'):
            if precomputed_coords is not None and precomputed_coords.shape[0] >= S_total:
                try:
                    off_val = int(rope_offset.item()) if hasattr(rope_offset, "item") else int(rope_offset)
                    Q_3d = self._precomputed_coords_3d[off_val : off_val + S]
                    K_3d = self._precomputed_coords_3d[:S_total]

                    # Optimize memory to avoid O(S * S_total) temporary expansion
                    Q_sq = mx.sum(mx.square(Q_3d), axis=-1, keepdims=True)
                    K_sq_T = mx.expand_dims(mx.sum(mx.square(K_3d), axis=-1), 0)
                    QK_prod = mx.matmul(Q_3d, K_3d.T)
                    dists = Q_sq + K_sq_T - 2.0 * QK_prod

                    dists_bias = mx.array(-0.01, dtype=dtype) * dists.astype(dtype)
                    dists_bias = mx.expand_dims(mx.expand_dims(dists_bias, 0), 0)
                    return attn_scores + dists_bias
                except Exception:
                    pass

        # Win 313: GPU E8 Coordinate Pre-fetching
        if precomputed_coords is not None and precomputed_coords.shape[0] >= S_total:
            try:
                off_val = int(rope_offset.item()) if hasattr(rope_offset, "item") else int(rope_offset)
                Q_3d = self._precomputed_coords_3d[off_val : off_val + S]
                K_3d = self._precomputed_coords_3d[:S_total]
                
                # Optimize memory to avoid O(S * S_total) temporary expansion
                Q_sq = mx.sum(mx.square(Q_3d), axis=-1, keepdims=True)
                K_sq_T = mx.expand_dims(mx.sum(mx.square(K_3d), axis=-1), 0)
                QK_prod = mx.matmul(Q_3d, K_3d.T)
                dists = Q_sq + K_sq_T - 2.0 * QK_prod

                dists_bias = mx.array(-0.01, dtype=dtype) * dists.astype(dtype)
                dists_bias = mx.expand_dims(mx.expand_dims(dists_bias, 0), 0)
                return attn_scores + dists_bias
            except Exception:
                pass
                
        # Fallback to CPU-side lookup (for decoding phase / single token step)
        if not is_prefill and getattr(self.__class__, "organism", None) is not None and getattr(self.__class__, "current_token_ids", None) is not None:
            token_ids = self.__class__.current_token_ids
            if len(token_ids) >= S_total:
                try:
                    off_val = int(rope_offset.item()) if hasattr(rope_offset, "item") else int(rope_offset)
                    q_ids = token_ids[off_val : off_val + S]
                    k_ids = token_ids[:S_total]
                    
                    q_coords, _ = self.__class__.organism.get_sequence_lattice_metadata(q_ids)
                    k_coords, _ = self.__class__.organism.get_sequence_lattice_metadata(k_ids)
                    
                    Q_coords = mx.array(q_coords, dtype=mx.float32)
                    K_coords = mx.array(k_coords, dtype=mx.float32)
                    
                    Q_3d = mx.matmul(Q_coords, self.P_8_3)
                    K_3d = mx.matmul(K_coords, self.P_8_3)
                    
                    # Optimize memory to avoid O(S * S_total) temporary expansion
                    Q_sq = mx.sum(mx.square(Q_3d), axis=-1, keepdims=True)
                    K_sq_T = mx.expand_dims(mx.sum(mx.square(K_3d), axis=-1), 0)
                    QK_prod = mx.matmul(Q_3d, K_3d.T)
                    dists = Q_sq + K_sq_T - 2.0 * QK_prod

                    dists_bias = mx.array(-0.01, dtype=dtype) * dists.astype(dtype)
                    dists_bias = mx.expand_dims(mx.expand_dims(dists_bias, 0), 0)
                    
                    attn_scores = attn_scores + dists_bias
                except Exception:
                    pass
        return attn_scores

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        shared_kv: Optional[tuple] = None,
        offset: Optional[Any] = None,
        *args,
        **kwargs
    ) -> Any:
        B, S, D = x.shape
        dtype = x.dtype
        
        # Win 314: Lightweight Attention Prefill Branch Detection
        is_prefill_val = kwargs.get("is_prefill")
        if is_prefill_val is not None:
            is_prefill = is_prefill_val
        else:
            is_prefill = getattr(self, "is_prefill", False)
        
        # 1. Dynamic Shell Scaling based on attention entropy
        layer_idx = getattr(self, "layer_idx", 0)
        
        # Asynchronous delayed sync: read the pending entropy from previous step
        if layer_idx == 0 and not getattr(self, "is_draft", False) and QuasicrystallineAttention._pending_entropy is not None:
            val = float(QuasicrystallineAttention._pending_entropy.item())
            QuasicrystallineAttention._shared_prev_entropy = val
            QuasicrystallineAttention._pending_entropy = None
            
            # Update history and calculate dynamic firewall interval (Enzymatic Gating)
            QuasicrystallineAttention._shared_entropy_history.append(val)
            if len(QuasicrystallineAttention._shared_entropy_history) > 5:
                QuasicrystallineAttention._shared_entropy_history.pop(0)
                
            if len(QuasicrystallineAttention._shared_entropy_history) >= 2:
                history = QuasicrystallineAttention._shared_entropy_history
                mean_val = sum(history) / len(history)
                variance = sum((x - mean_val) ** 2 for x in history) / len(history)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
                
            if volatility < 0.02 and val < 1.8:
                # Highly stable: check rarely
                interval = 128
            elif volatility < 0.05 and val < 2.2:
                # Moderately stable
                interval = 64
            elif volatility < 0.10:
                # Fluctuating / normal
                interval = 16
            else:
                # High volatility / high risk: check frequently
                interval = 2
                
            QuasicrystallineAttention._shared_firewall_interval = interval
            
        if QuasicrystallineAttention._shared_prev_entropy is not None:
            self.prev_entropy = QuasicrystallineAttention._shared_prev_entropy
            
        if not self.training and self.prev_entropy is not None:
            if self.prev_entropy < 1.5:
                shell_level = 3
            elif self.prev_entropy > 3.0:
                shell_level = 1
            else:
                shell_level = 2
        else:
            shell_level = 1

            
        self.roots_3d = self.cached_roots[shell_level]
        # Use cached attribute references instead of f-string construction per call
        if not hasattr(self, "_roots_3d_norm_by_level"):
            self._roots_3d_norm_by_level = {
                lvl: getattr(self, f"roots_3d_norm_lvl_{lvl}") for lvl in self.cached_roots
            }
        self.roots_3d_norm = self._roots_3d_norm_by_level[shell_level]
        
        # Project states using native linear layers to avoid expensive manual dequantization on every step
        if shared_kv is not None:
            queries = self.q_proj(x)
            queries = mx.transpose(mx.reshape(queries, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
            if hasattr(self, "q_norm"):
                queries = self.q_norm(queries)
            
            if len(shared_kv) == 3:
                keys, values, _ = shared_kv
            else:
                keys, values = shared_kv
        else:
            if not getattr(self, "has_kv", True):
                raise ValueError("Layer is a KV-shared layer but received no shared_kv")
            
            queries = self.q_proj(x)
            queries = mx.transpose(mx.reshape(queries, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
            if hasattr(self, "q_norm"):
                queries = self.q_norm(queries)
                
            use_sparse_projection = getattr(self, "_use_sparse_projection_override", None)
            if use_sparse_projection is None:
                use_sparse_projection = (
                    not self.training 
                    and S > 64 
                    and not getattr(self, "is_draft", False) 
                    and getattr(self, "sparse_ratio", 0.15) < 1.0
                    and not getattr(QuasicrystallineAttention, "in_jit", False)
                )
            
            if use_sparse_projection:
                keys = None
                values = None
            else:
                keys = self.k_proj(x)
                keys = mx.transpose(mx.reshape(keys, (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
                
                values = self.v_proj(x)
                values = mx.transpose(mx.reshape(values, (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
                
                if hasattr(self, "k_norm"):
                    keys = self.k_norm(keys)
                if hasattr(self, "v_norm"):
                    values = self.v_norm(values)

            
        # Determine positional encoding offset
        rope_offset = offset if offset is not None else (cache.offset if cache is not None else 0)
        
        # Apply RoPE
        if hasattr(self, "rope") and self.rope is not None:
            queries = self.rope(queries, offset=rope_offset)
            if shared_kv is None and keys is not None:
                keys = self.rope(keys, offset=rope_offset)
            
        # Swap out Keys and Values to DB — gated to every 4th token to reduce overhead
        if not self.training and shared_kv is None and keys is not None and self.swap_db is not None and getattr(self.swap_db, "enabled", True) and (self._token_count % 4 == 0):
            K_flat = mx.reshape(mx.transpose(keys, (0, 2, 1, 3)), (-1, self.head_dim))
            V_flat = mx.reshape(mx.transpose(values, (0, 2, 1, 3)), (-1, self.head_dim))
            if getattr(self, "is_draft", False) and hasattr(self.swap_db, "swap_out_draft"):
                self.swap_db.swap_out_draft(K_flat, V_flat, token_seq_len=S)
            elif hasattr(self.swap_db, "swap_out_target"):
                self.swap_db.swap_out_target(K_flat, V_flat, token_seq_len=S)
            else:
                self.swap_db.swap_out(K_flat, V_flat, token_seq_len=S)
            
        if getattr(QuasicrystallineAttention, "in_jit", False):
            S_total = rope_offset + S
            if shared_kv is not None:
                if len(shared_kv) == 3:
                    K_sparse, V_sparse, indices_sparse = shared_kv
                else:
                    K_sparse, V_sparse = shared_kv
                    indices_sparse = (mx.arange(K_sparse.shape[2]) + rope_offset)[None, :]
                offset = rope_offset
                
                # Fixed-shape causal mask for shared_kv
                q_pos = rope_offset + mx.arange(S)[None, None, :, None]
                k_pos = indices_sparse[:, None, None, :]
                valid_mask = k_pos <= q_pos
                mask_val = -65000.0 if dtype in (mx.float16, mx.bfloat16) else -1e9
                attn_mask_sparse = mx.where(valid_mask, 0.0, mask_val)
            else:
                if cache is not None:
                    if cache.keys is not None:
                        idx = rope_offset % cache.keys.shape[2]
                        # In-place slice assignment (keeps shape static)
                        cache.keys = _jit_slice_assignment(cache.keys, keys, idx, S)
                        cache.values = _jit_slice_assignment(cache.values, values, idx, S)
                        cache.offset = rope_offset + S
                        
                        K_sparse = cache.keys
                        V_sparse = cache.values
                        
                        # Fixed-shape causal mask
                        q_pos = rope_offset + mx.arange(S)[None, None, :, None]
                        k_pos = mx.arange(K_sparse.shape[2])[None, None, None, :]
                        valid_mask = k_pos <= q_pos
                        mask_val = -65000.0 if dtype in (mx.float16, mx.bfloat16) else -1e9
                        attn_mask_sparse = mx.where(valid_mask, 0.0, mask_val)
                        
                        indices_sparse = mx.arange(K_sparse.shape[2])[None, :]
                    else:
                        K_sparse = keys
                        V_sparse = values
                        cache.keys = keys
                        cache.values = values
                        cache.offset = rope_offset + S
                        indices_sparse = (mx.arange(K_sparse.shape[2]) + rope_offset)[None, :]
                        attn_mask_sparse = 0.0
                else:
                    K_sparse = keys
                    V_sparse = values
                    indices_sparse = (mx.arange(K_sparse.shape[2]) + rope_offset)[None, :]
                    attn_mask_sparse = 0.0
                S_total = rope_offset + S
                offset = rope_offset
                
            K_sparse_unrepeated = K_sparse
            V_sparse_unrepeated = V_sparse
            
            num_key_value_groups = self.num_heads // self.num_key_value_heads
            if num_key_value_groups > 1:
                shape_key = K_sparse.shape
                if not hasattr(self, "_gqa_shape_cache"):
                    self._gqa_shape_cache = {}
                if shape_key in self._gqa_shape_cache:
                    expand_shape, reshape_shape = self._gqa_shape_cache[shape_key]
                else:
                    B_kv, H_kv, S_kv, D_kv = shape_key
                    expand_shape = (B_kv, H_kv, num_key_value_groups, S_kv, D_kv)
                    reshape_shape = (B_kv, H_kv * num_key_value_groups, S_kv, D_kv)
                    self._gqa_shape_cache[shape_key] = (expand_shape, reshape_shape)
                K_sparse = mx.reshape(mx.broadcast_to(mx.expand_dims(K_sparse, 2), expand_shape), reshape_shape)
                V_sparse = mx.reshape(mx.broadcast_to(mx.expand_dims(V_sparse, 2), expand_shape), reshape_shape)
                
            attn_scores = (queries @ mx.transpose(K_sparse, (0, 1, 3, 2))) * self.scale
            attn_scores = self._apply_e8_lattice_bias(attn_scores, S, S_total, rope_offset, dtype, is_prefill=is_prefill)
            attn_scores = attn_scores + attn_mask_sparse
            
            attn_scores_max = mx.max(attn_scores, axis=-1, keepdims=True)
            is_masked_row = attn_scores_max <= -60000.0
            
            if getattr(self, "use_poly_softmax", False):
                # Prevent nan in fully masked rows (-inf - -inf)
                safe_max = mx.where(is_masked_row, 0.0, attn_scores_max)
                y = attn_scores - safe_max
                exp_approx = mx.maximum(0.0, 1.0 + 0.5 * y)
                exp_approx = exp_approx * exp_approx
                exp_approx = exp_approx * (~is_masked_row)
                sum_exp = mx.sum(exp_approx, axis=-1, keepdims=True)
                attn_weights = exp_approx / (sum_exp + 1e-6)
            else:
                attn_weights = mx.softmax(attn_scores, axis=-1)
                attn_weights = attn_weights * (~is_masked_row)
            self.last_attn_weights = attn_weights
            out = attn_weights @ V_sparse
            
            out = mx.transpose(out, (0, 2, 1, 3))
            out = mx.reshape(out, (B, S, -1))
            if kwargs.get("return_unprojected", False):
                if getattr(self, "is_gemma4", False):
                    return out, (K_sparse_unrepeated, V_sparse_unrepeated, indices_sparse), rope_offset
                else:
                    return out
            out = self.o_proj(out)
            
            if getattr(self, "is_gemma4", False):
                return out, (K_sparse_unrepeated, V_sparse_unrepeated, indices_sparse), rope_offset
            else:
                return out
        else:
            # Determine caching properties
            offset = cache.offset if (cache is not None and hasattr(cache, "offset")) else 0
            use_cache = cache is not None or (not self.training)
            
            is_prefill_start = (S > 1 and offset == 0 and (not hasattr(self, "locked_book_cache") or self.locked_book_cache is None))
            
            if not hasattr(self, "custom_kv_cache") or self.custom_kv_cache is None or not use_cache or is_prefill_start:
                self.custom_kv_cache = {
                    "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                }
                self._cpu_cache_len_history = {}
                if self.swap_db is not None and not getattr(self.swap_db, "d_model_draft", None):
                    self.swap_db.clear()
            elif not is_prefill and cache is not None and hasattr(cache, "offset") and self.custom_kv_cache["seq_len"] > cache.offset:
                # Rollback detected! Trim custom_kv_cache and swap_db to cache.offset (Bypassed during prefill)
                new_len = cache.offset
                k_cache = self.custom_kv_cache
                if k_cache["K"] is not None:
                    # Win 61: CPU-Side Cache Size History for MLX Rollback replacing mx.sum(valid_mask).item()
                    if hasattr(self, "_cpu_cache_len_history") and new_len in self._cpu_cache_len_history:
                        num_true = self._cpu_cache_len_history[new_len]
                    else:
                        valid_mask = k_cache["indices"][0] < new_len
                        num_true = int(mx.sum(valid_mask).item())
                    if num_true > 0:
                        self.custom_kv_cache = {
                            "K": k_cache["K"][:, :, :num_true, :],
                            "V": k_cache["V"][:, :, :num_true, :],
                            "indices": k_cache["indices"][:, :num_true],
                            "alignment_scores": k_cache["alignment_scores"][:, :num_true],
                            "seq_len": new_len
                        }
                    else:
                        self.custom_kv_cache = {
                            "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                        }
                if hasattr(self.swap_db, "rollback"):
                    self.swap_db.rollback(new_len)
                
            # Sliding window trimming for custom cache (Win 315: Bypass during prefill)
            is_sliding = getattr(self, "is_sliding", False) or (cache is not None and cache.__class__.__name__ == "RotatingKVCache")
            if not is_prefill and is_sliding and cache is not None:
                window_size = getattr(cache, "max_size", getattr(self, "window_size", 4096))
                limit = self.custom_kv_cache["seq_len"] - window_size
                if limit > 0:
                    k_cache = self.custom_kv_cache
                    if k_cache["K"] is not None:
                        valid_mask = k_cache["indices"][0] >= limit
                        num_true = int(mx.sum(valid_mask).item())
                        if num_true > 0:
                            sorted_idx = mx.argsort(valid_mask.astype(mx.int32))[::-1]
                            keep_idx = sorted_idx[:num_true]
                            keep_idx = mx.sort(keep_idx)
                            self.custom_kv_cache = {
                                "K": k_cache["K"][:, :, keep_idx, :],
                                "V": k_cache["V"][:, :, keep_idx, :],
                                "indices": k_cache["indices"][:, keep_idx],
                                "alignment_scores": k_cache["alignment_scores"][:, keep_idx],
                                "seq_len": k_cache["seq_len"]
                            }
                        else:
                            self.custom_kv_cache = {
                                "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": k_cache["seq_len"]
                            }

            min_keep = getattr(self, "min_keep", 0)
            offset = self.custom_kv_cache["seq_len"]
            
            if offset == 0:
                self.prompt_len = S
                
            if shared_kv is not None:
                if len(shared_kv) == 3:
                    K_sparse, V_sparse, indices_sparse = shared_kv
                else:
                    K_sparse, V_sparse = shared_kv
                    # Fallback for indices_sparse
                    indices_sparse = (mx.arange(K_sparse.shape[2]) + rope_offset)[None, :]
                S_total = rope_offset + S
                offset = rope_offset
            else:
                if (not self.training and (S <= 1024 or (offset > 0 and S <= 64))) or getattr(self, "is_draft", False) or getattr(self, "sparse_ratio", 0.15) >= 1.0:
                    K_sparse = keys
                    V_sparse = values
                    topk_scores = mx.full((B, S), 60000.0, dtype=dtype)
                    absolute_topk_indices = mx.broadcast_to((mx.arange(S, dtype=mx.int32) + offset)[None, :], (B, S))
                else:
                    # E8 projection to 3D in a single step (fused optimization)
                    if not hasattr(self, "_e8_proj_3d_weight") or self.training:
                        self._e8_proj_3d_weight = _get_dequantized_weight(self.e8_proj, dtype).T @ self.P_8_3
                        if hasattr(self.e8_proj, "bias") and self.e8_proj.bias is not None:
                            self._e8_proj_3d_bias = self.e8_proj.bias @ self.P_8_3
                        else:
                            self._e8_proj_3d_bias = None
                    
                    if self._e8_proj_3d_bias is not None:
                        seq_3d = x @ self._e8_proj_3d_weight + self._e8_proj_3d_bias
                    else:
                        seq_3d = x @ self._e8_proj_3d_weight
                    
                    seq_3d_norm = seq_3d / (mx.linalg.norm(seq_3d, axis=-1, keepdims=True) + 1e-6)
                    cos_sim = seq_3d_norm @ self.roots_3d_norm.T
                    alignment_score = mx.max(cos_sim, axis=-1)
                    
                    # BREAK GPU JIT GRAPH FOR PREFILL:
                    if S > 16 and mx.default_device().type == mx.DeviceType.gpu:
                        mx.eval(alignment_score)
                    
                    in_wave_solver = getattr(self.__class__, "in_wave_solver", False)
                    if in_wave_solver:
                        K_size = S
                        topk_indices = mx.broadcast_to(mx.arange(S)[None, :], (B, S))
                        topk_scores = alignment_score
                        absolute_topk_indices = topk_indices + offset
                    else:
                        K_size = min(S, max(min(S, min_keep) if min_keep > 0 else 1, int(S * self.sparse_ratio)))
                        if not self.training and S > 4 and offset == 0:
                            alignment_score = mx.array(alignment_score)
                            alignment_score[..., :4] = 60000.0
                        idx = mx.argpartition(alignment_score, kth=K_size - 1, axis=-1)
                        topk_indices = idx[..., :K_size]
                        topk_indices = mx.sort(topk_indices, axis=-1)
                        topk_scores = mx.take_along_axis(alignment_score, topk_indices, axis=-1)
                        absolute_topk_indices = topk_indices + offset
                    
                    # Index active keys/values
                    if use_sparse_projection:
                        # Sparse activation projection optimization (Sparse Projection Fusion)
                        x_sparse = mx.take_along_axis(x, mx.expand_dims(topk_indices, -1), axis=1)
                        
                        K_sparse = self.k_proj(x_sparse)
                        K_sparse = mx.transpose(mx.reshape(K_sparse, (B, K_size, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
                        if hasattr(self, "k_norm"):
                            K_sparse = self.k_norm(K_sparse)
                        if hasattr(self, "rope") and self.rope is not None:
                            K_sparse = apply_rope_with_indices(K_sparse, absolute_topk_indices, self.rope)
                            
                        V_sparse = self.v_proj(x_sparse)
                        V_sparse = mx.transpose(mx.reshape(V_sparse, (B, K_size, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
                        if hasattr(self, "v_norm"):
                            V_sparse = self.v_norm(V_sparse)
                    else:
                        gather_indices = mx.reshape(topk_indices, (B, 1, -1, 1))
                        K_sparse = mx.take_along_axis(keys, gather_indices, axis=2)
                        V_sparse = mx.take_along_axis(values, gather_indices, axis=2)
                    
                    # BREAK GPU JIT GRAPH FOR PREFILL:
                    if S > 16 and mx.default_device().type == mx.DeviceType.gpu:
                        mx.eval(K_sparse, V_sparse)
                        
                S_total = offset + S
                if use_cache:
                    cache_kv = self.custom_kv_cache
                    if cache_kv["K"] is not None:
                        K_combined = mx.concatenate([cache_kv["K"], K_sparse], axis=2)
                        V_combined = mx.concatenate([cache_kv["V"], V_sparse], axis=2)
                        indices_combined = mx.concatenate([cache_kv["indices"], absolute_topk_indices], axis=1)
                        scores_combined = mx.concatenate([cache_kv["alignment_scores"], topk_scores], axis=1)
                    else:
                        K_combined = K_sparse
                        V_combined = V_sparse
                        indices_combined = absolute_topk_indices
                        scores_combined = topk_scores
                        
                    if getattr(self, "sparse_ratio", 0.15) >= 1.0 and not getattr(self, "rg_enabled", False):
                        K_sparse = K_combined
                        V_sparse = V_combined
                        indices_sparse = indices_combined
                        scores_sparse = scores_combined
                    else:
                        K_total = min(indices_combined.shape[1], max(min(indices_combined.shape[1], min_keep) if min_keep > 0 else 1, int(S_total * self.sparse_ratio)))
                        num_extra = indices_combined.shape[1] - K_total
                        is_prefill_val = kwargs.get("is_prefill")
                        if is_prefill_val is not None:
                            is_prefill = is_prefill_val
                        else:
                            is_prefill = (S > 8 or offset == 0)
                        in_wave_solver = getattr(self.__class__, "in_wave_solver", False)
                        if not getattr(self, "is_draft", False) and (is_prefill or num_extra >= 16) and indices_combined.shape[1] > K_total and not in_wave_solver:
                            K_sparse, V_sparse, indices_sparse, scores_sparse = self._morse_collapse_cache(
                                K_combined, V_combined, indices_combined, scores_combined, K_total, offset
                            )
                        else:
                            K_sparse = K_combined
                            V_sparse = V_combined
                            indices_sparse = indices_combined
                            scores_sparse = scores_combined
                            
                        # Apply Information-Theoretic KV Cache RG Flow (Win 316: Bypass during prefill)
                        K_sparse, V_sparse, indices_sparse, scores_sparse = self._apply_rg_flow(
                            K_sparse, V_sparse, indices_sparse, scores_sparse, is_prefill=is_prefill
                        )
                    
                    self.custom_kv_cache["K"] = K_sparse
                    self.custom_kv_cache["V"] = V_sparse
                    self.custom_kv_cache["indices"] = indices_sparse
                    self.custom_kv_cache["alignment_scores"] = scores_sparse
                    self.custom_kv_cache["seq_len"] = S_total
                    if cache is not None:
                        self._update_cache_keys_values(cache, K_sparse, V_sparse, S_total)
                    
                    # Win 61: Record CPU-side cache size history (capped to last 256 entries) (Bypassed during prefill)
                    if not is_prefill:
                        if not hasattr(self, "_cpu_cache_len_history"):
                            self._cpu_cache_len_history = {}
                        self._cpu_cache_len_history[S_total] = indices_sparse.shape[1]
                        if len(self._cpu_cache_len_history) > 256:
                            oldest_key = next(iter(self._cpu_cache_len_history))
                            del self._cpu_cache_len_history[oldest_key]
                else:
                    indices_sparse = absolute_topk_indices
                    scores_sparse = topk_scores
                
            # Save unrepeated keys/values for cache and KV-sharing layers (which expect 1 KV head)
            K_sparse_unrepeated = K_sparse
            V_sparse_unrepeated = V_sparse
    
            # GQA repeat keys/values using zero-copy broadcast_to and reshape
            # Win 91: Cached Block-Masks/Shapes for GQA Key-Value Replication in MLX Attention
            num_key_value_groups = self.num_heads // self.num_key_value_heads
            if num_key_value_groups > 1:
                shape_key = K_sparse.shape
                if not hasattr(self, "_gqa_shape_cache"):
                    self._gqa_shape_cache = {}
                if shape_key in self._gqa_shape_cache:
                    expand_shape, reshape_shape = self._gqa_shape_cache[shape_key]
                else:
                    B_kv, H_kv, S_kv, D_kv = shape_key
                    expand_shape = (B_kv, H_kv, num_key_value_groups, S_kv, D_kv)
                    reshape_shape = (B_kv, H_kv * num_key_value_groups, S_kv, D_kv)
                    self._gqa_shape_cache[shape_key] = (expand_shape, reshape_shape)
                    
                K_sparse = mx.reshape(mx.broadcast_to(mx.expand_dims(K_sparse, 2), expand_shape), reshape_shape)
                V_sparse = mx.reshape(mx.broadcast_to(mx.expand_dims(V_sparse, 2), expand_shape), reshape_shape)
                
            # Retrieve paged items from Swap DB — gated to every 4th token to amortize neuromorphic search cost
            max_matches = 8 if (not self.training and getattr(self.swap_db, "enabled", True)) else 0
            if not self.training and queries.shape[2] == 1 and max_matches > 0 and (self._token_count % 4 == 0):
                if getattr(self, "is_draft", False) and hasattr(self.swap_db, "swap_in_batch_draft"):
                    swapped_k, swapped_v = self.swap_db.swap_in_batch_draft(queries, max_matches=max_matches)
                else:
                    swapped_k, swapped_v = self.swap_db.swap_in_batch(queries, max_matches=max_matches)
                K_sparse = mx.concatenate([K_sparse, swapped_k], axis=2)
                V_sparse = mx.concatenate([V_sparse, swapped_v], axis=2)
                
            # Win 212: Skip causal mask evaluation on any single-token generation when mask is None
            if S == 1 and mask is None:
                attn_mask_sparse = 0.0
            else:
                if not hasattr(self, "_cached_positions") or getattr(self, "_cached_positions_key", None) != (offset, S):
                    if offset + S > self.positions_buffer.shape[0]:
                        new_len = max(offset + S, self.positions_buffer.shape[0] * 2)
                        self.positions_buffer = mx.arange(new_len, dtype=mx.int32)
                    self._cached_positions = self.positions_buffer[offset : offset + S]
                    self._cached_positions_key = (offset, S)
                q_positions = mx.reshape(self._cached_positions, (1, 1, S, 1))
                k_positions = mx.reshape(indices_sparse, (B, 1, 1, -1))
                causal_mask = (k_positions > q_positions).astype(dtype)
                
                # Win 27: MLX Attention Pad Caching (Causal Pad)
                if not self.training and queries.shape[2] == 1 and max_matches > 0:
                    pad_shape = causal_mask.shape[:-1] + (max_matches,)
                    # Win 104: Fused Arithmetic Masking for Causal Attention in MLX (cache zero pad with dtype)
                    if not hasattr(self, "_cached_unmasked_causal_pad") or self._cached_unmasked_causal_pad.shape != pad_shape or self._cached_unmasked_causal_pad.dtype != dtype:
                        self._cached_unmasked_causal_pad = mx.zeros(pad_shape, dtype=dtype)
                    unmasked_causal_pad = self._cached_unmasked_causal_pad
                    causal_mask = mx.concatenate([causal_mask, unmasked_causal_pad], axis=-1)
                    
                mask_val = -float('inf')
                attn_mask_sparse = mx.where(causal_mask > 0, mask_val, 0.0)
            
            if mask is not None and isinstance(mask, mx.array):
                if S > 16:
                    print(f"[DEBUG MASK] mask shape: {mask.shape}, indices_sparse shape: {indices_sparse.shape}, S={S}, offset={offset}", flush=True)
                mask_ndim = mask.ndim
                if mask_ndim == 1:
                    user_mask_sparse = mask[None, None, None, :]
                elif mask_ndim == 2:
                    user_mask_sparse = mask[None, None, :, :]
                elif mask_ndim == 3:
                    user_mask_sparse = mask[None, :, :, :]
                else:
                    user_mask_sparse = mask
                
                # Broadcast/align batch dimension
                if user_mask_sparse.shape[0] != B:
                    user_mask_sparse = mx.broadcast_to(user_mask_sparse, (B,) + user_mask_sparse.shape[1:])
                # Broadcast/align head dimension
                if user_mask_sparse.shape[1] != self.num_heads and user_mask_sparse.shape[1] == 1:
                    user_mask_sparse = mx.broadcast_to(user_mask_sparse, (B, self.num_heads) + user_mask_sparse.shape[2:])
                
                # Align query dimension (-2)
                mask_S = user_mask_sparse.shape[-2]
                if mask_S != S:
                    if mask_S == 1:
                        user_mask_sparse = mx.broadcast_to(user_mask_sparse, user_mask_sparse.shape[:-2] + (S, user_mask_sparse.shape[-1]))
                    elif mask_S > S:
                        if mask_S >= offset + S:
                            user_mask_sparse = user_mask_sparse[..., offset : offset + S, :]
                        else:
                            user_mask_sparse = user_mask_sparse[..., :S, :]
                    else:
                        pad_q_shape = user_mask_sparse.shape[:-2] + (S - mask_S, user_mask_sparse.shape[-1])
                        pad_q = mx.zeros(pad_q_shape, dtype=user_mask_sparse.dtype)
                        user_mask_sparse = mx.concatenate([user_mask_sparse, pad_q], axis=-2)

                # Gather user mask along key dimension using indices_sparse
                # instead of slicing the first sparse_kv_len columns.
                # During prefill the base model passes a causal/window mask spanning the entire sequence length
                # but our sparse path only retains active keys at specific indices_sparse.
                sparse_kv_len = indices_sparse.shape[1]
                if user_mask_sparse.shape[-1] > sparse_kv_len:
                    gather_idx = mx.reshape(indices_sparse, (B, 1, 1, -1))
                    gather_idx = mx.clip(gather_idx, 0, user_mask_sparse.shape[-1] - 1)
                    user_mask_sparse = mx.take_along_axis(user_mask_sparse, gather_idx, axis=-1)
                
                # Align user_mask_sparse length to sparse_kv_len to prevent broadcast shape errors
                if user_mask_sparse.shape[-1] != sparse_kv_len:
                    diff_len = sparse_kv_len - user_mask_sparse.shape[-1]
                    if diff_len > 0:
                        pad_shape = user_mask_sparse.shape[:-1] + (diff_len,)
                        pad_val = mx.zeros(pad_shape, dtype=user_mask_sparse.dtype)
                        user_mask_sparse = mx.concatenate([user_mask_sparse, pad_val], axis=-1)
                    else:
                        user_mask_sparse = user_mask_sparse[..., :sparse_kv_len]
                
                # Convert boolean mask to additive mask if it's boolean
                if user_mask_sparse.dtype == mx.bool_:
                    mask_val = -float('inf')
                    user_mask_sparse = mx.where(user_mask_sparse, 0.0, mask_val)
                    
                if max_matches > 0:
                    # User mask pad caching
                    pad_shape = user_mask_sparse.shape[:-1] + (max_matches,)
                    cache_key = (pad_shape, dtype)
                    if not hasattr(self, "_cached_unmasked_pad_dict"):
                        self._cached_unmasked_pad_dict = {}
                    if cache_key not in self._cached_unmasked_pad_dict:
                        self._cached_unmasked_pad_dict[cache_key] = mx.zeros(pad_shape, dtype=dtype)
                    unmasked_pad = self._cached_unmasked_pad_dict[cache_key]
                    user_mask_sparse = mx.concatenate([user_mask_sparse, unmasked_pad], axis=-1)
                    
                try:
                    attn_mask_sparse = attn_mask_sparse + user_mask_sparse
                except Exception as e:
                    print(f"[DEBUG SHAPES] S={S}, offset={offset}, training={self.training}", flush=True)
                    print(f"  attn_mask_sparse shape: {attn_mask_sparse.shape}", flush=True)
                    print(f"  user_mask_sparse shape: {user_mask_sparse.shape}", flush=True)
                    print(f"  indices_sparse shape: {indices_sparse.shape}", flush=True)
                    if cache is not None:
                        print(f"  cache.offset: {cache.offset}", flush=True)
                        if cache.keys is not None:
                            print(f"  cache.keys shape: {cache.keys.shape}", flush=True)
                    print(f"  self.custom_kv_cache seq_len: {self.custom_kv_cache['seq_len']}", flush=True)
                    if self.custom_kv_cache['K'] is not None:
                        print(f"  self.custom_kv_cache K shape: {self.custom_kv_cache['K'].shape}", flush=True)
                    raise e
                
            # Fused Scaled Dot-Product Attention Optimization (Prefill Fusion)
            if not getattr(self, "use_poly_softmax", False):
                combined_mask = attn_mask_sparse
                
                # Check for E8 precomputed coordinates or fallback to current_token_ids lookup
                has_e8_coords = getattr(self.__class__, "precomputed_e8_coords", None) is not None
                has_cpu_coords = getattr(self.__class__, "organism", None) is not None and getattr(self.__class__, "current_token_ids", None) is not None
                
                if has_e8_coords or (has_cpu_coords and not is_prefill):
                    # Apply bias to zeros, then add to mask
                    bias_scores = self._apply_e8_lattice_bias(
                        mx.zeros((B, self.num_heads, S, K_sparse.shape[2]), dtype=dtype),
                        S, S_total, rope_offset, dtype, is_prefill=is_prefill
                    )
                    combined_mask = combined_mask + bias_scores
                
                # Ensure mask is cast to correct dtype and handle python scalars
                if isinstance(combined_mask, (int, float)):
                    if combined_mask == 0.0:
                        combined_mask = None
                    else:
                        combined_mask = mx.array(combined_mask, dtype=dtype)
                elif combined_mask is not None:
                    combined_mask = combined_mask.astype(dtype)
                
                if S == 1:
                    # For S==1, we compute attention weights manually using standard softmax
                    # to make attn_weights available for the firewall and entropy checks.
                    attn_scores = (queries @ mx.transpose(K_sparse, (0, 1, 3, 2))) * self.scale
                    if combined_mask is not None:
                        attn_scores = attn_scores + combined_mask
                    attn_weights = mx.softmax(attn_scores, axis=-1)
                    out = attn_weights @ V_sparse
                else:
                    from mlx.core.fast import scaled_dot_product_attention
                    self.last_combined_mask = combined_mask
                    out = scaled_dot_product_attention(
                        queries, K_sparse, V_sparse,
                        scale=self.scale,
                        mask=combined_mask
                    )
                    attn_weights = None
            else:
                self.last_combined_mask = attn_mask_sparse
                # Scaled dot-product attention
                attn_scores = (queries @ mx.transpose(K_sparse, (0, 1, 3, 2))) * self.scale
                attn_scores = self._apply_e8_lattice_bias(attn_scores, S, S_total, rope_offset, dtype, is_prefill=is_prefill)
                attn_scores = attn_scores + attn_mask_sparse
                
                # Win 104: Fused broadcast-based row masking in MLX attention
                attn_scores_max = mx.max(attn_scores, axis=-1, keepdims=True)
                is_masked_row = attn_scores_max <= -60000.0
                
                # Prevent nan in fully masked rows (-inf - -inf)
                safe_max = mx.where(is_masked_row, 0.0, attn_scores_max)
                
                # 2nd-order Taylor approximation: exp(y) approx (1 + y/2)^2 for y >= -2, else 0
                y = attn_scores - safe_max
                exp_approx = mx.maximum(0.0, 1.0 + 0.5 * y)
                exp_approx = exp_approx * exp_approx
                exp_approx = exp_approx * (~is_masked_row)
                
                sum_exp = mx.sum(exp_approx, axis=-1, keepdims=True)
                attn_weights = exp_approx / (sum_exp + 1e-6)
                out = attn_weights @ V_sparse
                
            self.last_attn_weights = attn_weights
            
            # Increment token counter (used to gate expensive CPU-sync operations)
            self._token_count += 1
            
            # Cohomology firewall inline check — gated to every Nth token.
            interval = getattr(QuasicrystallineAttention, "_shared_firewall_interval", 16)
            if (not self.training and not getattr(self, "is_draft", False)
                and hasattr(self, "firewall") and self.firewall is not None
                and S == 1
                and (self._token_count % interval == 0 or interval == 2)):
                is_fractured, cfi, alt_idx = self.firewall.check_obstruction(attn_weights)
                
            # Entropy tracking — async evaluation to avoid GPU pipeline stall (P0-B fix).
            layer_idx = getattr(self, "layer_idx", 0)
            if layer_idx == 0 and S == 1 and not getattr(self, "is_draft", False) and (QuasicrystallineAttention._shared_prev_entropy is None or self._token_count % interval == 0 or interval == 2):
                entropy = -mx.sum(attn_weights * mx.log(mx.clip(attn_weights, 1e-9, 1.0)), axis=-1).mean()
                mx.async_eval(entropy)
                QuasicrystallineAttention._pending_entropy = entropy
                
            out = mx.transpose(out, (0, 2, 1, 3))
            out = mx.reshape(out, (B, S, -1))
            if kwargs.get("return_unprojected", False):
                if cache is not None:
                    self._update_cache_keys_values(cache, K_sparse_unrepeated, V_sparse_unrepeated, S_total)
                if getattr(self, "is_gemma4", False):
                    return out, (K_sparse_unrepeated, V_sparse_unrepeated, indices_sparse), rope_offset
                else:
                    return out
            out = self.o_proj(out)
        

        
        if cache is not None:
            self._update_cache_keys_values(cache, K_sparse_unrepeated, V_sparse_unrepeated, S_total)
            
        if getattr(self, "is_gemma4", False):
            return out, (K_sparse_unrepeated, V_sparse_unrepeated, indices_sparse), rope_offset
        else:
            return out
        
    def _update_cache_keys_values(self, cache, K, V, S_total):
        max_size = getattr(cache, "max_size", S_total)
        if max_size > S_total:
            B, H, _, D = K.shape
            cache_k = mx.zeros((B, H, max_size, D), dtype=K.dtype)
            cache_v = mx.zeros((B, H, max_size, D), dtype=V.dtype)
            cache_k[..., :S_total, :] = K
            cache_v[..., :S_total, :] = V
            cache.keys = cache_k
            cache.values = cache_v
            cache.offset = S_total
            cache._idx = S_total
        else:
            cache.keys = K
            cache.values = V
            cache.offset = S_total
            cache._idx = S_total % max_size

    def _morse_collapse_cache(self, K_combined, V_combined, indices_combined, scores_combined, K_total, offset=0):
        B, H, K_len, head_dim = K_combined.shape
        
        # Compute row-sum vertex energies linearly O(N) using summation associativity in MLX
        K_sum = K_combined.sum(axis=2, keepdims=True)  # [B, H, 1, head_dim]
        # Win 64: Memory-Efficient GEMV for MLX Morse Collapse replacing mx.sum of broadcasted products
        row_sums = mx.squeeze(K_combined @ mx.transpose(K_sum, (0, 1, 3, 2)), -1) * self.scale  # [B, H, K_len]
        vertex_energies = mx.mean(row_sums, axis=(0, 1))  # [K_len]
        token_indices = indices_combined[0]
        
        book_len = 0
        if hasattr(self, "locked_book_cache") and self.locked_book_cache is not None:
            book_len = self.locked_book_cache.get("seq_len", 0)
            
        prompt_len = getattr(self, "prompt_len", offset)
        protect_start = book_len if book_len > 0 else max(0, prompt_len - 64)
        # Win 36: Fuse boolean condition for sink and convo token protection
        is_sink = (token_indices < 4) | (token_indices >= protect_start)
        
        # Win 50: Branch-Free Node Energy Boosting in MLX replacing mx.where
        vertex_energies = vertex_energies + is_sink.astype(vertex_energies.dtype) * 60000.0
        
        sorted_indices = mx.argsort(vertex_energies)[::-1]
        critical_summits = sorted_indices[:K_total]
        critical_summits = mx.sort(critical_summits)
        
        K_ret = K_combined[:, :, critical_summits, :]
        V_ret = V_combined[:, :, critical_summits, :]
        indices_ret = indices_combined[:, critical_summits]
        scores_ret = scores_combined[:, critical_summits]
        
        return K_ret, V_ret, indices_ret, scores_ret

    def _apply_rg_flow(self, K, V, indices, scores, is_prefill=False):
        if is_prefill:
            return K, V, indices, scores
        B, H, S_cache, D = K.shape
        if not self.rg_enabled or S_cache <= self.uv_window + self.rg_chunk_size:
            return K, V, indices, scores
            
        S_old = S_cache - self.uv_window
        start_idx = 1 if S_old % 2 != 0 else 0
        end_idx = S_old
        
        K_old = K[:, :, start_idx:end_idx, :]
        V_old = V[:, :, start_idx:end_idx, :]
        indices_old = indices[:, start_idx:end_idx]
        scores_old = scores[:, start_idx:end_idx]
        
        # Group into blocks of 2
        S_old_even = end_idx - start_idx
        S_blocks = S_old_even // 2
        
        K_old_blocks = mx.reshape(K_old, (B, H, S_blocks, 2, -1))
        V_old_blocks = mx.reshape(V_old, (B, H, S_blocks, 2, -1))
        indices_old_blocks = mx.reshape(indices_old, (B, S_blocks, 2))
        scores_old_blocks = mx.reshape(scores_old, (B, S_blocks, 2))
        
        # Compute local density matrix components
        k1 = K_old_blocks[:, :, :, 0, :]
        k2 = K_old_blocks[:, :, :, 1, :]
        
        a = mx.sum(k1 * k1, axis=-1)
        b = mx.sum(k1 * k2, axis=-1)
        c = mx.sum(k2 * k2, axis=-1)
        
        # Trigonometric diagonalization
        theta = 0.5 * mx.arctan2(2.0 * b, a - c)
        alpha = mx.cos(theta)
        beta = mx.sin(theta)
        
        # Project K and V
        K_proj = mx.expand_dims(alpha, axis=-1) * k1 + mx.expand_dims(beta, axis=-1) * k2
        V_proj = mx.expand_dims(alpha, axis=-1) * V_old_blocks[:, :, :, 0, :] + mx.expand_dims(beta, axis=-1) * V_old_blocks[:, :, :, 1, :]
        
        # Project indices and scores
        indices_proj = indices_old_blocks[:, :, 1]
        scores_proj = mx.max(scores_old_blocks, axis=-1)
        
        # Concatenate parts
        K_recent = K[:, :, end_idx:, :]
        V_recent = V[:, :, end_idx:, :]
        indices_recent = indices[:, end_idx:]
        scores_recent = scores[:, end_idx:]
        
        K_parts = []
        V_parts = []
        indices_parts = []
        scores_parts = []
        if start_idx > 0:
            K_parts.append(K[:, :, :start_idx, :])
            V_parts.append(V[:, :, :start_idx, :])
            indices_parts.append(indices[:, :start_idx])
            scores_parts.append(scores[:, :start_idx])
            
        K_parts.extend([K_proj, K_recent])
        V_parts.extend([V_proj, V_recent])
        indices_parts.extend([indices_proj, indices_recent])
        scores_parts.extend([scores_proj, scores_recent])
        
        K_comp = mx.concatenate(K_parts, axis=2)
        V_comp = mx.concatenate(V_parts, axis=2)
        indices_comp = mx.concatenate(indices_parts, axis=1)
        scores_comp = mx.concatenate(scores_parts, axis=1)
        
        return K_comp, V_comp, indices_comp, scores_comp

class UltrametricAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparse_ratio=0.15, num_key_value_heads=None, is_draft=False, depth=5, leaf_size=128):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparse_ratio = sparse_ratio
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_draft = is_draft
        self.depth = depth
        self.leaf_size = leaf_size
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / np.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Coordinate projection mapping embed_dim to 3 coordinates
        self.coordinate_proj = nn.Linear(embed_dim, 3)
        
        # Learnable decay parameter
        self.gamma = mx.array(1.0)
        
    def __call__(self, x, kv_cache=None, attn_mask=None):
        import inspect
        device = mx.default_device()
        dtype = x.dtype
        B, S, D = x.shape
        
        # Projects hidden states into 3D continuous coordinates
        coords = mx.sigmoid(self.coordinate_proj(x))
        coords = mx.where(mx.isnan(coords), 0.5, coords)
        
        # Extract depth digits for bases 2, 3, and 5
        c0 = coords[..., 0]
        c1 = coords[..., 1]
        c2 = coords[..., 2]
        
        # Base 2 digits
        current_0 = c0
        digits_2 = []
        for _ in range(self.depth):
            current_0 = current_0 * 2
            d = mx.floor(current_0)
            d = mx.clip(d, 0.0, 1.0)
            digits_2.append(d)
            current_0 = current_0 - d
            
        # Base 3 digits
        current_1 = c1
        digits_3 = []
        for _ in range(self.depth):
            current_1 = current_1 * 3
            d = mx.floor(current_1)
            d = mx.clip(d, 0.0, 2.0)
            digits_3.append(d)
            current_1 = current_1 - d
            
        # Base 5 digits
        current_2 = c2
        digits_5 = []
        for _ in range(self.depth):
            current_2 = current_2 * 5
            d = mx.floor(current_2)
            d = mx.clip(d, 0.0, 4.0)
            digits_5.append(d)
            current_2 = current_2 - d
            
        # Interleave and pack digits into base-30 Morton code integers
        morton_code = mx.zeros(c0.shape, dtype=mx.int64)
        for k in range(self.depth):
            d_30 = digits_2[k] + 2 * digits_3[k] + 6 * digits_5[k]
            morton_code = morton_code * 30 + d_30.astype(mx.int64)
            
        # Linear projections for Q, K, V
        Q = mx.transpose(mx.reshape(self.q_proj(x), (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(self.k_proj(x), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(self.v_proj(x), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        
        # Concatenate with cached K and V if kv_cache is provided
        if kv_cache is not None:
            if "K" in kv_cache and kv_cache["K"] is not None:
                K_combined = mx.concatenate([kv_cache["K"], K], axis=2)
                V_combined = mx.concatenate([kv_cache["V"], V], axis=2)
                morton_combined = mx.concatenate([kv_cache["morton_codes"], morton_code], axis=1)
            else:
                K_combined = K
                V_combined = V
                morton_combined = morton_code
            kv_cache["K"] = K_combined
            kv_cache["V"] = V_combined
            kv_cache["morton_codes"] = morton_combined
        else:
            K_combined = K
            V_combined = V
            morton_combined = morton_code
            
        S_total = K_combined.shape[2]
        
        # Repeat KV helper function (local)
        def repeat_kv(kv_arr, n_rep):
            if n_rep == 1:
                return kv_arr
            bs, n_kv, sl, hd = kv_arr.shape
            kv_arr = mx.expand_dims(kv_arr, axis=3)
            kv_arr = mx.broadcast_to(kv_arr, (bs, n_kv, sl, n_rep, hd))
            return mx.reshape(kv_arr, (bs, n_kv * n_rep, sl, hd))
            
        # S_total < 2048 (dynamic fallback to standard dot-product attention)
        if S_total < 2048:
            K_rep = repeat_kv(K_combined, self.num_key_value_groups)
            V_rep = repeat_kv(V_combined, self.num_key_value_groups)
            
            scores = (Q @ mx.transpose(K_rep, (0, 1, 3, 2))) * self.scaling
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_probs = mx.softmax(scores, axis=-1)
            out = attn_probs @ V_rep
        else:
            # S_total >= 2048: Fast Multipole Method (FMM) attention loop
            N = S_total
            H = self.num_heads
            
            K_rep = repeat_kv(K_combined, self.num_key_value_groups)
            V_rep = repeat_kv(V_combined, self.num_key_value_groups)
            
            # Sort sequence along sequence dimension by Morton codes (with chronological priority to prevent causal leakage)
            orig_idx = mx.broadcast_to(mx.expand_dims(mx.arange(N, dtype=mx.int64), axis=0), (B, N))
            sort_key = orig_idx * 100000000 + morton_combined
            sort_idx = mx.argsort(sort_key, axis=-1)
            unsort_idx = mx.argsort(sort_idx, axis=-1)
            
            # Pad Q to N if S < N
            if S < N:
                Q_padded = mx.pad(Q, [(0, 0), (0, 0), (0, N - S), (0, 0)], constant_values=0.0)
            else:
                Q_padded = Q
                
            # Gather sorted tensors along sequence dimension (dim 2)
            gather_idx = mx.broadcast_to(mx.reshape(sort_idx, (B, 1, N, 1)), (B, H, N, self.head_dim))
            Q_sorted = mx.take_along_axis(Q_padded, gather_idx, axis=2)
            K_sorted = mx.take_along_axis(K_rep, gather_idx, axis=2)
            V_sorted = mx.take_along_axis(V_rep, gather_idx, axis=2)
            
            # Determine tree dimensions
            B_sz = self.leaf_size
            M = (N + B_sz - 1) // B_sz
            L = int(np.ceil(np.log2(max(M, 1))))
            M_pow = 2**L
            N_tree = M_pow * B_sz
            pad_len = N_tree - N
            
            # Construct sorted mask
            mask_val = -65000.0 if dtype in (mx.float16, mx.bfloat16) else -1e9
            mask_sorted = mx.zeros((B, H, N_tree), dtype=dtype)
            
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    mask_seq = mx.broadcast_to(mx.expand_dims(attn_mask, axis=1), (B, H, attn_mask.shape[-1]))
                elif attn_mask.ndim == 3:
                    if attn_mask.shape[1] in (H, 1):
                        mask_seq = mx.broadcast_to(attn_mask, (B, H, attn_mask.shape[-1]))
                    else:
                        mask_seq = mx.broadcast_to(mx.expand_dims(attn_mask[:, 0, :], axis=1), (B, H, attn_mask.shape[-1]))
                elif attn_mask.ndim == 4:
                    if attn_mask.shape[1] == 1:
                        mask_seq = mx.broadcast_to(mx.expand_dims(attn_mask[:, 0, 0, :], axis=1), (B, H, attn_mask.shape[-1]))
                    else:
                        mask_seq = attn_mask[:, :, 0, :]
                else:
                    mask_seq = attn_mask
                
                if mask_seq.shape[-1] < N:
                    pad_size = N - mask_seq.shape[-1]
                    mask_seq = mx.pad(mask_seq, [(0, 0), (0, 0), (0, pad_size)], constant_values=0.0)
                elif mask_seq.shape[-1] > N:
                    mask_seq = mask_seq[..., :N]
                
                gather_idx_mask = mx.broadcast_to(mx.expand_dims(sort_idx, axis=1), (B, H, N))
                mask_sorted_seq = mx.take_along_axis(mask_seq, gather_idx_mask, axis=2)
                
                if pad_len > 0:
                    mask_sorted = mx.pad(mask_sorted_seq, [(0, 0), (0, 0), (0, pad_len)], constant_values=mask_val)
                else:
                    mask_sorted = mask_sorted_seq
            else:
                if pad_len > 0:
                    mask_sorted = mx.concatenate([mx.zeros((B, H, N), dtype=dtype), mx.full((B, H, pad_len), mask_val, dtype=dtype)], axis=2)
            
            # Pad sorted tensors to N_tree
            if pad_len > 0:
                Q_tree = mx.pad(Q_sorted, [(0, 0), (0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
                K_tree = mx.pad(K_sorted, [(0, 0), (0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
                V_tree = mx.pad(V_sorted, [(0, 0), (0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
            else:
                Q_tree = Q_sorted
                K_tree = K_sorted
                V_tree = V_sorted
                
            # Reshape to block structure
            Q_blocks = mx.reshape(Q_tree, (B, H, M_pow, B_sz, self.head_dim))
            K_blocks = mx.reshape(K_tree, (B, H, M_pow, B_sz, self.head_dim))
            V_blocks = mx.reshape(V_tree, (B, H, M_pow, B_sz, self.head_dim))
            
            # Upward Pass (Aggregate nodes)
            K_tree_nodes = {}
            V_tree_nodes = {}
            # Leaf level L nodes
            is_active = (mask_sorted == 0.0).astype(dtype)
            active_count = mx.sum(mx.reshape(is_active, (B, H, M_pow, B_sz)), axis=-1, keepdims=True)
            K_tree_nodes[L] = mx.sum(K_blocks, axis=-2) / mx.maximum(active_count, 1.0)
            V_tree_nodes[L] = mx.sum(V_blocks, axis=-2) / mx.maximum(active_count, 1.0)
            
            for l in range(L - 1, -1, -1):
                parent_K = mx.mean(mx.reshape(K_tree_nodes[l+1], (B, H, 2**l, 2, self.head_dim)), axis=3)
                parent_V = mx.mean(mx.reshape(V_tree_nodes[l+1], (B, H, 2**l, 2, self.head_dim)), axis=3)
                K_tree_nodes[l] = parent_K
                V_tree_nodes[l] = parent_V
                
            # Upward Pass for mask nodes
            mask_nodes = {}
            mask_nodes[L] = mx.max(mx.reshape(mask_sorted, (B, H, M_pow, B_sz)), axis=-1)
            for l in range(L - 1, -1, -1):
                reshaped_mask = mx.reshape(mask_nodes[l+1], (B, H, 2**l, 2))
                mask_nodes[l] = mx.maximum(reshaped_mask[..., 0], reshaped_mask[..., 1])
                
            # Near-field direct block attention
            attn_scores_near = (Q_blocks @ mx.transpose(K_blocks, (0, 1, 2, 4, 3))) * self.scaling
            
            # Causal and padding masks in near-field
            if pad_len > 0:
                sort_idx_tree = mx.pad(sort_idx, [(0, 0), (0, pad_len)], constant_values=N + 1000)
            else:
                sort_idx_tree = sort_idx
                
            orig_indices_blocks = mx.reshape(sort_idx_tree, (B, 1, M_pow, B_sz))
            
            idx_us_1 = mx.expand_dims(orig_indices_blocks, axis=-1)
            idx_us_2 = mx.expand_dims(orig_indices_blocks, axis=-2)
            causal_mask_near = (idx_us_1 < idx_us_2)
            padding_mask_near = (idx_us_2 >= N)
            
            mask_near = (causal_mask_near | padding_mask_near).astype(dtype) * mask_val
            
            # Extract block-local slice of sorted mask and add to near-field scores
            mask_blocks = mx.reshape(mask_sorted, (B, H, M_pow, B_sz))
            attn_scores_near = attn_scores_near + mask_near + mx.expand_dims(mask_blocks, axis=-2)
            
            # Upward hierarchy of minimum and maximum original indices for causal checks
            min_orig_node = {}
            max_orig_node = {}
            min_orig_node[L] = mx.min(orig_indices_blocks, axis=-1)
            max_orig_node[L] = mx.max(orig_indices_blocks, axis=-1)
            for l in range(L - 1, -1, -1):
                min_reshaped = mx.reshape(min_orig_node[l+1], (B, 1, 2**l, 2))
                max_reshaped = mx.reshape(max_orig_node[l+1], (B, 1, 2**l, 2))
                min_orig_node[l] = mx.minimum(min_reshaped[..., 0], min_reshaped[..., 1])
                max_orig_node[l] = mx.maximum(max_reshaped[..., 0], max_reshaped[..., 1])
                
            # Far-field aggregated sibling nodes attention
            j_indices = mx.arange(M_pow, dtype=mx.int64)
            K_sibs = []
            V_sibs = []
            is_sib_padded_list = []
            is_sib_causal_violation_list = []
            sibling_mask_list = []
            
            for l in range(1, L + 1):
                ancestor_indices = j_indices // (2**(L - l))
                sibling_indices = ancestor_indices ^ 1
                
                gather_idx_sib = mx.broadcast_to(mx.reshape(sibling_indices, (1, 1, M_pow, 1)), (B, H, M_pow, self.head_dim))
                K_sib_l = mx.take_along_axis(K_tree_nodes[l], gather_idx_sib, axis=2)
                V_sib_l = mx.take_along_axis(V_tree_nodes[l], gather_idx_sib, axis=2)
                
                K_sibs.append(mx.expand_dims(K_sib_l, axis=3))
                V_sibs.append(mx.expand_dims(V_sib_l, axis=3))
                
                # Check sibling padding
                sibling_start_block = sibling_indices * (2**(L - l))
                is_sib_padded_l = (sibling_start_block >= M)
                is_sib_padded_list.append(mx.reshape(is_sib_padded_l, (1, 1, M_pow, 1, 1)))
                
                # Check sibling causal violation
                gather_idx_max = mx.broadcast_to(mx.reshape(sibling_indices, (1, 1, M_pow)), (B, 1, M_pow))
                sibling_max_orig_l = mx.take_along_axis(max_orig_node[l], gather_idx_max, axis=2)
                causal_mask_sib_l = (mx.expand_dims(sibling_max_orig_l, axis=-1) > orig_indices_blocks)
                is_sib_causal_violation_list.append(mx.expand_dims(causal_mask_sib_l, axis=-1))
                
                # Sibling mask
                gather_idx_mask = mx.broadcast_to(mx.reshape(sibling_indices, (1, 1, M_pow)), (B, H, M_pow))
                sibling_mask_l = mx.take_along_axis(mask_nodes[l], gather_idx_mask, axis=2)
                sibling_mask_list.append(mx.expand_dims(sibling_mask_l, axis=-1))
                
            K_sib_all = mx.concatenate(K_sibs, axis=3) # Shape: [B, H, M_pow, L, D]
            V_sib_all = mx.concatenate(V_sibs, axis=3) # Shape: [B, H, M_pow, L, D]
            is_sib_padded_all = mx.concatenate(is_sib_padded_list, axis=4) # Shape: [1, 1, M_pow, 1, L]
            is_sib_causal_violation_all = mx.concatenate(is_sib_causal_violation_list, axis=-1) # Shape: [B, 1, M_pow, B_sz, L]
            sibling_mask_all = mx.concatenate(sibling_mask_list, axis=-1) # Shape: [B, H, M_pow, L]
            
            # Sibling scores
            scores_sib = (Q_blocks @ mx.transpose(K_sib_all, (0, 1, 2, 4, 3))) * self.scaling
            
            # Scale factor for sibling nodes: -self.gamma * 30**(-level)
            levels = mx.arange(1, L + 1, dtype=dtype)
            scale_factors = -self.gamma * (30.0 ** (-levels))
            scores_sib_scaled = scores_sib * mx.reshape(scale_factors, (1, 1, 1, 1, L))
            
            # Mask padded and causal violated sibling nodes
            is_sib_invalid = (is_sib_padded_all | is_sib_causal_violation_all)
            scores_sib_scaled = mx.where(is_sib_invalid, mask_val, scores_sib_scaled)
            
            # Add sibling mask
            scores_sib_scaled = scores_sib_scaled + mx.expand_dims(sibling_mask_all, axis=-2)
            
            # Joint attention scores and softmax
            total_scores = mx.concatenate([attn_scores_near, scores_sib_scaled], axis=-1)
            total_weights = mx.softmax(total_scores, axis=-1)
            
            weights_near = total_weights[..., :B_sz]
            weights_sib = total_weights[..., B_sz:]
            
            # Outputs
            out_near = weights_near @ V_blocks
            out_sib = weights_sib @ V_sib_all
            out_block = out_near + out_sib
            
            # Restore shape and unsort
            out_tree = mx.reshape(out_block, (B, H, N_tree, self.head_dim))
            out_sorted = out_tree[:, :, :N, :]
            
            unsort_gather_idx = mx.broadcast_to(mx.reshape(unsort_idx, (B, 1, N, 1)), (B, H, N, self.head_dim))
            out_unsorted = mx.take_along_axis(out_sorted, unsort_gather_idx, axis=2)
            
            out = out_unsorted[:, :, :S, :]
            
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, S, self.embed_dim))
        out = self.out_proj(out)
        
        # nan to num equivalent
        out = mx.where(mx.isnan(out), 0.0, out)
        out = mx.where(out == float('inf'), 20.0, out)
        out = mx.where(out == -float('inf'), -20.0, out)
        
        # Differentiable hook to populate gradients for coordinate_proj
        out = out + mx.mean(coords - mx.stop_gradient(coords), axis=-1, keepdims=True)
        
        if kv_cache is not None:
            return out, kv_cache
        return out

class HyperbolicAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparse_ratio=0.15, num_key_value_heads=None, is_draft=False, depth=5, leaf_size=128):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparse_ratio = sparse_ratio
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_draft = is_draft
        self.depth = depth
        self.leaf_size = leaf_size
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / np.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Coordinate projection mapping embed_dim to 2D Poincaré coordinates
        self.coordinate_proj = nn.Linear(embed_dim, 2)
        
        # Learnable decay / scaling parameter for hyperbolic distance
        self.gamma = mx.array(1.0)
        
    def poincare_project(self, x):
        u = self.coordinate_proj(x)
        norm_u = mx.sqrt(mx.sum(u**2, axis=-1, keepdims=True) + 1e-9)
        # Use (1.0 - 1e-5) multiplier to ensure coordinates lie strictly inside the open unit disk (|z| < 1.0)
        z = (1.0 - 1e-5) * mx.tanh(norm_u) * (u / norm_u)
        return z
        
    def hyperbolic_distance(self, u, v):
        # Computes pairwise geodesic distance in the Poincaré Disk
        norm_u_sq = mx.sum(u**2, axis=-1, keepdims=True)
        norm_v_sq = mx.sum(v**2, axis=-1, keepdims=True)
        dist_sq = mx.sum((u - v)**2, axis=-1, keepdims=True)
        
        denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq)
        denom = mx.maximum(denom, 1e-9)
        delta = dist_sq / denom
        
        # Use the arcsinh formulation: arcosh(1 + 2*delta) = 2 * arcsinh(sqrt(delta))
        # arcsinh(x) = log(x + sqrt(x^2 + 1))
        sqrt_delta = mx.sqrt(mx.maximum(delta, 0.0) + 1e-12)
        dist = 2.0 * mx.log(sqrt_delta + mx.sqrt(sqrt_delta**2 + 1.0))
        # Force distance to be exactly 0 when u and v are identical
        dist = mx.where(dist_sq == 0.0, 0.0, dist)
        return mx.squeeze(dist, axis=-1)
        
    def __call__(self, x, kv_cache=None, attn_mask=None):
        dtype = x.dtype
        B, S, D = x.shape
        
        # Project inputs to Poincaré Disk coordinates
        coords = self.poincare_project(x)
        
        # Extract depth digits for bases 2 and 3
        c0 = coords[..., 0]
        c1 = coords[..., 1]
        
        # Base 2 digits
        current_0 = c0
        digits_2 = []
        for _ in range(self.depth):
            current_0 = current_0 * 2
            d = mx.floor(current_0)
            d = mx.clip(d, 0.0, 1.0)
            digits_2.append(d)
            current_0 = current_0 - d
            
        # Base 3 digits
        current_1 = c1
        digits_3 = []
        for _ in range(self.depth):
            current_1 = current_1 * 3
            d = mx.floor(current_1)
            d = mx.clip(d, 0.0, 2.0)
            digits_3.append(d)
            current_1 = current_1 - d
            
        # Interleave and pack digits into base-6 Morton code integers
        morton_code = mx.zeros(c0.shape, dtype=mx.int64)
        for k in range(self.depth):
            d_6 = digits_2[k] + 2 * digits_3[k]
            morton_code = morton_code * 6 + d_6.astype(mx.int64)
            
        # Linear projections for Q, K, V
        Q = mx.transpose(mx.reshape(self.q_proj(x), (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(self.k_proj(x), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(self.v_proj(x), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        
        # Concatenate with cached K, V, coords, and Morton codes
        if kv_cache is not None:
            if "K" in kv_cache and kv_cache["K"] is not None:
                K_combined = mx.concatenate([kv_cache["K"], K], axis=2)
                V_combined = mx.concatenate([kv_cache["V"], V], axis=2)
                coords_combined = mx.concatenate([kv_cache["coords"], coords], axis=1)
                morton_combined = mx.concatenate([kv_cache["morton_codes"], morton_code], axis=1)
            else:
                K_combined = K
                V_combined = V
                coords_combined = coords
                morton_combined = morton_code
            kv_cache["K"] = K_combined
            kv_cache["V"] = V_combined
            kv_cache["coords"] = coords_combined
            kv_cache["morton_codes"] = morton_combined
        else:
            K_combined = K
            V_combined = V
            coords_combined = coords
            morton_combined = morton_code
            
        S_total = K_combined.shape[2]
        
        # Repeat KV helper function (local)
        def repeat_kv(kv_arr, n_rep):
            if n_rep == 1:
                return kv_arr
            bs, n_kv, sl, hd = kv_arr.shape
            kv_arr = mx.expand_dims(kv_arr, axis=3)
            kv_arr = mx.broadcast_to(kv_arr, (bs, n_kv, sl, n_rep, hd))
            return mx.reshape(kv_arr, (bs, n_kv * n_rep, sl, hd))
            
        # S_total < 2048 (dynamic fallback to standard dot-product attention)
        if S_total < 2048:
            K_rep = repeat_kv(K_combined, self.num_key_value_groups)
            V_rep = repeat_kv(V_combined, self.num_key_value_groups)
            
            # Pairwise hyperbolic distance mapping
            u_coords = mx.expand_dims(coords, axis=2)
            v_coords = mx.expand_dims(coords_combined, axis=1)
            dist = self.hyperbolic_distance(u_coords, v_coords)
            
            # Hyperbolic scores
            scores = -self.gamma * dist
            scores = mx.expand_dims(scores, axis=1) # Broadcast to heads
            
            if attn_mask is not None:
                scores = scores + attn_mask
            attn_probs = mx.softmax(scores, axis=-1)
            out = attn_probs @ V_rep
        else:
            # S_total >= 2048: Fast Multipole Method (FMM) attention loop
            N = S_total
            H = self.num_heads
            
            K_rep = repeat_kv(K_combined, self.num_key_value_groups)
            V_rep = repeat_kv(V_combined, self.num_key_value_groups)
            
            # Sort sequence along sequence dimension by Morton codes (with chronological priority to prevent causal leakage)
            orig_idx = mx.broadcast_to(mx.expand_dims(mx.arange(N, dtype=mx.int64), axis=0), (B, N))
            sort_key = orig_idx * 100000000 + morton_combined
            sort_idx = mx.argsort(sort_key, axis=-1)
            unsort_idx = mx.argsort(sort_idx, axis=-1)
            
            # Pad Q to N if S < N
            if S < N:
                Q_padded = mx.pad(Q, [(0, 0), (0, 0), (0, N - S), (0, 0)], constant_values=0.0)
                coords_padded = mx.pad(coords, [(0, 0), (0, N - S), (0, 0)], constant_values=0.0)
            else:
                Q_padded = Q
                coords_padded = coords
                
            # Gather sorted tensors along sequence dimension (dim 2)
            gather_idx = mx.broadcast_to(mx.reshape(sort_idx, (B, 1, N, 1)), (B, H, N, self.head_dim))
            Q_sorted = mx.take_along_axis(Q_padded, gather_idx, axis=2)
            K_sorted = mx.take_along_axis(K_rep, gather_idx, axis=2)
            V_sorted = mx.take_along_axis(V_rep, gather_idx, axis=2)
            
            gather_idx_coords = mx.broadcast_to(mx.reshape(sort_idx, (B, N, 1)), (B, N, 2))
            coords_sorted = mx.take_along_axis(coords_padded, gather_idx_coords, axis=1)
            
            # Determine tree dimensions
            B_sz = self.leaf_size
            M = (N + B_sz - 1) // B_sz
            L = int(np.ceil(np.log2(max(M, 1))))
            M_pow = 2**L
            N_tree = M_pow * B_sz
            pad_len = N_tree - N
            
            # Construct sorted mask
            mask_val = -65000.0 if dtype in (mx.float16, mx.bfloat16) else -1e9
            mask_sorted = mx.zeros((B, H, N_tree), dtype=dtype)
            
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    mask_seq = mx.broadcast_to(mx.expand_dims(attn_mask, axis=1), (B, H, attn_mask.shape[-1]))
                elif attn_mask.ndim == 3:
                    if attn_mask.shape[1] in (H, 1):
                        mask_seq = mx.broadcast_to(attn_mask, (B, H, attn_mask.shape[-1]))
                    else:
                        mask_seq = mx.broadcast_to(mx.expand_dims(attn_mask[:, 0, :], axis=1), (B, H, attn_mask.shape[-1]))
                elif attn_mask.ndim == 4:
                    if attn_mask.shape[1] == 1:
                        mask_seq = mx.broadcast_to(mx.expand_dims(attn_mask[:, 0, 0, :], axis=1), (B, H, attn_mask.shape[-1]))
                    else:
                        mask_seq = attn_mask[:, :, 0, :]
                else:
                    mask_seq = attn_mask
                
                if mask_seq.shape[-1] < N:
                    pad_size = N - mask_seq.shape[-1]
                    mask_seq = mx.pad(mask_seq, [(0, 0), (0, 0), (0, pad_size)], constant_values=0.0)
                elif mask_seq.shape[-1] > N:
                    mask_seq = mask_seq[..., :N]
                
                gather_idx_mask = mx.broadcast_to(mx.expand_dims(sort_idx, axis=1), (B, H, N))
                mask_sorted_seq = mx.take_along_axis(mask_seq, gather_idx_mask, axis=2)
                
                if pad_len > 0:
                    mask_sorted = mx.pad(mask_sorted_seq, [(0, 0), (0, 0), (0, pad_len)], constant_values=mask_val)
                else:
                    mask_sorted = mask_sorted_seq
            else:
                if pad_len > 0:
                    mask_sorted = mx.concatenate([mx.zeros((B, H, N), dtype=dtype), mx.full((B, H, pad_len), mask_val, dtype=dtype)], axis=2)
            
            # Pad sorted tensors to N_tree
            if pad_len > 0:
                Q_tree = mx.pad(Q_sorted, [(0, 0), (0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
                K_tree = mx.pad(K_sorted, [(0, 0), (0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
                V_tree = mx.pad(V_sorted, [(0, 0), (0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
                coords_tree = mx.pad(coords_sorted, [(0, 0), (0, pad_len), (0, 0)], constant_values=0.0)
            else:
                Q_tree = Q_sorted
                K_tree = K_sorted
                V_tree = V_sorted
                coords_tree = coords_sorted
                
            # Reshape to block structure
            Q_blocks = mx.reshape(Q_tree, (B, H, M_pow, B_sz, self.head_dim))
            K_blocks = mx.reshape(K_tree, (B, H, M_pow, B_sz, self.head_dim))
            V_blocks = mx.reshape(V_tree, (B, H, M_pow, B_sz, self.head_dim))
            coords_blocks = mx.reshape(coords_tree, (B, M_pow, B_sz, 2))
            
            # Upward Pass (Aggregate nodes)
            K_tree_nodes = {}
            V_tree_nodes = {}
            coords_tree_nodes = {}
            
            is_active = (mask_sorted == 0.0).astype(dtype)
            active_count = mx.sum(mx.reshape(is_active, (B, H, M_pow, B_sz)), axis=-1, keepdims=True)
            K_tree_nodes[L] = mx.sum(K_blocks, axis=-2) / mx.maximum(active_count, 1.0)
            V_tree_nodes[L] = mx.sum(V_blocks, axis=-2) / mx.maximum(active_count, 1.0)
            
            active_count_coords = mx.sum(mx.reshape(is_active[:, 0, :], (B, M_pow, B_sz)), axis=-1, keepdims=True)
            coords_tree_nodes[L] = mx.sum(coords_blocks, axis=-2) / mx.maximum(active_count_coords, 1.0)
            
            for l in range(L - 1, -1, -1):
                parent_K = mx.mean(mx.reshape(K_tree_nodes[l+1], (B, H, 2**l, 2, self.head_dim)), axis=3)
                parent_V = mx.mean(mx.reshape(V_tree_nodes[l+1], (B, H, 2**l, 2, self.head_dim)), axis=3)
                parent_coords = mx.mean(mx.reshape(coords_tree_nodes[l+1], (B, 2**l, 2, 2)), axis=2)
                
                K_tree_nodes[l] = parent_K
                V_tree_nodes[l] = parent_V
                coords_tree_nodes[l] = parent_coords
                
            # Upward Pass for mask nodes
            mask_nodes = {}
            mask_nodes[L] = mx.max(mx.reshape(mask_sorted, (B, H, M_pow, B_sz)), axis=-1)
            for l in range(L - 1, -1, -1):
                reshaped_mask = mx.reshape(mask_nodes[l+1], (B, H, 2**l, 2))
                mask_nodes[l] = mx.maximum(reshaped_mask[..., 0], reshaped_mask[..., 1])
                
            # Near-field block distance calculations
            Q_coords_blocks = mx.reshape(coords_tree, (B, M_pow, B_sz, 2))
            u_coords_near = mx.expand_dims(mx.expand_dims(Q_coords_blocks, axis=1), axis=4)
            v_coords_near = mx.expand_dims(mx.expand_dims(coords_blocks, axis=1), axis=3)
            
            # Near-field scores based on negative hyperbolic distance
            attn_scores_near = -self.gamma * self.hyperbolic_distance(u_coords_near, v_coords_near)
            
            # Causal and padding masks in near-field
            if pad_len > 0:
                sort_idx_tree = mx.pad(sort_idx, [(0, 0), (0, pad_len)], constant_values=N + 1000)
            else:
                sort_idx_tree = sort_idx
                
            orig_indices_blocks = mx.reshape(sort_idx_tree, (B, 1, M_pow, B_sz))
            
            idx_us_1 = mx.expand_dims(orig_indices_blocks, axis=-1)
            idx_us_2 = mx.expand_dims(orig_indices_blocks, axis=-2)
            causal_mask_near = (idx_us_1 < idx_us_2)
            padding_mask_near = (idx_us_2 >= N)
            
            mask_near = (causal_mask_near | padding_mask_near).astype(dtype) * mask_val
            
            # Extract block-local slice of sorted mask and add to near-field scores
            mask_blocks = mx.reshape(mask_sorted, (B, H, M_pow, B_sz))
            attn_scores_near = attn_scores_near + mask_near + mx.expand_dims(mask_blocks, axis=-2)
            
            # Upward hierarchy of minimum and maximum original indices for causal checks
            min_orig_node = {}
            max_orig_node = {}
            min_orig_node[L] = mx.min(orig_indices_blocks, axis=-1)
            max_orig_node[L] = mx.max(orig_indices_blocks, axis=-1)
            for l in range(L - 1, -1, -1):
                min_reshaped = mx.reshape(min_orig_node[l+1], (B, 1, 2**l, 2))
                max_reshaped = mx.reshape(max_orig_node[l+1], (B, 1, 2**l, 2))
                min_orig_node[l] = mx.minimum(min_reshaped[..., 0], min_reshaped[..., 1])
                max_orig_node[l] = mx.maximum(max_reshaped[..., 0], max_reshaped[..., 1])
                
            # Far-field aggregated sibling nodes attention
            j_indices = mx.arange(M_pow, dtype=mx.int64)
            K_sibs = []
            V_sibs = []
            coords_sibs = []
            is_sib_padded_list = []
            is_sib_causal_violation_list = []
            sibling_mask_list = []
            
            for l in range(1, L + 1):
                ancestor_indices = j_indices // (2**(L - l))
                sibling_indices = ancestor_indices ^ 1
                
                gather_idx_sib = mx.broadcast_to(mx.reshape(sibling_indices, (1, 1, M_pow, 1)), (B, H, M_pow, self.head_dim))
                K_sib_l = mx.take_along_axis(K_tree_nodes[l], gather_idx_sib, axis=2)
                V_sib_l = mx.take_along_axis(V_tree_nodes[l], gather_idx_sib, axis=2)
                
                gather_idx_sib_coords = mx.broadcast_to(mx.reshape(sibling_indices, (1, M_pow, 1)), (B, M_pow, 2))
                coords_sib_l = mx.take_along_axis(coords_tree_nodes[l], gather_idx_sib_coords, axis=1)
                
                K_sibs.append(mx.expand_dims(K_sib_l, axis=3))
                V_sibs.append(mx.expand_dims(V_sib_l, axis=3))
                coords_sibs.append(mx.expand_dims(coords_sib_l, axis=2))
                
                # Check sibling padding
                sibling_start_block = sibling_indices * (2**(L - l))
                is_sib_padded_l = (sibling_start_block >= M)
                is_sib_padded_list.append(mx.reshape(is_sib_padded_l, (1, 1, M_pow, 1, 1)))
                
                # Check sibling causal violation
                gather_idx_max = mx.broadcast_to(mx.reshape(sibling_indices, (1, 1, M_pow)), (B, 1, M_pow))
                sibling_max_orig_l = mx.take_along_axis(max_orig_node[l], gather_idx_max, axis=2)
                causal_mask_sib_l = (mx.expand_dims(sibling_max_orig_l, axis=-1) > orig_indices_blocks)
                is_sib_causal_violation_list.append(mx.expand_dims(causal_mask_sib_l, axis=-1))
                
                # Sibling mask
                gather_idx_mask = mx.broadcast_to(mx.reshape(sibling_indices, (1, 1, M_pow)), (B, H, M_pow))
                sibling_mask_l = mx.take_along_axis(mask_nodes[l], gather_idx_mask, axis=2)
                sibling_mask_list.append(mx.expand_dims(sibling_mask_l, axis=-1))
                
            K_sib_all = mx.concatenate(K_sibs, axis=3) # Shape: [B, H, M_pow, L, D]
            V_sib_all = mx.concatenate(V_sibs, axis=3) # Shape: [B, H, M_pow, L, D]
            coords_sib_all = mx.concatenate(coords_sibs, axis=2) # Shape: [B, M_pow, L, 2]
            
            is_sib_padded_all = mx.concatenate(is_sib_padded_list, axis=4) # Shape: [1, 1, M_pow, 1, L]
            is_sib_causal_violation_all = mx.concatenate(is_sib_causal_violation_list, axis=-1) # Shape: [B, 1, M_pow, B_sz, L]
            sibling_mask_all = mx.concatenate(sibling_mask_list, axis=-1) # Shape: [B, H, M_pow, L]
            
            # Sibling scores
            u_coords_sib = mx.expand_dims(mx.expand_dims(Q_coords_blocks, axis=1), axis=4)
            v_coords_sib = mx.expand_dims(mx.expand_dims(coords_sib_all, axis=1), axis=3)
            
            scores_sib = -self.gamma * self.hyperbolic_distance(u_coords_sib, v_coords_sib)
            
            # Mask padded and causal violated sibling nodes
            is_sib_invalid = (is_sib_padded_all | is_sib_causal_violation_all)
            scores_sib_scaled = mx.where(is_sib_invalid, mask_val, scores_sib)
            
            # Add sibling mask
            scores_sib_scaled = scores_sib_scaled + mx.expand_dims(sibling_mask_all, axis=-2)
            
            # Joint attention scores and softmax
            total_scores = mx.concatenate([attn_scores_near, scores_sib_scaled], axis=-1)
            total_weights = mx.softmax(total_scores, axis=-1)
            
            weights_near = total_weights[..., :B_sz]
            weights_sib = total_weights[..., B_sz:]
            
            # Outputs
            out_near = weights_near @ V_blocks
            out_sib = weights_sib @ V_sib_all
            out_block = out_near + out_sib
            
            # Restore shape and unsort
            out_tree = mx.reshape(out_block, (B, H, N_tree, self.head_dim))
            out_sorted = out_tree[:, :, :N, :]
            
            unsort_gather_idx = mx.broadcast_to(mx.reshape(unsort_idx, (B, 1, N, 1)), (B, H, N, self.head_dim))
            out_unsorted = mx.take_along_axis(out_sorted, unsort_gather_idx, axis=2)
            
            out = out_unsorted[:, :, :S, :]
            
        out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, S, self.embed_dim))
        out = self.out_proj(out)
        
        # nan to num equivalent
        out = mx.where(mx.isnan(out), 0.0, out)
        out = mx.where(out == float('inf'), 20.0, out)
        out = mx.where(out == -float('inf'), -20.0, out)
        
        # Differentiable hook to populate gradients for coordinate_proj
        out = out + mx.mean(coords - mx.stop_gradient(coords), axis=-1, keepdims=True)
        
        if kv_cache is not None:
            return out, kv_cache
        return out

class AnyonicBraidLinear(nn.Module):
    def __init__(self, input_dim, output_dim, depth=16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        
        # Determine internal size (must be even)
        self.N = max(input_dim, output_dim)
        if self.N % 2 != 0:
            self.N += 1
            
        self.num_pairs_odd = self.N // 2
        self.num_pairs_even = (self.N - 2) // 2
        
        # Trainable parameters
        self.weight_odd = mx.random.uniform(-0.1, 0.1, (self.num_pairs_odd, depth))
        self.weight_even = mx.random.uniform(-0.1, 0.1, (self.num_pairs_even, depth))
        self.theta = mx.array(3.141592653589793 / 4.0)
        
    def __call__(self, x):
        B, S, C = x.shape
        dtype = x.dtype
        
        # Pad input to N if C < N
        if C < self.N:
            x_padded = mx.pad(x, [(0, 0), (0, 0), (0, self.N - C)], constant_values=0.0)
        else:
            x_padded = x
            
        # Straight-Through Estimator for discrete weights
        W_odd_discrete = mx.where(self.weight_odd >= 0.0, 1.0, -1.0)
        W_odd = W_odd_discrete - mx.stop_gradient(self.weight_odd) + self.weight_odd
        
        W_even_discrete = mx.where(self.weight_even >= 0.0, 1.0, -1.0)
        W_even = W_even_discrete - mx.stop_gradient(self.weight_even) + self.weight_even
        
        cos_theta = mx.cos(self.theta)
        sin_theta = mx.sin(self.theta)
        
        current_x = x_padded
        for d in range(self.depth):
            # Alternating brickwork layers
            if d % 2 == 0:
                # Odd layer (0-indexed pairs: (0, 1), (2, 3), ...)
                reshaped = mx.reshape(current_x, (B, S, self.num_pairs_odd, 2))
                x1 = reshaped[..., 0]
                x2 = reshaped[..., 1]
                s = mx.reshape(W_odd[:, d], (1, 1, self.num_pairs_odd))
                
                new_x1 = cos_theta * x1 + s * sin_theta * x2
                new_x2 = -s * sin_theta * x1 + cos_theta * x2
                
                # Stack along last axis to restore pair size 2
                stacked = mx.stack([new_x1, new_x2], axis=-1)
                current_x = mx.reshape(stacked, (B, S, self.N))
            else:
                # Even layer (0-indexed middle pairs: (1, 2), (3, 4), ...)
                x_middle = current_x[..., 1:self.N-1]
                reshaped = mx.reshape(x_middle, (B, S, self.num_pairs_even, 2))
                x1 = reshaped[..., 0]
                x2 = reshaped[..., 1]
                s = mx.reshape(W_even[:, d], (1, 1, self.num_pairs_even))
                
                new_x1 = cos_theta * x1 + s * sin_theta * x2
                new_x2 = -s * sin_theta * x1 + cos_theta * x2
                
                stacked = mx.stack([new_x1, new_x2], axis=-1)
                new_middle = mx.reshape(stacked, (B, S, self.N - 2))
                
                current_x = mx.concatenate([current_x[..., :1], new_middle, current_x[..., -1:]], axis=-1)
                
        # Slice output to output_dim
        out = current_x[..., :self.output_dim]
        return out.astype(dtype)

class AutomorphicSpectralAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        num_modes: int = 32
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
        if num_modes % 2 != 0:
            raise ValueError(f"num_modes ({num_modes}) must be an even integer")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = embed_dim // num_heads
        self.num_modes = num_modes
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Project tokens to upper half-plane H = { u + i*v : v > 0 }
        self.coordinate_proj = nn.Linear(embed_dim, 2)
        
        # Learnable spectral eigenvalues/frequencies (positive)
        self.eigenvalues = mx.array(np.random.uniform(0.1, 2.0, (num_modes // 2,)), dtype=mx.float32)

    def _softplus(self, val: mx.array) -> mx.array:
        return mx.log(1.0 + mx.exp(-mx.abs(val))) + mx.maximum(val, 0.0)

    def modular_reduce(self, u: mx.array, v: mx.array, steps: int = 5) -> tuple[mx.array, mx.array]:
        # Reduces coordinates to the fundamental domain of SL_2(Z)
        # Fundamental domain is: |u| <= 0.5 and u^2 + v^2 >= 1.0
        for _ in range(steps):
            # T-shift: shift u to be in [-0.5, 0.5]
            shift = mx.round(u)
            u = u - shift
            
            # S-inversion: if u^2 + v^2 < 1.0, invert tau -> -1/tau
            denom = u**2 + v**2
            cond = denom < 1.0
            
            u_inv = -u / mx.maximum(denom, 1e-9)
            v_inv = v / mx.maximum(denom, 1e-9)
            
            u = mx.where(cond, u_inv, u)
            v = mx.where(cond, v_inv, v)
        return u, v

    def __call__(
        self,
        x: mx.array,
        kv_cache: Optional[dict] = None,
        attn_mask: Optional[mx.array] = None
    ) -> mx.array:
        B, S, D = x.shape
        dtype = x.dtype
        
        # 1. Project to upper half-plane
        coords = self.coordinate_proj(x)
        u = coords[..., 0]
        v = self._softplus(coords[..., 1]) + 1e-5
        
        # 2. Reduce to fundamental domain of SL_2(Z)
        u_star, v_star = self.modular_reduce(u, v)
        
        # 3. Compute Automorphic Spectral Basis Psi
        freqs = mx.arange(1, self.num_modes // 2 + 1, dtype=mx.float32)
        angle = 2.0 * np.pi * mx.expand_dims(u_star, axis=-1) * freqs
        cos_part = mx.cos(angle)
        sin_part = mx.sin(angle)
        
        # v-dependent spectral decay factor (eigenvalues represent Laplacian eigenvalues lambda_k)
        lambda_k = self._softplus(self.eigenvalues)
        v_expanded = mx.expand_dims(v_star, axis=-1)
        v_part = mx.power(mx.maximum(v_expanded, 1e-9), -lambda_k)
        
        psi_cos = cos_part * v_part
        psi_sin = sin_part * v_part
        psi = mx.concatenate([psi_cos, psi_sin], axis=-1) # Shape: [B, S, F]
        
        # Normalize Psi for numerical stability
        psi = psi / (mx.linalg.norm(psi, axis=-1, keepdims=True) + 1e-6)
        psi_expanded = mx.expand_dims(psi, axis=1) # Shape: [B, 1, S, F]
        
        # 4. Project Q, K, V
        Q = mx.transpose(mx.reshape(self.q_proj(x), (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(self.k_proj(x), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(self.v_proj(x), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        
        # GQA replication if necessary
        def repeat_kv(kv_arr, n_rep):
            if n_rep == 1:
                return kv_arr
            bs, n_kv, sl, hd = kv_arr.shape
            kv_arr = mx.expand_dims(kv_arr, axis=3)
            kv_arr = mx.broadcast_to(kv_arr, (bs, n_kv, sl, n_rep, hd))
            return mx.reshape(kv_arr, (bs, n_kv * n_rep, sl, hd))
            
        K_rep = repeat_kv(K, self.num_key_value_groups)
        V_rep = repeat_kv(V, self.num_key_value_groups)
        
        # 5. Non-negative Linear Causal Attention Feature Mapping
        # phi(Q) = softplus(Q) * psi, phi(K) = softplus(K) * psi
        phi_Q = self._softplus(Q)[..., None, :] * psi_expanded[..., :, None]
        phi_K = self._softplus(K_rep)[..., None, :] * psi_expanded[..., :, None]
        
        if kv_cache is not None:
            # Cache stores cumulative states: K_state [B, H, F, D] and denom_state [B, H, F, D]
            if "K_state" in kv_cache and kv_cache["K_state"] is not None:
                K_state_prev = kv_cache["K_state"]
                denom_state_prev = kv_cache["denom_state"]
                offset = kv_cache["seq_len"]
            else:
                K_state_prev = mx.zeros((B, self.num_heads, self.num_modes, self.head_dim), dtype=dtype)
                denom_state_prev = mx.zeros((B, self.num_heads, self.num_modes, self.head_dim), dtype=dtype)
                offset = 0
            
            # term = phi_K * V (expanded) -> shape [B, H, S, F, D]
            term = phi_K * mx.expand_dims(V_rep, axis=3) # [B, H, S, F, D]
            
            if S == 1:
                K_state = K_state_prev + term[:, :, 0]
                denom_state = denom_state_prev + phi_K[:, :, 0]
                
                num = mx.sum(phi_Q[:, :, 0] * K_state, axis=2)
                den = mx.sum(phi_Q[:, :, 0] * denom_state, axis=2)
                out = num / (den + 1e-6)
                out = mx.expand_dims(out, axis=2) # [B, H, 1, D]
            else:
                term_cumsum = mx.cumsum(term, axis=2) # [B, H, S, F, D]
                K_state_seq = K_state_prev[:, :, None] + term_cumsum
                
                denom_cumsum = mx.cumsum(phi_K, axis=2) # [B, H, S, F, D]
                denom_state_seq = denom_state_prev[:, :, None] + denom_cumsum
                
                num = mx.sum(phi_Q * K_state_seq, axis=3)
                den = mx.sum(phi_Q * denom_state_seq, axis=3)
                out = num / (den + 1e-6) # [B, H, S, D]
                
                # Store final states
                K_state = K_state_seq[:, :, -1]
                denom_state = denom_state_seq[:, :, -1]
                
            kv_cache["K_state"] = K_state
            kv_cache["denom_state"] = denom_state
            kv_cache["seq_len"] = offset + S
        else:
            term = phi_K * mx.expand_dims(V_rep, axis=3) # [B, H, S, F, D]
            K_state_seq = mx.cumsum(term, axis=2) # [B, H, S, F, D]
            denom_state_seq = mx.cumsum(phi_K, axis=2) # [B, H, S, F, D]
            
            num = mx.sum(phi_Q * K_state_seq, axis=3)
            den = mx.sum(phi_Q * denom_state_seq, axis=3)
            out = num / (den + 1e-6) # [B, H, S, D]
            
        out = mx.transpose(out, (0, 2, 1, 3)) # [B, S, H, D]
        out = mx.reshape(out, (B, S, -1)) # [B, S, embed_dim]
        out = self.out_proj(out)
        return out

class SymplecticHamiltonianAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        num_steps: int = 3,
        dt: float = 0.2
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = embed_dim // num_heads
        self.num_steps = num_steps
        self.dt = dt
        
        self.p_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Step-wise projections for each integration step
        self.q_projs = [nn.Linear(embed_dim, embed_dim) for _ in range(num_steps)]
        self.k_projs = [nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim) for _ in range(num_steps)]
        self.v_projs = [nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim) for _ in range(num_steps)]
        
        for k in range(num_steps):
            setattr(self, f"q_proj_{k}", self.q_projs[k])
            setattr(self, f"k_proj_{k}", self.k_projs[k])
            setattr(self, f"v_proj_{k}", self.v_projs[k])
            
    def _softplus(self, val: mx.array) -> mx.array:
        return mx.log(1.0 + mx.exp(-mx.abs(val))) + mx.maximum(val, 0.0)

    def __call__(
        self,
        x: mx.array,
        kv_cache: Optional[dict] = None,
        attn_mask: Optional[mx.array] = None
    ) -> mx.array:
        B, S, D = x.shape
        dtype = x.dtype
        
        # Initialize Hamiltonian state: position q0 and momentum p0
        q = x
        p = self.p_proj(x)
        
        # GQA replication helper
        def repeat_kv(kv_arr, n_rep):
            if n_rep == 1:
                return kv_arr
            bs, n_kv, sl, hd = kv_arr.shape
            kv_arr = mx.expand_dims(kv_arr, axis=3)
            kv_arr = mx.broadcast_to(kv_arr, (bs, n_kv, sl, n_rep, hd))
            return mx.reshape(kv_arr, (bs, n_kv * n_rep, sl, hd))
            
        for k in range(self.num_steps):
            q_proj_k = getattr(self, f"q_proj_{k}")
            k_proj_k = getattr(self, f"k_proj_{k}")
            v_proj_k = getattr(self, f"v_proj_{k}")
            
            Q_k = mx.transpose(mx.reshape(q_proj_k(q), (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
            K_k = mx.transpose(mx.reshape(k_proj_k(q), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
            V_k = mx.transpose(mx.reshape(v_proj_k(q), (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
            
            K_rep = repeat_kv(K_k, self.num_key_value_groups)
            V_rep = repeat_kv(V_k, self.num_key_value_groups)
            
            phi_Q = self._softplus(Q_k)
            phi_K = self._softplus(K_rep)
            
            # Causal force F_k (Causal Linear Attention)
            if kv_cache is not None:
                k_state_key = f"K_state_{k}"
                denom_state_key = f"denom_state_{k}"
                
                if k_state_key in kv_cache and kv_cache[k_state_key] is not None:
                    K_state_prev = kv_cache[k_state_key]
                    denom_state_prev = kv_cache[denom_state_key]
                    offset = kv_cache[f"seq_len_{k}"]
                else:
                    K_state_prev = mx.zeros((B, self.num_heads, self.head_dim, self.head_dim), dtype=dtype)
                    denom_state_prev = mx.zeros((B, self.num_heads, self.head_dim), dtype=dtype)
                    offset = 0
                    
                term = phi_K[..., :, None] * V_rep[..., None, :] # [B, H, S, D, D]
                
                if S == 1:
                    K_state = K_state_prev + term[:, :, 0]
                    denom_state = denom_state_prev + phi_K[:, :, 0]
                    
                    num = phi_Q @ K_state
                    den = mx.sum(phi_Q * mx.expand_dims(denom_state, axis=2), axis=-1, keepdims=True)
                    F_k = num / (den + 1e-6)
                else:
                    term_cumsum = mx.cumsum(term, axis=2)
                    K_state_seq = K_state_prev[:, :, None] + term_cumsum
                    
                    denom_cumsum = mx.cumsum(phi_K, axis=2)
                    denom_state_seq = denom_state_prev[:, :, None] + denom_cumsum
                    
                    num = mx.squeeze(mx.expand_dims(phi_Q, axis=3) @ K_state_seq, axis=3)
                    den = mx.sum(phi_Q * denom_state_seq, axis=-1, keepdims=True)
                    F_k = num / (den + 1e-6)
                    
                    K_state = K_state_seq[:, :, -1]
                    denom_state = denom_state_seq[:, :, -1]
                    
                kv_cache[k_state_key] = K_state
                kv_cache[denom_state_key] = denom_state
                kv_cache[f"seq_len_{k}"] = offset + S
            else:
                term = phi_K[..., :, None] * V_rep[..., None, :]
                K_state_seq = mx.cumsum(term, axis=2)
                denom_state_seq = mx.cumsum(phi_K, axis=2)
                
                num = mx.squeeze(mx.expand_dims(phi_Q, axis=3) @ K_state_seq, axis=3)
                den = mx.sum(phi_Q * denom_state_seq, axis=-1, keepdims=True)
                F_k = num / (den + 1e-6)
                
            F_k_spatial = mx.transpose(F_k, (0, 2, 1, 3))
            F_k_spatial = mx.reshape(F_k_spatial, (B, S, -1))
            
            p = p + self.dt * F_k_spatial
            q = q + self.dt * p
            
        return self.out_proj(q)


class GaloisAdapterLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, num_tasks: int = 4, scale: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.num_tasks = num_tasks
        self.scale = scale
        
        # 1. Precompute GF(2^8) tables
        exp_table_np = np.zeros(512, dtype=np.int32)
        log_table_np = np.zeros(256, dtype=np.int32)
        val = 1
        for i in range(255):
            exp_table_np[i] = val
            log_table_np[val] = i
            val <<= 1
            if val & 256:
                val ^= 285
        for i in range(255, 512):
            exp_table_np[i] = exp_table_np[i - 255]
            
        self.exp_table = mx.array(exp_table_np, dtype=mx.uint8)
        self.log_table = mx.array(log_table_np, dtype=mx.uint8)
        
        # 2. Setup A_proj
        self.A_proj = nn.Linear(input_dim, rank, bias=False)
        
        # 3. Setup points and Lagrange weights
        x_pts = list(range(1, num_tasks + 1))
        y_pts = list(range(num_tasks + 1, 2 * num_tasks + 1))
        
        exp_py = exp_table_np.tolist()
        log_py = log_table_np.tolist()
        
        def gf_add_py(u, v):
            return u ^ v
        def gf_mul_py(u, v):
            if u == 0 or v == 0:
                return 0
            return exp_py[(log_py[u] + log_py[v]) % 255]
        def gf_div_py(u, v):
            if u == 0:
                return 0
            return exp_py[(log_py[u] - log_py[v] + 255) % 255]
            
        lagrange_weights_np = np.zeros((num_tasks, num_tasks), dtype=np.uint8)
        for c in range(num_tasks):
            for i in range(num_tasks):
                w_val = 1
                for j in range(num_tasks):
                    if j != i:
                        num = gf_add_py(y_pts[c], x_pts[j])
                        den = gf_add_py(x_pts[i], x_pts[j])
                        term = gf_div_py(num, den)
                        w_val = gf_mul_py(w_val, term)
                lagrange_weights_np[c, i] = w_val
                
        self.lagrange_weights = mx.array(lagrange_weights_np, dtype=mx.uint8)
        
        # Precompute encoding weights matrix v_np
        v_np = np.zeros((num_tasks, num_tasks), dtype=np.uint8)
        for i in range(num_tasks):
            for c in range(num_tasks):
                v_val = 1
                for d in range(num_tasks):
                    if d != c:
                        num = gf_add_py(x_pts[i], y_pts[d])
                        den = gf_add_py(y_pts[c], y_pts[d])
                        term = gf_div_py(num, den)
                        v_val = gf_mul_py(v_val, term)
                v_np[i, c] = v_val
                
        # Initialize original W to 128 (corresponding to float 0.0)
        W_int = np.ones((num_tasks, rank, output_dim), dtype=np.uint8) * 128
        
        # Encode W_int to M
        M_np = np.zeros((num_tasks, rank, output_dim), dtype=np.uint8)
        for i in range(num_tasks):
            val_mat = np.zeros((rank, output_dim), dtype=np.uint8)
            for c in range(num_tasks):
                v_expanded = v_np[i, c]
                for r in range(rank):
                    for d_idx in range(output_dim):
                        term = gf_mul_py(v_expanded, W_int[c, r, d_idx])
                        val_mat[r, d_idx] = gf_add_py(val_mat[r, d_idx], term)
            M_np[i] = val_mat
            
        self.M = mx.array(M_np, dtype=mx.uint8)
        
    def gf_add(self, u: mx.array, v: mx.array) -> mx.array:
        return u ^ v
        
    def gf_mul(self, u: mx.array, v: mx.array) -> mx.array:
        log_u = self.log_table[u].astype(mx.int32)
        log_v = self.log_table[v].astype(mx.int32)
        exp_idx = (log_u + log_v) % 255
        prod = self.exp_table[exp_idx]
        return mx.where((u == 0) | (v == 0), mx.array(0, dtype=mx.uint8), prod)
        
    def decode_adapter(self, task_idx: int) -> mx.array:
        w_target = self.lagrange_weights[task_idx]
        w_expanded = mx.reshape(w_target, (self.num_tasks, 1, 1))
        
        term = self.gf_mul(w_expanded, self.M)
        
        decoded = term[0]
        for idx in range(1, self.num_tasks):
            decoded = decoded ^ term[idx]
            
        W_float = (decoded.astype(mx.float32) - 128.0) * self.scale
        return W_float.astype(mx.float16)
        
    def __call__(self, x: mx.array, task_idx: int = 0) -> mx.array:
        r = self.A_proj(x)
        W_B = self.decode_adapter(task_idx)
        return r @ W_B


class OctonionicHyperbolicAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        scale: Optional[float] = None
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim % 8 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 8 for octonionic representation.")
            
        self.C_oct = self.head_dim // 8
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Imaginary units Fano plane triplets definition
        table = {}
        for i in range(8):
            table[(0, i)] = (i, 1)
            table[(i, 0)] = (i, 1)
        for i in range(1, 8):
            table[(i, i)] = (0, -1)
        triplets = [
            (1, 2, 3), (1, 4, 5), (1, 7, 6), (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5)
        ]
        for a, b, c in triplets:
            table[(a, b)] = (c, 1)
            table[(b, c)] = (a, 1)
            table[(c, a)] = (b, 1)
            table[(b, a)] = (c, -1)
            table[(c, b)] = (a, -1)
            table[(a, c)] = (b, -1)
            
        # Structure constants C_str for imaginary units cross products (shape: 7, 7, 7)
        C_str_np = np.zeros((7, 7, 7), dtype=np.float32)
        for r in range(1, 8):
            for s in range(1, 8):
                if r != s:
                    m, sign = table[(r, s)]
                    if m >= 1:
                        C_str_np[r-1, s-1, m-1] = sign
                        
        self.C_str = mx.array(C_str_np)
        
        # Learnable parameters
        self.gamma = mx.array([0.1] * 7, dtype=mx.float32)
        self.beta = mx.array(scale if scale is not None else (1.0 / np.sqrt(self.head_dim)), dtype=mx.float32)
        
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        kv_cache: Optional[dict] = None
    ) -> mx.array:
        B, S, D = x.shape
        
        # Project states
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to multi-head / octonionic layout
        Q = mx.transpose(mx.reshape(Q, (B, S, self.num_heads, self.C_oct, 8)), (0, 2, 1, 3, 4))
        K = mx.transpose(mx.reshape(K, (B, S, self.num_key_value_heads, self.C_oct, 8)), (0, 2, 1, 3, 4))
        V = mx.transpose(mx.reshape(V, (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        
        # Caching logic
        if kv_cache is not None:
            if "K" in kv_cache and kv_cache["K"] is not None:
                K = mx.concatenate([kv_cache["K"], K], axis=2)
                V = mx.concatenate([kv_cache["V"], V], axis=2)
            kv_cache["K"] = K
            kv_cache["V"] = V
            
        # GQA replication helper
        def repeat_kv(arr, n_rep):
            if n_rep == 1:
                return arr
            bs, n_kv, sl = arr.shape[:3]
            extra_dims = arr.shape[3:]
            arr = mx.expand_dims(arr, axis=2)
            broadcast_shape = (bs, n_kv, n_rep, sl) + extra_dims
            arr = mx.broadcast_to(arr, broadcast_shape)
            reshape_shape = (bs, n_kv * n_rep, sl) + extra_dims
            return mx.reshape(arr, reshape_shape)
            
        K_rep = repeat_kv(K, self.num_key_value_groups)
        V_rep = repeat_kv(V, self.num_key_value_groups)
        
        S_q = Q.shape[2]
        S_k = K_rep.shape[2]
        
        # Temporal coordinates (hyperboloid time components)
        t_Q = mx.sqrt(1.0 + mx.sum(Q**2, axis=-1))
        t_K = mx.sqrt(1.0 + mx.sum(K_rep**2, axis=-1))
        t_prod = mx.expand_dims(t_Q, 3) * mx.expand_dims(t_K, 2)  # [B, H, S_q, S_k, C_oct]
        
        # Algebraic optimized octonionic product calculations (avoiding 8x8 outer product)
        # 1. Real part: Q_t @ K_t^T
        Q_t = mx.transpose(Q, (0, 1, 3, 2, 4))
        K_t = mx.transpose(K_rep, (0, 1, 3, 2, 4))
        real_part = Q_t @ mx.transpose(K_t, (0, 1, 2, 4, 3))  # [B, H, C_oct, S_q, S_k]
        real_part = mx.transpose(real_part, (0, 1, 3, 4, 2))  # [B, H, S_q, S_k, C_oct]
        
        # 2. Linear imaginary part: k0 * A - q0 * B
        Q_imag = Q[..., 1:]
        K_imag = K_rep[..., 1:]
        
        A = mx.sum(Q_imag * self.gamma, axis=-1)
        B_val = mx.sum(K_imag * self.gamma, axis=-1)
        
        q_0 = Q[..., 0]
        k_0 = K_rep[..., 0]
        
        A_exp = mx.expand_dims(A, 3)
        B_exp = mx.expand_dims(B_val, 2)
        q0_exp = mx.expand_dims(q_0, 3)
        k0_exp = mx.expand_dims(k_0, 2)
        
        linear_part = k0_exp * A_exp - q0_exp * B_exp
        
        # 3. Bilinear cross part: Q_imag @ G @ K_imag^T
        G = mx.sum(self.C_str * mx.reshape(self.gamma, (1, 1, 7)), axis=-1)
        
        Q_imag_t = mx.transpose(Q_imag, (0, 1, 3, 2, 4))
        K_imag_t = mx.transpose(K_imag, (0, 1, 3, 2, 4))
        
        Q_g = Q_imag_t @ G
        cross_part = Q_g @ mx.transpose(K_imag_t, (0, 1, 2, 4, 3))  # [B, H, C_oct, S_q, S_k]
        cross_part = mx.transpose(cross_part, (0, 1, 3, 4, 2))  # [B, H, S_q, S_k, C_oct]
        
        # Total Lorentzian inner product
        lorentz_inner = -t_prod + real_part + linear_part - cross_part
        
        # Hyperbolic distance via acosh
        y = mx.maximum(1.0 + 1e-6, -lorentz_inner)
        dist = mx.log(y + mx.sqrt(y**2 - 1.0))
        dist_sum = mx.sum(dist, axis=-1)  # [B, H, S_q, S_k]
        
        # Scores and attention
        scores = -self.beta * dist_sum
        if mask is not None:
            scores = scores + mask
            
        attn_probs = mx.softmax(scores, axis=-1)
        out = attn_probs @ V_rep
        
        # Project back to embedding dimension
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (B, S_q, -1))
        out = self.out_proj(out)
        
        return out


class KnotEntanglementAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        scale: Optional[float] = None
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = embed_dim // num_heads
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.t_proj = nn.Linear(embed_dim, self.num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Learnable scale / parameter beta
        self.beta = mx.array(scale if scale is not None else (1.0 / np.sqrt(self.head_dim)), dtype=mx.float32)
        
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        kv_cache: Optional[dict] = None,
        attn_mask: Optional[mx.array] = None
    ) -> mx.array:
        mask = mask if mask is not None else attn_mask
        B, S, D = x.shape
        
        # Project states
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape to multi-head layout
        Q = mx.transpose(mx.reshape(Q, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        K = mx.transpose(mx.reshape(K, (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        V = mx.transpose(mx.reshape(V, (B, S, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        
        # Project parameter t (input-dependent braid crossings)
        t = mx.transpose(mx.sigmoid(self.t_proj(x)) + 1e-4, (0, 2, 1)) # (B, H, S)
        
        # Caching logic
        if kv_cache is not None:
            if "K" in kv_cache and kv_cache["K"] is not None:
                K = mx.concatenate([kv_cache["K"], K], axis=2)
                V = mx.concatenate([kv_cache["V"], V], axis=2)
                
                t_prev = kv_cache["t"]
                P_prev = kv_cache["P"]
                S_sum_prev = kv_cache["S_sum"]
                
                # Align batch/head dimensions if they got modified (e.g. padding/batch changes)
                P_last = P_prev[..., -1:]
                P_new = P_last * mx.cumprod(t, axis=-1)
                S_sum_last = S_sum_prev[..., -1:]
                S_sum_new = S_sum_last + mx.cumsum(P_new, axis=-1)
                
                t = mx.concatenate([t_prev, t], axis=-1)
                P = mx.concatenate([P_prev, P_new], axis=-1)
                S_sum = mx.concatenate([S_sum_prev, S_sum_new], axis=-1)
            else:
                P = mx.cumprod(t, axis=-1)
                S_sum = mx.cumsum(P, axis=-1)
            
            kv_cache["K"] = K
            kv_cache["V"] = V
            kv_cache["t"] = t
            kv_cache["P"] = P
            kv_cache["S_sum"] = S_sum
        else:
            P = mx.cumprod(t, axis=-1)
            S_sum = mx.cumsum(P, axis=-1)
            
        # GQA replication helper
        def repeat_kv(arr, n_rep):
            if n_rep == 1:
                return arr
            bs, n_kv, sl = arr.shape[:3]
            extra_dims = arr.shape[3:]
            arr = mx.expand_dims(arr, axis=2)
            broadcast_shape = (bs, n_kv, n_rep, sl) + extra_dims
            arr = mx.broadcast_to(arr, broadcast_shape)
            reshape_shape = (bs, n_kv * n_rep, sl) + extra_dims
            return mx.reshape(arr, reshape_shape)
            
        K_rep = repeat_kv(K, self.num_key_value_groups)
        V_rep = repeat_kv(V, self.num_key_value_groups)
        
        S_kv = K_rep.shape[2]
        S_q = Q.shape[2]
        
        # Pad S_sum and P along sequence dimension
        S_pad = mx.concatenate([mx.zeros((B, self.num_heads, 1), dtype=S_sum.dtype), S_sum], axis=-1)
        P_pad = mx.concatenate([mx.ones((B, self.num_heads, 1), dtype=P.dtype), P], axis=-1)
        
        # Slice corresponding query and key positions from padded tensors
        S_i = S_pad[..., S_kv - S_q : S_kv] # (B, H, S_q)
        S_j = S_pad[..., :S_kv] # (B, H, S_kv)
        P_j = P_pad[..., :S_kv] # (B, H, S_kv)
        
        # Expand dimensions for broadcasting
        S_i_exp = mx.expand_dims(S_i, axis=-1) # (B, H, S_q, 1)
        S_j_exp = mx.expand_dims(S_j, axis=-2) # (B, H, 1, S_kv)
        P_j_exp = mx.expand_dims(P_j, axis=-2) # (B, H, 1, S_kv)
        
        # Compute active block determinants of Burau matrices (Alexander polynomial)
        dets = 1.0 + (S_i_exp - S_j_exp) / mx.clip(P_j_exp, 1e-12, None) # (B, H, S_q, S_kv)
        
        # Topological/semantic attention scores combining QK-dot product and Alexander Polynomial knot invariant
        semantic_scores = (Q @ mx.transpose(K_rep, (0, 1, 3, 2))) * (1.0 / np.sqrt(self.head_dim))
        knot_scores = -self.beta * mx.log(1.0 + dets**2)
        scores = semantic_scores + knot_scores
        
        if mask is not None:
            scores = scores + mask
            
        attn_probs = mx.softmax(scores, axis=-1)
        out = attn_probs @ V_rep
        
        # Project back to embedding dimension
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (B, S_q, -1))
        out = self.out_proj(out)
        
        return out


class GrothendieckTopologyMoE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int = 4,
        hidden_dim: Optional[int] = None,
        overlap_dim: int = 16,
        temperature: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim if hidden_dim is not None else 4 * embed_dim
        self.overlap_dim = overlap_dim
        self.temperature = temperature
        
        # Router
        self.router = nn.Linear(embed_dim, num_experts)
        
        # Experts FFNs
        self.experts = [
            nn.Sequential(
                nn.Linear(embed_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, embed_dim)
            ) for _ in range(num_experts)
        ]
        for k in range(num_experts):
            setattr(self, f"expert_{k}", self.experts[k])
            
        # Restriction maps (sheaf projection to overlap space)
        self.restriction_maps = [
            nn.Linear(embed_dim, overlap_dim) for _ in range(num_experts)
        ]
        for k in range(num_experts):
            setattr(self, f"restriction_{k}", self.restriction_maps[k])
            
        # Gluing projection (overlap space back to embedding dimension)
        self.glue_proj = nn.Linear(overlap_dim, embed_dim)
        
    def __call__(self, x: mx.array) -> mx.array:
        B, S, D = x.shape
        
        # 1. Compute Cover routing coefficients (partition of unity)
        router_scores = self.router(x) # (B, S, E)
        w = mx.softmax(router_scores / self.temperature, axis=-1) # (B, S, E)
        
        # 2. Compute individual expert outputs and project to overlap space
        Y_list = []
        O_list = []
        for k in range(self.num_experts):
            expert = getattr(self, f"expert_{k}")
            restriction = getattr(self, f"restriction_{k}")
            
            y_k = expert(x) # (B, S, D)
            o_k = restriction(y_k) # (B, S, d_overlap)
            
            Y_list.append(mx.expand_dims(y_k, axis=2)) # (B, S, 1, D)
            O_list.append(mx.expand_dims(o_k, axis=2)) # (B, S, 1, d_overlap)
            
        Y = mx.concatenate(Y_list, axis=2) # (B, S, E, D)
        O = mx.concatenate(O_list, axis=2) # (B, S, E, d_overlap)
        
        # 3. Sheaf gluing transition corrections
        O_j = mx.expand_dims(O, axis=3) # (B, S, E, 1, d_overlap)
        O_k = mx.expand_dims(O, axis=2) # (B, S, 1, E, d_overlap)
        discrepancies = O_j - O_k # (B, S, E, E, d_overlap)
        
        # w has shape (B, S, E), expand for broadcasting with discrepancies (B, S, E, E, d_overlap)
        # We multiply by w_j (weight of expert j)
        w_j = mx.expand_dims(mx.expand_dims(w, axis=2), axis=-1) # (B, S, 1, E, 1)
        weighted_discrepancies = w_j * discrepancies # (B, S, E, E, d_overlap)
        
        # Sum over expert j (axis 3) to get the aggregated transition adjustment for each expert k
        sum_discrepancies = mx.sum(weighted_discrepancies, axis=3) # (B, S, E, d_overlap)
        
        # Project back to embedding dimension
        correction = self.glue_proj(sum_discrepancies) # (B, S, E, D)
        
        # 4. Glued output
        Y_corrected = Y + correction # (B, S, E, D)
        w_k = mx.expand_dims(w, axis=-1) # (B, S, E, 1)
        out = mx.sum(w_k * Y_corrected, axis=2) # (B, S, D)
        
        return out


class DiracSpinorAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Split dimension must be divisible by 4 for the 4 components of the Dirac spinor
        assert self.head_dim % 4 == 0, "Head dimension must be divisible by 4 for Dirac Spinors"
        self.spinor_dim = self.head_dim // 4
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache = None) -> mx.array:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        
        # Transpose to [B, num_heads, S, head_dim]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        if cache is not None:
            if cache.keys is not None:
                k = mx.concatenate([cache.keys, k], axis=2)
                v = mx.concatenate([cache.values, v], axis=2)
            cache.keys = k
            cache.values = v
            
        # Reshape to split spinor components: [B, H, S, 4, spinor_dim]
        q_spinor = mx.reshape(q, (B, self.num_heads, q.shape[2], 4, self.spinor_dim))
        k_spinor = mx.reshape(k, (B, self.num_heads, k.shape[2], 4, self.spinor_dim))
        
        # Relativistic Dirac spinor contraction: \bar{\psi} \chi = \psi_1^* \chi_1 + \psi_2^* \chi_2 - \psi_3^* \chi_3 - \psi_4^* \chi_4
        # We split the 4 components along axis 3
        q1, q2, q3, q4 = q_spinor[..., 0, :], q_spinor[..., 1, :], q_spinor[..., 2, :], q_spinor[..., 3, :]
        k1, k2, k3, k4 = k_spinor[..., 0, :], k_spinor[..., 1, :], k_spinor[..., 2, :], k_spinor[..., 3, :]
        
        # Compute dot products
        scores1 = q1 @ mx.transpose(k1, (0, 1, 3, 2))
        scores2 = q2 @ mx.transpose(k2, (0, 1, 3, 2))
        scores3 = q3 @ mx.transpose(k3, (0, 1, 3, 2))
        scores4 = q4 @ mx.transpose(k4, (0, 1, 3, 2))
        
        # Relativistic metric signature (1, 1, -1, -1)
        scores = (scores1 + scores2 - scores3 - scores4) / mx.sqrt(self.spinor_dim)
        
        if mask is not None:
            scores = scores + mask
            
        attn_weights = mx.softmax(scores, axis=-1)
        out = attn_weights @ v
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (B, S, self.embed_dim))
        return self.out_proj(out)


class FibonacciQuasicrystallineEncoding(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims
        self.phi = (1.0 + mx.sqrt(mx.array(5.0))) / 2.0
        
    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        # Computes coordinates of a 1D Fibonacci quasicrystal: position_coordinate = n + (phi - 1) * floor((n + 1)/phi)
        S = x.shape[1]
        n = mx.arange(offset, offset + S, dtype=mx.float32)
        
        # Fibonacci projection map
        fib_coords = n + (self.phi - 1.0) * mx.floor((n + 1.0) / self.phi)
        
        # Apply quasicrystalline scale modulation to standard RoPE frequencies
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims))
        
        # Compute dynamic phase offset: fib_coords[:, None] * inv_freq[None, :]
        sinusoid_inp = mx.expand_dims(fib_coords, axis=-1) @ mx.expand_dims(inv_freq, axis=0)
        
        sin = mx.sin(sinusoid_inp)
        cos = mx.cos(sinusoid_inp)
        
        # Apply scale-invariant rotation
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        
        # Merge back
        rx = mx.stack([rx1, rx2], axis=-1)
        return mx.reshape(rx, x.shape)


class AperiodicPenroseMasking:
    def __init__(self, size: int = 2048):
        self.size = size
        # de Bruijn pentagrid parameters for self-similar Penrose masks
        indices = mx.arange(size)
        angles = 2.0 * mx.array(np.pi) * mx.arange(5) / 5.0
        
        # Compute coordinates mapping Penrose tiling intersections
        self.x = mx.cos(angles[0]) * indices[:, None]
        self.y = mx.sin(angles[0]) * indices[:, None]
        
    def get_mask(self, q_len: int, k_len: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        # Create a self-similar aperiodic sparse masking layout based on Penrose pentagrid symmetry
        i = mx.arange(q_len)[:, None]
        j = mx.arange(k_len)[None, :]
        
        # Golden ratio scaling
        diff = i - j
        p_val = mx.cos(2.0 * mx.array(np.pi) * diff / 5.0)
        
        # Aperiodic Penrose sparse channels: keep connections where Penrose condition holds
        sparse_mask = p_val >= 0.0
        
        # Convert to additive mask: 0.0 for active paths, -1e9 for masked paths
        mask = mx.where(sparse_mask, mx.array(0.0, dtype=dtype), mx.array(-1e9, dtype=dtype))
        return mask


class LegendrePolynomialSoftmax:
    def __call__(self, x: mx.array, axis: int = -1) -> mx.array:
        # 3rd-order Legendre orthogonal polynomial approximation of exp(x) on [-1.5, 1.5]
        # exp(x) ~ 1.0 + 1.026*x + 0.54*(x^2) + 0.17*(x^3)
        x_clipped = mx.clip(x, -2.0, 2.0)
        
        x2 = x_clipped * x_clipped
        x3 = x2 * x_clipped
        
        approx_exp = 1.0 + 1.026 * x_clipped + 0.54 * x2 + 0.17 * x3
        approx_exp = mx.clip(approx_exp, 1e-6, 1000.0)
        
        sum_exp = mx.sum(approx_exp, axis=axis, keepdims=True)
        return approx_exp / sum_exp


class ThermodynamicAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, decay_rate: float = 0.01, cold_threshold: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.decay_rate = decay_rate
        self.cold_threshold = cold_threshold
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache = None) -> mx.array:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim)
        
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        
        if cache is not None:
            if cache.keys is not None:
                k_prev = cache.keys
                v_prev = cache.values
                offset = k_prev.shape[2]
                
                # Thermodynamic cooling: tokens decay temperature based on age
                ages = mx.arange(offset, dtype=mx.float32)
                temps = mx.exp(-self.decay_rate * (offset - ages))
                
                # Identify "cold" keys/values (below threshold) and pool/compact them
                cold_mask = temps < self.cold_threshold
                cold_mask_int = cold_mask.astype(mx.int32)
                num_cold = mx.sum(cold_mask_int).item()
                num_hot = cold_mask.size - num_cold
                sorted_indices = mx.argsort(cold_mask_int)
                cold_indices = mx.sort(sorted_indices[-num_cold:]) if num_cold > 0 else mx.array([], dtype=mx.int32)
                hot_indices = mx.sort(sorted_indices[:num_hot]) if num_hot > 0 else mx.array([], dtype=mx.int32)
                
                if cold_indices.size > 1:
                    # Compact cold tokens by averaging adjacent pairs (2x reduction)
                    # We pad cold index length to even number for pairing
                    pad_len = cold_indices.size % 2
                    if pad_len > 0:
                        cold_indices = mx.concatenate([cold_indices, cold_indices[-1:]])
                    
                    indices_paired = mx.reshape(cold_indices, (-1, 2))
                    k_cold_paired = mx.take(k_prev, indices_paired, axis=2)
                    k_cold_compact = mx.mean(k_cold_paired, axis=-2)
                    
                    v_cold_paired = mx.take(v_prev, indices_paired, axis=2)
                    v_cold_compact = mx.mean(v_cold_paired, axis=-2)
                    
                    k_hot = mx.take(k_prev, hot_indices, axis=2)
                    v_hot = mx.take(v_prev, hot_indices, axis=2)
                    
                    k_prev_compact = mx.concatenate([k_cold_compact, k_hot], axis=2)
                    v_prev_compact = mx.concatenate([v_cold_compact, v_hot], axis=2)
                else:
                    k_prev_compact = k_prev
                    v_prev_compact = v_prev
                
                k = mx.concatenate([k_prev_compact, k], axis=2)
                v = mx.concatenate([v_prev_compact, v], axis=2)
                
            cache.keys = k
            cache.values = v
            
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / mx.sqrt(self.head_dim)
        if mask is not None:
            # Broadcast mask if its shape matches
            if mask.ndim == 2:
                mask_eval = mask[None, None, :, :]
            else:
                mask_eval = mask
            
            # Pad mask to match scores sequence length if compaction changed k shape
            if mask_eval.shape[-1] != scores.shape[-1]:
                diff_len = scores.shape[-1] - mask_eval.shape[-1]
                if diff_len > 0:
                    zeros_pad = mx.zeros(mask_eval.shape[:-1] + (diff_len,), dtype=mask_eval.dtype)
                    mask_eval = mx.concatenate([zeros_pad, mask_eval], axis=-1)
                else:
                    mask_eval = mask_eval[..., :scores.shape[-1]]
            scores = scores + mask_eval
            
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        out = mx.transpose(out, (0, 2, 1, 3))
        out = mx.reshape(out, (B, S, self.embed_dim))
        return self.out_proj(out)


class SolitonicAttentionWaveguide:
    def __init__(self, dt: float = 0.01, dx: float = 1.0):
        self.dt = dt
        self.dx = dx

    def __call__(self, x: mx.array) -> mx.array:
        # x shape: [B, H, S, D]
        if x.shape[2] < 5:
            return x
        
        u = x
        u_p1 = mx.concatenate([u[:, :, 1:], u[:, :, -1:]], axis=2)
        u_m1 = mx.concatenate([u[:, :, :1], u[:, :, :-1]], axis=2)
        u_p2 = mx.concatenate([u[:, :, 2:], u[:, :, -2:]], axis=2)
        u_m2 = mx.concatenate([u[:, :, :2], u[:, :, :-2]], axis=2)
        
        du_dx = (u_p1 - u_m1) / (2.0 * self.dx)
        d3u_dx3 = (u_p2 - 2.0 * u_p1 + 2.0 * u_m1 - u_m2) / (2.0 * (self.dx ** 3))
        
        u_next = u - self.dt * (6.0 * u * du_dx + d3u_dx3)
        return u_next


class DiracKählerSpectralCompression:
    def __init__(self, keep_ratio: float = 0.5):
        self.keep_ratio = keep_ratio

    def compress(self, x: mx.array) -> mx.array:
        B, H, S, D = x.shape
        if S < 4:
            return x
        
        indices = mx.arange(S)
        grid_i = indices[:, None]
        grid_j = indices[None, :]
        dct_matrix = mx.cos((np.pi / S) * (grid_j + 0.5) * grid_i)
        
        x_reshaped = mx.transpose(x, (2, 0, 1, 3)).reshape(S, B * H * D)
        freqs = dct_matrix @ x_reshaped
        
        keep_k = int(max(1, self.keep_ratio * S))
        mask = mx.zeros((S, 1))
        mask[:keep_k] = 1.0
        freqs_compressed = freqs * mask
        
        idct_matrix = dct_matrix.T * (2.0 / S)
        x_reconstructed = idct_matrix @ freqs_compressed
        
        x_out = x_reconstructed.reshape(S, B, H, D)
        return mx.transpose(x_out, (1, 2, 0, 3))


class ZeroCopyOctonionicAttention:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        assert embed_dim % 8 == 0, "Embedding dimension must be divisible by 8 for Octonionic attention"
        self.oct_dim = embed_dim // 8

    def multiply(self, q: mx.array, k: mx.array) -> mx.array:
        B, H, S, D = q.shape
        # Reshape to insert octonion dimension
        q_oct = q.reshape(B, H, S, 8, self.oct_dim)
        k_oct = k.reshape(B, H, S, 8, self.oct_dim)
        
        # Cayley table for octonions
        T = np.zeros((8, 8, 8), dtype=np.float32)
        for i in range(8):
            T[0, i, i] = 1.0
            T[i, 0, i] = 1.0
            T[i, i, 0] = -1.0 if i > 0 else 1.0
            
        triples = [
            (1, 2, 3), (1, 4, 5), (2, 4, 6), (3, 4, 7),
            (1, 7, 6), (2, 5, 7), (3, 5, 6)
        ]
        for a, b, c in triples:
            T[a, b, c] = 1.0
            T[b, a, c] = -1.0
            T[b, c, a] = 1.0
            T[c, b, a] = -1.0
            T[c, a, b] = 1.0
            T[a, c, b] = -1.0
            
        T_mx = mx.array(T)
        
        q_exp = mx.reshape(q_oct, (B, H, S, 8, 1, 1, self.oct_dim))
        k_exp = mx.reshape(k_oct, (B, H, S, 1, 8, 1, self.oct_dim))
        T_exp = mx.reshape(T_mx, (1, 1, 1, 8, 8, 8, 1))
        
        res = mx.sum(q_exp * k_exp * T_exp, axis=(3, 4))
        return res.reshape(B, H, S, self.embed_dim)


class AutomorphicModularFormsPooling:
    def __init__(self, k_weight: int = 2):
        self.k_weight = k_weight

    def pool(self, x: mx.array) -> mx.array:
        S = x.shape[2]
        if S < 2:
            return x
        
        s_coords = mx.arange(S, dtype=mx.float32) / float(S)
        magnitudes = mx.sqrt(s_coords ** 2 + 1.0)
        weights = mx.power(magnitudes, -float(self.k_weight))
        weights = weights / mx.sum(weights)
        
        weighted_x = x * weights[None, None, :, None]
        pooled = mx.sum(weighted_x, axis=2, keepdims=True)
        return mx.broadcast_to(pooled, x.shape)


class SchwarzschildMetricPositionalWarp:
    def __init__(self, rs_factor: float = 0.5):
        self.rs_factor = rs_factor

    def warp(self, positions: mx.array, token_importance: mx.array) -> mx.array:
        rs = self.rs_factor * mx.sum(token_importance)
        safe_r = mx.maximum(positions, rs + 1e-3)
        warped = safe_r / (1.0 - rs / safe_r)
        return warped


class FeynmanDiagramPerturbationAttention:
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        B, H, S, D = q.shape
        A_0 = mx.ones((B, H, S, S)) / float(S)
        A_1 = (q @ mx.transpose(k, (0, 1, 3, 2))) / mx.sqrt(D)
        
        attn = A_0 + self.epsilon * A_1
        attn = attn / mx.sum(mx.abs(attn), axis=-1, keepdims=True)
        return attn @ v


class GrassmannianSubspaceProjection:
    def __init__(self, subspace_dim: int):
        self.subspace_dim = subspace_dim

    def project(self, x: mx.array) -> mx.array:
        B, H, S, D = x.shape
        flat_x = x.reshape(-1, D)
        k = min(self.subspace_dim, D)
        with mx.stream(mx.cpu):
            Q, R = mx.linalg.qr(flat_x.T)
        basis = Q[:, :k]
        projected = (flat_x @ basis) @ basis.T
        return projected.reshape(B, H, S, D)


class ThermodynamicPartitionFunctionSoftmax:
    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def __call__(self, logits: mx.array, axis: int = -1) -> mx.array:
        max_val = mx.max(logits, axis=axis, keepdims=True)
        x = self.beta * (logits - max_val)
        x = mx.maximum(x, -10.0)
        approx_exp = 1.0 + x + (x ** 2) / 2.0 + (x ** 3) / 6.0
        approx_exp = mx.maximum(approx_exp, 1e-9)
        partition_function = mx.sum(approx_exp, axis=axis, keepdims=True)
        return approx_exp / partition_function

