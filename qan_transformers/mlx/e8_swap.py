import mlx.core as mx
import numpy as np
from typing import Tuple

_DEFAULT_MLX_PROJECTION_CACHE = {}

def _compile_swap_in_helper(quantized: mx.array, coords_norm: mx.array, coords_norm2: mx.array, k_val: int, active_len: int):
    q_norm2 = mx.sum(mx.square(quantized), axis=-1, keepdims=True)  # [S_q, 1]
    prod = quantized @ coords_norm.T  # [S_q, M]
    dist2 = q_norm2 + coords_norm2 - 2 * prod  # [S_q, M]
    
    # Mask out inactive elements
    capacity = coords_norm.shape[0]
    mask = mx.arange(capacity) >= active_len
    mask = mx.reshape(mask, (1, -1))
    # Win 51: Branch-Free Masking in MLX Swap-In Helper replacing mx.where
    dist2 = dist2 + mask.astype(dist2.dtype) * 1e9
    
    # Find top-k closest elements for each query
    idx = mx.argpartition(dist2, kth=k_val - 1, axis=-1)
    topk_idx = idx[..., :k_val]
    topk_idx = mx.sort(topk_idx, axis=-1)
    topk_val = mx.take_along_axis(dist2, topk_idx, axis=-1)
    
    is_neighbor = topk_val <= 2.05
    db_idx = mx.reshape(topk_idx, (-1,))
    is_neighbor_flat = mx.reshape(is_neighbor, (-1,))
    return db_idx, is_neighbor_flat

def _compile_swap_in_batch_helper(quantized: mx.array, coords_norm: mx.array, coords_norm2: mx.array, k_val: int, B: int, H: int, S: int, active_len: int):
    q_norm2 = mx.sum(mx.square(quantized), axis=-1, keepdims=True)  # [B, H, S, 1]
    
    quantized_flat = mx.reshape(quantized, (-1, 8))
    prod_flat = quantized_flat @ coords_norm.T  # [B*H*S, M]
    prod = mx.reshape(prod_flat, (B, H, S, -1))
    
    dist2 = q_norm2 + coords_norm2 - 2 * prod  # [B, H, S, M]
    
    # Mask out inactive elements
    capacity = coords_norm.shape[0]
    mask = mx.arange(capacity) >= active_len
    mask = mx.reshape(mask, (1, 1, 1, -1))
    # Win 52: Branch-Free Masking in MLX Swap-In Batch Helper replacing mx.where
    dist2 = dist2 + mask.astype(dist2.dtype) * 1e9
    
    M = dist2.shape[-1]
    dist2_flat = mx.reshape(dist2, (B, H, S * M))
    
    # top-k selection using argpartition on MLX arrays
    idx = mx.argpartition(dist2_flat, kth=k_val - 1, axis=-1)
    topk_idx = idx[..., :k_val]
    topk_idx = mx.sort(topk_idx, axis=-1)
    topk_val = mx.take_along_axis(dist2_flat, topk_idx, axis=-1)
    
    is_neighbor = topk_val <= 2.05
    db_idx = topk_idx % M
    return db_idx, is_neighbor

_compiled_swap_in_helper = mx.compile(_compile_swap_in_helper)
_compiled_swap_in_batch_helper = mx.compile(_compile_swap_in_batch_helper)

def _decode_d8_compiled(x: mx.array, arange_8: mx.array) -> mx.array:
    y = mx.round(x)
    y_sum = mx.sum(y, axis=-1, keepdims=True)
    # Win 101: Fast bitwise parity check without redundant mx.round
    odd_mask = ((y_sum.astype(mx.int32) & 1) != 0)
    
    errors = mx.abs(x - y)
    k = mx.argmax(errors, axis=-1, keepdims=True)
    
    # Win 102: Fused Scale-and-Shift inside MLX ConwaySloane E8 Decoder
    k_mask = (arange_8 == k)
    apply_adjustment = k_mask & odd_mask
    y = y + apply_adjustment.astype(x.dtype) * ((x >= y).astype(x.dtype) * 2.0 - 1.0)
    return y

def _decode_impl_compiled(x_flat: mx.array, half_ones: mx.array, arange_8: mx.array) -> mx.array:
    x_shifted = x_flat - half_ones
    
    # Batch the two calls to _decode_d8 by stacking along a new axis
    stacked = mx.stack([x_flat, x_shifted], axis=0)
    decoded = _decode_d8_compiled(stacked, arange_8)
    # Win 213: Batch distance calculations using stacked diff tensor
    diff = stacked - decoded
    dists = mx.sum(mx.square(diff), axis=-1)
    dist_f = dists[0]
    dist_g = dists[1]
    
    # Win 43: Branch-Free Choice in MLX E8 Decoder using multiplication-based interpolation
    g_x = decoded[1] + half_ones
    g_closer = mx.expand_dims(dist_g < dist_f, axis=-1).astype(x_flat.dtype)
    nearest = decoded[0] + g_closer * (g_x - decoded[0])
    return nearest

_compiled_decode_impl = mx.compile(_decode_impl_compiled)

class ConwaySloaneE8DecoderMLX:
    def __init__(self):
        self._half_ones_cache = {}
        self.arange_8 = mx.arange(8)

    def decode(self, x: mx.array) -> mx.array:
        """
        Vectorized E8 lattice decoder in MLX.
        Args:
            x: mlx.core.array of shape [..., 8]
        Returns:
            nearest_points: mlx.core.array of shape [..., 8]
        """
        orig_shape = x.shape
        x_flat = mx.reshape(x, (-1, 8))
        
        # Win 83: Fast Buffer Caching in MLX E8 Decoder
        if getattr(self, "_cached_half_ones", None) is not None and self._cached_half_ones.dtype == x.dtype:
            half_ones = self._cached_half_ones
        else:
            half_ones = mx.full((8,), 0.5, dtype=x.dtype)
            self._cached_half_ones = half_ones
            
        nearest = _compiled_decode_impl(x_flat, half_ones, self.arange_8)
        return mx.reshape(nearest, orig_shape)

class AdelicMemorySwapGridDB:
    def __init__(self, d_model: int, cache_limit_ratio: float = 0.15, d_model_draft: int = None):
        """
        Adelic E8 Memory Swap Grid DB in MLX.
        Manages coordinate-quantized key-value arrays in Unified Memory,
        allowing zero-copy paging and GPU-native vectorized lookups.
        
        Supports target and draft models sharing the same coordinate index grid.
        """
        self.d_model = d_model
        self.d_model_target = d_model
        self.d_model_draft = d_model_draft
        self.cache_limit_ratio = cache_limit_ratio
        self.enabled = True
        
        self.decoder = ConwaySloaneE8DecoderMLX()
        
        # Pre-cache Coxeter roots for E8
        from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates, project_e8_to_quasicrystal
        self.shell_1_roots = mx.array(generate_dynamic_e8_coordinates(1), dtype=mx.float32)
        
        # Precompute E8 root neighbors for O(1) Geodesic Pruned Search
        roots_8d = generate_dynamic_e8_coordinates(1)
        roots_3d = mx.array(project_e8_to_quasicrystal(roots_8d), dtype=mx.float32)
        roots_3d_norm = roots_3d / (mx.linalg.norm(roots_3d, axis=-1, keepdims=True) + 1e-6)
        sims = roots_3d_norm @ roots_3d_norm.T
        
        self.root_neighbors = []
        for i in range(240):
            row = np.array(sims[i])
            sorted_idx = np.argsort(-row)
            neighbors = sorted_idx[1:13].tolist() # 12 closest neighbors
            self.root_neighbors.append(neighbors)
            
        # Build Markov transition matrix for E8 heat diffusion
        A_E8_np = np.zeros((240, 240), dtype=np.float32)
        for i in range(240):
            A_E8_np[i, i] = 0.5
            for n in self.root_neighbors[i]:
                A_E8_np[i, n] = 0.5 / len(self.root_neighbors[i])
        self.A_E8 = mx.array(A_E8_np)
        
        # Cytoskeletal routing state
        self.W_route = mx.eye(240, dtype=mx.float32)
        self.alpha_grow = 0.1
        self.alpha_decay = 0.02
        self.noise_scale = 0.05
        
        # Single unified grid coordinates
        self._grid_coords = None
        self.grid_coords_len = 0
        self.grid_coords_capacity = 0
        self._empty_batch_cache = {}
        
        # CPU paging buffers, indexed by d_model dimension
        self._cpu_k_target_bufs = {}  # d_model -> mx.array
        self._cpu_v_target_bufs = {}  # d_model -> mx.array
        self._cpu_k_target_capacities = {}
        self.target_len = 0
        
        self._cpu_k_draft_bufs = {}   # d_model -> mx.array
        self._cpu_v_draft_bufs = {}   # d_model -> mx.array
        self._cpu_k_draft_capacities = {}
        self.draft_len = 0
        
        # Projection weight dictionaries, indexed by d_model dimension
        self.W_p_target_dict = {}  # d_model -> mx.array
        self.W_p_draft_dict = {}   # d_model -> mx.array
        
        # Backwards compatibility pointers
        self.cpu_k = None        # shape [M, d_model]
        self.cpu_v = None        # shape [M, d_model]
        
        # Private cache for sliced views
        self._cached_cpu_k_target = None
        self._cached_cpu_v_target = None
        self._cached_target_len = -1
        self._cached_cpu_k_target_buf_id = None
        self._cached_cpu_v_target_buf_id = None
        
        self._cached_cpu_k_draft = None
        self._cached_cpu_v_draft = None
        self._cached_draft_len = -1
        self._cached_cpu_k_draft_buf_id = None
        self._cached_cpu_v_draft_buf_id = None
        
        self.target_ratio = 8
        self.draft_ratio = 8
        self._coords_cache = {}


    def _update_adjacency_matrix(self, K_neighbors=32):
        # Thermodynamic diffusion is now executed on the fixed 240-node E8 root lattice graph (self.A_E8).
        # Dynamic coordinate-to-coordinate adjacency updates are no longer required, saving O(M^2) compute/VRAM.
        pass

    def _polymerize_cytoskeleton(self, q_flat):
        # 1. Calculate query entropy and specificity
        proj = q_flat @ self.shell_1_roots.astype(q_flat.dtype).T
        p = mx.softmax(proj, axis=-1)
        
        # Compute entropy H = -sum(p * log(p + 1e-9))
        H = -mx.sum(p * mx.log(p + 1e-9), axis=-1)
        
        # Map entropy to specificity in [0.0, 1.0] (ln(240) = 5.48)
        specificity = 1.0 - H / 5.48
        specificity = mx.maximum(specificity, 0.0)
        
        # 2. Co-activation matrix representing semantic pathways
        co_activation = p.T @ p
        
        # 3. Growth rate driven by specificity
        growth = self.alpha_grow * mx.mean(specificity) * co_activation
        
        # 4. Polymerize (growth) and Depolymerize (catastrophe)
        W_decayed = self.W_route * (1.0 - self.alpha_decay)
        self.W_route = W_decayed + growth * (1.0 - W_decayed)

    def neuromorphic_search(self, quantized, k_val, steps=3):
        coords = self._grid_coords[:self.grid_coords_len]
        M = coords.shape[0]
        
        if M == 0:
            return mx.zeros((quantized.shape[0], k_val), dtype=mx.int32), mx.zeros((quantized.shape[0], k_val), dtype=mx.bool_)
            
        q_flat = mx.reshape(quantized, (-1, 8))
        B_flat = q_flat.shape[0]
        
        # Update cytoskeletal filaments
        self._polymerize_cytoskeleton(q_flat)
        
        # 1. Initialize heat on the 240 E8 roots (one-hot point source at nearest E8 centroid)
        proj = q_flat @ self.shell_1_roots.astype(quantized.dtype).T
        best_root = mx.argmax(proj, axis=-1)
        u_roots = (mx.expand_dims(best_root, axis=-1) == mx.arange(240)).astype(quantized.dtype)
        
        # Route query heat through self-assembled cytoskeletal pathways
        u_roots = u_roots @ self.W_route.astype(quantized.dtype)
        
        # 2. Run thermodynamic diffusion on the E8 root lattice graph
        if hasattr(self, "A_E8") and self.A_E8 is not None:
            A_E8 = self.A_E8.astype(quantized.dtype)
            for _ in range(steps):
                u_roots = u_roots @ A_E8
                
        # 3. Gather heat to the database items using their E8 root assignments
        if not hasattr(self, "_coords_roots") or self._coords_roots.shape[0] != M:
            self._coords_roots = mx.argmax(coords.astype(quantized.dtype) @ self.shell_1_roots.astype(quantized.dtype).T, axis=-1)
        coords_roots = self._coords_roots
        
        u_db = u_roots[..., coords_roots]
        
        # 4. Retrieve top-k highest potential nodes
        k_val = min(k_val, M)
        idx = mx.argpartition(-u_db, kth=k_val - 1, axis=-1)
        topk_idx = idx[..., :k_val]
        topk_idx = mx.sort(topk_idx, axis=-1)
        
        # Calculate exact distance squared ONLY for these top-k retrieved items
        gathered_coords = coords[topk_idx] # [B_flat, k_val, 8]
        q_expanded = mx.expand_dims(q_flat, axis=1) # [B_flat, 1, 8]
        dist2_topk = mx.sum(mx.square(q_expanded - gathered_coords), axis=-1) # [B_flat, k_val]
        is_neighbor = dist2_topk <= 2.05
        
        db_idx = mx.reshape(topk_idx, (-1,))
        is_neighbor_flat = mx.reshape(is_neighbor, (-1,))
        
        return db_idx, is_neighbor_flat

    def neuromorphic_search_batch(self, quantized, k_val, B, H, S, steps=3):
        coords = self._grid_coords[:self.grid_coords_len]
        M = coords.shape[0]
        
        if M == 0:
            return mx.zeros((B, H, k_val), dtype=mx.int32), mx.zeros((B, H, k_val), dtype=mx.bool_)
            
        q_flat = mx.reshape(quantized, (-1, 8))
        
        # Update cytoskeletal filaments
        self._polymerize_cytoskeleton(q_flat)
        
        # 1. Initialize heat on the 240 E8 roots (one-hot point source at nearest E8 centroid)
        proj = q_flat @ self.shell_1_roots.astype(quantized.dtype).T
        best_root = mx.argmax(proj, axis=-1)
        u_roots = (mx.expand_dims(best_root, axis=-1) == mx.arange(240)).astype(quantized.dtype)
        
        # Route query heat through self-assembled cytoskeletal pathways
        u_roots = u_roots @ self.W_route.astype(quantized.dtype)
        
        # 2. Run thermodynamic diffusion on the E8 root lattice graph
        if hasattr(self, "A_E8") and self.A_E8 is not None:
            A_E8 = self.A_E8.astype(quantized.dtype)
            for _ in range(steps):
                u_roots = u_roots @ A_E8
                
        # 3. Gather heat to the database items using their E8 root assignments
        if not hasattr(self, "_coords_roots") or self._coords_roots.shape[0] != M:
            self._coords_roots = mx.argmax(coords.astype(quantized.dtype) @ self.shell_1_roots.astype(quantized.dtype).T, axis=-1)
        coords_roots = self._coords_roots
        
        u_db = u_roots[..., coords_roots]
        u_db_batch = mx.reshape(u_db, (B, H, S * M))
        
        # 4. Retrieve top-k highest potential nodes
        k_val = min(k_val, S * M)
        idx = mx.argpartition(-u_db_batch, kth=k_val - 1, axis=-1)
        topk_idx = idx[..., :k_val]
        topk_idx = mx.sort(topk_idx, axis=-1)
        
        # Calculate exact distance squared ONLY for these top-k retrieved items
        db_idx = topk_idx % M # [B, H, k_val]
        q_orig = mx.reshape(q_flat, (B, H, S, 8))
        
        gathered_coords = coords[db_idx] # [B, H, k_val, 8]
        q_idx = topk_idx // M # [B, H, k_val]
        
        q_idx_expanded = mx.expand_dims(q_idx, axis=-1)
        q_idx_expanded = mx.broadcast_to(q_idx_expanded, (B, H, k_val, 8))
        gathered_queries = mx.take_along_axis(q_orig, q_idx_expanded, axis=2)
        
        dist2_topk = mx.sum(mx.square(gathered_queries - gathered_coords), axis=-1) # [B, H, k_val]
        is_neighbor = dist2_topk <= 2.05
        
        return db_idx, is_neighbor


    @property
    def cpu_k_target_dict(self):
        return {d_model: buf[:self.target_len] for d_model, buf in self._cpu_k_target_bufs.items() if buf is not None}

    @property
    def cpu_v_target_dict(self):
        return {d_model: buf[:self.target_len] for d_model, buf in self._cpu_v_target_bufs.items() if buf is not None}

    @property
    def cpu_k_draft_dict(self):
        return {d_model: buf[:self.draft_len] for d_model, buf in self._cpu_k_draft_bufs.items() if buf is not None}

    @property
    def cpu_v_draft_dict(self):
        return {d_model: buf[:self.draft_len] for d_model, buf in self._cpu_v_draft_bufs.items() if buf is not None}

    @property
    def cpu_k_target(self):
        buf = self._cpu_k_target_bufs.get(self.d_model_target)
        if buf is None:
            return None
        buf_id = id(buf)
        if (self._cached_cpu_k_target is None or 
            self._cached_target_len != self.target_len or 
            self._cached_cpu_k_target_buf_id != buf_id):
            self._cached_cpu_k_target = buf[:self.target_len]
            self._cached_target_len = self.target_len
            self._cached_cpu_k_target_buf_id = buf_id
        return self._cached_cpu_k_target
        
    @cpu_k_target.setter
    def cpu_k_target(self, value):
        if value is None:
            if self.d_model_target in self._cpu_k_target_bufs:
                del self._cpu_k_target_bufs[self.d_model_target]
                del self._cpu_k_target_capacities[self.d_model_target]
            self.target_len = 0
        else:
            self._cpu_k_target_bufs[self.d_model_target] = value
            self._cpu_k_target_capacities[self.d_model_target] = value.shape[0]
            self.target_len = value.shape[0]
        
    @property
    def cpu_v_target(self):
        buf = self._cpu_v_target_bufs.get(self.d_model_target)
        if buf is None:
            return None
        buf_id = id(buf)
        if (self._cached_cpu_v_target is None or 
            self._cached_target_len != self.target_len or 
            self._cached_cpu_v_target_buf_id != buf_id):
            self._cached_cpu_v_target = buf[:self.target_len]
            self._cached_target_len = self.target_len
            self._cached_cpu_v_target_buf_id = buf_id
        return self._cached_cpu_v_target
        
    @cpu_v_target.setter
    def cpu_v_target(self, value):
        if value is None:
            if self.d_model_target in self._cpu_v_target_bufs:
                del self._cpu_v_target_bufs[self.d_model_target]
        else:
            self._cpu_v_target_bufs[self.d_model_target] = value
        
    @property
    def cpu_k_draft(self):
        buf = self._cpu_k_draft_bufs.get(self.d_model_draft or self.d_model)
        if buf is None:
            return None
        buf_id = id(buf)
        if (self._cached_cpu_k_draft is None or 
            self._cached_draft_len != self.draft_len or 
            self._cached_cpu_k_draft_buf_id != buf_id):
            self._cached_cpu_k_draft = buf[:self.draft_len]
            self._cached_draft_len = self.draft_len
            self._cached_cpu_k_draft_buf_id = buf_id
        return self._cached_cpu_k_draft
        
    @cpu_k_draft.setter
    def cpu_k_draft(self, value):
        d_model_key = self.d_model_draft or self.d_model
        if value is None:
            if d_model_key in self._cpu_k_draft_bufs:
                del self._cpu_k_draft_bufs[d_model_key]
                del self._cpu_k_draft_capacities[d_model_key]
            self.draft_len = 0
        else:
            self._cpu_k_draft_bufs[d_model_key] = value
            self._cpu_k_draft_capacities[d_model_key] = value.shape[0]
            self.draft_len = value.shape[0]
        
    @property
    def cpu_v_draft(self):
        buf = self._cpu_v_draft_bufs.get(self.d_model_draft or self.d_model)
        if buf is None:
            return None
        buf_id = id(buf)
        if (self._cached_cpu_v_draft is None or 
            self._cached_draft_len != self.draft_len or 
            self._cached_cpu_v_draft_buf_id != buf_id):
            self._cached_cpu_v_draft = buf[:self.draft_len]
            self._cached_draft_len = self.draft_len
            self._cached_cpu_v_draft_buf_id = buf_id
        return self._cached_cpu_v_draft
        
    @cpu_v_draft.setter
    def cpu_v_draft(self, value):
        d_model_key = self.d_model_draft or self.d_model
        if value is None:
            if d_model_key in self._cpu_v_draft_bufs:
                del self._cpu_v_draft_bufs[d_model_key]
        else:
            self._cpu_v_draft_bufs[d_model_key] = value

    @property
    def grid_coords(self):
        if self._grid_coords is None:
            return None
        return self._grid_coords[:self.grid_coords_len]
        
    @grid_coords.setter
    def grid_coords(self, value):
        self._grid_coords = value
        if value is None:
            self.grid_coords_len = 0
            self.grid_coords_capacity = 0
        else:
            self.grid_coords_len = value.shape[0]
            self.grid_coords_capacity = value.shape[0]

    def _init_default_projection(self, dtype, is_draft: bool, d_model: int):
        W_p_dict = self.W_p_draft_dict if is_draft else self.W_p_target_dict
        
        # Win 84: Cached Default Orthogonal Projections in Swap DB
        cache_key = (d_model, dtype)
        if cache_key in _DEFAULT_MLX_PROJECTION_CACHE:
            W_p = _DEFAULT_MLX_PROJECTION_CACHE[cache_key]
        else:
            g = np.random.randn(d_model, 8)
            q, r = np.linalg.qr(g)
            norms = np.linalg.norm(q, axis=0, keepdims=True)
            W_p_np = q / (norms + 1e-6)
            W_p = mx.array(W_p_np, dtype=dtype)
            _DEFAULT_MLX_PROJECTION_CACHE[cache_key] = W_p
            
        W_p_dict[d_model] = W_p

    def initialize_projections(self, W_q: mx.array, W_k: mx.array, is_draft: bool = False):
        dtype = W_q.dtype
        d_model = W_q.shape[-1]
        
        W_stacked = mx.concatenate([W_q, W_k], axis=0)
        # Cast to float32 since MLX SVD is only supported on float32/float64
        W_stacked_f32 = W_stacked.astype(mx.float32)
        U, S, Vh = mx.linalg.svd(W_stacked_f32)
        
        # Convert SVD results to numpy for searchsorted and scaling
        S_np = np.array(S).astype(np.float32)
        Vh_np = np.array(Vh).astype(np.float32)
        
        cumulative_energy = np.cumsum(S_np ** 2) / np.sum(S_np ** 2)
        target_energy = 0.95
        
        cutoff_idx = np.searchsorted(cumulative_energy, target_energy)
        cutoff_idx = min(cutoff_idx, len(cumulative_energy) - 1)
        tau = float(S_np[cutoff_idx])
        
        S_subset = S_np[:8]
        Vt_subset = Vh_np[:8].copy()
        scales = np.maximum(0.0, 1.0 - (tau / (S_subset + 1e-6)))
        Vt_subset = Vt_subset * scales[:, np.newaxis]
            
        if np.allclose(Vt_subset, 0.0):
            Vt_subset = Vh_np[:8]
            
        W_p_np = Vt_subset.T
        norms = np.linalg.norm(W_p_np, axis=0, keepdims=True)
        W_p_np = W_p_np / (norms + 1e-6)
        
        W_p = mx.array(W_p_np, dtype=dtype)
        W_p_dict = self.W_p_draft_dict if is_draft else self.W_p_target_dict
        W_p_dict[d_model] = W_p
            
    def _quantize(self, keys: mx.array) -> mx.array:
        return self.decoder.decode(keys)
        
    def clear(self):
        self._grid_coords = None
        self.grid_coords_len = 0
        self.grid_coords_capacity = 0
        self._cpu_k_target_bufs.clear()
        self._cpu_v_target_bufs.clear()
        self._cpu_k_target_capacities.clear()
        self.target_len = 0
        self._cpu_k_draft_bufs.clear()
        self._cpu_v_draft_bufs.clear()
        self._cpu_k_draft_capacities.clear()
        self.draft_len = 0
        self.cpu_k = None
        self.cpu_v = None
        self._coords_cache.clear()

        
        # Invalidate view caches
        self._cached_cpu_k_target = None
        self._cached_cpu_v_target = None
        self._cached_target_len = -1
        self._cached_cpu_k_target_buf_id = None
        self._cached_cpu_v_target_buf_id = None
        
        self._cached_cpu_k_draft = None
        self._cached_cpu_v_draft = None
        self._cached_draft_len = -1
        self._cached_cpu_k_draft_buf_id = None
        self._cached_cpu_v_draft_buf_id = None
        
    def rollback(self, num_tokens_to_keep: int, current_len: int = None):
        """
        Rolls back the coordinate index and storage buffers in unified memory.
        """
        ratio_target = getattr(self, "target_ratio", 8)
        if current_len is not None and current_len > 0:
            for d_model, buf in self._cpu_k_target_bufs.items():
                if buf is not None and self.target_len >= current_len:
                    ratio_target = self.target_len // current_len
                    self.target_ratio = ratio_target
                    break
                        
        num_vectors_target = num_tokens_to_keep * ratio_target
        self.target_len = min(self.target_len, num_vectors_target)
        self.grid_coords_len = min(self.grid_coords_len, num_vectors_target)



        ratio_draft = getattr(self, "draft_ratio", 8)
        if current_len is not None and current_len > 0:
            for d_model, buf in self._cpu_k_draft_bufs.items():
                if buf is not None and self.draft_len >= current_len:
                    ratio_draft = self.draft_len // current_len
                    self.draft_ratio = ratio_draft
                    break

        num_vectors_draft = num_tokens_to_keep * ratio_draft
        self.draft_len = min(self.draft_len, num_vectors_draft)
        self._coords_cache.clear()
        

            
        # Set backwards compatibility pointers if d_model target is present
        target_buf = self._cpu_k_target_bufs.get(self.d_model)
        if target_buf is not None:
            self.cpu_k = target_buf[:self.target_len]
            self.cpu_v = self._cpu_v_target_bufs[self.d_model][:self.target_len]

    def swap_out(self, keys: mx.array, values: mx.array, token_seq_len: int = None):
        if not getattr(self, "enabled", True):
            return
        self.swap_out_target(keys, values, token_seq_len)

    def swap_out_target(self, keys: mx.array, values: mx.array, token_seq_len: int = None):
        if not getattr(self, "enabled", True):
            return
        if len(keys) == 0:
            return
            
        dtype = keys.dtype
        d_model = keys.shape[-1]
        n_new = keys.shape[0]
        
        if token_seq_len is not None and token_seq_len > 0:
            self.target_ratio = keys.shape[0] // token_seq_len
            
        W_p_dict = self.W_p_target_dict
        if d_model not in W_p_dict or W_p_dict[d_model].dtype != dtype:
            self._init_default_projection(dtype, is_draft=False, d_model=d_model)
            
        keys_8d = keys @ W_p_dict[d_model]
        quantized = self._quantize(keys_8d)
        
        buf_k = self._cpu_k_target_bufs.get(d_model)
        buf_v = self._cpu_v_target_bufs.get(d_model)
        capacity = self._cpu_k_target_capacities.get(d_model, 0)
        
        if buf_k is None or capacity < self.target_len + n_new:
            new_capacity = max(2048, capacity * 2)
            while new_capacity < self.target_len + n_new:
                new_capacity *= 2
                
            new_k = mx.zeros((new_capacity, d_model), dtype=dtype)
            new_v = mx.zeros((new_capacity, d_model), dtype=dtype)
            
            if buf_k is not None:
                new_k[:self.target_len] = buf_k[:self.target_len]
                new_v[:self.target_len] = buf_v[:self.target_len]
                
            self._cpu_k_target_bufs[d_model] = new_k
            self._cpu_v_target_bufs[d_model] = new_v
            self._cpu_k_target_capacities[d_model] = new_capacity
            buf_k = new_k
            buf_v = new_v
            capacity = new_capacity
            
        buf_k[self.target_len : self.target_len + n_new] = keys
        buf_v[self.target_len : self.target_len + n_new] = values
        

            
        self.target_len += n_new
        
        num_target_tokens = self.target_len
        
        if self._grid_coords is None or self.grid_coords_len < num_target_tokens:
            new_tokens_count = num_target_tokens - self.grid_coords_len
            new_coords = quantized[-new_tokens_count:] if new_tokens_count > 0 else mx.zeros((0, 8), dtype=quantized.dtype)
            
            if self._grid_coords is None or self.grid_coords_capacity < self.grid_coords_len + new_tokens_count:
                new_capacity = max(2048, self.grid_coords_capacity * 2)
                while new_capacity < self.grid_coords_len + new_tokens_count:
                    new_capacity *= 2
                    
                new_grid = mx.zeros((new_capacity, 8), dtype=quantized.dtype)
                if self._grid_coords is not None:
                    new_grid[:self.grid_coords_len] = self._grid_coords[:self.grid_coords_len]
                self._grid_coords = new_grid
                self.grid_coords_capacity = new_capacity
                
            if new_tokens_count > 0:
                self._grid_coords[self.grid_coords_len : self.grid_coords_len + new_tokens_count] = new_coords
                self.grid_coords_len += new_tokens_count
                self._coords_cache.clear()
                
        if d_model == self.d_model:
            self.cpu_k = buf_k[:self.target_len]
            self.cpu_v = buf_v[:self.target_len]

    def swap_out_draft(self, keys: mx.array, values: mx.array, token_seq_len: int = None):
        if not getattr(self, "enabled", True):
            return
        if len(keys) == 0:
            return
            
        dtype = keys.dtype
        d_model = keys.shape[-1]
        n_new = keys.shape[0]
        
        if token_seq_len is not None and token_seq_len > 0:
            self.draft_ratio = keys.shape[0] // token_seq_len
            
        W_p_dict = self.W_p_draft_dict
        if d_model not in W_p_dict or W_p_dict[d_model].dtype != dtype:
            self._init_default_projection(dtype, is_draft=True, d_model=d_model)
            
        keys_8d = keys @ W_p_dict[d_model]
        quantized = self._quantize(keys_8d)
        
        buf_k = self._cpu_k_draft_bufs.get(d_model)
        buf_v = self._cpu_v_draft_bufs.get(d_model)
        capacity = self._cpu_k_draft_capacities.get(d_model, 0)
        
        if buf_k is None or capacity < self.draft_len + n_new:
            new_capacity = max(2048, capacity * 2)
            while new_capacity < self.draft_len + n_new:
                new_capacity *= 2
                
            new_k = mx.zeros((new_capacity, d_model), dtype=dtype)
            new_v = mx.zeros((new_capacity, d_model), dtype=dtype)
            
            if buf_k is not None:
                new_k[:self.draft_len] = buf_k[:self.draft_len]
                new_v[:self.draft_len] = buf_v[:self.draft_len]
                
            self._cpu_k_draft_bufs[d_model] = new_k
            self._cpu_v_draft_bufs[d_model] = new_v
            self._cpu_k_draft_capacities[d_model] = new_capacity
            buf_k = new_k
            buf_v = new_v
            capacity = new_capacity
            
        buf_k[self.draft_len : self.draft_len + n_new] = keys
        buf_v[self.draft_len : self.draft_len + n_new] = values
        

            
        self.draft_len += n_new
        
        num_draft_tokens = self.draft_len
        
        if self._grid_coords is None or self.grid_coords_len < num_draft_tokens:
            new_tokens_count = num_draft_tokens - self.grid_coords_len
            new_coords = quantized[-new_tokens_count:] if new_tokens_count > 0 else mx.zeros((0, 8), dtype=quantized.dtype)
            
            if self._grid_coords is None or self.grid_coords_capacity < self.grid_coords_len + new_tokens_count:
                new_capacity = max(2048, self.grid_coords_capacity * 2)
                while new_capacity < self.grid_coords_len + new_tokens_count:
                    new_capacity *= 2
                    
                new_grid = mx.zeros((new_capacity, 8), dtype=quantized.dtype)
                if self._grid_coords is not None:
                    new_grid[:self.grid_coords_len] = self._grid_coords[:self.grid_coords_len]
                self._grid_coords = new_grid
                self.grid_coords_capacity = new_capacity
                
            if new_tokens_count > 0:
                self._grid_coords[self.grid_coords_len : self.grid_coords_len + new_tokens_count] = new_coords
                self.grid_coords_len += new_tokens_count
                self._coords_cache.clear()

    def swap_in(self, queries: mx.array) -> Tuple[mx.array, mx.array]:
        return self.swap_in_target(queries)
        
    def swap_in_target(self, queries: mx.array) -> Tuple[mx.array, mx.array]:
        return self._swap_in(queries, is_draft=False)
        
    def swap_in_draft(self, queries: mx.array) -> Tuple[mx.array, mx.array]:
        return self._swap_in(queries, is_draft=True)
        
    def _swap_in(self, queries: mx.array, is_draft: bool) -> Tuple[mx.array, mx.array]:
        d_model = queries.shape[-1]
        
        cpu_k_bufs = self._cpu_k_draft_bufs if is_draft else self._cpu_k_target_bufs
        cpu_v_bufs = self._cpu_v_draft_bufs if is_draft else self._cpu_v_target_bufs
        active_len = self.draft_len if is_draft else self.target_len
        W_p_dict = self.W_p_draft_dict if is_draft else self.W_p_target_dict
        
        buf_k = cpu_k_bufs.get(d_model)
        buf_v = cpu_v_bufs.get(d_model)
        
        if len(queries) == 0 or buf_k is None or active_len == 0:
            return (mx.zeros((0, d_model), dtype=queries.dtype),
                    mx.zeros((0, d_model), dtype=queries.dtype))
                    
        cpu_k_buf = buf_k
        cpu_v_buf = buf_v
                    
        grid_coords_buf = self._grid_coords
        if grid_coords_buf is None or self.grid_coords_len == 0:
            return (mx.zeros((1, d_model), dtype=queries.dtype),
                    mx.zeros((1, d_model), dtype=queries.dtype))
            
        dtype = queries.dtype
        if d_model not in W_p_dict or W_p_dict[d_model].dtype != dtype:
            self._init_default_projection(dtype, is_draft, d_model)
            
        W_p = W_p_dict[d_model]
        queries_8d = queries @ W_p
        
        # Stochastic Resonance Doping
        if hasattr(self, "noise_scale") and self.noise_scale > 0.0:
            std = mx.sqrt(mx.var(queries_8d) + 1e-9)
            noise = mx.random.normal(queries_8d.shape, dtype=queries_8d.dtype)
            queries_8d = queries_8d + self.noise_scale * std * noise
            
        quantized = self._quantize(queries_8d)
        
        cache_key = (id(self._grid_coords), grid_coords_buf.shape[0], dtype)
        if self._coords_cache.get("key") != cache_key:
            coords_norm = grid_coords_buf.astype(dtype)
            coords_norm2 = mx.sum(mx.square(coords_norm), axis=-1, keepdims=True).T
            self._coords_cache = {
                "key": cache_key,
                "coords_norm": coords_norm,
                "coords_norm2": coords_norm2
            }
        coords_norm = self._coords_cache["coords_norm"]
        coords_norm2 = self._coords_cache["coords_norm2"]
        
        k_val = min(8, coords_norm.shape[0])
        
        # Update adjacency representation if database changed
        if self._grid_coords is not None and self.grid_coords_len > 0:
            if not hasattr(self, "_last_adj_len") or self._last_adj_len != self.grid_coords_len:
                self._update_adjacency_matrix()
                self._last_adj_len = self.grid_coords_len
                
        db_idx, is_neighbor_flat = self.neuromorphic_search(quantized, k_val)
        
        retrieved_k = cpu_k_buf[db_idx]
        retrieved_v = cpu_v_buf[db_idx]
        
        retrieved_k = retrieved_k * mx.expand_dims(is_neighbor_flat, axis=-1)
        retrieved_v = retrieved_v * mx.expand_dims(is_neighbor_flat, axis=-1)
        
        return retrieved_k, retrieved_v
        
    def swap_in_batch(self, queries: mx.array, max_matches: int = 16) -> Tuple[mx.array, mx.array]:
        return self.swap_in_batch_target(queries, max_matches)
        
    def swap_in_batch_target(self, queries: mx.array, max_matches: int = 16) -> Tuple[mx.array, mx.array]:
        return self._swap_in_batch(queries, max_matches, is_draft=False)
        
    def swap_in_batch_draft(self, queries: mx.array, max_matches: int = 16) -> Tuple[mx.array, mx.array]:
        return self._swap_in_batch(queries, max_matches, is_draft=True)
        
    def _swap_in_batch(self, queries: mx.array, max_matches: int, is_draft: bool) -> Tuple[mx.array, mx.array]:
        B, H, S, d_model = queries.shape
        dtype = queries.dtype
        
        cpu_k_bufs = self._cpu_k_draft_bufs if is_draft else self._cpu_k_target_bufs
        cpu_v_bufs = self._cpu_v_draft_bufs if is_draft else self._cpu_v_target_bufs
        active_len = self.draft_len if is_draft else self.target_len
        W_p_dict = self.W_p_draft_dict if is_draft else self.W_p_target_dict
        
        buf_k = cpu_k_bufs.get(d_model)
        buf_v = cpu_v_bufs.get(d_model)
        
        grid_coords_buf = self._grid_coords
        
        if buf_k is None or grid_coords_buf is None or self.grid_coords_len == 0 or active_len == 0:
            cache_key = (B, H, max_matches, d_model, dtype)
            if cache_key not in self._empty_batch_cache:
                zeros = mx.zeros((B, H, max_matches, d_model), dtype=dtype)
                self._empty_batch_cache[cache_key] = (zeros, zeros)
            return self._empty_batch_cache[cache_key]
            
        cpu_k_buf = buf_k
        cpu_v_buf = buf_v
        
        if d_model not in W_p_dict or W_p_dict[d_model].dtype != dtype:
            self._init_default_projection(dtype, is_draft, d_model)
            
        W_p = W_p_dict[d_model]
        queries_8d = queries @ W_p
        
        # Stochastic Resonance Doping
        if hasattr(self, "noise_scale") and self.noise_scale > 0.0:
            std = mx.sqrt(mx.var(queries_8d) + 1e-9)
            noise = mx.random.normal(queries_8d.shape, dtype=queries_8d.dtype)
            queries_8d = queries_8d + self.noise_scale * std * noise
            
        quantized = self._quantize(queries_8d)
        
        # Update adjacency representation if database changed
        if self._grid_coords is not None and self.grid_coords_len > 0:
            if not hasattr(self, "_last_adj_len") or self._last_adj_len != self.grid_coords_len:
                self._update_adjacency_matrix()
                self._last_adj_len = self.grid_coords_len
                
        # Perform neuromorphic search on the full database
        k_val = min(max_matches, S * self.grid_coords_len)
        db_idx, is_neighbor = self.neuromorphic_search_batch(quantized, max_matches, B, H, S)
        
        # Zero-copy index lookup
        matched_k = cpu_k_buf[db_idx]
        matched_v = cpu_v_buf[db_idx]
        
        matched_k = matched_k * mx.expand_dims(is_neighbor, axis=-1)
        matched_v = matched_v * mx.expand_dims(is_neighbor, axis=-1)
        
        if k_val < max_matches:
            pad_len = max_matches - k_val
            cache_key = (B, H, pad_len, d_model, dtype)
            if cache_key not in self._empty_batch_cache:
                self._empty_batch_cache[cache_key] = mx.zeros((B, H, pad_len, d_model), dtype=dtype)
            pad_zeros = self._empty_batch_cache[cache_key]
            
            k_out = mx.concatenate([matched_k, pad_zeros], axis=2)
            v_out = mx.concatenate([matched_v, pad_zeros], axis=2)
            return k_out, v_out
        else:
            return matched_k, matched_v


class CellularAutomataCompactor:
    def __init__(self, size: int):
        self.size = size
        
    def compact(self, states: mx.array) -> mx.array:
        # Rule 90 1D Cellular Automaton: next_state = state_left XOR state_right
        # Pad or slice to size
        S = states.shape[0]
        if S <= 1:
            return states
            
        left = mx.concatenate([states[-1:], states[:-1]], axis=0)
        right = mx.concatenate([states[1:], states[:1]], axis=0)
        
        # Bitwise XOR operation in MLX
        next_states = (left.astype(mx.int32) ^ right.astype(mx.int32)).astype(mx.bool_)
        return next_states


class QuantumWalkMetropolisSearch:
    def __init__(self, size: int = 240):
        self.size = size
        # Grover diffusion coin: C = (2/N)*J - I
        self.coin = (2.0 / size) * mx.ones((size, size)) - mx.eye(size)
        
    def walk(self, psi: mx.array, steps: int = 2) -> mx.array:
        # Simulates a discrete-time quantum walk (DTQW) over the E8 transition graph
        # input psi has shape [240]
        for _ in range(steps):
            # Apply coin operator
            psi = self.coin @ psi
            # Shift operator (diffusive absolute values)
            psi = mx.abs(psi)
            psi_sum = mx.sum(psi) + 1e-9
            psi = psi / psi_sum
        return psi


class DNASelfAssemblingLookup:
    def __init__(self, size: int):
        self.size = size
        
    def hybridize(self, q_dna: mx.array, db_dna: mx.array) -> mx.array:
        # DNA complementary base matching: A-T (00 ^ 11 = 11) and C-G (01 ^ 10 = 11)
        # Match score is the count of matches of base pair XOR = 3
        # q_dna: [B, H, S, 4], db_dna: [M, 4]
        q_exp = mx.expand_dims(q_dna, axis=-2) # [B, H, S, 1, 4]
        db_exp = mx.expand_dims(db_dna, axis=0) # [1, M, 4]
        
        xor_result = q_exp ^ db_exp
        matches = (xor_result == 3).astype(mx.int32)
        match_scores = mx.sum(matches, axis=-1) # [B, H, S, M]
        return match_scores


class FloquetResonanceDoping:
    def __init__(self, d_model: int, omega: float = 10.0):
        self.d_model = d_model
        self.omega = omega
        
    def modulate(self, coords: mx.array, time_step: int) -> mx.array:
        # Periodic temporal high-frequency acoustic modulation
        # Prevent spatial coordinate clustering: coords * (1 + 0.05 * sin(omega * t))
        phase = self.omega * time_step
        amplitude = 1.0 + 0.05 * mx.sin(mx.array(phase))
        return coords * amplitude


class SlimeMoldPaging:
    def __init__(self, capacity: int, decay: float = 0.02):
        self.capacity = capacity
        self.decay = decay
        self.flux = mx.zeros((capacity,))
        
    def update_flux(self, active_indices: mx.array, capacity_len: int):
        # Strengthen Physarum polycephalum tube networks on co-activation, decay others
        # Broadcasted comparison: shape (num_active, capacity)
        comparison = active_indices[:, None] == mx.arange(self.capacity)
        # Mask out invalid indices (those >= capacity_len)
        valid_mask = (active_indices < capacity_len)[:, None]
        valid_comparison = comparison & valid_mask
        
        # Reduce along axis 0 (any) to see which capacity slots were activated
        update = mx.any(valid_comparison, axis=0).astype(mx.float32)
            
        self.flux = (1.0 - self.decay) * self.flux + self.decay * update
        
    def prune_inactive(self, threshold: float = 0.05) -> mx.array:
        # Returns boolean mask where flux is above threshold
        return self.flux >= threshold


class DNAReassociationCotAnalysis:
    def __init__(self, k_rate: float = 0.1, c0: float = 1.0):
        self.k_rate = k_rate
        self.c0 = c0

    def get_reassociation_fraction(self, time_step: mx.array) -> mx.array:
        # Cot fraction: C/C0 = 1 / (1 + k * C0 * t)
        return 1.0 / (1.0 + self.k_rate * self.c0 * time_step)


class AntColonyPheromoneTrails:
    def __init__(self, num_pages: int, decay: float = 0.05, alpha: float = 1.0):
        self.num_pages = num_pages
        self.decay = decay
        self.alpha = alpha
        self.pheromones = mx.ones((num_pages,))

    def update_trail(self, visited_pages: mx.array):
        # Reinforce pheromone level of visited pages: tau = (1 - decay) * tau + d_tau
        update = mx.zeros((self.num_pages,))
        if visited_pages.size > 0:
            comparison = visited_pages[:, None] == mx.arange(self.num_pages)
            update = mx.any(comparison, axis=0).astype(mx.float32) * self.alpha
            
        self.pheromones = (1.0 - self.decay) * self.pheromones + update

    def get_priority_pages(self, k: int) -> mx.array:
        sorted_indices = mx.argsort(self.pheromones)
        return sorted_indices[-k:]


class AdelicAdjointGroupLayout:
    def __init__(self, prime: int = 2, max_depth: int = 8):
        self.prime = prime
        self.max_depth = max_depth

    def get_physical_offset(self, padic_coords: mx.array) -> mx.array:
        powers = mx.array([self.prime ** i for i in range(self.max_depth)], dtype=mx.int32)
        offset = mx.sum(padic_coords * powers[None, :], axis=-1)
        return offset


class CytoskeletalMicrotubuleRouting:
    def __init__(self, num_slots: int, velocity: int = 1):
        self.num_slots = num_slots
        self.velocity = velocity
        self.tubulins = mx.zeros((num_slots,), dtype=mx.int32)

    def step(self, motor_positions: mx.array) -> mx.array:
        if motor_positions.size == 0:
            return motor_positions
        next_positions = (motor_positions + self.velocity) % self.num_slots
        return next_positions


class STDPPlasticitySwappingPriority:
    def __init__(self, num_pages: int, tau_stdp: float = 10.0, a_plus: float = 0.1, a_minus: float = 0.12):
        self.num_pages = num_pages
        self.tau_stdp = tau_stdp
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.weights = mx.ones((num_pages, num_pages)) * 0.5
        self.last_spike_times = mx.zeros((num_pages,))

    def register_spike(self, page_idx: int, current_time: float):
        t_pre = self.last_spike_times
        dt = current_time - t_pre
        
        dw_plus = self.a_plus * mx.exp(-dt / self.tau_stdp)
        dw_minus = -self.a_minus * mx.exp(dt / self.tau_stdp)
        
        dw = mx.where(dt > 0, dw_plus, dw_minus)
        self.weights[:, page_idx] = mx.clip(self.weights[:, page_idx] + dw, 0.0, 1.0)
        self.last_spike_times[page_idx] = current_time

    def get_eviction_candidate(self, active_pages: mx.array) -> int:
        connectivity = mx.sum(self.weights[active_pages, :], axis=0)
        mask = mx.zeros((self.num_pages,))
        if active_pages.size > 0:
            comparison = active_pages[:, None] == mx.arange(self.num_pages)
            in_active = mx.any(comparison, axis=0)
            connectivity = mx.where(in_active, 1e9, connectivity)
            
        return int(mx.argmin(connectivity).item())


class SIREpidemicSemanticHubbing:
    def __init__(self, num_nodes: int, beta: float = 0.3, gamma: float = 0.1):
        self.num_nodes = num_nodes
        self.beta = beta
        self.gamma = gamma
        self.states = mx.zeros((num_nodes,))
        self.infection_durations = mx.zeros((num_nodes,))

    def step_epidemic(self, active_nodes: mx.array, adjacency_matrix: mx.array):
        if active_nodes.size > 0:
            comparison = active_nodes[:, None] == mx.arange(self.num_nodes)
            is_active = mx.any(comparison, axis=0)
            self.states = mx.where(is_active, mx.array(1.0), self.states)
            
        is_infected = (self.states == 1.0).astype(mx.float32)
        infected_neighbors = (adjacency_matrix @ is_infected[:, None])[:, 0]
        
        new_infections = (self.states == 0.0) & (infected_neighbors > 0.0)
        self.states = mx.where(new_infections, mx.array(1.0), self.states)
        
        self.infection_durations = mx.where(self.states == 1.0, self.infection_durations + 1.0, 0.0)
        should_recover = self.infection_durations > 3.0
        self.states = mx.where(should_recover, mx.array(2.0), self.states)
        
        should_susc = self.states == 2.0
        self.states = mx.where(should_susc & (mx.random.uniform(shape=(self.num_nodes,)) < 0.1), mx.array(0.0), self.states)

    def get_semantic_hubs(self, k: int) -> mx.array:
        sorted_indices = mx.argsort(self.infection_durations)
        return sorted_indices[-k:]


class ZeroCopyCacheSliceMapping:
    def __init__(self):
        pass

    def map_slice(self, cache_array: mx.array, start: int, end: int) -> mx.array:
        return cache_array[start:end]


class PrecompiledMetalJITKernels:
    def __init__(self):
        self.kernel_cache = {}

    def get_kernel(self, name: str) -> str:
        if name not in self.kernel_cache:
            self.kernel_cache[name] = f"// Precompiled Metal Kernel: {name}\nkernel void {name}() {{}}"
        return self.kernel_cache[name]

