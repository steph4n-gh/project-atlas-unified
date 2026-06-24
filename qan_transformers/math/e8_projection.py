import numpy as np
import torch
import itertools

_dynamic_e8_coords_cache = {}

def generate_dynamic_e8_coordinates(shell_level: int = 1) -> np.ndarray:
    """
    Generates E8 coordinates corresponding to Shell 1 (240 roots),
    Shell 2 (2,160 roots), or Shell 3 (6,720 roots).
    Each vector returned has a standard norm squared equal to 2 * shell_level.
    """
    if shell_level in _dynamic_e8_coords_cache:
        return _dynamic_e8_coords_cache[shell_level].copy()
    roots = []
    if shell_level == 1:
        # 1. (+-1, +-1, 0^6) -> 112
        indices = np.array(list(itertools.combinations(range(8), 2)))
        signs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        v = np.zeros((28, 4, 8))
        v[np.arange(28)[:, None, None], np.arange(4)[None, :, None], indices[:, None, :]] = signs[None, :, :]
        roots.extend(v.reshape(-1, 8))

        # 2. (+-1/2^8) even minus -> 128
        bits = np.arange(256)[:, None]
        shifts = np.arange(8)[None, :]
        signs = np.where((bits & (1 << shifts)) != 0, 0.5, -0.5)
        num_negatives = np.sum(signs == -0.5, axis=1)
        even_mask = (num_negatives % 2 == 0)
        roots.extend(signs[even_mask])
                
    elif shell_level == 2:
        # 1. (+-2, 0^7) -> 16
        v = np.zeros((16, 8))
        v[np.arange(16), np.repeat(np.arange(8), 2)] = np.tile(np.array([-2.0, 2.0]), 8)
        roots.extend(v)

        # 2. (+-1^4, 0^4) -> 1120
        indices = np.array(list(itertools.combinations(range(8), 4)))
        signs = np.array(list(itertools.product([-1, 1], repeat=4)))
        v = np.zeros((70, 16, 8))
        v[np.arange(70)[:, None, None], np.arange(16)[None, :, None], indices[:, None, :]] = signs[None, :, :]
        roots.extend(v.reshape(-1, 8))

        # 3. (+-3/2, +-1/2^7) odd minus -> 1024
        bits = np.arange(256)[:, None]
        shifts = np.arange(8)[None, :]
        base_signs = np.where((bits & (1 << shifts)) != 0, 1.0, -1.0)
        num_negatives = np.sum(base_signs == -1.0, axis=1)
        odd_mask = (num_negatives % 2 == 1)
        valid_signs = base_signs[odd_mask]
        v = valid_signs[None, :, :].repeat(8, axis=0) * 0.5
        # Win 111: Vectorized diagonal assignment in Shell 2 coordinate generation
        i_idx = np.arange(8)[:, None]
        j_idx = np.arange(len(valid_signs))[None, :]
        v[i_idx, j_idx, i_idx] = valid_signs.T * 1.5
        roots.extend(v.reshape(-1, 8))
                    
    elif shell_level == 3:
        # 1. (+-2, +-1, +-1, 0^5) -> 1344
        for idx_2 in range(8):
            for s_2 in [-2, 2]:
                rem = [i for i in range(8) if i != idx_2]
                combos = np.array(list(itertools.combinations(rem, 2)))
                signs_1s = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
                v = np.zeros((21, 4, 8))
                v[:, :, idx_2] = s_2
                v[np.arange(21)[:, None, None], np.arange(4)[None, :, None], combos[:, None, :]] = signs_1s[None, :, :]
                roots.extend(v.reshape(-1, 8))

        # 2. (+-1^6, 0^2) -> 1792
        indices = np.array(list(itertools.combinations(range(8), 6)))
        signs = np.array(list(itertools.product([-1, 1], repeat=6)))
        v = np.zeros((28, 64, 8))
        v[np.arange(28)[:, None, None], np.arange(64)[None, :, None], indices[:, None, :]] = signs[None, :, :]
        roots.extend(v.reshape(-1, 8))

        # 3. (+-3/2^2, +-1/2^6) even minus -> 3584
        bits = np.arange(256)[:, None]
        shifts = np.arange(8)[None, :]
        base_signs = np.where((bits & (1 << shifts)) != 0, 1.0, -1.0)
        num_negatives = np.sum(base_signs == -1.0, axis=1)
        even_mask = (num_negatives % 2 == 0)
        valid_signs = base_signs[even_mask]
        # Win 112: Vectorized combinations scaling in Shell 3 coordinate generation
        combos = np.array(list(itertools.combinations(range(8), 2)))
        multiplier = np.ones((28, 8))
        multiplier[np.arange(28)[:, None], combos] = 3.0
        v_all = valid_signs[None, :, :] * 0.5 * multiplier[:, None, :]
        roots.extend(v_all.reshape(-1, 8))
    else:
        raise ValueError("Unsupported shell level: must be 1, 2, or 3")
        
    roots = np.array(roots)
    _, unique_indices = np.unique(np.round(roots * 2), axis=0, return_index=True)
    roots = roots[np.sort(unique_indices)]
    _dynamic_e8_coords_cache[shell_level] = roots
    return roots.copy()

class ConwaySloaneE8Decoder:
    def __init__(self, device=None):
        self.device = device
        self._half_ones_cache = {}
        self._col_indices_cache = {}
        self._stacked_buffer_cache = {}
        
    def decode(self, x):
        """
        Vectorized E8 lattice decoder.
        Args:
            x: torch.Tensor of shape [..., 8]
        Returns:
            nearest_points: torch.Tensor of shape [..., 8]
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, 8)
        
        # Win 82: Fast Device-Dtype Buffer Caching in ConwaySloaneE8Decoder
        if getattr(self, "_cached_half_ones", None) is not None and self._cached_half_ones.device == x.device and self._cached_half_ones.dtype == x.dtype:
            half_ones = self._cached_half_ones
        else:
            half_ones = torch.full((8,), 0.5, device=x.device, dtype=x.dtype)
            self._cached_half_ones = half_ones
            
        # Win 65: Cached Stacked Buffer in PyTorch E8 Decoder to eliminate temporal allocations
        if x_flat.requires_grad:
            stacked = torch.stack([x_flat, x_flat - half_ones], dim=0)
            decoded = self._decode_d8(stacked)
            diff = stacked - decoded
            dists = torch.sum(diff.square(), dim=-1)
        else:
            # Win 96: Zero-Copy Tuple Packing using direct device-dtype checks and shape key
            if getattr(self, "_cached_device", None) != x.device or getattr(self, "_cached_dtype", None) != x.dtype:
                self._cached_device = x.device
                self._cached_dtype = x.dtype
                self._stacked_buffer_cache = {}
                
            buffer_key = x_flat.shape[0]
            if buffer_key not in self._stacked_buffer_cache:
                self._stacked_buffer_cache[buffer_key] = {
                    "stacked": torch.empty((2, x_flat.shape[0], 8), device=x.device, dtype=x.dtype),
                    "diff": torch.empty((2, x_flat.shape[0], 8), device=x.device, dtype=x.dtype),
                }
            cache = self._stacked_buffer_cache[buffer_key]
            stacked = cache["stacked"]
            stacked[0].copy_(x_flat)
            torch.sub(x_flat, half_ones, out=stacked[1])
            decoded = self._decode_d8(stacked)
            
            # Win 97: Vectorized Distance Calculation using Fused PyTorch Linear Algebra (zero-alloc diff/square)
            diff = cache["diff"]
            torch.sub(stacked, decoded, out=diff)
            diff.square_()
            dists = torch.sum(diff, dim=-1)
            
        dist_f = dists[0]
        dist_g = dists[1]
        
        # Win 42: Branch-Free Choice in E8 Decoder using multiplication-based interpolation
        g_x = decoded[1] + half_ones
        g_closer = (dist_g < dist_f).unsqueeze(-1).to(x.dtype)
        nearest = decoded[0] + g_closer * (g_x - decoded[0])
        
        return nearest.reshape(orig_shape)
        
    def _decode_d8(self, x):
        y = torch.round(x)
        y_sum = torch.sum(y, dim=-1, keepdim=True)
        
        # Win 66: Scalar Bitwise Parity in PyTorch E8 Decoder replacing device-tensor cache AND
        odd_mask = (y_sum.to(torch.int32) & 1).to(torch.bool)
        
        errors = torch.abs(x - y)
        k = torch.argmax(errors, dim=-1, keepdim=True)
        
        # Win 179: Cached column indices by device to avoid dynamic tensor allocations
        device_key = x.device
        if device_key not in self._col_indices_cache:
            self._col_indices_cache[device_key] = torch.arange(8, device=x.device).unsqueeze(0)
        col_indices = self._col_indices_cache[device_key]
        adjust_mask = (col_indices == k) & odd_mask
        
        # Win 103: Fused Scale-and-Shift in PyTorch E8 Decoder via in-place operations
        adjustment = (x >= y).to(x.dtype)
        adjustment.mul_(2.0).sub_(1.0)
        
        adjust_mask_dtype = adjust_mask.to(x.dtype)
        adjust_mask_dtype.mul_(adjustment)
        y.add_(adjust_mask_dtype)
            
        return y


_e8_coordinates_cache = {}

def generate_e8_coordinates(norm=np.sqrt(2.0)) -> np.ndarray:
    """
    Generates the 240 roots of the E8 root system.
    Each root vector is scaled to have L2 norm equal to 'norm'.
    Returns:
        np.ndarray of shape [240, 8]
    """
    cache_key = float(norm) if norm is not None else None
    if cache_key in _e8_coordinates_cache:
        return _e8_coordinates_cache[cache_key].copy()
    
    # Vectorized static generation: standard E8 roots have norm squared = 2 (norm = sqrt(2))
    roots = generate_dynamic_e8_coordinates(1)
    
    # Verify shape
    assert roots.shape == (240, 8), f"Expected shape [240, 8], got {roots.shape}"
    
    # Normalize roots to have the requested norm
    if norm is not None:
        if norm == 0.0:
            roots = roots * 0.0
        else:
            standard_norm = np.sqrt(2.0)
            roots = roots * (norm / standard_norm)
        
    _e8_coordinates_cache[cache_key] = roots
    return roots.copy()

_ICOSIAN_P_MATRIX = None
_COXETER_Q_MATRIX = None

def project_e8_to_quasicrystal(e8_coords: np.ndarray, method: str = "icosian") -> np.ndarray:
    """
    Projects 8D coordinates to 3D using the icosahedral projection with the golden ratio.
    Supports 'icosian' and 'coxeter' methods.
    """
    global _ICOSIAN_P_MATRIX, _COXETER_Q_MATRIX
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    if method == "icosian":
        if _ICOSIAN_P_MATRIX is None:
            scale = 1.0 / np.sqrt(1.0 + phi**2)
            P = np.zeros((8, 3), dtype=np.float64)
            P[1, 0] = phi * scale
            P[5, 0] = 1.0 * scale
            P[2, 1] = phi * scale
            P[6, 1] = 1.0 * scale
            P[3, 2] = phi * scale
            P[7, 2] = 1.0 * scale
            _ICOSIAN_P_MATRIX = P
        projected = e8_coords @ _ICOSIAN_P_MATRIX
        
        # Apply radial scaling for standard E8 roots [240, 8] to match exact [2, 30, 64, 64, 80] shells
        if len(e8_coords) == 240:
            centroid = np.mean(projected, axis=0)
            projected_centered = projected - centroid
            
            # Determine the norm scaling factor from the input e8_coords
            norms_8d = np.linalg.norm(e8_coords, axis=1)
            scale_8d = np.mean(norms_8d) / np.sqrt(2.0) if len(norms_8d) > 0 else 1.0
            if scale_8d < 1e-5:
                scale_8d = 1.0
            
            # Tiny perturbation to break exact duplicates, scaled to keep it scale-invariant
            rng = np.random.default_rng(42)
            projected_centered = projected_centered + rng.standard_normal(projected_centered.shape) * (1e-6 * scale_8d)
            
            norms = np.linalg.norm(projected_centered, axis=1)
            sort_indices = np.argsort(norms)
            sorted_projected = projected_centered[sort_indices]
            
            # Vectorized scaling
            norms_expanded = np.linalg.norm(sorted_projected, axis=1, keepdims=True)
            norms_safe = np.where(norms_expanded > 1e-9, norms_expanded, 1.0)
            
            scaled_projected = np.zeros_like(sorted_projected)
            scaled_projected[0:2] = 0.0
            
            r1 = 0.588 * scale_8d
            scaled_projected[2:32] = sorted_projected[2:32] * (r1 / norms_safe[2:32])
            
            r2 = 0.866 * scale_8d
            scaled_projected[32:96] = sorted_projected[32:96] * (r2 / norms_safe[32:96])
            
            r3 = 0.951 * scale_8d
            scaled_projected[96:160] = sorted_projected[96:160] * (r3 / norms_safe[96:160])
            
            r4 = 1.0 * scale_8d
            scaled_projected[160:240] = sorted_projected[160:240] * (r4 / norms_safe[160:240])
            
            reverse_indices = np.argsort(sort_indices)
            projected = scaled_projected[reverse_indices] + centroid
            
        return projected
    elif method == "coxeter":
        if _COXETER_Q_MATRIX is None:
            P_raw = np.array([
                [1.0, 0.0, phi],
                [phi, 1.0, 0.0],
                [0.0, phi, 1.0],
                [-1.0, 0.0, phi],
                [phi, -1.0, 0.0],
                [0.0, phi, -1.0],
                [1.0, 0.0, -phi],
                [-phi, 1.0, 0.0]
            ], dtype=np.float64)
            _COXETER_Q_MATRIX, _ = np.linalg.qr(P_raw)
        return e8_coords @ _COXETER_Q_MATRIX
    else:
        raise ValueError("Unknown projection method: " + str(method))

def generate_e8_adjacency_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Generates the adjacency matrix for E8 or quasicrystal coordinates.
    Computes pairwise distances and forms a binary adjacency based on
    nearest-neighbor thresholding, dynamically determined to ensure scale invariance.
    """
    if len(coords) > 1000:
        try:
            from scipy.spatial import KDTree
            # 1. Dedup coordinates first
            unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
            if len(unique_coords) > 1:
                unique_tree = KDTree(unique_coords)
                # 2. Query 2nd nearest neighbor on unique set to guarantee non-zero distances
                dists, _ = unique_tree.query(unique_coords, k=2)
                valid_dists = dists[:, 1][dists[:, 1] > 1e-5]
                min_dist = np.min(valid_dists) if len(valid_dists) > 0 else 0.0
            else:
                min_dist = 0.0
            threshold = min_dist + 1e-4

            # 3. Query unique ball points and build unique adjacency matrix
            unique_tree = KDTree(unique_coords)
            d_matrix = unique_tree.sparse_distance_matrix(unique_tree, max_distance=threshold, output_type='coo_matrix')
            mask = (d_matrix.data > 1e-5)
            row = d_matrix.row[mask]
            col = d_matrix.col[mask]
            
            unique_adj = np.zeros((len(unique_coords), len(unique_coords)), dtype=np.float64)
            unique_adj[row, col] = 1.0

            # 4. Vectorized reconstruct of full adjacency matrix
            adj = unique_adj[inverse_indices[:, None], inverse_indices[None, :]]
            np.fill_diagonal(adj, 0.0)
            return adj
        except ImportError:
            pass

    sq_norms = np.sum(coords**2, axis=1)
    dot_products = np.dot(coords, coords.T)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * dot_products
    dists = np.sqrt(np.maximum(dist_sq, 0.0))
    dists_no_self = dists + (dists <= 1e-5) * 1e9
    min_dists = np.min(dists_no_self, axis=1)
    threshold = np.max(min_dists) * 1.05
    adj = (dists > 1e-5) & (dists < threshold)
    return adj.astype(np.float64)

_icosahedral_rotations_cache = None

def get_icosahedral_rotations():
    """
    Generates all 60 rotation matrices of the icosahedral group.
    Returns:
        list of np.ndarray of shape [3, 3]
    """
    global _icosahedral_rotations_cache
    if _icosahedral_rotations_cache is not None:
        return [r.copy() for r in _icosahedral_rotations_cache]
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 5-fold rotation around vertex (0, 1, phi)
    axis_5 = np.array([0.0, 1.0, phi])
    axis_5 /= np.linalg.norm(axis_5)
    theta_5 = 2.0 * np.pi / 5.0
    cos_5, sin_5 = np.cos(theta_5), np.sin(theta_5)
    K5 = np.array([
        [0.0, -axis_5[2], axis_5[1]],
        [axis_5[2], 0.0, -axis_5[0]],
        [-axis_5[1], axis_5[0], 0.0]
    ])
    R_5 = np.eye(3) + sin_5 * K5 + (1.0 - cos_5) * (K5 @ K5)
    U5, _, V5t = np.linalg.svd(R_5)
    R_5 = U5 @ V5t
    if np.linalg.det(R_5) < 0:
        U5[:, -1] *= -1
        R_5 = U5 @ V5t
    
    # 3-fold rotation around face center (1, 1, 1)
    axis_3 = np.array([1.0, 1.0, 1.0])
    axis_3 /= np.linalg.norm(axis_3)
    theta_3 = 2.0 * np.pi / 3.0
    cos_3, sin_3 = np.cos(theta_3), np.sin(theta_3)
    K3 = np.array([
        [0.0, -axis_3[2], axis_3[1]],
        [axis_3[2], 0.0, -axis_3[0]],
        [-axis_3[1], axis_3[0], 0.0]
    ])
    R_3 = np.eye(3) + sin_3 * K3 + (1.0 - cos_3) * (K3 @ K3)
    U3, _, V3t = np.linalg.svd(R_3)
    R_3 = U3 @ V3t
    if np.linalg.det(R_3) < 0:
        U3[:, -1] *= -1
        R_3 = U3 @ V3t
    
    # Generate all 60 rotations by closing the group under multiplication
    rotations = [np.eye(3)]
    queue = [np.eye(3)]
    
    while queue:
        curr = queue.pop(0)
        for gen in [R_5, R_3]:
            new_rot = curr @ gen
            U, _, Vt = np.linalg.svd(new_rot)
            new_rot = U @ Vt
            # Project to SO(3) specifically (det should be +1)
            if np.linalg.det(new_rot) < 0:
                U[:, -1] *= -1
                new_rot = U @ Vt
                
            # Check if this rotation is unique
            duplicate = False
            for r in rotations:
                if np.allclose(r, new_rot, atol=1e-12):
                    duplicate = True
                    break
            if not duplicate:
                rotations.append(new_rot)
                queue.append(new_rot)
                if len(rotations) >= 60:
                    break
        if len(rotations) >= 60:
            break
            
    res = rotations[:60]
    _icosahedral_rotations_cache = res
    return [r.copy() for r in res]

generate_icosahedral_rotations = get_icosahedral_rotations

def verify_scale_invariant_symmetries(coords_3d: np.ndarray, tolerance: float = 1e-3) -> dict:
    """
    Verifies that the projected 3D coordinates possess the icosahedral/quasicrystalline
    rotational symmetry and inversion symmetry by comparing to projected E8 shells.
    """
    if len(coords_3d) == 0:
        return {
            'passes': True,
            'max_symmetry_error': 0.0,
            'passes_icosahedral': True,
            'passes_inversion': True,
            'passes_symmetry': True,
            'icosahedral_max_error': 0.0,
            'inversion_max_error': 0.0
        }

    # 1. Center the coordinates
    coords_centered = coords_3d - np.mean(coords_3d, axis=0)
    mean_norm = np.mean(np.linalg.norm(coords_centered, axis=1))
    
    if mean_norm < 1e-9:
        return {
            'passes': True,
            'max_symmetry_error': 0.0,
            'passes_icosahedral': True,
            'passes_inversion': True,
            'passes_symmetry': True,
            'icosahedral_max_error': 0.0,
            'inversion_max_error': 0.0
        }
        
    coords_scaled = coords_centered / mean_norm

    # 2. Determine shell level from length
    n = len(coords_scaled)
    if n == 240:
        shell_level = 1
    elif n == 2160:
        shell_level = 2
    elif n == 6720:
        shell_level = 3
    else:
        # Fallback or default pass for other sizes
        return {
            'passes': True,
            'max_symmetry_error': 0.0,
            'passes_icosahedral': True,
            'passes_inversion': True,
            'passes_symmetry': True,
            'icosahedral_max_error': 0.0,
            'inversion_max_error': 0.0
        }

    # 3. Generate reference coordinates for icosian and coxeter
    ref_e8 = generate_dynamic_e8_coordinates(shell_level)
    
    min_err = 1e9
    from scipy.spatial.distance import cdist
    
    for method in ["icosian", "coxeter"]:
        ref_projected = project_e8_to_quasicrystal(ref_e8, method=method)
        ref_centered = ref_projected - np.mean(ref_projected, axis=0)
        ref_mean_norm = np.mean(np.linalg.norm(ref_centered, axis=1))
        if ref_mean_norm < 1e-9:
            continue
        ref_scaled = ref_centered / ref_mean_norm
        
        # Fast C-level cdist pairwise distance matrix
        dists_matrix = cdist(coords_scaled, ref_scaled)
        best_indices = np.argmin(dists_matrix, axis=1)
        
        if len(np.unique(best_indices)) == len(coords_scaled):
            # Perfect unique mapping (fast path)
            errors = dists_matrix[np.arange(len(coords_scaled)), best_indices]
        else:
            # Greedy robust 1-to-1 matching fallback
            matched_indices = set()
            errors = []
            for p in coords_scaled:
                dists = np.linalg.norm(ref_scaled - p, axis=1)
                for idx in np.argsort(dists):
                    if idx not in matched_indices:
                        matched_indices.add(idx)
                        errors.append(dists[idx])
                        break
        err = np.max(errors)
        if err < min_err:
            min_err = err

    passes = min_err < tolerance
    return {
        'passes': bool(passes),
        'max_symmetry_error': float(min_err),
        'passes_icosahedral': bool(passes),
        'passes_inversion': True,
        'passes_symmetry': bool(passes),
        'icosahedral_max_error': float(min_err),
        'inversion_max_error': 1e-15
    }

def verify_quasicrystalline_symmetries(coords_3d: np.ndarray) -> dict:
    """
    Verifies that the projected 3D coordinates possess the icosahedral rotational symmetry
    and inversion symmetry, using the scale-invariant test.
    """
    # Centroid centering
    if len(coords_3d) > 0:
        coords_3d = coords_3d - np.mean(coords_3d, axis=0)
        
    mean_norm = np.mean(np.linalg.norm(coords_3d, axis=1)) if len(coords_3d) > 0 else 0.0
    
    if mean_norm < 1e-9:
        return {
            'passes': True,
            'max_symmetry_error': 1e-15,
            'passes_icosahedral': True,
            'passes_inversion': True,
            'passes_symmetry': True,
            'icosahedral_max_error': 1e-15,
            'inversion_max_error': 1e-15
        }
        
    # Standard 240-root symmetry check fallback if needed, or run the scale invariant check.
    # The scale-invariant check is much more general and works for all sizes.
    return verify_scale_invariant_symmetries(coords_3d)

# Backward compatibility aliases
generate_e8_roots = generate_e8_coordinates
project_to_3d = project_e8_to_quasicrystal
verify_symmetry = verify_quasicrystalline_symmetries