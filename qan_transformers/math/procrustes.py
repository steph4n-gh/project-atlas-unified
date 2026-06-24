import torch
import torch.nn.functional as F

_PROCRUSTES_WORKSPACE_CACHE = {}

def compute_procrustes_alignment(X_src: torch.Tensor, X_tgt: torch.Tensor) -> torch.Tensor:
    """
    Computes the orthogonal Procrustes alignment matrix M_align such that:
    X_src @ M_align approximates X_tgt.
    Supports batched or multi-dimensional inputs by flattening leading dimensions.
    """
    if X_src.dim() > 2:
        X_src = X_src.reshape(-1, X_src.size(-1))
    if X_tgt.dim() > 2:
        X_tgt = X_tgt.reshape(-1, X_tgt.size(-1))

    device = X_src.device
    dtype = X_src.dtype
    N, D_src = X_src.shape
    _, D_tgt = X_tgt.shape

    # Win 99: Zero-Copy Integer Key in Procrustes Workspace Cache to avoid tuple allocations
    if getattr(compute_procrustes_alignment, "_cached_device", None) != device or getattr(compute_procrustes_alignment, "_cached_dtype", None) != dtype:
        compute_procrustes_alignment._cached_device = device
        compute_procrustes_alignment._cached_dtype = dtype
        _PROCRUSTES_WORKSPACE_CACHE.clear()

    key = (N << 32) | (D_src << 16) | D_tgt
    if key not in _PROCRUSTES_WORKSPACE_CACHE:
        _PROCRUSTES_WORKSPACE_CACHE[key] = {
            "A": torch.empty((N, D_src), device=device, dtype=dtype),
            "B": torch.empty((N, D_tgt), device=device, dtype=dtype),
            "C": torch.empty((D_src, D_tgt), device=device, dtype=dtype),
        }

    cache = _PROCRUSTES_WORKSPACE_CACHE[key]
    A = cache["A"]
    B = cache["B"]
    C = cache["C"]

    # Centering hidden states is standard to align translation offsets
    torch.sub(X_src, X_src.mean(dim=0, keepdim=True), out=A)
    torch.sub(X_tgt, X_tgt.mean(dim=0, keepdim=True), out=B)
    
    torch.matmul(A.t(), B, out=C)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    # Vh is already Vt. Vt has shape [d_src, d_tgt] for full_matrices=False when d_src <= d_tgt
    M_align = torch.matmul(U, Vh)
    return M_align

_ALIGNMENT_WORKSPACE_CACHE = {}

def validate_alignment(X_src_val: torch.Tensor, X_tgt_val: torch.Tensor, M_align: torch.Tensor) -> float:
    """
    Validates alignment by mapping X_src_val to target space and computing
    the correlation of their pairwise cosine similarities.
    Supports batched or multi-dimensional inputs by flattening leading dimensions.
    """
    if X_src_val.dim() > 2:
        X_src_val = X_src_val.reshape(-1, X_src_val.size(-1))
    if X_tgt_val.dim() > 2:
        X_tgt_val = X_tgt_val.reshape(-1, X_tgt_val.size(-1))

    device = X_src_val.device
    dtype = X_src_val.dtype
    N, D_src = X_src_val.shape
    _, D_tgt = X_tgt_val.shape
    
    # Win 99: Zero-Copy Integer Key in Alignment Workspace Cache to avoid tuple allocations
    if getattr(validate_alignment, "_cached_val_device", None) != device or getattr(validate_alignment, "_cached_val_dtype", None) != dtype:
        validate_alignment._cached_val_device = device
        validate_alignment._cached_val_dtype = dtype
        _ALIGNMENT_WORKSPACE_CACHE.clear()
        
    key = (N << 32) | (D_src << 16) | D_tgt
    
    if key not in _ALIGNMENT_WORKSPACE_CACHE:
        _ALIGNMENT_WORKSPACE_CACHE[key] = {
            "X_mapped_val": torch.empty((N, D_tgt), device=device, dtype=dtype),
            "X_tgt_norm": torch.empty((N, D_tgt), device=device, dtype=dtype),
            "X_mapped_norm": torch.empty((N, D_tgt), device=device, dtype=dtype),
            "norm_tgt_buf": torch.empty((N, 1), device=device, dtype=dtype),
            "norm_mapped_buf": torch.empty((N, 1), device=device, dtype=dtype),
            "cos_tgt": torch.empty((N, N), device=device, dtype=dtype),
            "cos_mapped": torch.empty((N, N), device=device, dtype=dtype),
            "x_centered_buf": torch.empty((N * N,), device=device, dtype=dtype),
            "y_centered_buf": torch.empty((N * N,), device=device, dtype=dtype),
        }
        
    cache = _ALIGNMENT_WORKSPACE_CACHE[key]
    X_mapped_val = cache["X_mapped_val"]
    X_tgt_norm = cache["X_tgt_norm"]
    X_mapped_norm = cache["X_mapped_norm"]
    norm_tgt_buf = cache["norm_tgt_buf"]
    norm_mapped_buf = cache["norm_mapped_buf"]
    cos_tgt = cache["cos_tgt"]
    cos_mapped = cache["cos_mapped"]
    x_centered = cache["x_centered_buf"]
    y_centered = cache["y_centered_buf"]
    
    # 1. Map source to target space: X_src_val @ M_align
    torch.matmul(X_src_val, M_align, out=X_mapped_val)
    
    # 2. Normalize target: X_tgt_val
    torch.linalg.vector_norm(X_tgt_val, ord=2, dim=-1, keepdim=True, out=norm_tgt_buf)
    norm_tgt_buf.clamp_(min=1e-12)
    torch.div(X_tgt_val, norm_tgt_buf, out=X_tgt_norm)
    
    # 3. Normalize mapped source: X_mapped_val
    torch.linalg.vector_norm(X_mapped_val, ord=2, dim=-1, keepdim=True, out=norm_mapped_buf)
    norm_mapped_buf.clamp_(min=1e-12)
    torch.div(X_mapped_val, norm_mapped_buf, out=X_mapped_norm)
    
    # 4. Pairwise cosine similarities
    torch.matmul(X_tgt_norm, X_tgt_norm.t(), out=cos_tgt)
    torch.matmul(X_mapped_norm, X_mapped_norm.t(), out=cos_mapped)
    
    # Flatten and compute correlation without torch.stack or corrcoef
    cos_tgt_flat = cos_tgt.flatten()
    cos_mapped_flat = cos_mapped.flatten()
    
    # Win 115: Pre-allocated centering buffers in Procrustes validation
    torch.sub(cos_tgt_flat, cos_tgt_flat.mean(), out=x_centered)
    torch.sub(cos_mapped_flat, cos_mapped_flat.mean(), out=y_centered)
    
    norm_x = torch.linalg.vector_norm(x_centered)
    norm_y = torch.linalg.vector_norm(y_centered)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
        
    correlation = (torch.dot(x_centered, y_centered) / (norm_x * norm_y)).item()
    return correlation

