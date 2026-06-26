import torch
import torch.nn as nn
import torch.nn.functional as F

_REPEAT_KV_CACHE = {}

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    
    # Win 90: Cached Block-Masks/Shapes for GQA Key-Value Replication in PyTorch Attention
    shape_key = (hidden_states.shape, n_rep)
    if shape_key in _REPEAT_KV_CACHE:
        expand_shape, reshape_shape = _REPEAT_KV_CACHE[shape_key]
    else:
        batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
        expand_shape = (batch, num_key_value_heads, n_rep, seq_len, head_dim)
        reshape_shape = (batch, num_key_value_heads * n_rep, seq_len, head_dim)
        _REPEAT_KV_CACHE[shape_key] = (expand_shape, reshape_shape)
        
    return (
        hidden_states.unsqueeze(2)
        .expand(expand_shape)
        .reshape(reshape_shape)
    )

def enforce_orthogonality(A: torch.Tensor, B: torch.Tensor):
    with torch.no_grad():
        Q, R = torch.linalg.qr(A)
        A.copy_(Q)
        B.copy_(torch.matmul(B, R.t()))

def cayley_orthogonal_adapter(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, cache: dict = None) -> torch.Tensor:
    d, r = A.shape
    orig_shape = X.shape
    X_flat = X.reshape(-1, d)
    
    version_A = getattr(A, "_version", 0)
    version_B = getattr(B, "_version", 0)
    
    if (cache is not None and "U_inv" in cache and 
        cache.get("device") == X.device and 
        cache.get("dtype") == X.dtype and
        cache.get("version_A") == version_A and
        cache.get("version_B") == version_B):
        U_inv = cache["U_inv"]
        V_t = cache["V_t"]
    else:
        device = A.device
        dtype = A.dtype
        U = torch.cat([A, -B], dim=1)  # [d, 2r]
        V = torch.cat([B, A], dim=1)  # [d, 2r]
        
        # Win 128: Cache pre-allocated identity matrix in Cayley Orthogonal Adapter to avoid dynamic eye allocation
        if cache is not None and "I_2r" in cache and cache["I_2r"].device == device:
            I_2r = cache["I_2r"]
        else:
            I_2r = torch.eye(2 * r, device=device, dtype=torch.float32)
            if cache is not None:
                cache["I_2r"] = I_2r
        VT_U_f32 = torch.matmul(V.t(), U).to(torch.float32)
        inv_M = torch.linalg.inv(I_2r + VT_U_f32).to(dtype)
        
        U_inv = torch.matmul(U, inv_M)
        V_t = V.t()
        
        if cache is not None:
            cache["U_inv"] = U_inv
            cache["V_t"] = V_t
            cache["device"] = X.device
            cache["dtype"] = X.dtype
            cache["version_A"] = version_A
            cache["version_B"] = version_B
            
    X_U_inv = torch.matmul(X_flat, U_inv)
    # Win 80: Fused matrix multiplication and subtraction using torch.addmm
    X_adapted = torch.addmm(X_flat, X_U_inv, V_t, beta=1.0, alpha=-2.0)
    
    return X_adapted.reshape(orig_shape)
