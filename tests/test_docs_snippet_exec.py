import torch
import torch.nn as nn
import pytest

# Copied directly from docs/woodbury_cayley_overview.md
def cayley_orthogonal_adapter(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, cache: dict = None) -> torch.Tensor:
    # A and B have shape [D, r], where D is model dimension and r is low rank (typically r=16)
    d, r = A.shape
    orig_shape = X.shape
    X_flat = X.reshape(-1, d)
    
    # Track parameter versions to handle weight updates during training
    version_A = getattr(A, "_version", 0)
    version_B = getattr(B, "_version", 0)
    
    # Return cached projections if parameters have not changed
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
        
        # Pack factor matrices into low-rank matrices of shape [D, 2r]
        U = torch.cat([A, -B], dim=1)  # [D, 2r]
        V = torch.cat([B, A], dim=1)   # [D, 2r]
        
        # Pre-allocate identity matrix to avoid dynamic allocations
        if cache is not None and "I_2r" in cache and cache["I_2r"].device == device:
            I_2r = cache["I_2r"]
        else:
            I_2r = torch.eye(2 * r, device=device, dtype=torch.float32)
            if cache is not None:
                cache["I_2r"] = I_2r
                
        # Perform matrix inversion in float32 for numerical stability
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
            
    # Apply Woodbury Cayley rotation to the hidden state X using a fused addmm
    X_U_inv = torch.matmul(X_flat, U_inv)
    X_adapted = torch.addmm(X_flat, X_U_inv, V_t, beta=1.0, alpha=-2.0)
    
    return X_adapted.reshape(orig_shape)

def test_docs_snippet_execution():
    torch.manual_seed(42)
    d = 64
    r = 16
    
    X = torch.randn(10, d)
    A = torch.randn(d, r) * 0.01
    B = torch.randn(d, r) * 0.01
    
    # 1. Test basic execution
    out = cayley_orthogonal_adapter(X, A, B)
    assert out.shape == X.shape
    
    # 2. Test cache usage
    cache = {}
    out_cached = cayley_orthogonal_adapter(X, A, B, cache=cache)
    assert "U_inv" in cache
    assert "V_t" in cache
    assert "I_2r" in cache
    
    # Run again with cache and check it returns correct results
    out_cached_2 = cayley_orthogonal_adapter(X, A, B, cache=cache)
    assert torch.allclose(out_cached, out_cached_2)
    
    # 3. Verify parameter version update resets cache
    # Simulate parameter update by modifying A in-place, which increments _version
    with torch.no_grad():
        A.add_(1.0)
    out_updated = cayley_orthogonal_adapter(X, A, B, cache=cache)
    assert cache["version_A"] > 0
    
    # 4. Verify distance preservation (orthogonality check)
    dists_orig = torch.cdist(X, X)
    dists_adapted = torch.cdist(out, out)
    assert torch.allclose(dists_orig, dists_adapted, atol=1e-4)

if __name__ == "__main__":
    test_docs_snippet_execution()
    print("All doc snippet tests passed!")
