import pytest
import torch
from qan_transformers.kernels.mps_scatter import mps_coordinate_gather_scatter

def pytorch_reference_gather(k, v, indices):
    B, H, S, d = k.shape
    if indices.dim() == 1:
        gather_indices = indices.view(1, 1, -1, 1).expand(B, H, -1, d)
    else:
        gather_indices = indices.view(B, 1, -1, 1).expand(B, H, -1, d)
    K_ref = torch.gather(k, 2, gather_indices)
    V_ref = torch.gather(v, 2, gather_indices)
    return K_ref, V_ref

@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("is_shared", [True, False])
@pytest.mark.parametrize("B, H, S, d, K_size", [
    (1, 4, 30, 16, 5),
    (2, 4, 30, 16, 5),
])
def test_forward_backward_equivalence(device, is_shared, B, H, S, d, K_size):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device not available on this platform")
        
    dtype = torch.float32  # Use float32 for safety across both CPU and MPS
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    q = torch.randn(B, H, 1, d, device=device, dtype=dtype)
    
    if is_shared:
        indices = torch.randint(0, S, (K_size,), device=device, dtype=torch.long)
    else:
        indices = torch.randint(0, S, (B, K_size), device=device, dtype=torch.long)
        
    k = torch.randn(B, H, S, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, d, device=device, dtype=dtype, requires_grad=True)
    
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    
    # 1. Run fused MPS kernel
    K_test, V_test = mps_coordinate_gather_scatter(q, k, v, indices)
    
    # 2. Run standard PyTorch reference
    K_ref, V_ref = pytorch_reference_gather(k_ref, v_ref, indices)
    
    # 3. Assert forward outputs match exactly
    atol = 1e-4 if device == "mps" else 1e-6
    assert torch.allclose(K_test, K_ref, atol=atol), f"K mismatch on {device}"
    assert torch.allclose(V_test, V_ref, atol=atol), f"V mismatch on {device}"
    
    # 4. Assert backward pass gradients match
    grad_K = torch.randn_like(K_test)
    grad_V = torch.randn_like(V_test)
    
    (K_test * grad_K + V_test * grad_V).sum().backward()
    (K_ref * grad_K + V_ref * grad_V).sum().backward()
    
    assert torch.allclose(k.grad, k_ref.grad, atol=atol), f"k gradient mismatch on {device}"
    assert torch.allclose(v.grad, v_ref.grad, atol=atol), f"v gradient mismatch on {device}"

@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("is_shared", [True, False])
@pytest.mark.parametrize("B, H, S, d, K_size", [
    (1, 2, 8, 4, 3),
    (2, 2, 8, 4, 3),
])
def test_gradcheck(device, is_shared, B, H, S, d, K_size):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS device not available on this platform")
        
    dtype = torch.float32 if device == "mps" else torch.float64
    
    torch.manual_seed(42)
    q = torch.randn(B, H, 1, d, device=device, dtype=dtype)
    
    if is_shared:
        indices = torch.randint(0, S, (K_size,), device=device, dtype=torch.long)
    else:
        indices = torch.randint(0, S, (B, K_size), device=device, dtype=torch.long)
        
    k = torch.randn(B, H, S, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, d, device=device, dtype=dtype, requires_grad=True)
    
    def wrapper(k_in, v_in):
        return mps_coordinate_gather_scatter(q, k_in, v_in, indices)
        
    if device == "mps":
        eps = 1e-2
        atol = 1e-2
        rtol = 1e-2
    else:
        eps = 1e-6
        atol = 1e-5
        rtol = 1e-4
        
    # Run autograd gradcheck
    # We suppress user warning about float32 for MPS gradcheck
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input #.* requires gradient and is not a double precision")
        passed = torch.autograd.gradcheck(wrapper, (k, v), eps=eps, atol=atol, rtol=rtol)
        assert passed, f"Gradcheck failed on {device}"
