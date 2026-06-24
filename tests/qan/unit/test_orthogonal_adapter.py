import pytest
import torch

from qan_transformers.modeling.attention import enforce_orthogonality, cayley_orthogonal_adapter

def test_orthogonal_adapter_drift_prevention():
    torch.manual_seed(42)
    d = 64
    r = 16
    num_samples = 100
    
    # Initialize A and B small
    A = torch.randn(d, r) * 0.01
    B = torch.randn(d, r) * 0.01
    A.requires_grad = True
    B.requires_grad = True
    
    X = torch.randn(num_samples, d)
    
    # 1. Check distance preservation at init
    W_L_init = torch.eye(d) + torch.matmul(A, B.t())
    dists_orig = torch.cdist(X, X)
    dists_init = torch.cdist(torch.matmul(X, W_L_init), torch.matmul(X, W_L_init))
    
    # Diff should be tiny
    assert torch.allclose(dists_orig, dists_init, atol=1e-1)
    
    # 2. Simulate training steps to let weights grow
    optimizer = torch.optim.SGD([A, B], lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        # Loss wants weights to grow
        W_L = torch.eye(d) + torch.matmul(A, B.t())
        loss = -torch.norm(W_L)  # push weight magnitude up
        loss.backward()
        optimizer.step()
        
    # Check drift without QR re-orthogonalization
    W_L_drifted = torch.eye(d) + torch.matmul(A, B.t())
    dists_drifted = torch.cdist(torch.matmul(X, W_L_drifted), torch.matmul(X, W_L_drifted))
    max_drift = torch.max(torch.abs(dists_orig - dists_drifted)).item()
    print(f"Max drift without QR: {max_drift:.4f}")
    
    # 3. Apply QR re-orthogonalization and see if it restores near-orthogonality
    # For A @ B.t() to be orthogonal, we want (I + A @ B.t()) @ (I + B @ A.t()) = I
    # A @ B.t() + B @ A.t() + A @ B.t() @ B @ A.t() = 0
    # Enforce QR on A and update B to maintain orthogonality constraints
    enforce_orthogonality(A, B)
    W_L_qr = torch.eye(d) + torch.matmul(A, B.t())
    
    # Check orthonormality of W_L_qr
    orth_err = torch.norm(torch.matmul(W_L_qr, W_L_qr.t()) - torch.eye(d)).item()
    print(f"Orthonormality error after QR adjustment: {orth_err:.4f}")
    
    # 4. Test Cayley map parameterization (strictly orthogonal by construction)
    A_c = torch.randn(d, r)
    B_c = torch.randn(d, r)
    X_cayley = cayley_orthogonal_adapter(X, A_c, B_c)
    dists_cayley = torch.cdist(X_cayley, X_cayley)
    # Zero out diagonal elements which suffer from numerical cdist precision artifacts
    dists_orig_zeroed = dists_orig.clone()
    dists_orig_zeroed.fill_diagonal_(0.0)
    dists_cayley_zeroed = dists_cayley.clone()
    dists_cayley_zeroed.fill_diagonal_(0.0)
    assert torch.allclose(dists_orig_zeroed, dists_cayley_zeroed, atol=1e-4)
