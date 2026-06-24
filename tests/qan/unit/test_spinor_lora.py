import pytest
import torch
import torch.nn as nn
from qan_transformers.lora.pipeline import SpinorRotationLoRA

def test_spinor_lora_shapes_and_forward():
    # Square layer (in_features == out_features)
    linear_sq = nn.Linear(32, 32)
    lora_sq = SpinorRotationLoRA(linear_sq, r=4)
    x = torch.randn(2, 5, 32)
    
    out_sq = lora_sq(x)
    assert out_sq.shape == (2, 5, 32)
    
    # Non-square layer (in_features != out_features)
    linear_ns = nn.Linear(32, 16)
    lora_ns = SpinorRotationLoRA(linear_ns, r=4)
    out_ns = lora_ns(x)
    assert out_ns.shape == (2, 5, 16)

def test_spinor_lora_exponential_equivalence():
    d = 16
    r = 2
    linear = nn.Linear(d, d)
    lora = SpinorRotationLoRA(linear, r=r)
    
    # Randomly set weights
    torch.manual_seed(42)
    with torch.no_grad():
        nn.init.normal_(lora.lora_A)
        nn.init.normal_(lora.lora_B)
        
    A = lora.lora_A.data
    B = lora.lora_B.data
    
    # 1. Full d x d matrix exponential
    Omega_full = torch.mm(A, B.t()) - torch.mm(B, A.t()) # (d, d)
    exp_full = torch.linalg.matrix_exp(Omega_full) # (d, d)
    
    # 2. Subspace matrix exponential
    AB = torch.cat([A, B], dim=1)
    Q, _ = torch.linalg.qr(AB)
    hat_A = torch.mm(Q.t(), A)
    hat_B = torch.mm(Q.t(), B)
    hat_Omega = torch.mm(hat_A, hat_B.t()) - torch.mm(hat_B, hat_A.t())
    exp_sub = torch.linalg.matrix_exp(hat_Omega)
    
    # Reconstructed full matrix from subspace: R = I + Q @ (exp_sub - I) @ Q^T
    eye_2r = torch.eye(2 * r, device=Q.device)
    exp_recon = torch.eye(d) + torch.mm(torch.mm(Q, exp_sub - eye_2r), Q.t())
    
    # Verify they match perfectly
    assert torch.allclose(exp_full, exp_recon, atol=1e-5)

def test_spinor_lora_orthogonality():
    d = 16
    r = 2
    linear = nn.Linear(d, d)
    lora = SpinorRotationLoRA(linear, r=r)
    
    # Compute rotor R
    A = lora.lora_A
    B = lora.lora_B
    AB = torch.cat([A, B], dim=1)
    Q, _ = torch.linalg.qr(AB)
    hat_A = torch.mm(Q.t(), A)
    hat_B = torch.mm(Q.t(), B)
    hat_Omega = torch.mm(hat_A, hat_B.t()) - torch.mm(hat_B, hat_A.t())
    exp_sub = torch.linalg.matrix_exp(hat_Omega)
    
    # Check that the sub-exponential matrix is strictly orthogonal: R^T R = I
    eye_2r = torch.eye(2 * r)
    prod = torch.mm(exp_sub.t(), exp_sub)
    assert torch.allclose(prod, eye_2r, atol=1e-5)

def test_spinor_lora_backpropagation():
    linear = nn.Linear(16, 16)
    lora = SpinorRotationLoRA(linear, r=4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    
    out = lora(x)
    loss = torch.sum(out ** 2)
    loss.backward()
    
    # Verify gradients flow back to A, B and input x
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert x.grad is not None
    assert not torch.allclose(lora.lora_A.grad, torch.zeros_like(lora.lora_A.grad))
