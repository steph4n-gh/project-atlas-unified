import pytest
import torch
import torch.nn.functional as F

from qan_transformers.math.procrustes import compute_procrustes_alignment, validate_alignment

def test_procrustes_alignment_correctness():
    torch.manual_seed(42)
    d_src = 16
    d_tgt = 32
    num_samples_calib = 200
    num_samples_val = 100
    
    # 1. Generate a ground truth projection matrix (near-orthogonal)
    W_truth = torch.randn(d_src, d_tgt)
    U, _, Vh = torch.linalg.svd(W_truth, full_matrices=False)
    W_truth_orth = torch.matmul(U, Vh)
    
    # 2. Generate source hidden states and target states mapped with some noise
    X_src_calib = torch.randn(num_samples_calib, d_src)
    X_tgt_calib = torch.matmul(X_src_calib, W_truth_orth) + 0.1 * torch.randn(num_samples_calib, d_tgt)
    
    # 3. Compute alignment matrix
    M_align = compute_procrustes_alignment(X_src_calib, X_tgt_calib)
    
    # Assert row orthonormality (M_align @ M_align.t() = I)
    identity_approx = torch.matmul(M_align, M_align.t())
    assert torch.allclose(identity_approx, torch.eye(d_src), atol=1e-3)
    
    # 4. Validate on held-out set
    X_src_val = torch.randn(num_samples_val, d_src)
    X_tgt_val = torch.matmul(X_src_val, W_truth_orth) + 0.1 * torch.randn(num_samples_val, d_tgt)
    
    X_mapped_val = torch.matmul(X_src_val, M_align)
    
    correlation = validate_alignment(X_src_val, X_tgt_val, M_align)
    print(f"Held-out cosine similarity correlation: {correlation:.4f}")
    assert correlation >= 0.85

def test_procrustes_alignment_batched():
    torch.manual_seed(42)
    B, S, D_src, D_tgt = 2, 50, 16, 32
    
    W_truth = torch.randn(D_src, D_tgt)
    U, _, Vh = torch.linalg.svd(W_truth, full_matrices=False)
    W_truth_orth = torch.matmul(U, Vh)
    
    X_src_calib = torch.randn(B, S, D_src)
    X_tgt_calib = torch.matmul(X_src_calib, W_truth_orth) + 0.05 * torch.randn(B, S, D_tgt)
    
    M_align = compute_procrustes_alignment(X_src_calib, X_tgt_calib)
    
    identity_approx = torch.matmul(M_align, M_align.t())
    assert torch.allclose(identity_approx, torch.eye(D_src), atol=1e-3)
    
    X_src_val = torch.randn(B, S, D_src)
    X_tgt_val = torch.matmul(X_src_val, W_truth_orth) + 0.05 * torch.randn(B, S, D_tgt)
    
    correlation = validate_alignment(X_src_val, X_tgt_val, M_align)
    assert correlation >= 0.85

