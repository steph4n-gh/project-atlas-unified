import pytest
import torch
import numpy as np
from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB

def shannon_entropy(coords: torch.Tensor) -> float:
    # Convert coordinates to round representation for unique counting
    rounded = torch.round(coords * 10) / 10
    unique_coords, counts = torch.unique(rounded, dim=0, return_counts=True)
    probs = counts.float() / len(coords)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-12))
    return entropy.item()

def test_w_p_rank_1_vs_orthogonal_entropy():
    # Set seed for reproducibility
    torch.manual_seed(42)
    device = "cpu"
    d_model = 64
    num_samples = 500
    
    # 1. Generate diverse keys
    keys = torch.randn(num_samples, d_model, device=device)
    
    # 2. Test rank-1 projection (what was there originally: torch.ones)
    # Every column is identical, so projection collapses all dimensions to 1D
    W_p_rank1 = torch.nn.functional.normalize(torch.ones(d_model, 8, device=device), dim=0)
    keys_8d_rank1 = keys @ W_p_rank1
    
    db = AdelicMemorySwapGridDB(d_model=d_model, device=device)
    quantized_rank1 = db._quantize(keys_8d_rank1)
    entropy_rank1 = shannon_entropy(quantized_rank1)
    
    print(f"Rank-1 projection entropy: {entropy_rank1:.4f} bits")
    # Rank-1 should collapse to very few E8 bins compared to high-rank projections
    assert entropy_rank1 < 3.2
    
    # 3. Test default initialized projection (should be rank-8 orthogonal)
    db_orth = AdelicMemorySwapGridDB(d_model=d_model, device=device)
    # Trigger default initialization
    db_orth.swap_out_target(keys, keys)
    quantized_orth = db_orth.grid_coords
    entropy_orth = shannon_entropy(quantized_orth)
    
    print(f"Orthogonal projection entropy: {entropy_orth:.4f} bits")
    assert entropy_orth > 4.5

def test_svd_bisection_entropy():
    torch.manual_seed(42)
    device = "cpu"
    d_model = 64
    num_samples = 500
    
    # Generate keys and mock weights
    keys = torch.randn(num_samples, d_model, device=device)
    W_q = torch.randn(128, d_model, device=device)
    W_k = torch.randn(128, d_model, device=device)
    
    db = AdelicMemorySwapGridDB(d_model=d_model, device=device)
    db.initialize_projections(W_q, W_k, is_draft=False)
    
    # Project keys through the SVD-initialized matrix
    keys_8d = keys @ db.W_p_target
    quantized = db._quantize(keys_8d)
    
    entropy_svd = shannon_entropy(quantized)
    print(f"SVD projection entropy: {entropy_svd:.4f} bits")
    assert entropy_svd > 4.5
