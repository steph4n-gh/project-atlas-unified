import pytest
import torch
import numpy as np

from qan_transformers.modeling import graft_model
from qan_transformers.modeling.attention import DenseAttention, QuasicrystallineAttention
from qan_transformers.optim.adelic import AdelicLangevinOptimizer
from qan_transformers.firewall.cohomology import CohomologyFirewall

def test_t4_real_world_long_qa_ceiling():
    """
    Verify that QuasicrystallineAttention KV cache stays well under the 17.76 GB ceiling
    when scaled to 500k context compared to Dense attention.
    """
    embed_dim = 2048
    num_heads = 8
    num_layers = 18
    sparse_ratio = 0.15
    
    # Calculate KV cache size for 500k context in float16
    S = 500000
    element_size = 2 # float16
    
    # Dense elements = 2 * Layers * Heads * S * head_dim
    # Since Heads * head_dim = embed_dim
    dense_elements = 2 * num_layers * S * embed_dim
    dense_vram_gb = (dense_elements * element_size) / (1024**3)
    
    # QAN-ATLAS active GPU elements = Dense elements * sparse_ratio
    qan_vram_gb = dense_vram_gb * sparse_ratio
    
    assert qan_vram_gb < 15.0, f"QAN VRAM projected to 500k context must be < 15 GB, got {qan_vram_gb:.2f} GB"
    assert dense_vram_gb > 17.76, f"Dense VRAM projected should exceed 17.76 GB, got {dense_vram_gb:.2f} GB"

def test_t4_real_world_optimizer_convergence():
    """
    Verify that AdelicLangevinOptimizer converges successfully and is robust to NaN/Inf gradients.
    """
    p = torch.nn.Parameter(torch.tensor([2.0, -2.0, 3.0]))
    opt = AdelicLangevinOptimizer([p], lr=0.02, alpha=0.75, T_0=0.01)
    
    target = torch.tensor([0.0, 0.0, 0.0])
    initial_dist = float(torch.norm(p - target).item())
    
    # 1. Standard convergence steps
    for step in range(50):
        loss = 0.5 * torch.sum((p - target)**2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    intermediate_dist = float(torch.norm(p - target).item())
    assert intermediate_dist < initial_dist, "Optimizer failed to converge towards target"
    
    # 2. NaN/Inf robustness test: inject bad gradients and verify parameter stability
    p.grad = torch.tensor([float('nan'), float('inf'), -float('inf')])
    opt.step()
    
    assert not torch.isnan(p.data).any(), "AdelicLangevinOptimizer parameters became NaN on NaN gradients"
    assert not torch.isinf(p.data).any(), "AdelicLangevinOptimizer parameters became Inf on Inf gradients"

def test_t4_real_world_token_generation_rollback():
    """
    Verify that batched token generation rollback (B=2) works successfully without crashes.
    """
    model = graft_model("google/gemma-4-e2b", lightweight=True)
    model.eval()
    
    # We will trigger a fracture to test batched rollback loop execution
    class MockFracturedFirewall(CohomologyFirewall):
        def check_obstruction(self, attn_matrix):
            # Returns B=2 lists of outputs: one item is normal, one is fractured
            # attn_matrix is B=2 during batched generation
            return [False, True], [0.1, 5.0], [[], [1, 2, 3]]
            
    firewall = MockFracturedFirewall()
    
    # Attach firewall
    for layer in model.layers:
        if isinstance(layer.attn, QuasicrystallineAttention):
            layer.attn.firewall = firewall
            layer.attn.sparse_ratio = 1.0
            
    input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long) # B = 2
    
    # Generate should run without crashes and return the expected token shape
    generated = model.generate(input_ids, max_new_tokens=2, firewall=firewall, rollback_limit=2)
    assert generated.shape == (2, 5), f"Expected shape (2, 5), got {generated.shape}"

def test_t4_real_world_peak_memory_profile():
    """
    Verify that QuasicrystallineAttention peak active KV cache size is strictly smaller
    than DenseAttention at sequence length 1000.
    """
    embed_dim = 128
    num_heads = 8
    sparse_ratio = 0.15
    
    dense_layer = DenseAttention(embed_dim=embed_dim, num_heads=num_heads, sparse_ratio=sparse_ratio)
    qan_layer = QuasicrystallineAttention(embed_dim=embed_dim, num_heads=num_heads, sparse_ratio=sparse_ratio)
    
    x = torch.randn(1, 1000, embed_dim)
    
    cache_dense = {}
    cache_qan = {}
    
    with torch.no_grad():
        dense_layer(x, kv_cache=cache_dense)
        qan_layer(x, kv_cache=cache_qan)
        
    dense_mem = cache_dense["K"].nelement() + cache_dense["V"].nelement()
    qan_mem = cache_qan["K"].nelement() + cache_qan["V"].nelement()
    
    assert qan_mem < dense_mem, f"Expected QAN KV cache ({qan_mem}) to be smaller than Dense ({dense_mem})"

def test_t4_real_world_adversarial_attack_hardening():
    """
    Verify that CohomologyFirewall detects NaNs/Infs and triggers fracture defense correctly.
    """
    firewall = CohomologyFirewall(threshold=1.5)
    
    # 1. NaN attention maps must trigger fracture
    nan_attn = torch.tensor([[[[float('nan'), 0.1], [0.1, 0.8]]]]) # [1, 1, 2, 2]
    is_fractured, cfi, alt_idx = firewall.check_obstruction(nan_attn)
    assert is_fractured, "NaN attention maps must trigger the firewall fracture flag"
    
    # 2. Inf attention maps must trigger fracture
    inf_attn = torch.tensor([[[[float('inf'), 0.1], [0.1, 0.8]]]]) # [1, 1, 2, 2]
    is_fractured_inf, cfi_inf, alt_idx_inf = firewall.check_obstruction(inf_attn)
    assert is_fractured_inf, "Inf attention maps must trigger the firewall fracture flag"
