import torch
import torch.nn.functional as F
import pytest
from qan_transformers.modeling.attention.base import QuasicrystallineAttention
from qan_transformers.lora.pipeline import inject_lora
from qan_transformers.modeling.persistent_homology_loss import PersistentHomologyLoss
from qan_transformers.modeling.conformal import ConformalPositionalEncoding
from qan_transformers.firewall.cohomology import CohomologyFirewall

def test_marsshot_e2e_integration():
    embed_dim = 16
    num_heads = 2
    
    # 1. Instantiate the attention block with Marsshot configurations
    attn = QuasicrystallineAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparse_ratio=0.5,
        attention_mode='octonionic',
        temperature_mode='tropical',
        use_derived_composition=True,
        use_braiding=True
    )
    
    # 2. Inject Galois adapters
    inject_lora(attn, r=4, adapter_type="galois", num_tasks=3)
    
    # 3. Instantiate other components like Conformal Positional Encoding and Topology Loss
    cpe = ConformalPositionalEncoding(max_positions=64)
    ph_loss_fn = PersistentHomologyLoss()
    
    # Configure mock config withprev attn weights placeholder for derived composition
    class MockConfig:
        pass
    attn.config = MockConfig()
    attn.layer_idx = 1
    
    # Setup inputs
    x = torch.randn(2, 8, embed_dim)
    
    # Run conformal positional encoding
    x_encoded = cpe(x)
    assert x_encoded.shape == x.shape
    
    # Run forward pass of QuasicrystallineAttention (which uses octonionic, tropical, symplectic, braiding)
    # Set M to non-128 so adapter is active
    with torch.no_grad():
        attn.q_proj.galois_adapter.M.fill_(130)
        attn.k_proj.galois_adapter.M.fill_(130)
        attn.v_proj.galois_adapter.M.fill_(130)
        attn.out_proj.galois_adapter.M.fill_(130)
        
    attn.config.shared_prev_attn_weights = torch.softmax(torch.randn(2, num_heads, 8, 4), dim=-1)
    
    out = attn(x_encoded)
    assert out.shape == (2, 8, embed_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    # 4. Compute persistent homology loss on the attention weights
    # Retrieve attention weights from forward pass
    attn_weights = attn.config.shared_prev_attn_weights
    
    # Pad key dimension K to query dimension S to form a square [B, H, S, S] matrix
    B, H, S, K_total = attn_weights.shape
    attn_weights_full = F.pad(attn_weights, (0, S - K_total))
    
    # Average over heads to get a [B, S, S] skeleton
    skeleton = attn_weights_full.mean(dim=1)
    
    ph_loss = ph_loss_fn(skeleton)
    assert ph_loss.item() >= 0.0
    
    # 5. Backward pass through everything (loss + autograd graph check)
    total_loss = out.sum() + 0.1 * ph_loss
    total_loss.backward()
    
    # Verify that gradients flow to all components
    assert attn.q_proj.galois_adapter.A_proj.weight.grad is not None
    assert (attn.q_proj.galois_adapter.A_proj.weight.grad != 0.0).any()
    
    assert attn.braid_attention.r_matrices[0].raw_t.grad is not None
    assert cpe.delta.grad is not None
    assert attn.e8_proj.weight.grad is not None
    assert attn.e8_proj_momentum.weight.grad is not None
    
    print("E2E pass completed successfully! Gradients verified.")
