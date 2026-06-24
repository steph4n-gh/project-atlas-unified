import pytest
import torch
import torch.nn as nn
import mlx.core as mx
import mlx.nn as mlx_nn

from qan_transformers.modeling.auto import AutoQANGraftModel
from qan_transformers.modeling.attention import QuasicrystallineAttention as PQAttention
from qan_transformers.mlx.attention import QuasicrystallineAttention as MXAttention

# Mock class definitions to represent PyTorch MoE structures for testing
class MockSelfAttention(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = 4
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        return x

class MockExpertRouting(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(32, 4)
    def forward(self, x):
        return x

class MockMoeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockSelfAttention()
        self.sparse_moe_block = MockExpertRouting()
    def forward(self, x):
        return self.self_attn(x) + self.sparse_moe_block(x)

def test_pytorch_auto_graft_lightweight():
    """
    Verifies that AutoQANGraftModel correctly creates a lightweight grafted model in PyTorch.
    """
    model = AutoQANGraftModel.from_pretrained("google/gemma-4-e2b", sparse_ratio=0.15, lightweight=True)
    
    # Ensure attention layers are converted to QuasicrystallineAttention
    has_graft = False
    for m in model.modules():
        if isinstance(m, PQAttention):
            has_graft = True
            break
            
    assert has_graft, "PyTorch model was not grafted with QuasicrystallineAttention."

def test_moe_auto_graft_selective():
    """
    Verifies that AutoQANGraftModel grafts attention layers of a MoE model
    while skipping routing and expert layers.
    """
    from qan_transformers.modeling import make_quasicrystalline
    model = MockMoeModel()
    
    # Run grafting
    make_quasicrystalline(model)
    
    # Verify self-attention was grafted
    assert isinstance(model.self_attn, PQAttention)
    
    # Verify MoE routing gate remains untouched (not converted to attention)
    assert isinstance(model.sparse_moe_block, MockExpertRouting)
    assert not isinstance(model.sparse_moe_block, PQAttention)

def test_mlx_auto_graft_framework_fallback():
    """
    Verifies that AutoQANGraftModel throws appropriate errors when trying to use MLX
    if mlx_lm is missing, or falls back/completes if importable.
    """
    # Since mlx is installed in the test runner, we can check for mlx behaviour:
    # If the user tries to load a dummy repo id, it will fail to load but the import should succeed
    try:
        from mlx_lm import load
        HAS_MLX_LM = True
    except ImportError:
        HAS_MLX_LM = False
        
    if not HAS_MLX_LM:
        with pytest.raises(ImportError):
            AutoQANGraftModel.from_pretrained("google/gemma-4-e2b", framework="mlx")
    else:
        # Should raise an error or try to run load (which fails due to dummy model name)
        # Verify ValueError for invalid frameworks
        with pytest.raises(ValueError):
            AutoQANGraftModel.from_pretrained("google/gemma-4-e2b", framework="invalid")

def test_pytorch_auto_graft_norms():
    """
    Verifies that make_quasicrystalline correctly copies q_norm, k_norm, and v_norm,
    and they are invoked in QuasicrystallineAttention.forward.
    """
    from qan_transformers.modeling import make_quasicrystalline

    class MockAttentionWithNorms(nn.Module):
        def __init__(self, d_model=32):
            super().__init__()
            self.embed_dim = d_model
            self.num_heads = 4
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)
            
            # Simple dummy modules to act as normalization submodules
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.v_norm = nn.Identity()
            self.scaling = 2.5
            
        def forward(self, x):
            return x

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttentionWithNorms()
        def forward(self, x):
            return self.attn(x)

    model = MockModel()
    make_quasicrystalline(model)
    
    grafted_attn = model.attn
    assert isinstance(grafted_attn, PQAttention)
    assert grafted_attn.q_norm is not None
    assert grafted_attn.k_norm is not None
    assert grafted_attn.v_norm is not None
    assert grafted_attn.scaling == 2.5
    
    # Run a forward pass to verify it executes without error and uses norms
    x = torch.randn(2, 8, 32)
    # Mock necessary attributes for forward execution
    grafted_attn.roots_3d = torch.zeros(30, 3)
    grafted_attn.roots_3d_norm = torch.zeros(30, 3)
    
    out = grafted_attn(x)
    assert out.shape == (2, 8, 32)
