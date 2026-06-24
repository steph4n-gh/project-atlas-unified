import pytest
import torch
import numpy as np
from qan_transformers.modeling import graft_model, QANModel
from qan_transformers.firewall.cohomology import CohomologyFirewall
from qan_transformers.optim.adelic import AdelicLangevinOptimizer
from qan_transformers.modeling.attention import QuasicrystallineAttention

# ==========================================
# MOCK FIREWALLS FOR RELIABLE ADVERSARIAL TESTING
# ==========================================

class AlwaysFracturedFirewall(CohomologyFirewall):
    """
    Mock firewall that always reports a fracture and returns some alternative token indices.
    Used to reliably trigger and test the token generation rollback loop.
    """
    def __init__(self, threshold=0.0, rollback_limit=2, alt_idx=None):
        super().__init__(threshold=threshold, rollback_limit=rollback_limit)
        self.alt_idx = alt_idx if alt_idx is not None else [1, 2, 3]

    def check_obstruction(self, attn_matrix):
        return True, 5.0, self.alt_idx

# ==========================================
# 1. ČECH COHOMOLOGY FIREWALL ADVERSARIAL TESTS
# ==========================================

def test_firewall_batched_blindness():
    """
    Test if the firewall correctly detects fractures in batches other than the first.
    """
    firewall = CohomologyFirewall(threshold=0.1)

    # Normal attention matrix (returns low or zero CFI)
    normal_attn = torch.eye(8).unsqueeze(0).unsqueeze(0) # [1, 1, 8, 8]
    
    # Highly fractured attention matrix (returns high CFI)
    fractured_attn = torch.randn(8, 8).abs()
    fractured_attn = fractured_attn / fractured_attn.sum(dim=-1, keepdim=True)
    fractured_attn = fractured_attn.unsqueeze(0).unsqueeze(0)

    # Verify if fractured_attn alone triggers the firewall
    is_frac_single, cfi_single, _ = firewall.check_obstruction(fractured_attn)
    assert is_frac_single, f"Expected the mock fractured attention to trigger firewall, got cfi={cfi_single}"

    # Construct a batch: batch index 0 is normal, batch index 1 is fractured
    batched_attn = torch.cat([normal_attn, fractured_attn], dim=0) # [2, 1, 8, 8]

    # Evaluate batch
    is_fractured, cfi, alt_idx = firewall.check_obstruction(batched_attn)

    # We expect a list response with index 1 being fractured
    assert isinstance(is_fractured, list), "Expected list response for batched firewall check"
    assert not is_fractured[0], "First batch item should not be fractured"
    assert is_fractured[1], "Second batch item should be fractured"


def test_firewall_nan_inf_handling():
    """
    Test how the firewall handles NaN or Inf values in the attention matrix.
    """
    firewall = CohomologyFirewall(threshold=1.0)
    
    attn_nan = torch.tensor([[float('nan'), 0.5], [0.5, 0.5]])
    is_fractured, cfi, alt_idx = firewall.check_obstruction(attn_nan)
    assert is_fractured, "NaN values should trigger the firewall"
    
    attn_inf = torch.tensor([[float('inf'), 0.5], [0.5, 0.5]])
    is_fractured_inf, cfi_inf, alt_idx_inf = firewall.check_obstruction(attn_inf)
    assert is_fractured_inf, "Inf values should trigger the firewall"


# ==========================================
# 2. TOKEN GENERATION ROLLBACK CRASH TESTS
# ==========================================

def test_rollback_batch_size_mismatch_crash():
    """
    Test if QANModel.generate runs successfully without crashing when batch size > 1 and firewall triggers a rollback.
    """
    model = graft_model("google/gemma-4-e2b", lightweight=True)
    model.eval()

    # Create a firewall that always triggers and returns non-empty alt_idx
    firewall = AlwaysFracturedFirewall(alt_idx=[1, 2, 3])
    
    # Inject our mock firewall into all QuasicrystallineAttention layers
    for layer in model.layers:
        if isinstance(layer.attn, QuasicrystallineAttention):
            layer.attn.firewall = firewall
            layer.attn.min_keep = 10
            layer.attn.sparse_ratio = 1.0

    # Run generate with batch size 2. This should work successfully.
    input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]], dtype=torch.long) # B = 2
    
    generated = model.generate(input_ids, max_new_tokens=2, rollback_limit=2)
    assert generated.shape == (2, 5), f"Expected shape (2, 5), got {generated.shape}"
    print(f"\n[Rollback Batch Test] Succeeded without crash! Generated shape: {generated.shape}")



def test_rollback_no_alternatives_single_batch():
    """
    Test rollback behavior when B=1 but no alternative indices are available.
    """
    model = graft_model("google/gemma-4-e2b", lightweight=True)
    model.eval()

    # Low threshold firewall to force rollback but returning empty alt_idx
    firewall = AlwaysFracturedFirewall(alt_idx=[])
    for layer in model.layers:
        if isinstance(layer.attn, QuasicrystallineAttention):
            layer.attn.firewall = firewall
            layer.attn.min_keep = 10
            layer.attn.sparse_ratio = 1.0

    input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long) # B = 1
    
    generated = model.generate(input_ids, max_new_tokens=2, rollback_limit=2)
    assert generated.shape == (1, 5), f"Expected shape (1, 5), got {generated.shape}"
    print(f"[Rollback Single Batch Test] Succeeded! Generated shape: {generated.shape}")


# ==========================================
# 3. OPTIMIZER ADVERSARIAL TESTS
# ==========================================

def test_optimizer_nan_inf_gradients():
    """
    Test if AdelicLangevinOptimizer handles NaN/Inf gradients gracefully without blowing up parameters.
    """
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    optimizer = AdelicLangevinOptimizer([p], lr=0.1)
    
    # Assign NaN and Inf to gradients
    p.grad = torch.tensor([float('nan'), float('inf'), -float('inf')])
    
    optimizer.step()
    
    # Verify that parameter data did not become NaN or Inf
    assert not torch.isnan(p.data).any(), "Parameters became NaN after step with NaN/Inf gradients"
    assert not torch.isinf(p.data).any(), "Parameters became Inf after step with NaN/Inf gradients"


def test_optimizer_extreme_parameters():
    """
    Test optimizer behavior with invalid or extreme tree hyperparameters.
    """
    p = torch.nn.Parameter(torch.randn(3))
    p.grad = torch.randn(3)
    
    # 1. Learning rate negative (should fail on init)
    with pytest.raises(ValueError):
        AdelicLangevinOptimizer([p], lr=-0.01)

    # 2. Extreme tree depth and base values
    opt = AdelicLangevinOptimizer([p], lr=0.01, tree_depth=0, p_base=2)
    opt.step()
    
    # 3. What if alpha is extremely negative?
    opt_alpha = AdelicLangevinOptimizer([p], lr=0.01, alpha=-5.0)
    opt_alpha.step()


# ==========================================
# 4. ATTENTION MODEL SEQUENCE LIMITS
# ==========================================

def test_model_sequence_overflow():
    """
    Test if QANModel handles sequence lengths exceeding max_seq_len using position wrapping.
    """
    model = QANModel(vocab_size=100, embed_dim=16, num_heads=2, num_layers=1, max_seq_len=8)
    
    # Sequence length 12 exceeds max_seq_len of 8
    input_ids = torch.randint(0, 100, (1, 12))
    
    logits = model(input_ids)
    assert logits.shape == (1, 12, 100)
