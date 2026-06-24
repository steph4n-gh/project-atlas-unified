import pytest
import torch
import torch.nn as nn
from qan_transformers.modeling import graft_model
from qan_transformers.modeling.gemma import speculative_verify_superposition_gemma

def rollback_superposition_caches(kv_caches, winning_channel, accepted_len, T):
    for cache in kv_caches:
        if "K_superposition" in cache and cache["K_superposition"] is not None:
            K_super = cache["K_superposition"]
            V_super = cache["V_superposition"]
            S_seq = K_super.shape[3]
            offset = S_seq - T
            new_len = offset + accepted_len
            
            # Collapse superposition cache to standard reference cache
            cache["K"] = K_super[:, winning_channel, :, :new_len, :]
            cache["V"] = V_super[:, winning_channel, :, :new_len, :]
            cache["seq_len"] = new_len
            
            # Clear superposition cache
            cache["K_superposition"] = None
            cache["V_superposition"] = None

def test_unitary_tree_decoding_integration():
    # 1. Instantiate lightweight target and draft models
    target = graft_model("google/gemma-4-e2b", lightweight=True)
    draft = graft_model("google/gemma-4-e2b", lightweight=True)
    
    # Verify models have QuasicrystallineAttention grafted
    has_qan = False
    for m in target.modules():
        if m.__class__.__name__ == "QuasicrystallineAttention":
            has_qan = True
            break
    assert has_qan, "Target model must have QuasicrystallineAttention grafted."
    
    # 2. Set up a superposition input representing 4 candidate paths of depth 3
    # Shape: [B, C, S] = [1, 4, 3]
    B, C, S = 1, 4, 3
    input_ids = torch.randint(10, 100, (B, C, S))
    
    # 3. Execute target forward pass with superposition input and cache tracking
    kv_caches = [{} for _ in range(target.num_layers)]
    target_logits, target_caches = target(input_ids, kv_caches=kv_caches)
    
    assert target_logits.shape == (B, C, S, target.vocab_size)
    
    # Check that superposition keys/values are correctly stored in the cache
    for cache in target_caches:
        assert "K_superposition" in cache
        assert cache["K_superposition"] is not None
        assert cache["K_superposition"].shape[0] == B
        assert cache["K_superposition"].shape[1] == C
        assert cache["K_superposition"].shape[3] == S
        
    # 4. Set up candidate tokens and amplitudes for wave-packet collapse verification
    # Let's make channel 2 the winning channel by matching it with target's argmax logits
    candidate_tokens = input_ids.clone()
    for t in range(S):
        candidate_tokens[0, 2, t] = target_logits[0, 2, t].argmax(dim=-1)
        
    # Let's mismatch channel 1 at token 1
    target_pred_1 = target_logits[0, 1, 1].argmax(dim=-1)
    candidate_tokens[0, 1, 1] = (target_pred_1 + 1) % target.vocab_size
    
    amplitudes = torch.tensor([0.1, 0.2, 0.5, 0.2])  # channel 2 has highest amplitude
    
    # Run wave-packet collapse verification
    winning_channel, accepted_len, correction_token = speculative_verify_superposition_gemma(
        target_logits=target_logits,
        candidate_tokens=candidate_tokens,
        amplitudes=amplitudes
    )
    
    # Channel 2 should win because it has the longest match (3 tokens)
    assert winning_channel == 2
    assert accepted_len == 3
    assert correction_token is None  # All tokens accepted
    
    # 5. Run cache rollback/collapse
    rollback_superposition_caches(target_caches, winning_channel, accepted_len, S)
    
    # Check that cache has been rolled back and standard cache is active
    for cache in target_caches:
        assert cache["K_superposition"] is None
        assert cache["V_superposition"] is None
        assert cache["K"] is not None
        assert cache["K"].shape[2] == S
        assert cache["seq_len"] == S
