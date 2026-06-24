import mlx.core as mx
import numpy as np
import pytest

from qan_transformers.mlx.attention import QuasicrystallineAttention
from qan_transformers.mlx.e8_swap import AdelicMemorySwapGridDB
from scratch.run_mlx_speculative_chat import QANModelMLX, rollback_mlx_custom_caches

def test_mlx_shared_cache_db_consistency():
    """
    Verifies that target and draft models write to their respective CPU buffers
    while sharing a single unified E8 coordinate index in MLX.
    """
    # Initialize shared database with target dim 16 and draft dim 8
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Keys/values representing target and draft tokens
    # Sequence length S = 4, target head dim 16, draft head dim 8
    target_k = mx.random.normal((4, 16))
    target_v = mx.random.normal((4, 16))
    draft_k = mx.random.normal((4, 8))
    draft_v = mx.random.normal((4, 8))
    
    # Swap out target
    db.swap_out_target(target_k, target_v)
    assert db.grid_coords.shape == (4, 8)
    assert db.cpu_k_target.shape == (4, 16)
    assert db.cpu_v_target.shape == (4, 16)
    
    # Swap out draft
    # Since draft has the same length, coordinates index shape should not exceed 4
    db.swap_out_draft(draft_k, draft_v)
    assert db.grid_coords.shape == (4, 8)
    assert db.cpu_k_draft.shape == (4, 8)
    assert db.cpu_v_draft.shape == (4, 8)
    
    # Swap in target check
    queries_target = target_k[0:1] # [1, 16]
    ret_k_target, ret_v_target = db.swap_in_target(queries_target)
    assert ret_k_target.shape[0] >= 1
    assert ret_k_target.shape[1] == 16
    
    # Swap in draft check
    queries_draft = draft_k[0:1] # [1, 8]
    ret_k_draft, ret_v_draft = db.swap_in_draft(queries_draft)
    assert ret_k_draft.shape[0] >= 1
    assert ret_k_draft.shape[1] == 8

def test_mlx_speculative_rollback_alignment():
    """
    Verifies that rolling back the shared cache database to an accepted length
    preserves correct indexing and trims buffers properly in MLX.
    """
    db = AdelicMemorySwapGridDB(d_model=16, d_model_draft=8)
    
    # Start with prefill of 4 tokens.
    # We append 4 target head vectors and 4 draft head vectors (1 head per token)
    target_k_prefill = mx.random.normal((4, 16))
    target_v_prefill = mx.random.normal((4, 16))
    draft_k_prefill = mx.random.normal((4, 8))
    draft_v_prefill = mx.random.normal((4, 8))
    
    db.swap_out_target(target_k_prefill, target_v_prefill)
    db.swap_out_draft(draft_k_prefill, draft_v_prefill)
    
    # Propose 4 new candidate tokens (T = 4)
    # Total sequence length grows to 8 tokens
    target_k_candidates = mx.random.normal((4, 16))
    target_v_candidates = mx.random.normal((4, 16))
    draft_k_candidates = mx.random.normal((4, 8))
    draft_v_candidates = mx.random.normal((4, 8))
    
    db.swap_out_target(target_k_candidates, target_v_candidates)
    db.swap_out_draft(draft_k_candidates, draft_v_candidates)
    
    assert db.cpu_k_target.shape[0] == 8
    assert db.cpu_k_draft.shape[0] == 8
    assert db.grid_coords.shape[0] == 8
    
    # We accept 2 out of 4 candidate tokens, making the accepted length 6 tokens.
    # Rollback to 6 tokens. Ratio is 1 head vector per token.
    db.rollback(num_tokens_to_keep=6, current_len=8)
    
    assert db.cpu_k_target.shape[0] == 6
    assert db.cpu_k_draft.shape[0] == 6
    assert db.grid_coords.shape[0] == 6

def test_mlx_speculative_attention_forward():
    """
    Verifies that QAN-grafted target and draft models can run prefill,
    draft proposal, and parallel verification forward passes using a shared grid in MLX.
    """
    # Load lightweight configurations
    target = QANModelMLX(vocab_size=128, embed_dim=16, num_heads=2, num_layers=1, is_draft=False)
    draft = QANModelMLX(vocab_size=128, embed_dim=16, num_heads=2, num_layers=1, is_draft=True)
    
    target.train(False)
    draft.train(False)
    
    # Initialize shared grid
    target_head_dim = target.embed_dim // target.num_heads
    draft_head_dim = draft.embed_dim // draft.num_heads
    shared_db = AdelicMemorySwapGridDB(d_model=target_head_dim, d_model_draft=draft_head_dim)
    
    # Bind grid
    for m in target.modules():
        if isinstance(m, QuasicrystallineAttention):
            m.swap_db = shared_db
            m.is_draft = False
            
    for m in draft.modules():
        if isinstance(m, QuasicrystallineAttention):
            m.swap_db = shared_db
            m.is_draft = True
            
    # Prefill step
    input_ids = mx.array([[10, 20, 30, 40]], dtype=mx.int32)
    from scratch.run_mlx_speculative_chat import MLXLayerCache
    target_caches = [MLXLayerCache() for _ in range(len(target.layers))]
    draft_caches = [MLXLayerCache() for _ in range(len(draft.layers))]
    
    shared_db.clear()
    logits_t = target(input_ids, caches=target_caches)
    logits_d = draft(input_ids, caches=draft_caches)
    
    mx.eval(logits_t, logits_d)
    
    # Verify prefill is non-empty in shared db
    assert shared_db.grid_coords is not None
    assert shared_db.cpu_k_target is not None
    assert shared_db.cpu_k_draft is not None
    
    # Propose draft tokens (T = 2)
    next_token = mx.array([[40]], dtype=mx.int32)
    logits_d2 = draft(next_token, caches=draft_caches)
    mx.eval(logits_d2)
    
    # Parallel verification
    candidate_ids = mx.array([[50, 60]], dtype=mx.int32)
    logits_t2 = target(candidate_ids, caches=target_caches)
    mx.eval(logits_t2)
    
    # Ensure verification outputs do not contain NaNs
    assert not np.isnan(np.array(logits_t2)).any()
    assert logits_t2.shape == (1, 2, target.vocab_size)
