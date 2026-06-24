import pytest
import mlx.core as mx
import math
from qan_transformers.mlx.moonshots import (
    FusedJITSpeculativeVerifier,
    JITDraftARLoopGenerator,
    ZeroCopySlidingKVCacheManager,
    JITCompiledSpeculativeSampler,
    JITCompiledRoPECache,
    FusedDraftTargetProjection,
    UnifiedSpeculativeKVCache,
    FusedMHAExecution,
    FusedSwiGLUFFN,
    FusedSpeculativeTransformerBlock,
    JITCompiledPrefetchLookahead,
    DualStreamCommandQueuing,
    ContiguousFlashKVLinearization,
    DynamicDraftAdaptiveLength,
    UnifiedSpeculativePipeline,
    ZeroSyncSpeculativeVerifier,
    JITQuantizedWeightCache,
    DynamicDraftLengthAdjuster,
    FusedKVAppendVerify,
    BlockSparseLayerSkipping,
    SIMDCoalescedSoftmax,
    FusedAttentionProjections,
    PreAllocatedUnifiedKVCache,
    LookaheadEmbeddingPrefetch,
    BlockWiseWeightLoading,
    SpeculativeTreeVerifier,
    FusedSwiGLUFFNReg,
    MultiStreamOverlap,
    DynamicCachePruning,
    EndToEndJITSpeculativePipeline
)

def test_fused_jit_speculative_verifier():
    verifier = FusedJITSpeculativeVerifier()
    
    target_logits = mx.array([
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0]
        ],
        [
            [0.0, 1.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.5]
        ]
    ])
    
    candidate_tokens = mx.array([
        [0, 99, 4],
        [1, 3, 4]
    ])
    
    num_accepted, correction = verifier.verify(target_logits, candidate_tokens)
    
    assert num_accepted[0].item() == 1
    assert correction[0].item() == 2
    
    assert num_accepted[1].item() == 3
    assert correction[1].item() == 4


def test_jit_draft_ar_loop_generator():
    class DummyModel:
        def __call__(self, x):
            last_tok = x[:, -1:]
            logits = mx.zeros((x.shape[0], 10))
            indices = mx.minimum(last_tok + 1, 9)
            arange = mx.arange(10)[None, :]
            logits = mx.where(arange == indices, mx.array(10.0), mx.array(0.0))
            return logits

    model = DummyModel()
    generator = JITDraftARLoopGenerator(model)
    
    x = mx.array([[0]])
    tokens = generator.generate(x, num_tokens=3, temp=0.0)
    
    assert tokens.shape == (1, 3)
    assert tokens[0, 0].item() == 1
    assert tokens[0, 1].item() == 2
    assert tokens[0, 2].item() == 3


def test_zero_copy_sliding_kv_cache_manager():
    manager = ZeroCopySlidingKVCacheManager(batch_size=1, num_heads=2, max_length=10, head_dim=4)
    
    new_k = mx.ones((1, 2, 3, 4)) * 1.5
    new_v = mx.ones((1, 2, 3, 4)) * 2.5
    
    k_view, v_view = manager.update(new_k, new_v, offset=0)
    
    assert k_view.shape == (1, 2, 3, 4)
    assert v_view.shape == (1, 2, 3, 4)
    assert k_view[0, 0, 0, 0].item() == 1.5
    assert v_view[0, 0, 0, 0].item() == 2.5
    
    new_k2 = mx.ones((1, 2, 2, 4)) * 7.0
    new_v2 = mx.ones((1, 2, 2, 4)) * 8.0
    k_view2, v_view2 = manager.update(new_k2, new_v2, offset=3)
    
    assert k_view2.shape == (1, 2, 5, 4)
    assert k_view2[0, 0, 2, 0].item() == 1.5
    assert k_view2[0, 0, 3, 0].item() == 7.0
    assert v_view2[0, 0, 3, 0].item() == 8.0


def test_jit_compiled_speculative_sampler():
    sampler = JITCompiledSpeculativeSampler()
    
    logits = mx.zeros((1, 100))
    logits[0, 42] = 100.0
    
    sample = sampler.sample(logits, temp=1.0, k=5)
    
    assert sample.shape == (1, 1)
    assert sample[0, 0].item() == 42


def test_jit_compiled_rope_cache():
    rope = JITCompiledRoPECache(max_length=10, head_dim=4)
    x = mx.ones((1, 1, 2, 4))
    
    out = rope.apply(x, offset=2)
    assert out.shape == (1, 1, 2, 4)
    assert not mx.any(mx.isnan(out)).item()


def test_fused_draft_target_projection():
    w_draft = mx.ones((8, 4)) * 2.0
    w_target = mx.ones((8, 6)) * 3.0
    
    proj = FusedDraftTargetProjection(w_draft, w_target)
    
    x = mx.ones((1, 5, 8))
    
    d_out, t_out = proj.project(x)
    
    assert d_out.shape == (1, 5, 4)
    assert t_out.shape == (1, 5, 6)
    assert abs(d_out[0, 0, 0].item() - 16.0) < 1e-5
    assert abs(t_out[0, 0, 0].item() - 24.0) < 1e-5


def test_unified_speculative_kv_cache():
    cache = UnifiedSpeculativeKVCache(batch_size=1, h_draft=1, h_target=2, max_length=10, head_dim=4)
    
    kd = mx.ones((1, 1, 2, 4)) * 1.1
    vd = mx.ones((1, 1, 2, 4)) * 2.2
    
    kt = mx.ones((1, 2, 3, 4)) * 3.3
    vt = mx.ones((1, 2, 3, 4)) * 4.4
    
    (kd_view, vd_view), (kt_view, vt_view) = cache.update(kd, vd, kt, vt, offset=0)
    
    assert kd_view.shape == (1, 1, 2, 4)
    assert kt_view.shape == (1, 2, 3, 4)
    assert abs(kd_view[0, 0, 0, 0].item() - 1.1) < 1e-5
    assert abs(kt_view[0, 0, 0, 0].item() - 3.3) < 1e-5


def test_fused_mha_execution():
    w_out = mx.ones((8, 8)) * 0.5
    mha = FusedMHAExecution(w_out)
    
    # B = 1, H = 2, S = 3, D = 4
    # H * D = 8.
    q = mx.ones((1, 2, 3, 4))
    k = mx.ones((1, 2, 3, 4))
    v = mx.ones((1, 2, 3, 4))
    
    out = mha.execute(q, k, v)
    
    assert out.shape == (1, 3, 8)
    # Output of attention matrix product (all 1s scaled by softmax) should be 1s.
    # Output projection with all 0.5s over input size 8 should be: 8 * 0.5 = 4.0
    assert abs(out[0, 0, 0].item() - 4.0) < 1e-5


def test_fused_swiglu_ffn():
    w_gate = mx.ones((8, 16)) * 0.5
    w_up = mx.ones((8, 16)) * 0.25
    w_down = mx.ones((16, 8)) * 0.1
    
    ffn = FusedSwiGLUFFN(w_gate, w_up, w_down)
    x = mx.ones((1, 4, 8))
    
    out = ffn.forward(x)
    
    assert out.shape == (1, 4, 8)
    expected_val = 16.0 * (4.0 / (1.0 + math.exp(-4.0))) * 2.0 * 0.1
    assert abs(out[0, 0, 0].item() - expected_val) < 1e-4


def test_fused_speculative_transformer_block():
    B, S, D = 1, 2, 8
    H, D_head = 2, 4
    
    # Weights
    w_q = mx.ones((D, H * D_head)) * 0.1
    w_k = mx.ones((D, H * D_head)) * 0.2
    w_v = mx.ones((D, H * D_head)) * 0.3
    w_out = mx.ones((H * D_head, D)) * 0.4
    
    w_gate = mx.ones((D, 16)) * 0.5
    w_up = mx.ones((D, 16)) * 0.25
    w_down = mx.ones((16, D)) * 0.1
    
    rms_attn_w = mx.zeros((D,))
    rms_ffn_w = mx.zeros((D,))
    
    # RoPE cache
    rope_cache = JITCompiledRoPECache(max_length=10, head_dim=D_head)
    
    block = FusedSpeculativeTransformerBlock(
        w_q, w_k, w_v, w_out, w_gate, w_up, w_down,
        rms_attn_w, rms_ffn_w, rope_cache.cos_cache, rope_cache.sin_cache
    )
    
    x = mx.ones((B, S, D))
    k_cache = mx.zeros((B, H, 10, D_head))
    v_cache = mx.zeros((B, H, 10, D_head))
    
    out, k_cache, v_cache = block.forward(x, k_cache, v_cache, offset=0)
    
    assert out.shape == (B, S, D)
    assert not mx.any(mx.isnan(out)).item()
    assert k_cache[0, 0, 0, 0].item() != 0.0
    assert v_cache[0, 0, 0, 0].item() != 0.0


def test_jit_compiled_prefetch_lookahead():
    embeddings = mx.array([
        [1.0, 1.1],
        [2.0, 2.2],
        [3.0, 3.3],
        [4.0, 4.4]
    ])
    
    prefetcher = JITCompiledPrefetchLookahead(embeddings)
    
    candidate_ids = mx.array([[1, 2, 3]])
    
    curr, lookahead = prefetcher.prefetch(candidate_ids, lookahead_steps=1)
    
    assert curr.shape == (1, 3, 2)
    assert lookahead.shape == (1, 3, 2)
    
    # curr should be embedding of [1, 2, 3]
    assert abs(curr[0, 0, 0].item() - 2.0) < 1e-5
    assert abs(curr[0, 1, 0].item() - 3.0) < 1e-5
    assert abs(curr[0, 2, 0].item() - 4.0) < 1e-5
    
    # lookahead should be shifted by 1: [2, 3, 0]
    assert abs(lookahead[0, 0, 0].item() - 3.0) < 1e-5
    assert abs(lookahead[0, 1, 0].item() - 4.0) < 1e-5
    assert abs(lookahead[0, 2, 0].item() - 1.0) < 1e-5  # embedding of 0 is [1.0, 1.1]


def test_dual_stream_command_queuing():
    queuing = DualStreamCommandQueuing()
    
    def draft_work(x):
        return x * 2.0
        
    def target_work(y):
        return y + 10.0
        
    x = mx.array([1.0, 2.0])
    y = mx.array([5.0, 6.0])
    
    d_out, t_out = queuing.execute_dual(
        draft_work, target_work,
        draft_args=(x,), target_args=(y,)
    )
    
    mx.eval(d_out, t_out)
    
    assert abs(d_out[0].item() - 2.0) < 1e-5
    assert abs(t_out[0].item() - 15.0) < 1e-5


def test_contiguous_flash_kv_linearization():
    manager = ContiguousFlashKVLinearization(batch_size=1, num_heads=2, max_length=10, head_dim=4)
    
    # Input in standard layout: (B, H, S, D)
    new_k = mx.ones((1, 2, 3, 4)) * 1.5
    new_v = mx.ones((1, 2, 3, 4)) * 2.5
    
    k_view, v_view = manager.update_and_get(new_k, new_v, offset=0)
    
    # Expected layout in memory: (B, S, H, D) -> (1, 3, 2, 4)
    assert k_view.shape == (1, 3, 2, 4)
    assert v_view.shape == (1, 3, 2, 4)
    assert k_view[0, 0, 0, 0].item() == 1.5
    assert v_view[0, 0, 0, 0].item() == 2.5
    
    new_k2 = mx.ones((1, 2, 2, 4)) * 7.0
    new_v2 = mx.ones((1, 2, 2, 4)) * 8.0
    k_view2, v_view2 = manager.update_and_get(new_k2, new_v2, offset=3)
    
    assert k_view2.shape == (1, 5, 2, 4)
    assert k_view2[0, 2, 0, 0].item() == 1.5
    assert k_view2[0, 3, 0, 0].item() == 7.0
    assert v_view2[0, 3, 0, 0].item() == 8.0


def test_dynamic_draft_adaptive_length():
    class DummyEntropyModel:
        def __call__(self, x):
            # x shape (B, S)
            last_tok = x[:, -1:]
            B = x.shape[0]
            peaked = mx.zeros((B, 10))
            peaked[:, 1] = 100.0  # extremely peaked, zero entropy
            uniform = mx.zeros((B, 10))  # uniform, high entropy (log(10) ~ 2.3)
            
            cond = (last_tok == 0)
            cond = mx.broadcast_to(cond, (B, 10))
            return mx.where(cond, peaked, uniform)

    model = DummyEntropyModel()
    generator = DynamicDraftAdaptiveLength(model, max_draft_len=3, entropy_threshold=1.0)
    
    x = mx.array([[0]])
    tokens, active_counts = generator.generate_adaptive(x, temp=0.0)
    
    assert tokens.shape == (1, 3)
    assert active_counts.shape == (1,)
    # Step 0 (input 0 -> peaked output 1) is active: count becomes 1
    # Step 1 (input 1 -> uniform output) is high-entropy, stops: count remains 1
    assert active_counts[0].item() == 1


def test_unified_speculative_pipeline():
    class DummyDraftModel:
        def __call__(self, x):
            B = x.shape[0]
            peaked = mx.zeros((B, 10))
            peaked[:, 1] = 100.0  # peaked at 1
            return peaked
            
    class DummyTargetModel:
        def __call__(self, x):
            # Returns logits peaked at 1 for all tokens
            B, S = x.shape
            peaked = mx.zeros((B, S, 10))
            peaked[:, :, 1] = 100.0
            return peaked

    pipeline = UnifiedSpeculativePipeline(
        DummyDraftModel(), DummyTargetModel(),
        draft_max_len=2, entropy_threshold=2.0
    )
    
    x = mx.array([[0]])
    candidates, num_accepted, correction = pipeline.step(x, offset=0)
    
    assert candidates.shape == (1, 2)
    assert num_accepted.shape == (1,)
    assert correction.shape == (1,)


def test_zero_sync_speculative_verifier():
    verifier = ZeroSyncSpeculativeVerifier()
    
    target_logits = mx.array([
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ]
    ])  # peaked at 0, 1, 2
    
    candidate_tokens = mx.array([[0, 99, 2]])  # second token mismatch
    offsets = mx.array([10])
    
    num_accepted, correction, new_offsets = verifier.verify_async(target_logits, candidate_tokens, offsets)
    
    # We should be able to evaluate these at the end
    mx.eval(num_accepted, correction, new_offsets)
    
    assert num_accepted[0].item() == 1
    assert correction[0].item() == 1
    assert new_offsets[0].item() == 11


def test_jit_quantized_weight_cache():
    qw = mx.array([[1, 2], [3, 4]])
    s = mx.array([[0.5, 0.5], [1.0, 1.0]])
    b = mx.array([[0.1, 0.1], [0.2, 0.2]])
    
    cache = JITQuantizedWeightCache(qw, s, b)
    
    w1 = cache.get_dequantized()
    assert cache.cached_weight is not None
    
    # second call should return identical cached array object
    w2 = cache.get_dequantized()
    assert w1 is w2
    
    assert abs(w1[0, 0].item() - 0.6) < 1e-5
    assert abs(w1[1, 1].item() - 4.2) < 1e-5
    
    cache.clear()
    assert cache.cached_weight is None


def test_dynamic_draft_length_adjuster():
    adjuster = DynamicDraftLengthAdjuster(initial_k=3, min_k=1, max_k=5, alpha=0.5)
    
    # 1. High acceptance (e.g. 3 accepted out of 3)
    # new EMA = 0.5 * 0.5 + 0.5 * 1.0 = 0.75 > 0.7. K should increase to 4
    k = adjuster.update_and_get_k(num_accepted=3, num_drafted=3)
    assert k == 4
    
    # 2. High acceptance again (e.g. 4 accepted out of 4)
    # new EMA = 0.5 * 0.75 + 0.5 * 1.0 = 0.875 > 0.7. K should increase to 5 (max)
    k = adjuster.update_and_get_k(num_accepted=4, num_drafted=4)
    assert k == 5
    
    # 3. High acceptance again, should clamp to max_k (5)
    k = adjuster.update_and_get_k(num_accepted=5, num_drafted=5)
    assert k == 5
    
    # 4. Low acceptance (e.g. 0 accepted out of 5)
    # new EMA = 0.5 * 0.9375 + 0.5 * 0.0 = 0.46875. K stays 5.
    k = adjuster.update_and_get_k(num_accepted=0, num_drafted=5)
    assert k == 5
    
    # 5. Low acceptance again
    # new EMA = 0.5 * 0.46875 + 0.5 * 0.0 = 0.234375 < 0.3. K should decrease to 4.
    k = adjuster.update_and_get_k(num_accepted=0, num_drafted=5)
    assert k == 4


def test_fused_kv_append_verify():
    manager = FusedKVAppendVerify(batch_size=1, num_heads=2, max_length=10, head_dim=4)
    
    new_k = mx.ones((1, 2, 4, 4)) * 5.0
    new_v = mx.ones((1, 2, 4, 4)) * 6.0
    
    num_accepted = mx.array([2])  # only accept 2 out of 4 proposed candidates
    
    k_cache, v_cache, new_offset = manager.append_verified_only(new_k, new_v, num_accepted, offset=0)
    
    assert new_offset.item() == 2
    assert k_cache.shape == (1, 2, 10, 4)
    assert v_cache.shape == (1, 2, 10, 4)
    
    assert k_cache[0, 0, 0, 0].item() == 5.0
    assert k_cache[0, 0, 1, 0].item() == 5.0
    assert k_cache[0, 0, 2, 0].item() == 0.0


def test_block_sparse_layer_skipping():
    skipping = BlockSparseLayerSkipping(confidence_threshold=0.9)
    
    # 1. High confidence hidden state
    # We want top-1 probability to be >= 0.9.
    # hidden_state: (B=1, S=1, D=4)
    # lm_head_weight: (V=3, D=4)
    lm_head = mx.array([
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])
    h_high = mx.array([[[1.0, 0.0, 0.0, 0.0]]]) # product is [10.0, 0.0, 0.0], softmax is near 1.0 at index 0.
    
    should_exit_high, token_high = skipping.evaluate_exit(h_high, lm_head)
    
    assert should_exit_high[0].item() is True
    assert token_high[0].item() == 0
    
    # 2. Low confidence hidden state
    h_low = mx.array([[[0.0, 1.0, 0.0, 0.0]]]) # product is [0.0, 0.0, 0.0], softmax is 0.33 each
    should_exit_low, token_low = skipping.evaluate_exit(h_low, lm_head)
    
    assert should_exit_low[0].item() is False


def test_simd_coalesced_softmax():
    softmax_op = SIMDCoalescedSoftmax()
    
    # Test with size 32 (divisible by 16)
    logits_32 = mx.random.normal((2, 32))
    out_32 = softmax_op.softmax(logits_32)
    expected_32 = mx.softmax(logits_32, axis=-1)
    
    assert out_32.shape == (2, 32)
    assert mx.allclose(out_32, expected_32, atol=1e-5).item()
    
    # Test with size 25 (non-divisible by 16)
    logits_25 = mx.random.normal((2, 25))
    out_25 = softmax_op.softmax(logits_25)
    expected_25 = mx.softmax(logits_25, axis=-1)
    
    assert out_25.shape == (2, 25)
    assert mx.allclose(out_25, expected_25, atol=1e-5).item()


def test_fused_attention_projections():
    w_q = mx.ones((8, 4)) * 0.5
    w_k = mx.ones((8, 6)) * 0.25
    w_v = mx.ones((8, 8)) * 0.125
    
    proj = FusedAttentionProjections(w_q, w_k, w_v)
    
    x = mx.ones((1, 5, 8))
    
    q, k, v = proj.project(x)
    
    assert q.shape == (1, 5, 4)
    assert k.shape == (1, 5, 6)
    assert v.shape == (1, 5, 8)
    
    assert abs(q[0, 0, 0].item() - 4.0) < 1e-5
    assert abs(k[0, 0, 0].item() - 2.0) < 1e-5
    assert abs(v[0, 0, 0].item() - 1.0) < 1e-5


def test_pre_allocated_unified_kv_cache():
    # num_layers = 4, B = 1, H = 2, max_len = 10, D_head = 4
    cache = PreAllocatedUnifiedKVCache(num_layers=4, batch_size=1, total_heads=2, max_length=10, head_dim=4)
    
    # Update layer 2
    new_k = mx.ones((1, 2, 3, 4)) * 3.0
    new_v = mx.ones((1, 2, 3, 4)) * 4.0
    
    k_view, v_view = cache.update_layer(layer_idx=2, new_k=new_k, new_v=new_v, offset=0)
    
    assert k_view.shape == (1, 2, 3, 4)
    assert v_view.shape == (1, 2, 3, 4)
    assert k_view[0, 0, 0, 0].item() == 3.0
    assert v_view[0, 0, 0, 0].item() == 4.0
    
    # Verify other layers are still empty
    assert cache.k_cache[0, 0, 0, 0, 0].item() == 0.0
    assert cache.k_cache[2, 0, 0, 0, 0].item() == 3.0


def test_lookahead_embedding_prefetch():
    embeddings = mx.array([
        [1.0, 1.1],
        [2.0, 2.2],
        [3.0, 3.3],
        [4.0, 4.4]
    ])
    
    prefetcher = LookaheadEmbeddingPrefetch(embeddings)
    
    next_ids = mx.array([[1, 3]])
    embs = prefetcher.prefetch_async(next_ids)
    
    mx.eval(embs)
    
    assert embs.shape == (1, 2, 2)
    assert abs(embs[0, 0, 0].item() - 2.0) < 1e-5
    assert abs(embs[0, 1, 0].item() - 4.0) < 1e-5


def test_block_wise_weight_loading():
    w1 = mx.ones((8, 4)) * 1.5
    w2 = mx.ones((8, 6)) * 2.5
    w3 = mx.ones((8, 8)) * 3.5
    
    loader = BlockWiseWeightLoading([w1, w2, w3])
    
    ret_w1 = loader.get_layer_weight(0)
    ret_w2 = loader.get_layer_weight(1)
    ret_w3 = loader.get_layer_weight(2)
    
    assert ret_w1.shape == (8, 4)
    assert ret_w2.shape == (8, 6)
    assert ret_w3.shape == (8, 8)
    
    assert abs(ret_w1[0, 0].item() - 1.5) < 1e-5
    assert abs(ret_w2[0, 0].item() - 2.5) < 1e-5
    assert abs(ret_w3[0, 0].item() - 3.5) < 1e-5


def test_speculative_tree_verifier():
    verifier = SpeculativeTreeVerifier()
    
    # 5 nodes in the tree:
    # 0 (root), 1 (child of 0), 2 (child of 1) -> Branch 1
    # 3 (child of 0), 4 (child of 3)          -> Branch 2
    # parents array: [-1, 0, 1, 0, 3]
    # For testing, parents[0] can be 0 (no effect on take)
    parents = [0, 0, 1, 0, 3]
    
    # Let's say:
    # target predicts: [1, 2, 3, 4, 99]
    # candidates are:  [1, 2, 3, 4, 5]
    # matches will be: [1, 1, 1, 1, 0] (since last node 4 mismatches candidate 5 vs target 99)
    # Path 1 (0->1->2) matches fully: count = 3
    # Path 2 (0->3->4) matches partially (0->3): count = 2
    # best accepted count should be 3
    
    target_logits = mx.array([
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 1
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 2
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 3
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 4
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]   # argmax 6 (candidate is 5 -> mismatch)
        ]
    ])
    
    tree_tokens = mx.array([[1, 2, 3, 4, 5]])
    
    best_acc = verifier.verify_tree(target_logits, tree_tokens, parents)
    
    assert best_acc[0].item() == 3


def test_fused_swiglu_ffn_reg():
    w_gate = mx.ones((8, 16)) * 0.5
    w_up = mx.ones((8, 16)) * 0.25
    w_down = mx.ones((16, 8)) * 0.1
    rms_ffn_w = mx.ones((8,)) * 1.2
    
    ffn_reg = FusedSwiGLUFFNReg(w_gate, w_up, w_down, rms_ffn_w, eps=1e-6)
    x = mx.ones((1, 4, 8)) * 0.8
    
    # Standard computation:
    # 1. RMSNorm
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    norm_x = x * mx.rsqrt(variance + 1e-6) * rms_ffn_w
    # 2. FusedSwiGLUFFN
    from qan_transformers.mlx.moonshots import FusedSwiGLUFFN
    ffn = FusedSwiGLUFFN(w_gate, w_up, w_down)
    ffn_out = ffn.forward(norm_x)
    # 3. Residual
    expected_out = x + ffn_out
    
    out = ffn_reg.forward(x)
    
    assert out.shape == (1, 4, 8)
    assert mx.allclose(out, expected_out, atol=1e-5).item()


def test_multi_stream_overlap():
    overlap = MultiStreamOverlap()
    
    def target_fn(a, b):
        return a + b

    def draft_fn(c, d):
        return c * d

    a = mx.array([1.0, 2.0])
    b = mx.array([3.0, 4.0])
    c = mx.array([5.0, 6.0])
    d = mx.array([7.0, 8.0])
    
    t_out, d_out = overlap.verify_and_draft(target_fn, draft_fn, (a, b), (c, d))
    
    assert mx.allclose(t_out, mx.array([4.0, 6.0])).item()
    assert mx.allclose(d_out, mx.array([35.0, 48.0])).item()


def test_dynamic_cache_pruning():
    pruner = DynamicCachePruning(keep_ratio=0.8)
    
    # 5 sequence elements: index 4 has very small keys (norm ~0), indices 0..3 have larger keys
    k = mx.array([
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], # 0
            [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]], # 1
            [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]], # 2
            [[4.0, 4.0, 4.0, 4.0], [4.0, 4.0, 4.0, 4.0]], # 3
            [[0.01, 0.01, 0.01, 0.01], [0.01, 0.01, 0.01, 0.01]] # 4
        ]
    ])
    v = k * 2.0
    
    k_pruned, v_pruned = pruner.prune_cache(k, v)
    
    assert k_pruned.shape == (1, 4, 2, 4)
    assert v_pruned.shape == (1, 4, 2, 4)
    
    # Check that the smallest norm (index 4) was pruned and indices 0..3 remain
    assert mx.allclose(k_pruned[0, 0, 0], mx.array([1.0, 1.0, 1.0, 1.0])).item()
    assert mx.allclose(v_pruned[0, 3, 0], mx.array([8.0, 8.0, 8.0, 8.0])).item()


def test_end_to_end_jit_speculative_pipeline():
    # Draft model: always predicts the next integer token (1, 2, 3, ...) based on input length
    def draft_model_fn(x):
        B, S = x.shape
        logits = mx.zeros((B, S, 10))
        for i in range(S):
            pred_tok = i + 1
            if pred_tok < 10:
                logits[0, i, pred_tok] = 1.0
        return logits

    # Target model: predicts [1, 2, 99, 5] for the suffix positions
    def target_model_fn(x):
        B, S = x.shape
        logits = mx.zeros((B, S, 100))
        logits[0, 0, 1] = 1.0
        logits[0, 1, 2] = 1.0
        logits[0, 2, 99] = 1.0
        logits[0, 3, 5] = 1.0
        return logits

    pipeline = EndToEndJITSpeculativePipeline(draft_model_fn, target_model_fn)
    x = mx.array([[0]])
    
    accepted_count, draft_seq, next_val = pipeline.run_speculative_step(x, K=3)
    
    assert accepted_count.item() == 2
    assert mx.allclose(draft_seq, mx.array([[1, 2, 3]])).item()
    assert next_val.item() == 99


def test_dynamic_draft_length_adjuster_momentum():
    adjuster = DynamicDraftLengthAdjuster(initial_k=3, min_k=1, max_k=5, alpha=0.5, momentum=0.5)
    k = adjuster.update_and_get_k(num_accepted=3, num_drafted=3)
    assert k == 3
    k = adjuster.update_and_get_k(num_accepted=4, num_drafted=4)
    assert k == 4

