import pytest
import mlx.core as mx
import numpy as np
from scratch.run_mlx_speculative_chat import speculative_verify_adelic_path_integral

def test_exact_match_selection():
    B, C, S, V = 1, 4, 3, 32
    
    # Target logits: make target predictions argmax equal to some values
    # Let's say target predictions for all channels are [10, 20, 30]
    target_logits = mx.zeros((B, C, S, V))
    # Channel 2 predictions:
    target_logits[0, 2, 0, 10] = 10.0
    target_logits[0, 2, 1, 20] = 10.0
    target_logits[0, 2, 2, 30] = 10.0
    
    # Candidate tokens:
    # Let's make channel 2 match exactly, while others mismatch
    candidate_tokens = mx.array([
        [[5, 5, 5],
         [6, 6, 6],
         [10, 20, 30],  # Exactly matches prediction
         [7, 7, 7]]
    ], dtype=mx.int32)
    
    amplitudes = mx.array([0.25, 0.25, 0.25, 0.25])
    
    win, acc_len, corr = speculative_verify_adelic_path_integral(
        target_logits, candidate_tokens, amplitudes
    )
    
    assert win == 2
    assert acc_len == 3
    assert corr is None

def test_correction_token_generation():
    B, C, S, V = 1, 2, 4, 16
    
    # Target prediction: [5, 6, 7, 8]
    target_logits = mx.zeros((B, C, S, V))
    for c in range(C):
        target_logits[0, c, 0, 5] = 5.0
        target_logits[0, c, 1, 6] = 5.0
        target_logits[0, c, 2, 7] = 5.0
        target_logits[0, c, 3, 8] = 5.0
        
    # Candidates:
    # Channel 0: [5, 6, 99, 8] (mismatch at index 2, should accept 2 tokens, correction is 7)
    # Channel 1: [99, 6, 7, 8] (mismatch at index 0, should accept 0 tokens)
    candidate_tokens = mx.array([
        [[5, 6, 99, 8],
         [99, 6, 7, 8]]
    ], dtype=mx.int32)
    
    amplitudes = mx.array([0.5, 0.5])
    
    win, acc_len, corr = speculative_verify_adelic_path_integral(
        target_logits, candidate_tokens, amplitudes
    )
    
    assert win == 0
    assert acc_len == 2
    assert corr == 7

def test_prior_amplitude_tie_breaker():
    B, C, S, V = 1, 2, 3, 10
    
    # Target prediction: [1, 2, 3]
    target_logits = mx.zeros((B, C, S, V))
    for c in range(C):
        target_logits[0, c, 0, 1] = 5.0
        target_logits[0, c, 1, 2] = 5.0
        target_logits[0, c, 2, 3] = 5.0
        
    # Candidates: both match exactly [1, 2, 3]
    candidate_tokens = mx.array([
        [[1, 2, 3],
         [1, 2, 3]]
    ], dtype=mx.int32)
    
    # Prior amplitudes: channel 1 has higher prior probability amplitude
    amplitudes = mx.array([0.2, 0.8])
    
    win, acc_len, corr = speculative_verify_adelic_path_integral(
        target_logits, candidate_tokens, amplitudes
    )
    
    # Channel 1 should win due to higher prior amplitude
    assert win == 1
    assert acc_len == 3
    assert corr is None
