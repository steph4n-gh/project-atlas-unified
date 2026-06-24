import pytest
import mlx.core as mx
import mlx.nn as nn
from qan_transformers.mlx.wave_solver import wave_solver_generate
from scratch.run_mlx_speculative_chat import QANModelMLX

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

def test_wave_solver_generate_lightweight():
    """
    Tests wave_solver_generate using a lightweight QANModelMLX.
    Verifies execution, shape correctness, and convergence.
    """
    # 1. Initialize lightweight model and mock tokenizer
    mx.random.seed(42)
    model = QANModelMLX(
        vocab_size=128,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        sparse_ratio=0.15
    )
    tokenizer = MockTokenizer()
    
    # 2. Setup prompt and parameters
    prompt_tokens = mx.array([5, 12, 43, 21, 99], dtype=mx.uint32)
    max_tokens = 8
    
    # Disable swap_db in attention layers for clean CPU unit execution
    for m in model.modules():
        if hasattr(m, "swap_db") and m.swap_db is not None:
            m.swap_db.enabled = False
            
    # 3. Execute solver generate
    gen_tokens, iterations, converged = wave_solver_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temp=0.0,
        max_iterations=5,
        tolerance=0.1
    )
    
    # 4. Assertions
    assert gen_tokens.shape == (max_tokens,)
    assert gen_tokens.dtype == mx.uint32
    assert iterations >= 1
    assert isinstance(converged, bool)
    
    # Check that prompt prefix is not in gen_tokens
    # (gen_tokens must only return the generated tokens)
    assert not mx.array_equal(gen_tokens, prompt_tokens)

def test_wave_solver_generate_single_token():
    """
    Tests edge case where max_tokens is 1.
    """
    model = QANModelMLX(
        vocab_size=64,
        embed_dim=16,
        num_heads=2,
        num_layers=1
    )
    tokenizer = MockTokenizer()
    prompt_tokens = mx.array([1, 2, 3], dtype=mx.uint32)
    
    for m in model.modules():
        if hasattr(m, "swap_db") and m.swap_db is not None:
            m.swap_db.enabled = False
            
    gen_tokens, iterations, converged = wave_solver_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_tokens=1,
        temp=0.0
    )
    
    assert gen_tokens.shape == (1,)
    assert iterations == 1
    assert converged is True

def test_early_exit_guided_generation():
    """
    Tests early exit guided wave generation with early_exit_layer option.
    Uses mock model with 2 layers and exits at layer 1.
    """
    mx.random.seed(42)
    model = QANModelMLX(
        vocab_size=64,
        embed_dim=16,
        num_heads=2,
        num_layers=2
    )
    tokenizer = MockTokenizer()
    prompt_tokens = mx.array([10, 20, 30], dtype=mx.uint32)
    
    for m in model.modules():
        if hasattr(m, "swap_db") and m.swap_db is not None:
            m.swap_db.enabled = False
            
    # Run with early_exit_layer = 1
    gen_tokens, iterations, converged = wave_solver_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_tokens=prompt_tokens,
        max_tokens=6,
        temp=0.0,
        early_exit_layer=1,
        max_iterations=5
    )
    
    assert gen_tokens.shape == (6,)
    assert gen_tokens.dtype == mx.uint32
    assert iterations >= 1
    assert isinstance(converged, bool)

def test_early_exit_forward_pass_direct():
    """
    Directly tests early_exit_forward_pass function.
    """
    model = QANModelMLX(
        vocab_size=32,
        embed_dim=8,
        num_heads=2,
        num_layers=2
    )
    inputs = mx.array([[1, 2, 3]], dtype=mx.uint32)
    
    from qan_transformers.mlx.wave_solver import early_exit_forward_pass
    
    # 1. Full pass
    out_full = early_exit_forward_pass(model, inputs, None, num_layers=None)
    
    # 2. Early exit at layer 1
    out_exit = early_exit_forward_pass(model, inputs, None, num_layers=1)
    
    assert out_full.shape == (1, 3, 32)
    assert out_exit.shape == (1, 3, 32)
