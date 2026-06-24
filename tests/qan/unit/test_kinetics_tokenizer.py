import pytest
import os
from pathlib import Path
from qan_transformers.tokenizer import KineticsTokenizer

# Locate Gemma4 tokenizer JSON
TOKENIZER_JSON = Path("/Volumes/Storage/antigravity_worktrees/project_atlas/explain-git-worktree/benchmarks/data/gemma4-E4B-tokenizer.json")

def test_kinetics_tokenizer_initialization():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    # Initialize CPU KineticsTokenizer
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    assert len(tokenizer.vocab) > 200000
    assert len(tokenizer.merge_ranks) > 200000

def test_kinetics_tokenizer_encode_decode():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(text)
    
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    
    # SentencePiece normalizer prepends ' ' to the first word or spaces,
    # so we compare normalized text or strip.
    assert decoded.strip() == text

def test_rust_turbo_tokenizer():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    # Initialize with use_rust=True
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=True, use_mlx=False)
    
    if not tokenizer.rust_bin_path:
        pytest.skip("Rust tokenizer binary not compiled or not found")
        
    text = "Explain the benefits of BPE micro-optimizations."
    tokens = tokenizer.encode(text)
    
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert decoded.strip() == text

def test_mlx_gpu_tokenizer():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx is not installed")
        
    # Initialize with use_mlx=True
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=True)
    
    if not tokenizer.mlx_initialized:
        pytest.skip("MLX/Metal tokenizer not initialized (could be running on non-Apple Silicon)")
        
    text = "High-performance sequence modeling on Apple Silicon GPU."
    tokens = tokenizer.encode(text)
    
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens)
    assert decoded.strip() == text

def test_tokenizer_id_parity():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    try:
        from tokenizers import Tokenizer
    except ImportError:
        pytest.skip("tokenizers not installed")
        
    ref_tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    kin_tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False, use_kinetics=False)
    
    text = "The quick brown fox jumps over the lazy dog. Explain BPE optimization."
    ref_ids = ref_tokenizer.encode(text, add_special_tokens=False).ids
    kin_ids = kin_tokenizer.encode(text)
    
    assert kin_ids == ref_ids

def test_autopilot_routing():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    try:
        from tokenizers import Tokenizer
        base_tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    except ImportError:
        base_tokenizer = None
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False, base_tokenizer=base_tokenizer)
    
    # 1. Short text (< 5000 chars)
    text_short = "Short prompt for testing Autopilot routing logic."
    tokens_short = tokenizer.encode(text_short)
    assert len(tokens_short) > 0
    
    # 2. Large text (> 5000 chars)
    text_large = "Long text " * 1000
    tokens_large = tokenizer.encode(text_large)
    assert len(tokens_large) > 0

def test_mlx_gpu_graph_encode():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx is not installed")
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=True)
    if not tokenizer.mlx_initialized:
        pytest.skip("MLX not initialized")
        
    text = "Verify that zero-copy graph-fused tokenization works perfectly on Apple Silicon."
    
    # 1. Standard encode
    ref_ids = tokenizer.encode(text)
    
    # 2. Graph encode
    symbols_arr, num_blocks = tokenizer.prepare_fused_input(text)
    out_symbols, out_lengths, _ = tokenizer.graph_encode(symbols_arr)
    compacted = tokenizer.compact_token_ids(out_symbols, out_lengths)
    
    # Evaluate and compare
    graph_ids = compacted.tolist()
    assert graph_ids == ref_ids

def test_mlx_gpu_graph_encode_coords():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    try:
        import mlx.core as mx
        import numpy as np
    except ImportError:
        pytest.skip("mlx is not installed")
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=True)
    if not tokenizer.mlx_initialized:
        pytest.skip("MLX not initialized")
        
    text = "Verify that coordinates are returned and non-empty."
    symbols_arr, num_blocks = tokenizer.prepare_fused_input(text)
    out_symbols, out_lengths, out_coords = tokenizer.graph_encode(symbols_arr)
    
    # out_coords shape is [num_blocks * 64, 8]
    assert out_coords.size > 0
    assert out_coords.shape == (num_blocks * 64, 8)
    assert out_coords.dtype == mx.float32
    
    # Verify we can compact the coordinates
    compacted_coords = tokenizer.compact_coords(out_coords, out_lengths)
    compacted_syms = tokenizer.compact_token_ids(out_symbols, out_lengths)
    assert compacted_coords.shape[0] == compacted_syms.shape[0]
    assert compacted_coords.shape[1] == 8
    
    # Check that they contain float values and are not all zero (since the mapped E8 coordinates are non-zero)
    coords_np = np.array(compacted_coords)
    assert np.any(coords_np != 0.0)
    
    # Verify static compaction of coordinates works as well
    compacted_static, length = tokenizer.compact_coords_static(out_coords, out_lengths)
    assert compacted_static.shape == (num_blocks * 64, 8)
    assert length.item() == compacted_syms.shape[0]

    # Test dynamic temperature parameter does not crash the compilation/execution
    temp_arr = mx.array([1.5], dtype=mx.float32)
    out_symbols_temp, out_lengths_temp, out_coords_temp = tokenizer.graph_encode(symbols_arr, temperature=temp_arr)
    assert out_coords_temp.shape == (num_blocks * 64, 8)

def test_context_organism_memoization():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    from qan_transformers.tokenizer.context_organism import DeterministicContextOrganism
    
    db_file = Path("tests/unit/test_organism_db.json")
    if db_file.exists():
        db_file.unlink()
        
    organism = DeterministicContextOrganism(str(db_file))
    text = "Some random prompt text for testing context organism caching functionality."
    
    # Cache miss
    assert organism.get(text) is None
    
    # Set and snapshot
    token_ids = [10, 20, 30, 40]
    organism.set(text, token_ids)
    assert organism.get(text) == token_ids
    organism.snapshot()
    
    # Restore check
    new_organism = DeterministicContextOrganism(str(db_file))
    assert new_organism.get(text) == token_ids
    
    # Fork check
    forked = new_organism.fork()
    assert forked.get(text) == token_ids
    
    # Cleanup
    if db_file.exists():
        db_file.unlink()


def test_trie_context_caching():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    
    prefix_text = "The quick brown fox"
    suffix_text = " jumps over the lazy dog."
    full_text = prefix_text + suffix_text
    
    # Cache the prefix
    prefix_ids = tokenizer.encode(prefix_text)
    assert tokenizer.organism.get(prefix_text) == prefix_ids
    
    # Check that longest prefix match retrieves the prefix_ids and the suffix
    matched_ids, matched_suffix = tokenizer.organism.get_longest_prefix(full_text)
    assert matched_ids == prefix_ids
    assert matched_suffix == suffix_text
    
    # Now encode the full text. This should trigger the Trie caching logic:
    # retrieve prefix_ids and only encode suffix_text.
    full_ids = tokenizer.encode(full_text)
    
    # Verify correctness of output
    ref_ids = tokenizer._encode_raw(full_text)
    assert full_ids == ref_ids

def test_default_kinetics_integration():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    # By default, use_kinetics must be True
    tokenizer_default = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    assert tokenizer_default.use_kinetics is True
    
    # With use_kinetics=True, encode exact parameter defaults to None, resolving to exact=False (kinetics merge)
    text = "Explain the benefits of BPE micro-optimizations."
    
    # Verify that calling encode without exact uses exact=False
    # (By checking it matches direct _encode_raw with exact=False)
    ids_default = tokenizer_default.encode(text)
    ids_kinetics = tokenizer_default._encode_raw(text, exact=False)
    assert ids_default == ids_kinetics
    
    # With use_kinetics=False, encode exact resolves to True
    tokenizer_exact = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False, use_kinetics=False)
    assert tokenizer_exact.use_kinetics is False
    ids_no_kin = tokenizer_exact.encode(text)
    ids_exact = tokenizer_exact._encode_raw(text, exact=True)
    assert ids_no_kin == ids_exact

def test_mlx_gpu_graph_encode_static():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx is not installed")
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=True)
    if not tokenizer.mlx_initialized:
        pytest.skip("MLX not initialized")
        
    text = "Verify that zero-copy static graph compaction works beautifully without any host sync."
    
    # 1. Standard encode
    ref_ids = tokenizer.encode(text)
    
    # 2. Graph encode + Static Compaction
    symbols_arr, num_blocks = tokenizer.prepare_fused_input(text)
    
    # Compile a fused graph function that includes both encoding and static compaction
    def fused_encode_static(syms):
        out_syms, out_lens, _ = tokenizer.graph_encode(syms)
        compacted, length = tokenizer.compact_token_ids_static(out_syms, out_lens)
        return compacted, length
        
    compiled_fused = mx.compile(fused_encode_static)
    
    compacted, length = compiled_fused(symbols_arr)
    mx.eval(compacted, length)
    
    # Slice using the compiled length to get the active tokens
    graph_ids = compacted[:length.item()].tolist()
    
    # Compare with reference
    assert graph_ids == ref_ids

def test_transparent_autotokenizer_wrapping_and_grafting():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    from transformers import AutoTokenizer as HFAutoTokenizer
    from qan_transformers.tokenizer import AutoTokenizer as QANAutoTokenizer, graft_tokenizer
    
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_tok_path = Path(temp_dir) / "tokenizer.json"
        shutil.copy(str(TOKENIZER_JSON), str(temp_tok_path))
        
        # Test AutoTokenizer transparent global wrapping / patching
        # Since patch_global_autotokenizer is run automatically on import, HFAutoTokenizer.from_pretrained
        # should already return a grafted tokenizer! Let's test that:
        tokenizer = HFAutoTokenizer.from_pretrained(temp_dir, use_rust=False, use_mlx=False)
        assert hasattr(tokenizer, "kinetics")
        assert hasattr(tokenizer, "graph_encode")
        
        # Test the wrapped class explicitly
        tokenizer_qan = QANAutoTokenizer.from_pretrained(temp_dir, use_rust=False, use_mlx=False)
        assert hasattr(tokenizer_qan, "kinetics")
        
        # Encode and decode using grafted methods
        text = "Grafting check for AutoTokenizer."
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert decoded.strip() == text

def test_trie_metadata_save_load_hooks():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
        
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    
    # Add some cached entries
    tokenizer.encode("Keep learning.")
    tokenizer.encode("Topological quantum computing.")
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
        
    try:
        # Save trie metadata
        tokenizer.save_trie_metadata(temp_path)
        
        # Create a new tokenizer and load metadata
        new_tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
        assert len(new_tokenizer.organism.cache) == 0
        
        new_tokenizer.load_trie_metadata(temp_path)
        assert len(new_tokenizer.organism.cache) > 0
        
        # Check matching
        prefix_ids, suffix = new_tokenizer.organism.get_longest_prefix("Keep learning. Indeed.")
        assert prefix_ids is not None
        assert suffix == " Indeed."
    finally:
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_catalyst_based_backtracking():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found in benchmarks/data")
    tokenizer = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    
    # Let's populate the cache with "The quick brown"
    prefix = "The quick brown"
    prefix_ids = tokenizer.encode(prefix)
    
    # Now when we encode "The quick brown fox", we should backtrack to the last word boundary catalyst:
    # "brown" or similar, and encode the remainder.
    # We verify that full_ids matches the exact encoding.
    full_text = "The quick brown fox"
    full_ids = tokenizer.encode(full_text)
    exact_ids = tokenizer._encode_raw(full_text, exact=True)
    assert full_ids == exact_ids

def test_attention_metadata_store():
    import tempfile
    import shutil
    import torch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize organism with cache directory
        from qan_transformers.tokenizer.context_organism import DeterministicContextOrganism
        organism = DeterministicContextOrganism(cache_dir=temp_dir)
        
        text = "Hello world context"
        organism.set(text, [1, 2, 3])
        
        # Save attention metadata
        tensors = {
            "morse_collapse": torch.randn(2, 4),
            "adelic_grid": torch.randn(3, 3)
        }
        non_tensor = {"threshold": 0.85}
        
        organism.save_attention_metadata(text, tensors, non_tensor)
        
        # Load and check
        loaded_tensors, loaded_non_tensor = organism.load_attention_metadata(text)
        assert "morse_collapse" in loaded_tensors
        assert "adelic_grid" in loaded_tensors
        assert loaded_non_tensor == non_tensor
        assert torch.allclose(loaded_tensors["morse_collapse"], tensors["morse_collapse"])
        assert torch.allclose(loaded_tensors["adelic_grid"], tensors["adelic_grid"])

def test_grafted_dunder_methods():
    if not TOKENIZER_JSON.exists():
        pytest.skip("Gemma4 tokenizer JSON not found")
    
    import tempfile
    import shutil
    from transformers import AutoTokenizer as HFAutoTokenizer
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_tok_path = Path(temp_dir) / "tokenizer.json"
        shutil.copy(str(TOKENIZER_JSON), str(temp_tok_path))
        tokenizer = HFAutoTokenizer.from_pretrained(temp_dir, use_rust=False, use_mlx=False)
    
    # Check that dunder methods work
    assert len(tokenizer) > 0
    assert "gemma" in repr(tokenizer).lower() or "grafted" in repr(tokenizer).lower()
    assert str(tokenizer) is not None
    
    # Test __contains__, __getitem__, __iter__
    # For a Gemma tokenizer, 100 should be in vocabulary
    assert 100 in tokenizer
    assert isinstance(tokenizer[100], str)
    
    # Iteration check
    iterator = iter(tokenizer)
    assert next(iterator) is not None
