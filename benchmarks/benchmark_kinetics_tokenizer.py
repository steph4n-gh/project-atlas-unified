import time
import os
from pathlib import Path
from qan_transformers.tokenizer import KineticsTokenizer

# Paths
TOKENIZER_JSON = Path("/Volumes/Storage/antigravity_worktrees/project_atlas/explain-git-worktree/benchmarks/data/gemma4-E4B-tokenizer.json")
CODEBASE_FILE = Path("/Volumes/Storage/antigravity_worktrees/project_atlas/explain-git-worktree/qan_transformers/mlx/modeling.py")

def main():
    if not TOKENIZER_JSON.exists():
        print("Error: Gemma4 tokenizer JSON not found.")
        return
        
    if not CODEBASE_FILE.exists():
        print("Error: Codebase file modeling.py not found.")
        return
        
    print("=" * 60)
    print("       KINETICS TOKENIZER PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Load corpus text
    with open(CODEBASE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Replicate to simulate a large context (~1.5 MB of code corpus)
    large_text = text * 10
    char_len = len(large_text)
    print(f"Benchmark Corpus Size: {char_len:,} characters (~{char_len/1e6:.2f} MB)")
    
    # 1. Baseline HF Tokenizer
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(TOKENIZER_JSON))
        
        t0 = time.perf_counter()
        ref_ids = tok.encode(large_text, add_special_tokens=False).ids
        hf_dt = (time.perf_counter() - t0) * 1000
        print(f"1. HuggingFace tokenizers (Baseline): {hf_dt:.2f} ms | {len(ref_ids):,} tokens | {char_len/hf_dt/1e3:.2f} MB/s")
    except ImportError:
        print("1. HuggingFace tokenizers: package not installed")
        
    # 2. Pure Python exact merge (CPU)
    tokenizer_py_exact = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    t0 = time.perf_counter()
    py_exact_ids = tokenizer_py_exact.encode(large_text, exact=True)
    py_exact_dt = (time.perf_counter() - t0) * 1000
    print(f"2. Pure Python Exact Merge (CPU):       {py_exact_dt:.2f} ms | {len(py_exact_ids):,} tokens | {char_len/py_exact_dt/1e3:.2f} MB/s")
    
    # 3. Pure Python Parallel Kinetics waves (CPU)
    tokenizer_py_turbo = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=False)
    t0 = time.perf_counter()
    py_turbo_ids = tokenizer_py_turbo.encode(large_text, exact=False)
    py_turbo_dt = (time.perf_counter() - t0) * 1000
    print(f"3. Pure Python Parallel Waves (CPU):    {py_turbo_dt:.2f} ms | {len(py_turbo_ids):,} tokens | {char_len/py_turbo_dt/1e3:.2f} MB/s")

    # 4. Rust Turbo parallel reaction kinetics (CPU subprocess)
    tokenizer_rust = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=True, use_mlx=False)
    if tokenizer_rust.rust_bin_path:
        t0 = time.perf_counter()
        rust_ids = tokenizer_rust.encode(large_text)
        rust_dt = (time.perf_counter() - t0) * 1000
        print(f"4. Rust Parallel Reaction (CPU Turbo):  {rust_dt:.2f} ms | {len(rust_ids):,} tokens | {char_len/rust_dt/1e3:.2f} MB/s")
    else:
        print("4. Rust Parallel Reaction (CPU Turbo):  Binary not found")

    # 5. MLX GPU BlockBPE (GPU/Metal)
    try:
        import mlx.core as mx
        tokenizer_mlx = KineticsTokenizer(str(TOKENIZER_JSON), use_rust=False, use_mlx=True)
        if tokenizer_mlx.mlx_initialized:
            # Warm up GPU
            _ = tokenizer_mlx.encode(large_text)
            
            t0 = time.perf_counter()
            mlx_ids = tokenizer_mlx.encode(large_text)
            mlx_dt = (time.perf_counter() - t0) * 1000
            print(f"5. MLX BlockBPE (Metal GPU):            {mlx_dt:.2f} ms | {len(mlx_ids):,} tokens | {char_len/mlx_dt/1e3:.2f} MB/s")
        else:
            print("5. MLX BlockBPE (Metal GPU):            Not initialized")
    except ImportError:
        print("5. MLX BlockBPE (Metal GPU):            mlx package not installed")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
