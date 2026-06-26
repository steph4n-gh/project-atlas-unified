import time
import torch
import torch.nn as nn
import numpy as np
from qan_transformers.modeling.attention.base import QuasicrystallineAttention
from qan_transformers.marsshot_config import MarsshotConfig
from qan_transformers.lora.pipeline import inject_lora

def run_prefill_decode_benchmark(model_type, embed_dim=128, num_heads=4, seq_len=512, batch_size=2, device='cpu'):
    # Setup model
    if model_type == 'standard_dense':
        # Dense attention fallback
        attn = QuasicrystallineAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sparse_ratio=1.0,
            use_braiding=False
        )
    elif model_type == 'qan_projected':
        # Baseline QAN (E8 projected to 3D, fixed temperature, no derived composition/symplectic/braiding)
        attn = QuasicrystallineAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sparse_ratio=0.15,
            attention_mode='projected',
            temperature_mode='fixed',
            use_derived_composition=False,
            use_braiding=False
        )
    elif model_type == 'marsshot_full':
        # Marsshot full features: octonionic E8, tropical temp, derived composition, symplectic, braiding
        attn = QuasicrystallineAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sparse_ratio=0.15,
            attention_mode='octonionic',
            temperature_mode='tropical',
            use_derived_composition=True,
            use_braiding=True
        )
        # Inject Galois lora
        inject_lora(attn, r=4, adapter_type='galois')
        
    attn = attn.to(device)
    attn.train()
    
    # Input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    
    # Warmup
    for _ in range(2):
        attn._cayley_cache = {}
        _ = attn(x)
        
    # Latency benchmark
    start_time = time.perf_counter()
    num_runs = 5
    
    # Track peak memory on CUDA if available
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    for _ in range(num_runs):
        attn._cayley_cache = {}
        out = attn(x)
        loss = out.sum()
        loss.backward()
        attn.zero_grad()
        
    end_time = time.perf_counter()
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000.0
    
    peak_mem_mb = 0.0
    if device == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    return avg_latency_ms, peak_mem_mb

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on device: {device}...")
    
    configs = ['standard_dense', 'qan_projected', 'marsshot_full']
    seq_lengths = [256, 512]
    
    print("-" * 70)
    print(f"{'Model Config':<20} | {'Seq Len':<8} | {'Avg Latency (ms)':<18} | {'Peak Mem (MB)':<14}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        for config in configs:
            try:
                latency, mem = run_prefill_decode_benchmark(
                    config, seq_len=seq_len, device=device
                )
                mem_str = f"{mem:.2f}" if device == 'cuda' else "N/A"
                print(f"{config:<20} | {seq_len:<8} | {latency:<18.2f} | {mem_str:<14}")
            except Exception as e:
                print(f"{config:<20} | {seq_len:<8} | {'FAILED':<18} | {'N/A':<14} (Error: {e})")
                
    print("-" * 70)
