import os
import gc
import json
import time
import struct
import zlib
import numpy as np
import torch
import torch.nn as nn

from qan_transformers.modeling.attention import DenseAttention, QuasicrystallineAttention
from qan_transformers.optim.adelic import AdelicLangevinOptimizer
from qan_transformers.math.e8_swap import AdelicMemorySwapGridDB

# Try importing matplotlib, but fallback to custom pure-python drawing if unavailable
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ==========================================
# PURE-PYTHON PNG DRAWING UTILITIES (FALLBACK)
# ==========================================

# Vector/Stroke font definitions for drawing text without external libraries
STROKES = {
    '0': [(0,0, 8,0), (8,0, 8,12), (8,12, 0,12), (0,12, 0,0)],
    '1': [(4,0, 4,12)],
    '2': [(0,0, 8,0), (8,0, 8,6), (8,6, 0,6), (0,6, 0,12), (0,12, 8,12)],
    '3': [(0,0, 8,0), (8,0, 8,12), (8,12, 0,12), (0,6, 8,6)],
    '4': [(0,0, 0,6), (0,6, 8,6), (8,0, 8,12)],
    '5': [(8,0, 0,0), (0,0, 0,6), (0,6, 8,6), (8,6, 8,12), (8,12, 0,12)],
    '6': [(8,0, 0,0), (0,0, 0,12), (0,12, 8,12), (8,12, 8,6), (8,6, 0,6)],
    '7': [(0,0, 8,0), (8,0, 8,12)],
    '8': [(0,0, 8,0), (8,0, 8,12), (8,12, 0,12), (0,12, 0,0), (0,6, 8,6)],
    '9': [(0,0, 8,0), (8,0, 8,12), (0,0, 0,6), (0,6, 8,6)],
    '.': [(4,10, 4,12)],
    '-': [(2,6, 6,6)],
    ':': [(4,3, 4,5), (4,7, 4,9)],
    ',': [(4,10, 2,12)],
    ' ': [],
    'A': [(0,12, 4,0), (4,0, 8,12), (2,6, 6,6)],
    'B': [(0,0, 6,0), (6,0, 8,3), (8,3, 6,6), (6,6, 0,6), (6,6, 8,9), (8,9, 6,12), (6,12, 0,12), (0,12, 0,0)],
    'C': [(8,0, 0,0), (0,0, 0,12), (0,12, 8,12)],
    'D': [(0,0, 6,0), (6,0, 8,6), (8,6, 6,12), (6,12, 0,12), (0,12, 0,0)],
    'E': [(8,0, 0,0), (0,0, 0,12), (0,12, 8,12), (0,6, 6,6)],
    'F': [(8,0, 0,0), (0,0, 0,12), (0,6, 6,6)],
    'G': [(8,0, 0,0), (0,0, 0,12), (0,12, 8,12), (8,12, 8,6), (8,6, 4,6)],
    'H': [(0,0, 0,12), (8,0, 8,12), (0,6, 8,6)],
    'I': [(2,0, 6,0), (4,0, 4,12), (2,12, 6,12)],
    'L': [(0,0, 0,12), (0,12, 8,12)],
    'M': [(0,12, 0,0), (0,0, 4,6), (4,6, 8,0), (8,0, 8,12)],
    'N': [(0,12, 0,0), (0,0, 8,12), (8,12, 8,0)],
    'O': [(0,0, 8,0), (8,0, 8,12), (8,12, 0,12), (0,12, 0,0)],
    'P': [(0,12, 0,0), (0,0, 8,0), (8,0, 8,6), (8,6, 0,6)],
    'Q': [(0,0, 8,0), (8,0, 8,12), (8,12, 0,12), (0,12, 0,0), (4,8, 8,12)],
    'R': [(0,12, 0,0), (0,0, 8,0), (8,0, 8,6), (8,6, 0,6), (4,6, 8,12)],
    'S': [(8,0, 0,0), (0,0, 0,6), (0,6, 8,6), (8,6, 8,12), (8,12, 0,12)],
    'T': [(0,0, 8,0), (4,0, 4,12)],
    'U': [(0,0, 0,12), (0,12, 8,12), (8,12, 8,0)],
    'V': [(0,0, 4,12), (4,12, 8,0)],
    'W': [(0,0, 2,12), (2,12, 4,6), (4,6, 6,12), (6,12, 8,0)],
    'X': [(0,0, 8,12), (8,0, 0,12)],
    'Y': [(0,0, 4,6), (8,0, 4,6), (4,6, 4,12)],
    'Z': [(0,0, 8,0), (8,0, 0,12), (0,12, 8,12)],
    'k': [(0,0, 0,12), (0,6, 6,0), (0,6, 6,12)],
    'x': [(0,0, 8,12), (8,0, 0,12)],
    'v': [(0,0, 4,12), (4,12, 8,0)],
    '(': [(6,0, 2,4), (2,4, 2,8), (2,8, 6,12)],
    ')': [(2,0, 6,4), (6,4, 6,8), (6,8, 2,12)],
}

def draw_line(pixels, x0, y0, x1, y1, color):
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    h, w, _ = pixels.shape
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            pixels[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def draw_text(pixels, text, x, y, color, scale=1.0):
    curr_x = x
    for char in text:
        char_upper = char.upper() if char not in STROKES else char
        strokes = STROKES.get(char_upper, STROKES.get(char.upper(), []))
        for x0, y0, x1, y1 in strokes:
            draw_line(pixels, curr_x + x0 * scale, y + y0 * scale, curr_x + x1 * scale, y + y1 * scale, color)
        curr_x += int(12 * scale)

def save_png(pixels, filename):
    h, w, _ = pixels.shape
    png = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    
    def make_chunk(tag, data):
        length = struct.pack('>I', len(data))
        crc = struct.pack('>I', zlib.crc32(tag + data))
        return length + tag + data + crc

    png += make_chunk(b'IHDR', ihdr_data)
    
    # Fast row-based serialization using tobytes() to avoid quadratic string concatenation overhead
    raw_data = b''.join(b'\x00' + pixels[y].tobytes() for y in range(h))
            
    idat_data = zlib.compress(raw_data)
    png += make_chunk(b'IDAT', idat_data)
    png += make_chunk(b'IEND', b'')
    
    with open(filename, 'wb') as f:
        f.write(png)


# ==========================================
# PLOTTING LOGIC (WITH FALLBACK)
# ==========================================

def plot_memory_scaling(seq_lengths, dense_mem, qan_mem, dense_slope, dense_intercept, qan_slope, qan_intercept):
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        x_extrapolate = np.linspace(0, 500000, 100)
        y_dense = dense_slope * x_extrapolate + dense_intercept
        y_qan = qan_slope * x_extrapolate + qan_intercept
        
        plt.plot(x_extrapolate, y_dense, label="Dense Attention (Projected)", color="red", linestyle="--")
        plt.plot(x_extrapolate, y_qan, label="QAN-ATLAS (Projected)", color="blue", linewidth=2)
        
        plt.scatter(seq_lengths, dense_mem, color="darkred", label="Dense Attention (Measured)")
        plt.scatter(seq_lengths, qan_mem, color="darkblue", label="QAN-ATLAS (Measured)")
        
        plt.axhline(y=17.76, color="green", linestyle=":", label="VRAM Ceiling (17.76 GB)")
        plt.axhline(y=15.0, color="orange", linestyle="-.", label="15 GB Target")
        
        plt.xlabel("Context Length (Tokens)")
        plt.ylabel("KV-Cache VRAM Footprint (GB)")
        plt.title("KV Cache VRAM Footprint Scaling up to 500k Context")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig("results/context_v_vram.png", dpi=150)
        plt.close()
    else:
        # Custom pure-python rendering of the scaling plot
        pixels = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Grid area
        x_max, y_max = 500000.0, 80.0
        
        def map_coords(x_val, y_val):
            px = 80 + (x_val / x_max) * 680
            py = 520 - (y_val / y_max) * 440
            return px, py
            
        # Draw axes
        draw_line(pixels, 80, 80, 80, 520, [0, 0, 0])
        draw_line(pixels, 80, 520, 760, 520, [0, 0, 0])
        
        # Grid lines & Labels
        for y_val in [0, 20, 40, 60, 80]:
            px_start, py = map_coords(0, y_val)
            px_end, _ = map_coords(500000, y_val)
            if y_val > 0:
                for x in range(int(px_start), int(px_end), 8):
                    draw_line(pixels, x, py, min(x+4, px_end), py, [220, 220, 220])
            draw_text(pixels, f"{y_val}", px_start - 35, py - 6, [0, 0, 0], scale=0.8)
            
        for x_val in [0, 100000, 200000, 300000, 400000, 500000]:
            px, py_start = map_coords(x_val, 0)
            _, py_end = map_coords(x_val, 80)
            if x_val > 0:
                for y in range(int(py_end), int(py_start), 8):
                    draw_line(pixels, px, y, px, min(y+4, py_start), [220, 220, 220])
            label = f"{int(x_val//1000)}K" if x_val > 0 else "0"
            draw_text(pixels, label, px - 15, py_start + 10, [0, 0, 0], scale=0.8)
            
        # Extrapolated lines
        prev_x_dense, prev_y_dense = map_coords(0, dense_intercept)
        prev_x_qan, prev_y_qan = map_coords(0, qan_intercept)
        
        for step_x in np.linspace(0, 500000, 100):
            y_d = dense_slope * step_x + dense_intercept
            y_q = qan_slope * step_x + qan_intercept
            px_d, py_d = map_coords(step_x, y_d)
            px_q, py_q = map_coords(step_x, y_q)
            
            # Dense (red dashed)
            if int(step_x) % 10000 < 5000:
                draw_line(pixels, prev_x_dense, prev_y_dense, px_d, py_d, [255, 0, 0])
            prev_x_dense, prev_y_dense = px_d, py_d
            
            # QAN-ATLAS (blue solid)
            draw_line(pixels, prev_x_qan, prev_y_qan, px_q, py_q, [0, 0, 255])
            prev_x_qan, prev_y_qan = px_q, py_q
            
        # Draw measured points
        for s, d in zip(seq_lengths, dense_mem):
            px, py = map_coords(s, d)
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx*dx + dy*dy <= 16:
                        draw_line(pixels, px+dx, py+dy, px+dx, py+dy, [150, 0, 0])
                        
        for s, q in zip(seq_lengths, qan_mem):
            px, py = map_coords(s, q)
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx*dx + dy*dy <= 16:
                        draw_line(pixels, px+dx, py+dy, px+dx, py+dy, [0, 0, 150])
                        
        # 17.76 GB ceiling line (Green dashed)
        _, py_ceil = map_coords(0, 17.76)
        for x in range(80, 760, 8):
            draw_line(pixels, x, py_ceil, min(x+4, 760), py_ceil, [0, 150, 0])
            
        # 15.0 GB target line (Orange dashed)
        _, py_target = map_coords(0, 15.0)
        for x in range(80, 760, 8):
            draw_line(pixels, x, py_target, min(x+4, 760), py_target, [255, 128, 0])
            
        # Title and Labels
        draw_text(pixels, "VRAM SCALING: DENSE VS QAN-ATLAS (FLOAT16)", 150, 20, [0, 0, 0], scale=1.2)
        draw_text(pixels, "CONTEXT LENGTH (TOKENS)", 300, 560, [0, 0, 0], scale=0.9)
        draw_text(pixels, "VRAM (GB)", 15, 60, [0, 0, 0], scale=0.8)
        
        # Legend
        draw_line(pixels, 100, 100, 130, 100, [255, 0, 0])
        draw_text(pixels, "DENSE ATTENTION (PROJECTED)", 140, 94, [0, 0, 0], scale=0.7)
        draw_line(pixels, 100, 120, 130, 120, [0, 0, 255])
        draw_text(pixels, "QAN-ATLAS (PROJECTED)", 140, 114, [0, 0, 255], scale=0.7)
        draw_line(pixels, 100, 140, 130, 140, [0, 150, 0])
        draw_text(pixels, "CEILING (17.76 GB)", 140, 134, [0, 150, 0], scale=0.7)
        draw_line(pixels, 100, 160, 130, 160, [255, 128, 0])
        draw_text(pixels, "15 GB TARGET", 140, 154, [255, 128, 0], scale=0.7)
        
        save_png(pixels, "results/context_v_vram.png")

def plot_optimizer_convergence(adelic_losses, adamw_losses):
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(10, 6))
        plt.plot(adelic_losses, label="Adelic Langevin Optimizer", color="blue", linewidth=2)
        plt.plot(adamw_losses, label="AdamW Optimizer", color="red", linestyle="--")
        plt.xlabel("Optimization Steps")
        plt.ylabel("True Quadratic Loss (Noiseless)")
        plt.title("Optimizer Convergence Comparison under Noisy Quadratic Objective")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig("results/optimizer_convergence.png", dpi=150)
        plt.close()
    else:
        # Custom pure-python rendering of the optimizer convergence plot
        pixels = np.ones((600, 800, 3), dtype=np.uint8) * 255
        
        # Grid area
        x_max, y_max = 100.0, 1.5
        
        def map_coords(x_val, y_val):
            px = 80 + (x_val / x_max) * 680
            py = 520 - (y_val / y_max) * 440
            return px, py
            
        # Draw axes
        draw_line(pixels, 80, 80, 80, 520, [0, 0, 0])
        draw_line(pixels, 80, 520, 760, 520, [0, 0, 0])
        
        # Grid lines & Labels
        for y_val in [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]:
            px_start, py = map_coords(0, y_val)
            px_end, _ = map_coords(100, y_val)
            if y_val > 0:
                for x in range(int(px_start), int(px_end), 8):
                    draw_line(pixels, x, py, min(x+4, px_end), py, [220, 220, 220])
            draw_text(pixels, f"{y_val:.1f}", px_start - 35, py - 6, [0, 0, 0], scale=0.8)
            
        for x_val in [0, 20, 40, 60, 80, 100]:
            px, py_start = map_coords(x_val, 0)
            _, py_end = map_coords(x_val, 1.5)
            if x_val > 0:
                for y in range(int(py_end), int(py_start), 8):
                    draw_line(pixels, px, y, px, min(y+4, py_start), [220, 220, 220])
            draw_text(pixels, f"{x_val}", px - 10, py_start + 10, [0, 0, 0], scale=0.8)
            
        # Convergence curves
        prev_x_adel, prev_y_adel = map_coords(0, adelic_losses[0])
        prev_x_adam, prev_y_adam = map_coords(0, adamw_losses[0])
        
        for idx in range(1, 100):
            px_adel, py_adel = map_coords(idx, adelic_losses[idx])
            px_adam, py_adam = map_coords(idx, adamw_losses[idx])
            
            # Adelic (blue solid)
            draw_line(pixels, prev_x_adel, prev_y_adel, px_adel, py_adel, [0, 0, 255])
            prev_x_adel, prev_y_adel = px_adel, py_adel
            
            # AdamW (red dashed)
            if idx % 4 < 2:
                draw_line(pixels, prev_x_adam, prev_y_adam, px_adam, py_adam, [255, 0, 0])
            prev_x_adam, prev_y_adam = px_adam, py_adam
            
        # Title and Labels
        draw_text(pixels, "OPTIMIZER CONVERGENCE COMPARISON", 200, 20, [0, 0, 0], scale=1.2)
        draw_text(pixels, "OPTIMIZATION STEPS", 340, 560, [0, 0, 0], scale=0.9)
        draw_text(pixels, "TRUE LOSS", 15, 60, [0, 0, 0], scale=0.8)
        
        # Legend
        draw_line(pixels, 500, 100, 530, 100, [0, 0, 255])
        draw_text(pixels, "ADELIC LANGEVIN", 540, 94, [0, 0, 255], scale=0.7)
        draw_line(pixels, 500, 120, 530, 120, [255, 0, 0])
        draw_text(pixels, "ADAMW BASELINE", 540, 114, [255, 0, 0], scale=0.7)
        
        save_png(pixels, "results/optimizer_convergence.png")

# ==========================================
# BENCHMARK RUNNERS
# ==========================================

def run_memory_scaling():
    print("Running Memory Scaling Benchmark...")
    embed_dim = 2048
    num_heads = 8
    num_layers = 18
    sparse_ratio = 0.15
    
    dense_layer = DenseAttention(embed_dim=embed_dim, num_heads=num_heads, sparse_ratio=sparse_ratio).to(torch.float32)
    qan_layer = QuasicrystallineAttention(embed_dim=embed_dim, num_heads=num_heads, sparse_ratio=sparse_ratio).to(torch.float32)
    
    dense_layer.eval()
    qan_layer.train()
    
    seq_lengths = [1000, 2000, 4000, 8000]
    dense_mem = []
    qan_mem = []
    
    for S in seq_lengths:
        # 1. Dense Attention
        x_dense = torch.randn(1, S, embed_dim, dtype=torch.float32)
        cache_dense = {}
        with torch.no_grad():
            _ = dense_layer(x_dense, kv_cache=cache_dense)
            
        # Measure VRAM using 2 bytes per element (representing target float16 format)
        num_el_d = cache_dense["K"].nelement() + cache_dense["V"].nelement()
        mem_d = num_el_d * 2
        
        dense_mem.append((mem_d * num_layers) / (1024**3))
        
        del x_dense, cache_dense
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 2. QAN-ATLAS
        x_qan = torch.randn(1, S, embed_dim, dtype=torch.float32)
        cache_qan = {}
        with torch.no_grad():
            _ = qan_layer(x_qan, kv_cache=cache_qan)
            
        # Measure VRAM using 2 bytes per element (representing target float16 format)
        num_el_q = cache_qan["K"].nelement() + cache_qan["V"].nelement()
        mem_q = num_el_q * 2
        
        qan_mem.append((mem_q * num_layers) / (1024**3))
        
        del x_qan, cache_qan
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    dense_slope, dense_intercept = np.polyfit(seq_lengths, dense_mem, 1)
    qan_slope, qan_intercept = np.polyfit(seq_lengths, qan_mem, 1)
    
    projected_500k_dense = dense_slope * 500000 + dense_intercept
    projected_500k_qan = qan_slope * 500000 + qan_intercept
    
    print(f"Projected 500k Dense VRAM: {projected_500k_dense:.4f} GB")
    print(f"Projected 500k QAN-ATLAS VRAM: {projected_500k_qan:.4f} GB")
    
    plot_memory_scaling(seq_lengths, dense_mem, qan_mem, dense_slope, dense_intercept, qan_slope, qan_intercept)
    
    return seq_lengths, dense_mem, qan_mem, projected_500k_dense, projected_500k_qan

def run_optimizer_convergence():
    print("Running Optimizer Convergence Benchmark...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    p_adelic = torch.nn.Parameter(torch.tensor([1.5, -1.0, 2.0]))
    p_adamw = torch.nn.Parameter(torch.tensor([1.5, -1.0, 2.0]))
    
    opt_adelic = AdelicLangevinOptimizer([p_adelic], lr=0.05, alpha=0.75, T_0=0.01)
    opt_adamw = torch.optim.AdamW([p_adamw], lr=0.05)
    
    target = torch.tensor([0.0, 0.0, 0.0])
    noise_level = 0.08
    
    adelic_losses = []
    adamw_losses = []
    
    for step in range(100):
        noise = torch.randn(3) * noise_level
        
        # Adelic step
        loss_adelic = 0.5 * torch.sum((p_adelic - target) ** 2) + torch.sum(p_adelic * noise)
        opt_adelic.zero_grad()
        loss_adelic.backward()
        opt_adelic.step()
        
        true_loss_adel = float((0.5 * torch.sum((p_adelic - target) ** 2)).item())
        adelic_losses.append(true_loss_adel)
        
        # AdamW step
        loss_adamw = 0.5 * torch.sum((p_adamw - target) ** 2) + torch.sum(p_adamw * noise)
        opt_adamw.zero_grad()
        loss_adamw.backward()
        opt_adamw.step()
        
        true_loss_adam = float((0.5 * torch.sum((p_adamw - target) ** 2)).item())
        adamw_losses.append(true_loss_adam)
        
    plot_optimizer_convergence(adelic_losses, adamw_losses)
    
    return adelic_losses[-1], adamw_losses[-1], np.mean(adelic_losses), np.mean(adamw_losses)

def run_semantic_qa():
    print("Running Semantic QA Needle-in-Haystack Benchmark...")
    d_model = 64
    db = AdelicMemorySwapGridDB(d_model=d_model, device="cpu", cache_limit_ratio=0.15)
    
    corpus_keys = torch.randn(5000, d_model)
    corpus_vals = torch.randn(5000, d_model)
    
    needle_idx = 2500
    needle_key = corpus_keys[needle_idx:needle_idx+1].clone()
    needle_key[0, 0] = 5.0
    corpus_keys[needle_idx] = needle_key[0]
    
    db.swap_out(corpus_keys, corpus_vals)
    
    start_time = time.perf_counter()
    retrieved_keys, retrieved_vals = db.swap_in(needle_key)
    end_time = time.perf_counter()
    
    retrieval_latency_ms = (end_time - start_time) * 1000.0
    print(f"Needle-in-haystack retrieval latency: {retrieval_latency_ms:.4f} ms")
    
    found = False
    if retrieved_keys.shape[0] > 0:
        dists = torch.norm(retrieved_keys - needle_key, dim=-1)
        if (dists < 1e-4).any():
            found = True
            
    print(f"Needle found: {found}")
    return retrieval_latency_ms, found

def main():
    os.makedirs("results", exist_ok=True)
    
    seq_lengths, dense_mem, qan_mem, proj_dense, proj_qan = run_memory_scaling()
    adel_final, adam_final, adel_mean, adam_mean = run_optimizer_convergence()
    latency, found = run_semantic_qa()
    
    report = {
        "memory_scaling": {
            "sequence_lengths": seq_lengths,
            "dense_vram_gb": [float(v) for v in dense_mem],
            "qan_vram_gb": [float(v) for v in qan_mem],
            "projected_500k_dense_vram_gb": float(proj_dense),
            "projected_500k_qan_vram_gb": float(proj_qan),
            "vram_ceiling_gb": 17.76
        },
        "optimizer_convergence": {
            "adelic_final_loss": float(adel_final),
            "adamw_final_loss": float(adam_final),
            "adelic_mean_loss": float(adel_mean),
            "adamw_mean_loss": float(adam_mean)
        },
        "semantic_qa": {
            "corpus_size_tokens": 5000,
            "retrieval_latency_ms": float(latency),
            "needle_found": bool(found)
        }
    }
    
    with open("results/qa_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\nBenchmark Suite Completed Successfully!")
    print("Saved results to results/qa_report.json")

if __name__ == "__main__":
    main()
