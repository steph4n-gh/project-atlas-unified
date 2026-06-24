import os
import sys
import time
import subprocess
import torch
import numpy as np

# Add workspace path to system path
sys.path.append("/Volumes/Storage/project_atlas")

def get_original_code():
    return """import torch

class MPSCoordinateGatherScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, indices):
        ctx.k_shape = k.shape
        ctx.v_shape = v.shape
        
        B, H, S, d = k.shape
        K_size = indices.shape[-1]
        
        # Convert indices to int32 and ensure k, v are contiguous on MPS to prevent driver segmentation faults (EXC_BAD_ACCESS)
        if indices.device.type == "mps":
            indices = indices.to(torch.int32)
            k = k.contiguous()
            v = v.contiguous()
            
        ctx.save_for_backward(indices)
        
        # Ensure gather_indices is contiguous to prevent stride mismatch on MPS
        gather_indices = indices.view(B, 1, K_size, 1).expand(-1, H, -1, d).contiguous()
        K_sparse = torch.gather(k, 2, gather_indices)
        V_sparse = torch.gather(v, 2, gather_indices)
        
        return K_sparse, V_sparse

    @staticmethod
    def backward(ctx, grad_k_sparse, grad_v_sparse):
        indices, = ctx.saved_tensors
        B, H, S, d = ctx.k_shape
        K_size = indices.shape[-1]
        
        grad_k = torch.zeros(B, H, S, d, dtype=grad_k_sparse.dtype, device=grad_k_sparse.device)
        grad_v = torch.zeros(B, H, S, d, dtype=grad_v_sparse.dtype, device=grad_v_sparse.device)
        
        if indices.device.type == "mps":
            indices = indices.to(torch.int32)
            grad_k_sparse = grad_k_sparse.contiguous()
            grad_v_sparse = grad_v_sparse.contiguous()
            
        gather_indices = indices.view(B, 1, K_size, 1).expand(-1, H, -1, d).contiguous()
        
        # Use scatter_add_ to scatter gradients back to dense coordinates
        grad_k.scatter_add_(2, gather_indices, grad_k_sparse)
        grad_v.scatter_add_(2, gather_indices, grad_v_sparse)
        
        return grad_k, grad_v, None

def mps_coordinate_gather_scatter(q, k, v, indices):
    \"\"\"
    Apple Silicon MPS coordinate-sparse gather-scatter matrix execution.
    \"\"\"
    return MPSCoordinateGatherScatter.apply(k, v, indices)
"""

def get_optimized_code():
    with open("qan_transformers/kernels/mps_scatter.py", "r") as f:
        return f.read()

def run_benchmark(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", module_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    shapes = [
        {"B": 1, "H": 8, "S": 100, "d": 256, "K_size": 15},
        {"B": 1, "H": 8, "S": 1024, "d": 256, "K_size": 128},
        {"B": 2, "H": 12, "S": 2048, "d": 256, "K_size": 256}
    ]
    
    total_time = 0.0
    iterations = 300
    warmup = 50
    
    for shape in shapes:
        B, H, S, d, K_size = shape["B"], shape["H"], shape["S"], shape["d"], shape["K_size"]
        
        q = torch.randn(B, H, 1, d, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H, S, d, device=device, dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H, S, d, device=device, dtype=torch.float16, requires_grad=True)
        indices = torch.randint(0, S, (B, K_size), device=device, dtype=torch.long)
        
        # Warmup loop to prime cache and queue
        for _ in range(warmup):
            K_sparse, V_sparse = test_module.mps_coordinate_gather_scatter(q, k, v, indices)
            loss = K_sparse.sum() + V_sparse.sum()
            loss.backward()
            k.grad = None
            v.grad = None
            
        if device.type == "mps":
            torch.mps.synchronize()
            
        start_time = time.perf_counter()
        for _ in range(iterations):
            K_sparse, V_sparse = test_module.mps_coordinate_gather_scatter(q, k, v, indices)
            loss = K_sparse.sum() + V_sparse.sum()
            loss.backward()
            k.grad = None
            v.grad = None
            
        if device.type == "mps":
            torch.mps.synchronize()
        end_time = time.perf_counter()
        
        total_time += (end_time - start_time) * 1000.0  # ms
        
    return total_time / len(shapes)

def main():
    print("=====================================================================")
    print("          QAN-ATLAS: ORIGINAL VS OPTIMIZED KERNEL BENCHMARK          ")
    print("=====================================================================")
    
    original_file = "scratch/mps_scatter_original.py"
    optimized_file = "scratch/mps_scatter_optimized.py"
    
    try:
        # Write temporary files
        original_code = get_original_code()
        optimized_code = get_optimized_code()
        
        with open(original_file, "w") as f:
            f.write(original_code)
        with open(optimized_file, "w") as f:
            f.write(optimized_code)
            
        print("Running micro-benchmarks on target hardware...")
        print("  -> Benchmarking original baseline...")
        orig_ms = run_benchmark(original_file)
        
        print("  -> Benchmarking self-optimized candidate...")
        opt_ms = run_benchmark(optimized_file)
        
        # Calculations
        speedup = ((orig_ms - opt_ms) / orig_ms) * 100.0
        orig_ops = 1000.0 / orig_ms
        opt_ops = 1000.0 / opt_ms
        
        print("\n" + "="*60)
        print("                     BENCHMARK COMPARISON REPORT                     ")
        print("="*60)
        print(f"Original Baseline Mean Latency :  {orig_ms:10.4f} ms  ({orig_ops:8.2f} steps/sec)")
        print(f"Optimized Kernel Mean Latency  :  {opt_ms:10.4f} ms  ({opt_ops:8.2f} steps/sec)")
        print("-"*60)
        print(f"Net Hardware Speedup           :  \033[1;32m{speedup:10.2f}%\033[0m")
        print(f"Latency Reduction              :  \033[1;36m{orig_ms - opt_ms:10.4f} ms\033[0m")
        print("="*60 + "\n")
        
    finally:
        # Cleanup
        if os.path.exists(original_file):
            os.remove(original_file)
        if os.path.exists(optimized_file):
            os.remove(optimized_file)

if __name__ == "__main__":
    main()
