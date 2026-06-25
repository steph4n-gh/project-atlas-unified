import os
import fcntl
import threading
import torch
import queue
# Optimized with target transformation
import numpy as np
from threading import Thread

# Pinned memory CPU tensors are coerced onto the GPU device (mps:0) by PyTorch on Apple Silicon.
# To keep CPU buffers strictly offloaded from the M4 Pro VRAM ceiling, we disable pin_memory on macOS.
# We also only use pin_memory if CUDA is available, to avoid NVIDIA driver errors on CPU-only environments.
use_pin_memory = torch.cuda.is_available() and not torch.backends.mps.is_available()

class ThreadWithReturnValue(Thread):
    def __init__(self, target, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        super().__init__(target=target, args=args, kwargs=kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        super().join(*args)
        return self._return
from typing import Dict, List, Tuple
from qan_transformers.math.e8_projection import ConwaySloaneE8Decoder, generate_dynamic_e8_coordinates

_DEFAULT_PROJECTION_CACHE = {}

_SHARED_SHELL_1_ROOTS = None

def get_shared_shell_1_roots():
    global _SHARED_SHELL_1_ROOTS
    if _SHARED_SHELL_1_ROOTS is None:
        _SHARED_SHELL_1_ROOTS = generate_dynamic_e8_coordinates(1)
    return _SHARED_SHELL_1_ROOTS

class FileMutex:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.fd = None
        self.thread_lock = threading.Lock()
        self.disable_flock = os.environ.get("QAN_MULTI_PROCESS") != "1"

    def acquire(self):
        self.thread_lock.acquire()
        if self.disable_flock:
            return
        try:
            if self.fd is None:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_WRONLY)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        except Exception:
            self.thread_lock.release()
            raise

    def release(self):
        if self.disable_flock:
            self.thread_lock.release()
            return
        try:
            if self.fd is not None:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            self.thread_lock.release()

    def close(self):
        if self.disable_flock:
            return
        if self.fd is not None:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None

    def __del__(self):
        self.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class DummyMutex:
    def __init__(self, lock_path: str = None):
        self.lock_path = lock_path

    def acquire(self):
        pass

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class AdelicMemorySwapGridDB:
    def __init__(self, d_model: int, device: str = "cpu", cache_limit_ratio: float = 0.15, d_model_draft: int = None, lock_path: str = None):
        """
        Adelic E8 Memory Swap Grid DB.
        Manages CPU-offloaded KV state using E8 sphere-packing quantization,
        indexing it via a GPU-native coordinate tensor for fast vectorized lookups.
        
        Args:
            d_model: Dimension of the keys/values.
            device: GPU device or CPU.
            cache_limit_ratio: Maximum percentage of keys allowed to reside in GPU cache.
            d_model_draft: Optional dimension of the draft model's keys/values.
        """
        self.d_model = d_model
        self.d_model_target = d_model
        self.d_model_draft = d_model_draft
        self.device = device
        self.cache_limit_ratio = cache_limit_ratio
        
        # Mutex Lock
        if lock_path is None:
            self.lock_path = None
            self.mutex = DummyMutex()
        else:
            self.lock_path = lock_path
            self.mutex = FileMutex(self.lock_path)
        
        # Vectorized E8 decoder
        self.decoder = ConwaySloaneE8Decoder()
        
        # Shared shell 1 roots generated dynamically or retrieved from global cache
        
        # GPU index and CPU paging tensors
        self._grid_coords = None  # shape [M, 8], on GPU
        self.grid_coords_len = 0
        self.grid_coords_capacity = 0
        self._coords_cache = {}
        
        # Separate CPU paging tensors for Target and Draft
        self._cpu_k_target = None        # shape [M, d_model_target], on CPU
        self._cpu_v_target = None        # shape [M, d_model_target], on CPU
        self.cpu_k_target_capacity = 0
        self.cpu_v_target_capacity = 0
        self.target_len = 0
        
        self._cpu_k_draft = None         # shape [M, d_model_draft], on CPU
        self._cpu_v_draft = None         # shape [M, d_model_draft], on CPU
        self.cpu_k_draft_capacity = 0
        self.cpu_v_draft_capacity = 0
        self.draft_len = 0
        
        # Backwards compatible pointers
        # (defined as properties below)
        self.gpu_cache = {}
        self.shell_1_roots = torch.tensor(generate_dynamic_e8_coordinates(1), dtype=torch.float32)
        self._target_buckets = [[] for _ in range(240)]
        self._draft_buckets = [[] for _ in range(240)]
        
        # Background swap queue and thread
        self._start_uploader()
        
    def _start_uploader(self):
        self._swap_queue = queue.Queue()
        self._uploader_shutdown = False
        self._uploader_thread = threading.Thread(target=self._uploader_loop, daemon=True)
        self._uploader_thread.start()

    def _uploader_loop(self):
        while not self._uploader_shutdown:
            try:
                item = self._swap_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self._swap_queue.task_done()
                break
            
            is_draft, k_cpu, v_cpu, quantized_cpu = item
            try:
                if is_draft:
                    self._sync_swap_out_draft(k_cpu, v_cpu, quantized_cpu)
                else:
                    self._sync_swap_out_target(k_cpu, v_cpu, quantized_cpu)
            except Exception as e:
                import traceback
                traceback.print_exc()
            finally:
                self._swap_queue.task_done()

    def _wait_for_pending_swaps(self):
        if getattr(self, "_uploader_thread", None) == threading.current_thread():
            return
        if hasattr(self, "_swap_queue"):
            self._swap_queue.join()

    @property
    def grid_coords(self):
        self._wait_for_pending_swaps()
        if self._grid_coords is None:
            return None
        return self._grid_coords[:self.grid_coords_len]

    @grid_coords.setter
    def grid_coords(self, value):
        self._grid_coords = value
        if value is None:
            self.grid_coords_len = 0
            self.grid_coords_capacity = 0
        else:
            self.grid_coords_len = value.shape[0]
            self.grid_coords_capacity = value.shape[0]

    @property
    def cpu_k_target(self):
        self._wait_for_pending_swaps()
        if self._cpu_k_target is None:
            return None
        return self._cpu_k_target[:self.target_len]

    @cpu_k_target.setter
    def cpu_k_target(self, value):
        self._cpu_k_target = value
        if value is None:
            self.target_len = 0
            self.cpu_k_target_capacity = 0
        else:
            self.target_len = value.shape[0]
            self.cpu_k_target_capacity = value.shape[0]

    @property
    def cpu_v_target(self):
        self._wait_for_pending_swaps()
        if self._cpu_v_target is None:
            return None
        return self._cpu_v_target[:self.target_len]

    @cpu_v_target.setter
    def cpu_v_target(self, value):
        self._cpu_v_target = value
        if value is None:
            self.cpu_v_target_capacity = 0
        else:
            self.cpu_v_target_capacity = value.shape[0]

    @property
    def cpu_k_draft(self):
        self._wait_for_pending_swaps()
        if self._cpu_k_draft is None:
            return None
        return self._cpu_k_draft[:self.draft_len]

    @cpu_k_draft.setter
    def cpu_k_draft(self, value):
        self._cpu_k_draft = value
        if value is None:
            self.draft_len = 0
            self.cpu_k_draft_capacity = 0
        else:
            self.draft_len = value.shape[0]
            self.cpu_k_draft_capacity = value.shape[0]

    @property
    def cpu_v_draft(self):
        self._wait_for_pending_swaps()
        if self._cpu_v_draft is None:
            return None
        return self._cpu_v_draft[:self.draft_len]

    @cpu_v_draft.setter
    def cpu_v_draft(self, value):
        self._cpu_v_draft = value
        if value is None:
            self.cpu_v_draft_capacity = 0
        else:
            self.cpu_v_draft_capacity = value.shape[0]

    @property
    def cpu_k(self):
        self._wait_for_pending_swaps()
        if self._cpu_k_target is None:
            return None
        return self._cpu_k_target[:self.target_len]

    @cpu_k.setter
    def cpu_k(self, value):
        pass

    @property
    def cpu_v(self):
        self._wait_for_pending_swaps()
        if self._cpu_v_target is None:
            return None
        return self._cpu_v_target[:self.target_len]

    @cpu_v.setter
    def cpu_v(self, value):
        pass

    def _init_default_projection(self, device, dtype, is_draft: bool):
        d_model = self.d_model_draft if is_draft else self.d_model_target
        W_p_attr = "W_p_draft" if is_draft else "W_p_target"
        
        # Win 84: Cached Default Orthogonal Projections in Swap DB
        cache_key = (d_model, str(device), dtype)
        if cache_key in _DEFAULT_PROJECTION_CACHE:
            W_p = _DEFAULT_PROJECTION_CACHE[cache_key]
        else:
            # QR decomposition of a random Gaussian matrix to ensure W_p is rank-8 orthogonal
            # and has high entropy
            g = torch.randn(d_model, 8, device=device, dtype=torch.float32)
            q, r = torch.linalg.qr(g)
            W_p = torch.nn.functional.normalize(q, dim=0).to(dtype=dtype)
            _DEFAULT_PROJECTION_CACHE[cache_key] = W_p
            
        setattr(self, W_p_attr, W_p)
        
    def initialize_projections(self, W_q: torch.Tensor, W_k: torch.Tensor, is_draft: bool = False):
        """
        Initializes W_p using Singular Value Decomposition on query/key weights.
        Uses a bisection search on the cumulative singular values to dynamically
        determine the soft-thresholding cutoff tau that preserves 95% of spectral energy.
        """
        device = W_q.device
        dtype = W_q.dtype
        d_model = W_q.shape[-1]
        
        W_stacked = torch.cat([W_q.detach().to(torch.float32), W_k.detach().to(torch.float32)], dim=0)
        U, S, Vh = torch.linalg.svd(W_stacked, full_matrices=False)
        
        # Win 107: Loop-free GPU-based SVD energy cutoff selection using torch.searchsorted
        cumulative_energy = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        target_energy = torch.tensor([0.95], device=device, dtype=torch.float32)
        cutoff_idx = torch.searchsorted(cumulative_energy, target_energy).clamp(max=len(cumulative_energy) - 1).item()
        tau = S[cutoff_idx].item()
        
        Vt = Vh
        num_vectors = Vt.shape[0]
        W_p = torch.zeros(d_model, 8, device=device, dtype=dtype)
        copy_rows = min(8, num_vectors)
        
        with torch.no_grad():
            # Win 110: Vectorized projection scaling in PyTorch SVD initialization
            scales = torch.clamp(1.0 - (tau / (S[:copy_rows] + 1e-6)), min=0.0)
            Vt_subset = Vt[:copy_rows] * scales.unsqueeze(-1)
            
            if torch.allclose(Vt_subset, torch.zeros_like(Vt_subset)):
                Vt_subset = Vt[:copy_rows]
                
            W_p[:, :copy_rows] = Vt_subset.t().to(device=device, dtype=dtype)
            W_p = torch.nn.functional.normalize(W_p, dim=0)
            
        if is_draft:
            self.W_p_draft = W_p
        else:
            self.W_p_target = W_p
            
    def _quantize(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Quantizes keys of shape [..., 8] to the nearest E8 lattice points.
        """
        return self.decoder.decode(keys)

    def _update_coords_cache(self, new_coords: torch.Tensor, dtype):
        if not hasattr(self, "_coords_cache") or self._coords_cache is None:
            self._coords_cache = {}
            
        capacity = getattr(self, "grid_coords_capacity", 2048)
        current_len = self.grid_coords_len - new_coords.shape[0]
        needed_capacity = max(capacity, current_len + new_coords.shape[0])
        
        if ("coords_norm" not in self._coords_cache or 
            self._coords_cache.get("dtype") != dtype or 
            self._coords_cache.get("capacity", 0) < needed_capacity):
            
            alloc_capacity = max(2048, needed_capacity * 2)
            coords_buf = torch.empty((alloc_capacity, 8), dtype=dtype, device="cpu")
            coords_norm2_buf = torch.empty((alloc_capacity,), dtype=dtype, device="cpu")
            
            if current_len > 0:
                coords = self.grid_coords[:current_len].to(dtype=dtype, device="cpu")
                coords_buf[:current_len].copy_(coords)
                coords_norm2_buf[:current_len].copy_(torch.sum(coords.square(), dim=-1))
                
            self._coords_cache = {
                "dtype": dtype,
                "grid_id": id(self._grid_coords),
                "coords_norm": coords_buf,
                "coords_norm2": coords_norm2_buf,
                "capacity": alloc_capacity
            }
            
        new_c = new_coords.to(dtype=dtype, device="cpu")
        new_norm2 = torch.sum(new_c.square(), dim=-1)
        
        self._coords_cache["coords_norm"][current_len : current_len + new_coords.shape[0]].copy_(new_c)
        self._coords_cache["coords_norm2"][current_len : current_len + new_coords.shape[0]].copy_(new_norm2)
        
    def clear(self):
        self._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            self._grid_coords = None
            self.grid_coords_len = 0
            self.grid_coords_capacity = 0
            
            self._cpu_k_target = None
            self._cpu_v_target = None
            self.cpu_k_target_capacity = 0
            self.cpu_v_target_capacity = 0
            self.target_len = 0
            
            self._cpu_k_draft = None
            self._cpu_v_draft = None
            self.cpu_k_draft_capacity = 0
            self.cpu_v_draft_capacity = 0
            self.draft_len = 0
            
            self.cpu_k = None
            self.cpu_v = None
            self.gpu_cache.clear()
            self._coords_cache.clear()
            self._target_buckets = [[] for _ in range(240)]
            self._draft_buckets = [[] for _ in range(240)]
        finally:
            self.mutex.release()
            if hasattr(self.mutex, "close"):
                self.mutex.close()
        
    def rollback(self, num_tokens_to_keep: int, current_len: int = None):
        self._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            target_len = getattr(self, "target_len", 0)
            if current_len is None or current_len == 0:
                ratio = 2
            else:
                ratio = target_len // current_len if current_len > 0 else 2
                
            num_vectors_target = num_tokens_to_keep * ratio
            self.target_len = min(self.target_len, num_vectors_target)
            
            draft_len = getattr(self, "draft_len", 0)
            if current_len is None or current_len == 0:
                ratio_d = 2
            else:
                ratio_d = draft_len // current_len if current_len > 0 else 2
            num_vectors_draft = num_tokens_to_keep * ratio_d
            self.draft_len = min(self.draft_len, num_vectors_draft)
            
            # Grid coordinates are shared, truncate to the maximum required length
            max_len = max(self.target_len, self.draft_len)
            self.grid_coords_len = min(self.grid_coords_len, max_len)
            
            self._coords_cache.clear()
            
            for c in range(240):
                self._target_buckets[c] = [idx for idx in self._target_buckets[c] if idx < self.target_len]
            for c in range(240):
                self._draft_buckets[c] = [idx for idx in self._draft_buckets[c] if idx < self.draft_len]
            
            if self._cpu_k_target is not None:
                self.cpu_k = self._cpu_k_target[:self.target_len]
                self.cpu_v = self._cpu_v_target[:self.target_len]
        finally:
            self.mutex.release()

    def swap_out(self, keys: torch.Tensor, values: torch.Tensor):
        self.swap_out_target(keys, values)

    def swap_out_target(self, keys: torch.Tensor, values: torch.Tensor):
        if len(keys) == 0:
            return
            
        device = keys.device
        dtype = keys.dtype
        
        if not hasattr(self, "W_p_target") or self.W_p_target.device != device or self.W_p_target.dtype != dtype:
            self._init_default_projection(device, dtype, is_draft=False)
            
        # Win 78: Asynchronous CPU-Offloaded E8 Decoding during Swap-Out
        keys_8d = keys @ self.W_p_target
        
        k_cpu = keys.detach().to("cpu", non_blocking=True)
        v_cpu = values.detach().to("cpu", non_blocking=True)
        keys_8d_cpu = keys_8d.detach().to("cpu", non_blocking=True)
        
        if getattr(self, "_uploader_thread", None) is None:
            self._start_uploader()
            
        self._swap_queue.put((False, k_cpu, v_cpu, keys_8d_cpu))

    def _sync_swap_out_target(self, k_cpu: torch.Tensor, v_cpu: torch.Tensor, keys_8d_cpu: torch.Tensor):
        # Quantize on CPU
        quantized_cpu = self._quantize(keys_8d_cpu).to(dtype=k_cpu.dtype)
        
        self.mutex.acquire()
        try:
            n_new = k_cpu.shape[0]
            d_model = k_cpu.shape[-1]
            dtype = k_cpu.dtype
            
            target_capacity = getattr(self, "cpu_k_target_capacity", 0)
            target_len = getattr(self, "target_len", 0)
            
            if self._cpu_k_target is None or target_capacity < target_len + n_new:
                new_capacity = max(2048, target_capacity * 2)
                while new_capacity < target_len + n_new:
                    new_capacity *= 2
                
                new_k_buf = torch.empty((new_capacity, d_model), dtype=dtype, device="cpu", pin_memory=use_pin_memory)
                new_v_buf = torch.empty((new_capacity, d_model), dtype=dtype, device="cpu", pin_memory=use_pin_memory)
                
                if self._cpu_k_target is not None:
                    new_k_buf[:target_len].copy_(self._cpu_k_target[:target_len])
                    new_v_buf[:target_len].copy_(self._cpu_v_target[:target_len])
                    
                self._cpu_k_target = new_k_buf
                self._cpu_v_target = new_v_buf
                self.cpu_k_target_capacity = new_capacity
                self.cpu_v_target_capacity = new_capacity

            self._cpu_k_target[target_len:target_len+n_new].copy_(k_cpu)
            self._cpu_v_target[target_len:target_len+n_new].copy_(v_cpu)
            
            # Topological E8 Voronoi Hashing
            shell_roots_cpu = self.shell_1_roots.to(device="cpu", dtype=quantized_cpu.dtype)
            centroids = torch.argmax(quantized_cpu @ shell_roots_cpu.t(), dim=-1)
            centroids_np = centroids.numpy()
            for i, c_idx in enumerate(centroids_np):
                abs_idx = target_len + i
                self._target_buckets[int(c_idx)].append(abs_idx)
                
            num_target_tokens = target_len + n_new
            self.target_len = num_target_tokens
            
            if self._grid_coords is None or self.grid_coords_len < num_target_tokens:
                new_tokens_count = num_target_tokens - self.grid_coords_len
                new_coords = quantized_cpu[-new_tokens_count:]
                
                grid_coords_capacity = getattr(self, "grid_coords_capacity", 0)
                grid_coords_len = getattr(self, "grid_coords_len", 0)
                
                if self._grid_coords is None or grid_coords_capacity < grid_coords_len + new_tokens_count:
                    new_capacity = max(2048, grid_coords_capacity * 2)
                    while new_capacity < grid_coords_len + new_tokens_count:
                        new_capacity *= 2
                    
                    new_grid = torch.empty((new_capacity, 8), dtype=quantized_cpu.dtype, device="cpu")
                    if self._grid_coords is not None:
                        new_grid[:grid_coords_len].copy_(self._grid_coords[:grid_coords_len])
                    self._grid_coords = new_grid
                    self.grid_coords_capacity = new_capacity
                    
                self._grid_coords[grid_coords_len:grid_coords_len+new_tokens_count].copy_(new_coords)
                self.grid_coords_len = grid_coords_len + new_tokens_count
                if hasattr(self, "_coords_cache") and "dtype" in self._coords_cache:
                    self._update_coords_cache(new_coords, self._coords_cache["dtype"])
                else:
                    self._coords_cache.clear()
                
                # Win 41: Thread-Safe GPU Cache Invalidation
                if hasattr(self, "_coords_gpu_cache") and self._coords_gpu_cache is not None:
                    self._coords_gpu_cache.clear()
            
            self.cpu_k = self.cpu_k_target[:self.target_len]
            self.cpu_v = self.cpu_v_target[:self.target_len]
        finally:
            self.mutex.release()

    def swap_out_draft(self, keys: torch.Tensor, values: torch.Tensor):
        if len(keys) == 0:
            return
        if self.d_model_draft is None:
            raise ValueError("d_model_draft was not initialized in swap database")
            
        device = keys.device
        dtype = keys.dtype
        
        if not hasattr(self, "W_p_draft") or self.W_p_draft.device != device or self.W_p_draft.dtype != dtype:
            self._init_default_projection(device, dtype, is_draft=True)
            
        # Win 78: Asynchronous CPU-Offloaded E8 Decoding during Swap-Out
        keys_8d = keys @ self.W_p_draft
        
        k_cpu = keys.detach().to("cpu", non_blocking=True)
        v_cpu = values.detach().to("cpu", non_blocking=True)
        keys_8d_cpu = keys_8d.detach().to("cpu", non_blocking=True)
        
        if getattr(self, "_uploader_thread", None) is None:
            self._start_uploader()
            
        self._swap_queue.put((True, k_cpu, v_cpu, keys_8d_cpu))

    def _sync_swap_out_draft(self, k_cpu: torch.Tensor, v_cpu: torch.Tensor, keys_8d_cpu: torch.Tensor):
        # Quantize on CPU
        quantized_cpu = self._quantize(keys_8d_cpu).to(dtype=k_cpu.dtype)
        
        self.mutex.acquire()
        try:
            n_new = k_cpu.shape[0]
            d_model = k_cpu.shape[-1]
            dtype = k_cpu.dtype
            
            draft_capacity = getattr(self, "cpu_k_draft_capacity", 0)
            draft_len = getattr(self, "draft_len", 0)
            
            if self._cpu_k_draft is None or draft_capacity < draft_len + n_new:
                new_capacity = max(2048, draft_capacity * 2)
                while new_capacity < draft_len + n_new:
                    new_capacity *= 2
                
                new_k_buf = torch.empty((new_capacity, d_model), dtype=dtype, device="cpu", pin_memory=use_pin_memory)
                new_v_buf = torch.empty((new_capacity, d_model), dtype=dtype, device="cpu", pin_memory=use_pin_memory)
                
                if self._cpu_k_draft is not None:
                    new_k_buf[:draft_len].copy_(self._cpu_k_draft[:draft_len])
                    new_v_buf[:draft_len].copy_(self._cpu_v_draft[:draft_len])
                    
                self._cpu_k_draft = new_k_buf
                self._cpu_v_draft = new_v_buf
                self.cpu_k_draft_capacity = new_capacity
                self.cpu_v_draft_capacity = new_capacity

            self._cpu_k_draft[draft_len:draft_len+n_new].copy_(k_cpu)
            self._cpu_v_draft[draft_len:draft_len+n_new].copy_(v_cpu)
            
            # Topological E8 Voronoi Hashing
            shell_roots_cpu = self.shell_1_roots.to(device="cpu", dtype=quantized_cpu.dtype)
            centroids = torch.argmax(quantized_cpu @ shell_roots_cpu.t(), dim=-1)
            centroids_np = centroids.numpy()
            for i, c_idx in enumerate(centroids_np):
                abs_idx = draft_len + i
                self._draft_buckets[int(c_idx)].append(abs_idx)
                
            num_draft_tokens = draft_len + n_new
            self.draft_len = num_draft_tokens
            
            if self._grid_coords is None or self.grid_coords_len < num_draft_tokens:
                new_tokens_count = num_draft_tokens - self.grid_coords_len
                new_coords = quantized_cpu[-new_tokens_count:]
                
                grid_coords_capacity = getattr(self, "grid_coords_capacity", 0)
                grid_coords_len = getattr(self, "grid_coords_len", 0)
                
                if self._grid_coords is None or grid_coords_capacity < grid_coords_len + new_tokens_count:
                    new_capacity = max(2048, grid_coords_capacity * 2)
                    while new_capacity < grid_coords_len + new_tokens_count:
                        new_capacity *= 2
                    
                    new_grid = torch.empty((new_capacity, 8), dtype=quantized_cpu.dtype, device="cpu")
                    if self._grid_coords is not None:
                        new_grid[:grid_coords_len].copy_(self._grid_coords[:grid_coords_len])
                    self._grid_coords = new_grid
                    self.grid_coords_capacity = new_capacity
                    
                self._grid_coords[grid_coords_len:grid_coords_len+new_tokens_count].copy_(new_coords)
                self.grid_coords_len = grid_coords_len + new_tokens_count
                if hasattr(self, "_coords_cache") and "dtype" in self._coords_cache:
                    self._update_coords_cache(new_coords, self._coords_cache["dtype"])
                else:
                    self._coords_cache.clear()
                
                # Win 41: Thread-Safe GPU Cache Invalidation
                if hasattr(self, "_coords_gpu_cache") and self._coords_gpu_cache is not None:
                    self._coords_gpu_cache.clear()
        finally:
            self.mutex.release()
            
    def swap_in(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.swap_in_target(queries)
        
    def swap_in_target(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._swap_in(queries, is_draft=False)
        
    def swap_in_draft(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._swap_in(queries, is_draft=True)
        
    def _swap_in(self, queries: torch.Tensor, is_draft: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            d_model = self.d_model_draft if is_draft else self.d_model_target
            cpu_k_buf = self.cpu_k_draft if is_draft else self.cpu_k_target
            cpu_v_buf = self.cpu_v_draft if is_draft else self.cpu_v_target
            W_p_attr = "W_p_draft" if is_draft else "W_p_target"
            
            limit = self.draft_len if is_draft else self.target_len
            
            if len(queries) == 0:
                if not hasattr(self, "_empty_swap_in_cache"):
                    self._empty_swap_in_cache = {}
                cache_key = (0, d_model, queries.device, queries.dtype)
                if cache_key not in self._empty_swap_in_cache:
                    self._empty_swap_in_cache[cache_key] = (
                        torch.zeros(0, d_model, device=queries.device, dtype=queries.dtype),
                        torch.zeros(0, d_model, device=queries.device, dtype=queries.dtype)
                    )
                return self._empty_swap_in_cache[cache_key]
                        
            if cpu_k_buf is None or limit == 0 or self.grid_coords is None or self.grid_coords_len == 0:
                if not hasattr(self, "_empty_swap_in_cache"):
                    self._empty_swap_in_cache = {}
                cache_key = (1, d_model, queries.device, queries.dtype)
                if cache_key not in self._empty_swap_in_cache:
                    self._empty_swap_in_cache[cache_key] = (
                        torch.zeros(1, d_model, device=queries.device, dtype=queries.dtype),
                        torch.zeros(1, d_model, device=queries.device, dtype=queries.dtype)
                    )
                return self._empty_swap_in_cache[cache_key]
                
            device = queries.device
            dtype = queries.dtype
            
            cpu_k_buf = cpu_k_buf[:limit]
            cpu_v_buf = cpu_v_buf[:limit]
            grid_coords_sliced = self.grid_coords[:limit]
            
            if not hasattr(self, W_p_attr) or getattr(self, W_p_attr).device != device or getattr(self, W_p_attr).dtype != dtype:
                self._init_default_projection(device, dtype, is_draft)
                
            W_p = getattr(self, W_p_attr)
            
            # Win 88: Fused Query Projection & Distance in PyTorch Swap DB
            if queries.requires_grad:
                queries_8d = queries @ W_p
                quantized = self._quantize(queries_8d)
                q_norm2 = torch.sum(quantized.square(), dim=-1, keepdim=True)
            else:
                if not hasattr(self, "_queries_8d_buf") or self._queries_8d_buf.shape[0] < queries.shape[0] or self._queries_8d_buf.device != device or self._queries_8d_buf.dtype != dtype:
                    self._queries_8d_buf = torch.empty((queries.shape[0], 8), device=device, dtype=dtype)
                queries_8d = torch.matmul(queries, W_p, out=self._queries_8d_buf[:queries.shape[0]])
                quantized = self._quantize(queries_8d)
                
                if not hasattr(self, "_q_norm2_buf") or self._q_norm2_buf.shape[0] < queries.shape[0] or self._q_norm2_buf.device != device or self._q_norm2_buf.dtype != dtype:
                    self._q_norm2_buf = torch.empty((queries.shape[0], 1), device=device, dtype=dtype)
                q_norm2 = torch.sum(quantized.square(), dim=-1, keepdim=True, out=self._q_norm2_buf[:queries.shape[0]])
            
            if (not hasattr(self, "_coords_gpu_cache") or 
                self._coords_gpu_cache.get("dtype") != dtype or 
                self._coords_gpu_cache.get("device") != device or
                self._coords_gpu_cache.get("grid_id") != id(self._grid_coords) or
                "coords_norm" not in self._coords_gpu_cache):
                coords = self.grid_coords.to(dtype=dtype, device=device)
                coords_norm2 = torch.sum(coords.square(), dim=-1)
                self._coords_gpu_cache = {
                    "dtype": dtype,
                    "device": device,
                    "grid_id": id(self._grid_coords),
                    "coords_norm": coords,
                    "coords_norm2": coords_norm2
                }
            coords_norm = self._coords_gpu_cache["coords_norm"][:limit]
            coords_norm2 = self._coords_gpu_cache["coords_norm2"][:limit].unsqueeze(0)
            
            # Ultrametric 2-adic GPU-resident Search
            use_pruned = (queries.shape[0] < 8 and limit >= 512)
            if use_pruned:
                # 1. Pre-cache shell roots + zero
                if not hasattr(self, "_shell_roots_and_zero_gpu") or self._shell_roots_and_zero_gpu.device != device or self._shell_roots_and_zero_gpu.dtype != dtype:
                    roots = self.shell_1_roots.to(device=device, dtype=dtype)
                    zero = torch.zeros((1, 8), device=device, dtype=dtype)
                    self._shell_roots_and_zero_gpu = torch.cat([roots, zero], dim=0)
                
                # 2. Candidate coordinates
                candidates = quantized.unsqueeze(1) + self._shell_roots_and_zero_gpu.unsqueeze(0)
                
                # 3. 2-adic Level 1 mapping
                y = torch.round(2.0 * candidates).to(torch.long)
                y_mod_2 = torch.remainder(y, 2)
                
                if not hasattr(self, "_coset_multipliers") or self._coset_multipliers.device != device:
                    self._coset_multipliers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long, device=device).view(1, 1, 8)
                
                candidate_cosets = torch.sum(y_mod_2 * self._coset_multipliers, dim=-1)
                candidate_cosets_uniq = torch.unique(candidate_cosets.view(-1))
                
                # 4. Database cosets
                db_y = torch.round(2.0 * coords_norm).to(torch.long)
                db_y_mod_2 = torch.remainder(db_y, 2)
                db_cosets = torch.sum(db_y_mod_2 * self._coset_multipliers[0], dim=-1)
                
                # 5. Broadcast match
                match_mask = (db_cosets.unsqueeze(0) == candidate_cosets_uniq.unsqueeze(1)).any(dim=0)
                matched_db_indices = torch.where(match_mask)[0]
                
                if len(matched_db_indices) == 0:
                    if not hasattr(self, "_empty_swap_in_cache"):
                        self._empty_swap_in_cache = {}
                    cache_key = (1, d_model, device, dtype)
                    if cache_key not in self._empty_swap_in_cache:
                        self._empty_swap_in_cache[cache_key] = (
                            torch.zeros(1, d_model, device=device, dtype=dtype),
                            torch.zeros(1, d_model, device=device, dtype=dtype)
                        )
                    return self._empty_swap_in_cache[cache_key]
                
                coords_pruned = coords_norm[matched_db_indices]
                coords_norm2_pruned = coords_norm2[..., matched_db_indices]
                
                dist2 = torch.addmm(
                    q_norm2 + coords_norm2_pruned,
                    quantized,
                    coords_pruned.t(),
                    alpha=-2.0
                )
                mask = dist2 <= 2.05
                matched_mask = mask.any(dim=0)
                matched_indices_gpu = matched_db_indices[torch.where(matched_mask)[0]]
                matched_indices_cpu = matched_indices_gpu.cpu()
                
                retrieved_k = cpu_k_buf[matched_indices_cpu].to(device=device, dtype=dtype, non_blocking=True)
                retrieved_v = cpu_v_buf[matched_indices_cpu].to(device=device, dtype=dtype, non_blocking=True)
            else:
                if queries.requires_grad:
                    dist2 = torch.addmm(q_norm2 + coords_norm2, quantized, coords_norm.t(), alpha=-2.0)
                else:
                    if (not hasattr(self, "_dist2_buf") or 
                        self._dist2_buf.shape[0] < queries.shape[0] or 
                        self._dist2_buf.shape[1] < limit or 
                        self._dist2_buf.device != device or 
                        self._dist2_buf.dtype != dtype):
                        self._dist2_buf = torch.empty((queries.shape[0], max(2048, limit * 2)), device=device, dtype=dtype)
                    dist2 = torch.addmm(q_norm2 + coords_norm2, quantized, coords_norm.t(), alpha=-2.0, out=self._dist2_buf[:queries.shape[0], :limit])
                
                mask = dist2 <= 2.05
                matched_mask = mask.any(dim=0)
                matched_indices_cpu = torch.where(matched_mask)[0].cpu()
                
                if len(matched_indices_cpu) == 0:
                    if not hasattr(self, "_empty_swap_in_cache"):
                        self._empty_swap_in_cache = {}
                    cache_key = (1, d_model, device, dtype)
                    if cache_key not in self._empty_swap_in_cache:
                        self._empty_swap_in_cache[cache_key] = (
                            torch.zeros(1, d_model, device=device, dtype=dtype),
                            torch.zeros(1, d_model, device=device, dtype=dtype)
                        )
                    return self._empty_swap_in_cache[cache_key]
                    
                retrieved_k = cpu_k_buf[matched_indices_cpu].to(device=device, dtype=dtype, non_blocking=True)
                retrieved_v = cpu_v_buf[matched_indices_cpu].to(device=device, dtype=dtype, non_blocking=True)
            
            return retrieved_k, retrieved_v
        finally:
            self.mutex.release()

    def swap_in_batch(self, queries: torch.Tensor, max_matches: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.swap_in_batch_target(queries, max_matches)
        
    def swap_in_batch_target(self, queries: torch.Tensor, max_matches: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._swap_in_batch(queries, max_matches, is_draft=False)
        
    def swap_in_batch_draft(self, queries: torch.Tensor, max_matches: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._swap_in_batch(queries, max_matches, is_draft=True)
        
    def _swap_in_batch(self, queries: torch.Tensor, max_matches: int, is_draft: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            B, H, S, D = queries.shape
            device = queries.device
            dtype = queries.dtype
            
            d_model = self.d_model_draft if is_draft else self.d_model_target
            cpu_k_buf = self.cpu_k_draft if is_draft else self.cpu_k_target
            cpu_v_buf = self.cpu_v_draft if is_draft else self.cpu_v_target
            W_p_attr = "W_p_draft" if is_draft else "W_p_target"
            
            limit = self.draft_len if is_draft else self.target_len
            
            if cpu_k_buf is None or limit == 0 or self.grid_coords is None or self.grid_coords_len == 0:
                if not hasattr(self, "_empty_batch_cache"):
                    self._empty_batch_cache = {}
                cache_key = (B, H, max_matches, D, device, dtype)
                if cache_key not in self._empty_batch_cache:
                    zeros = torch.zeros(B, H, max_matches, D, device=device, dtype=dtype)
                    self._empty_batch_cache[cache_key] = (zeros, zeros)
                return self._empty_batch_cache[cache_key]
                
            cpu_k_buf = cpu_k_buf[:limit]
            cpu_v_buf = cpu_v_buf[:limit]
            grid_coords_sliced = self.grid_coords[:limit]
            
            if not hasattr(self, W_p_attr) or getattr(self, W_p_attr).device != device or getattr(self, W_p_attr).dtype != dtype:
                self._init_default_projection(device, dtype, is_draft)
                
            W_p = getattr(self, W_p_attr)
            
            # Win 88: Fused Query Projection & Distance in PyTorch Swap DB (Batch)
            if queries.requires_grad:
                queries_flat = queries.reshape(-1, D)
                queries_8d_flat = queries_flat @ W_p
                queries_8d = queries_8d_flat.reshape(B, H, S, 8)
                quantized = self._quantize(queries_8d).to(dtype=dtype)
                q_norm2 = torch.sum(quantized.square(), dim=-1, keepdim=True)
                quantized_flat = quantized.reshape(-1, 8)
            else:
                if not hasattr(self, "_queries_8d_batch_buf") or self._queries_8d_batch_buf.shape[0] < B * H * S or self._queries_8d_batch_buf.device != device or self._queries_8d_batch_buf.dtype != dtype:
                    self._queries_8d_batch_buf = torch.empty((B * H * S, 8), device=device, dtype=dtype)
                queries_flat = queries.reshape(-1, D)
                queries_8d_flat = torch.matmul(queries_flat, W_p, out=self._queries_8d_batch_buf[:B * H * S])
                queries_8d = queries_8d_flat.reshape(B, H, S, 8)
                quantized = self._quantize(queries_8d).to(dtype=dtype)
                
                if not hasattr(self, "_q_norm2_batch_buf") or self._q_norm2_batch_buf.shape[0] < B * H * S or self._q_norm2_batch_buf.device != device or self._q_norm2_batch_buf.dtype != dtype:
                    self._q_norm2_batch_buf = torch.empty((B * H * S, 1), device=device, dtype=dtype)
                quantized_flat = quantized.reshape(-1, 8)
                q_norm2_flat = torch.sum(quantized_flat.square(), dim=-1, keepdim=True, out=self._q_norm2_batch_buf[:B * H * S])
                q_norm2 = q_norm2_flat.reshape(B, H, S, 1)
            
            if (not hasattr(self, "_coords_gpu_cache") or 
                self._coords_gpu_cache.get("dtype") != dtype or 
                self._coords_gpu_cache.get("device") != device or
                self._coords_gpu_cache.get("grid_id") != id(self._grid_coords) or
                "coords_norm" not in self._coords_gpu_cache):
                coords = self.grid_coords.to(dtype=dtype, device=device)
                coords_norm2 = torch.sum(coords.square(), dim=-1)
                self._coords_gpu_cache = {
                    "dtype": dtype,
                    "device": device,
                    "grid_id": id(self._grid_coords),
                    "coords_norm": coords,
                    "coords_norm2": coords_norm2
                }
            coords_norm = self._coords_gpu_cache["coords_norm"][:limit]
            coords_norm2 = self._coords_gpu_cache["coords_norm2"][:limit].reshape(1, 1, 1, -1)
            
            # Compute distance matrix on GPU
            use_pruned = (S < 8 and limit >= 512)
            if use_pruned:
                # 1. Pre-cache shell roots + zero
                if not hasattr(self, "_shell_roots_and_zero_gpu") or self._shell_roots_and_zero_gpu.device != device or self._shell_roots_and_zero_gpu.dtype != dtype:
                    roots = self.shell_1_roots.to(device=device, dtype=dtype)
                    zero = torch.zeros((1, 8), device=device, dtype=dtype)
                    self._shell_roots_and_zero_gpu = torch.cat([roots, zero], dim=0)
                
                # 2. Candidate coordinates
                quantized_flat = quantized.reshape(-1, 8)
                candidates = quantized_flat.unsqueeze(1) + self._shell_roots_and_zero_gpu.unsqueeze(0)
                
                # 3. 2-adic Level 1 mapping
                y = torch.round(2.0 * candidates).to(torch.long)
                y_mod_2 = torch.remainder(y, 2)
                
                if not hasattr(self, "_coset_multipliers") or self._coset_multipliers.device != device:
                    self._coset_multipliers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.long, device=device).view(1, 1, 8)
                
                candidate_cosets = torch.sum(y_mod_2 * self._coset_multipliers, dim=-1)
                candidate_cosets_uniq = torch.unique(candidate_cosets.view(-1))
                
                # 4. Database cosets
                db_y = torch.round(2.0 * coords_norm).to(torch.long)
                db_y_mod_2 = torch.remainder(db_y, 2)
                db_cosets = torch.sum(db_y_mod_2 * self._coset_multipliers[0], dim=-1)
                
                # 5. Broadcast match
                match_mask = (db_cosets.unsqueeze(0) == candidate_cosets_uniq.unsqueeze(1)).any(dim=0)
                matched_db_indices = torch.where(match_mask)[0]
                
                if len(matched_db_indices) == 0:
                    cache_key = (B, H, max_matches, D, device, dtype)
                    if not hasattr(self, "_empty_batch_cache"):
                        self._empty_batch_cache = {}
                    if cache_key not in self._empty_batch_cache:
                        zeros = torch.zeros(B, H, max_matches, D, device=device, dtype=dtype)
                        self._empty_batch_cache[cache_key] = (zeros, zeros)
                    return self._empty_batch_cache[cache_key]
                
                coords_pruned = coords_norm[matched_db_indices]
                coords_norm2_pruned = coords_norm2[..., matched_db_indices]
                
                dist2_flat = torch.addmm(
                    (q_norm2 + coords_norm2_pruned).reshape(-1, coords_pruned.shape[0]),
                    quantized_flat,
                    coords_pruned.t(),
                    alpha=-2.0
                )
                dist2 = dist2_flat.reshape(B, H, S, -1)
                M_pruned = dist2.shape[-1]
                dist2_flat_view = dist2.reshape(B, H, S * M_pruned)
                k_val = min(max_matches, S * M_pruned)
                
                topk_val, topk_idx = torch.topk(dist2_flat_view, k_val, dim=-1, largest=False)
                is_neighbor = topk_val <= 2.05
                db_idx = topk_idx % M_pruned
                
                db_idx = matched_db_indices[db_idx]
                db_idx_cpu = db_idx.cpu()
                
                matched_k_cpu = cpu_k_buf[db_idx_cpu]
                matched_v_cpu = cpu_v_buf[db_idx_cpu]
                
                matched_k = matched_k_cpu.to(device=device, dtype=dtype, non_blocking=True)
                matched_v = matched_v_cpu.to(device=device, dtype=dtype, non_blocking=True)
                
                neighbor_mask = is_neighbor.unsqueeze(-1)
                matched_k = matched_k * neighbor_mask
                matched_v = matched_v * neighbor_mask
            else:
                if queries.requires_grad:
                    dist2_flat = torch.addmm((q_norm2 + coords_norm2).reshape(-1, coords_norm.shape[0]), quantized_flat, coords_norm.t(), alpha=-2.0)
                else:
                    if (not hasattr(self, "_dist2_batch_buf") or 
                        self._dist2_batch_buf.shape[0] < B * H * S or 
                        self._dist2_batch_buf.shape[1] < limit or 
                        self._dist2_batch_buf.device != device or 
                        self._dist2_batch_buf.dtype != dtype):
                        self._dist2_batch_buf = torch.empty((B * H * S, max(2048, limit * 2)), device=device, dtype=dtype)
                    dist2_flat = torch.addmm((q_norm2 + coords_norm2).reshape(-1, coords_norm.shape[0]), quantized_flat, coords_norm.t(), alpha=-2.0, out=self._dist2_batch_buf[:B * H * S, :limit])
                dist2 = dist2_flat.reshape(B, H, S, -1)
                
                M = dist2.shape[-1]
                dist2_flat_view = dist2.reshape(B, H, S * M)
                k_val = min(max_matches, S * M)
                
                topk_val, topk_idx = torch.topk(dist2_flat_view, k_val, dim=-1, largest=False)
                is_neighbor = topk_val <= 2.05
                db_idx = topk_idx % M
                db_idx_cpu = db_idx.cpu()
                
                matched_k_cpu = cpu_k_buf[db_idx_cpu]
                matched_v_cpu = cpu_v_buf[db_idx_cpu]
                
                matched_k = matched_k_cpu.to(device=device, dtype=dtype, non_blocking=True)
                matched_v = matched_v_cpu.to(device=device, dtype=dtype, non_blocking=True)
                
                neighbor_mask = is_neighbor.unsqueeze(-1)
                matched_k = matched_k * neighbor_mask
                matched_v = matched_v * neighbor_mask
            
            if k_val < max_matches:
                pad_len = max_matches - k_val
                cache_key = (B, H, pad_len, D, device, dtype)
                if not hasattr(self, "_empty_gpu_pad_cache"):
                    self._empty_gpu_pad_cache = {}
                if cache_key not in self._empty_gpu_pad_cache:
                    self._empty_gpu_pad_cache[cache_key] = torch.zeros(B, H, pad_len, D, device=device, dtype=dtype)
                pad_zeros = self._empty_gpu_pad_cache[cache_key]
                
                matched_k = torch.cat([matched_k, pad_zeros], dim=2)
                matched_v = torch.cat([matched_v, pad_zeros], dim=2)
                
            return matched_k, matched_v
        finally:
            self.mutex.release()


class CoWMemorySwapGridDB(AdelicMemorySwapGridDB):
    def __init__(self, parent: AdelicMemorySwapGridDB, lock_path: str = None):
        super().__init__(
            d_model=parent.d_model,
            device=parent.device,
            cache_limit_ratio=parent.cache_limit_ratio,
            d_model_draft=parent.d_model_draft,
            lock_path=lock_path if lock_path is not None else "/tmp/qan_e8_cow.lock"
        )
        self.parent = parent
        
        # Share projections
        if hasattr(parent, "W_p_target"):
            self.W_p_target = parent.W_p_target
        if hasattr(parent, "W_p_draft"):
            self.W_p_draft = parent.W_p_draft

    def _swap_in(self, queries: torch.Tensor, is_draft: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_pending_swaps()
        if self.parent is not None:
            self.parent._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            p_limit = self.parent.draft_len if is_draft else self.parent.target_len
            l_limit = self.draft_len if is_draft else self.target_len
            
            p_coords = self.parent.grid_coords[:p_limit] if self.parent.grid_coords is not None else None
            l_coords = self.grid_coords[:l_limit] if self.grid_coords is not None else None
            
            if p_coords is None or p_coords.shape[0] == 0:
                # Local only
                self.mutex.release()
                try:
                    return super()._swap_in(queries, is_draft)
                finally:
                    self.mutex.acquire()
            if l_coords is None or l_coords.shape[0] == 0:
                # Parent only
                return self.parent._swap_in(queries, is_draft)
                
            d_model = self.d_model_draft if is_draft else self.d_model_target
            p_cpu_k = self.parent.cpu_k_draft if is_draft else self.parent.cpu_k_target
            p_cpu_v = self.parent.cpu_v_draft if is_draft else self.parent.cpu_v_target
            l_cpu_k = self.cpu_k_draft if is_draft else self.cpu_k_target
            l_cpu_v = self.cpu_v_draft if is_draft else self.cpu_v_target
            
            device = queries.device
            dtype = queries.dtype
            W_p_attr = "W_p_draft" if is_draft else "W_p_target"
            
            if not hasattr(self, W_p_attr) or getattr(self, W_p_attr).device != device or getattr(self, W_p_attr).dtype != dtype:
                self._init_default_projection(device, dtype, is_draft)
            
            W_p = getattr(self, W_p_attr)
            queries_8d = queries @ W_p
            quantized = self._quantize(queries_8d)
            
            # Perform search and indexing entirely on CPU to avoid syncs
            quantized_cpu = quantized.to("cpu")
            q_norm2_cpu = torch.sum(quantized_cpu ** 2, dim=-1, keepdim=True)
            
            # Get parent cached norms/squares
            if (not hasattr(self.parent, "_coords_cache") or 
                self.parent._coords_cache.get("dtype") != dtype or
                "coords_norm" not in self.parent._coords_cache):
                p_coords_all = self.parent.grid_coords.to(dtype=dtype, device="cpu")
                self.parent._coords_cache = {
                    "dtype": dtype,
                    "grid_id": id(self.parent._grid_coords),
                    "coords_norm": p_coords_all,
                    "coords_norm2": torch.sum(p_coords_all ** 2, dim=-1)
                }
            p_norm = self.parent._coords_cache["coords_norm"][:p_limit]
            p_norm2 = self.parent._coords_cache["coords_norm2"][:p_limit]
            
            # Get local cached norms/squares
            if (not hasattr(self, "_coords_cache") or 
                self._coords_cache.get("dtype") != dtype or
                "coords_norm" not in self._coords_cache):
                l_coords_all = self.grid_coords.to(dtype=dtype, device="cpu")
                self._coords_cache = {
                    "dtype": dtype,
                    "grid_id": id(self._grid_coords),
                    "coords_norm": l_coords_all,
                    "coords_norm2": torch.sum(l_coords_all ** 2, dim=-1)
                }
            l_norm = self._coords_cache["coords_norm"][:l_limit]
            l_norm2 = self._coords_cache["coords_norm2"][:l_limit]
            
            coords_norm_cpu = torch.cat([p_norm, l_norm], dim=0)
            coords_norm2_cpu = torch.cat([p_norm2, l_norm2], dim=0).unsqueeze(0)
            
            dist2_cpu = torch.addmm(q_norm2_cpu + coords_norm2_cpu, quantized_cpu, coords_norm_cpu.t(), alpha=-2.0)
            
            mask_cpu = dist2_cpu <= 2.05
            matched_mask_cpu = mask_cpu.any(dim=0)
            matched_indices_cpu = torch.where(matched_mask_cpu)[0]
            
            if len(matched_indices_cpu) == 0:
                return (torch.zeros(1, d_model, device=device, dtype=dtype),
                        torch.zeros(1, d_model, device=device, dtype=dtype))
                        
            # Avoid CPU concatenations of full parent and local databases
            retrieved_k_cpu = torch.empty((matched_indices_cpu.shape[0], d_model), dtype=p_cpu_k.dtype, device="cpu")
            retrieved_v_cpu = torch.empty((matched_indices_cpu.shape[0], d_model), dtype=p_cpu_v.dtype, device="cpu")
            
            is_parent = matched_indices_cpu < p_limit
            parent_indices = matched_indices_cpu[is_parent]
            local_indices = matched_indices_cpu[~is_parent] - p_limit
            
            if len(parent_indices) > 0:
                retrieved_k_cpu[is_parent] = p_cpu_k[:p_limit][parent_indices]
                retrieved_v_cpu[is_parent] = p_cpu_v[:p_limit][parent_indices]
            if len(local_indices) > 0:
                retrieved_k_cpu[~is_parent] = l_cpu_k[:l_limit][local_indices]
                retrieved_v_cpu[~is_parent] = l_cpu_v[:l_limit][local_indices]
                
            retrieved_k = retrieved_k_cpu.to(device=device, dtype=dtype, non_blocking=True)
            retrieved_v = retrieved_v_cpu.to(device=device, dtype=dtype, non_blocking=True)
            return retrieved_k, retrieved_v
        finally:
            self.mutex.release()

    def _swap_in_batch(self, queries: torch.Tensor, max_matches: int, is_draft: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        self._wait_for_pending_swaps()
        if self.parent is not None:
            self.parent._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            p_limit = self.parent.draft_len if is_draft else self.parent.target_len
            l_limit = self.draft_len if is_draft else self.target_len
            
            p_coords = self.parent.grid_coords[:p_limit] if self.parent.grid_coords is not None else None
            l_coords = self.grid_coords[:l_limit] if self.grid_coords is not None else None
            
            if p_coords is None or p_coords.shape[0] == 0:
                self.mutex.release()
                try:
                    return super()._swap_in_batch(queries, max_matches, is_draft)
                finally:
                    self.mutex.acquire()
            if l_coords is None or l_coords.shape[0] == 0:
                return self.parent._swap_in_batch(queries, max_matches, is_draft)
                
            B, H, S, D = queries.shape
            device = queries.device
            dtype = queries.dtype
            
            d_model = self.d_model_draft if is_draft else self.d_model_target
            p_cpu_k = self.parent.cpu_k_draft if is_draft else self.parent.cpu_k_target
            p_cpu_v = self.parent.cpu_v_draft if is_draft else self.parent.cpu_v_target
            l_cpu_k = self.cpu_k_draft if is_draft else self.cpu_k_target
            l_cpu_v = self.cpu_v_draft if is_draft else self.cpu_v_target
            
            W_p_attr = "W_p_draft" if is_draft else "W_p_target"
            
            if not hasattr(self, W_p_attr) or getattr(self, W_p_attr).device != device or getattr(self, W_p_attr).dtype != dtype:
                self._init_default_projection(device, dtype, is_draft)
                
            W_p = getattr(self, W_p_attr)
            queries_8d = queries @ W_p
            quantized = self._quantize(queries_8d)
            
            # Perform search and indexing entirely on CPU to avoid syncs
            quantized_cpu = quantized.to("cpu")
            q_norm2_cpu = torch.sum(quantized_cpu ** 2, dim=-1, keepdim=True)
            
            # Get parent cached norms/squares
            if (not hasattr(self.parent, "_coords_cache") or 
                self.parent._coords_cache.get("dtype") != dtype or
                "coords_norm" not in self.parent._coords_cache):
                p_coords_all = self.parent.grid_coords.to(dtype=dtype, device="cpu")
                self.parent._coords_cache = {
                    "dtype": dtype,
                    "grid_id": id(self.parent._grid_coords),
                    "coords_norm": p_coords_all,
                    "coords_norm2": torch.sum(p_coords_all ** 2, dim=-1)
                }
            p_norm = self.parent._coords_cache["coords_norm"][:p_limit]
            p_norm2 = self.parent._coords_cache["coords_norm2"][:p_limit]
            
            # Get local cached norms/squares
            if (not hasattr(self, "_coords_cache") or 
                self._coords_cache.get("dtype") != dtype or
                "coords_norm" not in self._coords_cache):
                l_coords_all = self.grid_coords.to(dtype=dtype, device="cpu")
                self._coords_cache = {
                    "dtype": dtype,
                    "grid_id": id(self._grid_coords),
                    "coords_norm": l_coords_all,
                    "coords_norm2": torch.sum(l_coords_all ** 2, dim=-1)
                }
            l_norm = self._coords_cache["coords_norm"][:l_limit]
            l_norm2 = self._coords_cache["coords_norm2"][:l_limit]
            
            coords_norm_cpu = torch.cat([p_norm, l_norm], dim=0)
            coords_norm2_cpu = torch.cat([p_norm2, l_norm2], dim=0).view(1, 1, 1, -1)
            
            quantized_flat_cpu = quantized_cpu.view(-1, 8)
            dist2_flat_cpu = torch.addmm((q_norm2_cpu + coords_norm2_cpu).view(-1, coords_norm_cpu.shape[0]), quantized_flat_cpu, coords_norm_cpu.t(), alpha=-2.0)
            dist2_cpu = dist2_flat_cpu.view(B, H, S, -1)
            
            M_total = dist2_cpu.shape[-1]
            dist2_flat_cpu = dist2_cpu.reshape(B, H, S * M_total)
            k_val = min(max_matches, S * M_total)
            
            topk_val_cpu, topk_idx_cpu = torch.topk(dist2_flat_cpu, k_val, dim=-1, largest=False)
            is_neighbor_cpu = topk_val_cpu <= 2.05
            db_idx_cpu = topk_idx_cpu % M_total
            
            # Avoid CPU concatenations of full parent and local databases
            is_parent = db_idx_cpu < p_limit
            matched_k_cpu = torch.empty((*db_idx_cpu.shape, d_model), dtype=p_cpu_k.dtype, device="cpu")
            matched_v_cpu = torch.empty((*db_idx_cpu.shape, d_model), dtype=p_cpu_v.dtype, device="cpu")
            
            db_idx_flat = db_idx_cpu.view(-1)
            is_parent_flat = db_idx_flat < p_limit
            parent_indices = db_idx_flat[is_parent_flat]
            local_indices = db_idx_flat[~is_parent_flat] - p_limit
            
            matched_k_flat = matched_k_cpu.view(-1, d_model)
            matched_v_flat = matched_v_cpu.view(-1, d_model)
            
            if len(parent_indices) > 0:
                matched_k_flat[is_parent_flat] = p_cpu_k[:p_limit][parent_indices]
                matched_v_flat[is_parent_flat] = p_cpu_v[:p_limit][parent_indices]
            if len(local_indices) > 0:
                matched_k_flat[~is_parent_flat] = l_cpu_k[:l_limit][local_indices]
                matched_v_flat[~is_parent_flat] = l_cpu_v[:l_limit][local_indices]
                
            matched_k_cpu = matched_k_cpu * is_neighbor_cpu.unsqueeze(-1)
            matched_v_cpu = matched_v_cpu * is_neighbor_cpu.unsqueeze(-1)
            
            if k_val < max_matches:
                k_out_cpu = torch.zeros(B, H, max_matches, d_model, device="cpu", dtype=dtype)
                v_out_cpu = torch.zeros(B, H, max_matches, d_model, device="cpu", dtype=dtype)
                k_out_cpu[:, :, :k_val] = matched_k_cpu
                v_out_cpu[:, :, :k_val] = matched_v_cpu
                
                matched_k = k_out_cpu.to(device=device, dtype=dtype, non_blocking=True)
                matched_v = v_out_cpu.to(device=device, dtype=dtype, non_blocking=True)
            else:
                matched_k = matched_k_cpu.to(device=device, dtype=dtype, non_blocking=True)
                matched_v = matched_v_cpu.to(device=device, dtype=dtype, non_blocking=True)
                
            return matched_k, matched_v
        finally:
            self.mutex.release()

    def merge_to_parent(self):
        self._wait_for_pending_swaps()
        if self.parent is not None:
            self.parent._wait_for_pending_swaps()
        self.mutex.acquire()
        try:
            self.parent.mutex.acquire()
            try:
                if self._cpu_k_target is not None and self.target_len > 0:
                    if self.parent.grid_coords is None or self.parent.grid_coords_len == 0:
                        self.parent.grid_coords = self.grid_coords[:self.grid_coords_len].clone()
                        self.parent.grid_coords_len = self.grid_coords_len
                        self.parent.grid_coords_capacity = self.grid_coords_len
                        self.parent.cpu_k_target = self.cpu_k_target.clone()
                        self.parent.cpu_v_target = self.cpu_v_target.clone()
                        self.parent.cpu_k = self.parent.cpu_k_target
                        self.parent.cpu_v = self.parent.cpu_v_target
                    else:
                        occupied_set = {tuple(coord.tolist()) for coord in self.parent.grid_coords[:self.parent.grid_coords_len]}
                        merged_coords = []
                        shell_1_roots_np = get_shared_shell_1_roots()
                        
                        # Win 102: Unified Host Transfer to avoid individual GPU-to-CPU copies
                        grid_coords_cpu = self.grid_coords[:self.grid_coords_len].cpu().numpy()
                        for coord_np in grid_coords_cpu:
                            coord_tuple = tuple(coord_np.tolist())
                            if coord_tuple not in occupied_set:
                                merged_coords.append(torch.from_numpy(coord_np).to(device=self.parent.grid_coords.device, dtype=self.parent.grid_coords.dtype))
                                occupied_set.add(coord_tuple)
                            else:
                                relocated = False
                                candidates = np.round((coord_np[np.newaxis, :] + shell_1_roots_np) * 2.0) / 2.0
                                for cand_np in candidates:
                                    cand_tuple = tuple(cand_np.tolist())
                                    if cand_tuple not in occupied_set:
                                        merged_coords.append(torch.tensor(cand_np, dtype=self.parent.grid_coords.dtype, device=self.parent.grid_coords.device))
                                        occupied_set.add(cand_tuple)
                                        relocated = True
                                        break
                                if not relocated:
                                    merged_coords.append(torch.from_numpy(coord_np).to(device=self.parent.grid_coords.device, dtype=self.parent.grid_coords.dtype))
                                    
                        merged_coords_tensor = torch.stack(merged_coords, dim=0)
                        
                        self.parent.grid_coords = torch.cat([self.parent.grid_coords[:self.parent.grid_coords_len], merged_coords_tensor], dim=0)
                        self.parent.grid_coords_len = self.parent.grid_coords.shape[0]
                        self.parent.grid_coords_capacity = self.parent.grid_coords.shape[0]
                        self.parent.cpu_k_target = torch.cat([self.parent.cpu_k_target, self.cpu_k_target], dim=0)
                        self.parent.cpu_v_target = torch.cat([self.parent.cpu_v_target, self.cpu_v_target], dim=0)
                        self.parent.cpu_k = self.parent.cpu_k_target
                        self.parent.cpu_v = self.parent.cpu_v_target
                        
                if self._cpu_k_draft is not None and self.draft_len > 0:
                    if self.parent.cpu_k_draft is None or self.parent.draft_len == 0:
                        self.parent.cpu_k_draft = self.cpu_k_draft.clone()
                        self.parent.cpu_v_draft = self.cpu_v_draft.clone()
                    else:
                        self.parent.cpu_k_draft = torch.cat([self.parent.cpu_k_draft, self.cpu_k_draft], dim=0)
                        self.parent.cpu_v_draft = torch.cat([self.parent.cpu_v_draft, self.cpu_v_draft], dim=0)
                        
                self.cpu_k_target = None
                self.cpu_v_target = None
                self.cpu_k_draft = None
                self.cpu_v_draft = None
                self.grid_coords = None
                self.cpu_k = None
                self.cpu_v = None
            finally:
                self.parent.mutex.release()
        finally:
            self.mutex.release()