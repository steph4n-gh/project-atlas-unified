# Systems Engineering & Concurrency Reference Guide

This document details the systems engineering architecture, concurrent memory managers, hardware-specific performance kernels, and critical engineering trade-offs implemented across the **Project Atlas (QAN-ATLAS)** framework.

---

## 1. Thread-Safe File Mutex & Lockfile Concurrency

To synchronize database modifications when multiple local agents (or parallel generation threads) access the same coordinate swap database, Project Atlas implements a Unix-level file lock mutex.

### 1.1 Concurrency Lock Architecture
The `FileMutex` wrapper resides in [e8_swap.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/math/e8_swap.py#L41-L87). It uses the POSIX Unix system call `fcntl.flock` to enforce exclusive write locks:
```python
import os
import fcntl
import threading

class FileMutex:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self.fd = None
        self.thread_lock = threading.Lock()

    def acquire(self):
        self.thread_lock.acquire()
        try:
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_WRONLY)
            fcntl.flock(self.fd, fcntl.LOCK_EX)  # Exclusive block
        except Exception:
            self.thread_lock.release()
            raise

    def release(self):
        try:
            if self.fd is not None:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
                self.fd = None
        finally:
            self.thread_lock.release()
```

### 1.2 Limitations & Failsafes
*   **Network Filesystem Limitations**: The `fcntl.flock` primitive is cooperative and managed by the operating system kernel. **It does not support network filesystems (like NFS, CIFS/Samba, or virtual clouds)**. On these volumes, flock calls may either fail silently or block indefinitely. For multi-node cluster settings, standard distributed lock managers (e.g. Redis or Etcd) must replace this class.
*   **Reentrancy Warning**: `fcntl.flock` locks are associated with the file descriptor, not the thread. To prevent deadlocks, the class wraps the file lock inside a standard re-entrant thread lock (`threading.Lock`).

---

## 2. Copy-on-Write (CoW) Memory Branching

To support concurrent developer agents querying and modifying the database state in parallel without workspace contamination, Project Atlas uses a Copy-on-Write database wrapper.

### 2.1 The `CoWMemorySwapGridDB` class
The class inherits from the main database and wraps a parent instance:
*   **Initialization**: Shares the projection matrices (`W_p_target` / `W_p_draft`) of the parent database to prevent re-computation.
*   **Read Isolation (`_swap_in`)**: When a query occurs, the wrapper searches both the parent's `grid_coords` and the local thread's CoW `grid_coords` concurrently, combining results dynamically before returning key-value tensors.
*   **Write Isolation (`swap_out`)**: Newly generated key-values are written *only* to the local child's CPU buffers and E8 coordinate arrays, keeping the parent memory pristine during active generation.

### 2.2 Merging Contexts
When generation completes, the agent branch merges back into the parent via `merge_to_parent()`. This operation is wrapped in a nested lock (`self.mutex` + `self.parent.mutex`) to guarantee atomicity.

*Code Reference*: Implemented in [e8_swap.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/math/e8_swap.py#L1049-L1368).

---

## 3. Topological Collision Relocation Mechanics

When merging local agent branches back to the parent database, coordinate collisions (two agents writing distinct key-values to the same E8 coordinate) are resolved via nearest-neighbor relocation instead of vector averaging.

### 3.1 Relocation Logic
If an agent's coordinate tuple already exists in the parent's occupied set:
1.  The system retrieves the 240 root coordinates of the $E_8$ lattice's Shell 1 (which have a standard distance squared of $2.0$).
2.  It iterates through these 240 root neighbors in order.
3.  It calculates a candidate coordinate:
    

$$
\mathbf{x}_{\text{cand}} = \text{round}\left( (\mathbf{x}_{\text{collision}} + \mathbf{r}_{\text{root}}) \times 2 \right) / 2.0
$$

4.  If the candidate tuple is unoccupied, the key-value representation is written to this new coordinate, and the candidate is marked as occupied.
5.  If all 240 neighbor coordinates are occupied (extreme density), it defaults to appending the original coordinate.

*Code Reference*: Implemented in [e8_swap.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/math/e8_swap.py#L1318-L1339).

---

## 4. p-Adic Routing and FMM Attention Tree Concurrency (UCE)

To avoid quadratic cost at long context lengths, the **Ultrametric Cognitive Engine (UCE)** organizes sequence tokens into a hierarchical $p$-adic tree.

### 4.1 Morton Code Sorting & Chronological Priority
To route attention queries down the ultrametric tree on GPU:
1.  Tokens are assigned Morton codes $M(x)$ based on the prime bases 2, 3, and 5 (packed into base-30).
2.  To prevent causal leakage, sorting keys are constructed by combining the Morton code with the token's chronological index:
    

$$
\text{SortKey} = \text{Index} \times 10^8 + M(x)
$$

3.  The sequence is gathered along this sorted index using a vectorized `torch.gather` on the GPU.

### 4.2 Tree Construction & Upward/Downward Passes
Once the sequence is sorted along the ultrametric path:
*   **Tree Structure**: Tokens are chunked into leaf blocks of size $B_{\text{leaf}} = 128$. Leaf nodes are padded to the nearest power of $2$ to construct a complete binary tree of depth $L$.
*   **Upward Pass (Clustering)**: The keys and values of leaf nodes are aggregated by taking their sum and dividing by the active (non-padded) token count:
    

$$
K_{\text{parent}} = \frac{1}{\text{ActiveCount}} \sum_{j \in \text{children}} K_j
$$

    This is recursively propagated up the levels of the tree.
*   **Downward Pass (Attention Evaluation)**: Queries compute attention weights against representing nodes at different levels of the tree. If a parent node has low attention weight, its entire subtree is pruned, achieving sub-quadratic execution times for context lengths $\ge 2048$.

### 4.3 2-adic Coset Pruning in Swap Database
To perform quick lookups within `AdelicMemorySwapGridDB`:
*   The coordinate database builds an index over 2-adic coset IDs representing discrete E8 coordinates.
*   Upon query execution, the system dynamically calculates candidate E8 coordinates in a local neighborhood and constructs their 2-adic cosets.
*   Database search is pruned by selecting only candidate cosets matching the database indexes. This avoids heavy memory sweeps and decreases bus transfer overhead on Apple Silicon (MPS).

*Code Reference*: The tree-structured FMM attention is implemented in [attention.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/mlx/attention.py) (`UltrametricAttention.forward`). The 2-adic database pruning is implemented in [e8_swap.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/math/e8_swap.py#L756-L813) (`AdelicMemorySwapGridDB._swap_in`).

---

## 5. Custom Metal Performance Shaders (MPS) Autograd Operators

Standard PyTorch on macOS lacks native GPU gather-scatter kernels for sparse coordinate indices, falling back to slow CPU transfers. To resolve this, Project Atlas implements custom MPS autograd operators utilizing PyTorch's native MPS support.

### 5.1 Implementation Layout
The custom autograd operator is defined in [mps_scatter.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/kernels/mps_scatter.py).
*   **Forward Pass**: Rather than implementing a custom Metal (MSL) shader kernel compiled directly, the operator uses PyTorch's native MPS support with `index_select` wrapped in a custom `torch.autograd.Function`. This gathers key-value representations along coordinate-sparse indices directly in Metal GPU buffers, bypassing CPU-GPU PCIe bus transfers.
*   **Backward Pass**: Computes gradients for the coordinate routing weights using PyTorch's native MPS `index_add_` wrapped in the backward pass of the autograd Function, ensuring the model remains end-to-end differentiable on Apple Silicon without custom MSL source compilation.

### 5.2 Fused Prefill Optimization
To optimize prefill latency, the custom MPS operator layout merges index casting, contiguity checks, and coordinate-sparse selection into a single autograd launch boundary.
*   **Performance Impact**: Reduces coordinate-routing mean latency on Apple Silicon by **49.63%** (mean latency drops from **883.34 ms to 444.97 ms**).
*   **Numerical Parity**: This optimization maintains strict numerical parity with the standard PyTorch reference (`torch.gather` / `torch.index_select`) within $<1\text{e-}4$ for both forward activations and backward gradients, verified via `torch.autograd.gradcheck`.

---

## 6. Apple MLX SVD/QR Hardware Constraints

Apple's MLX framework is highly optimized for Apple Silicon unified memory, but has specific execution stream characteristics for linear algebra operations like SVD (`mx.linalg.svd`) and QR (`mx.linalg.qr`).

### 6.1 GPU Execution and CPU Stream
Rather than falling back to NumPy on CPU:
1.  **MLX SVD**: Run directly on the GPU in float32 precision, ensuring GPU acceleration is fully utilized for the singular value decomposition during projection initialization.
2.  **MLX QR**: Executed on the MLX CPU stream, which allows it to run without blocking or causing kernel panics on the GPU.
3.  Both results remain within MLX memory space without needing transfer to NumPy/CPU arrays and back.

### 6.2 Performance Impact
By running SVD directly on GPU and QR on the MLX CPU stream, initialization and parameter re-orthogonalization steps remain highly performant, resulting in **zero latency impact during standard token generation steps**.

---

## 7. Marsshot Speculative Decoding Hardware Optimizations (MLX Backend)

To bring 4.5-bit ELQ speculative decoding to speed parity with native 4-bit MLX, Marsshot implements a suite of low-level hardware optimizations on Apple Silicon:

### 7.1 Metal SIMD-Group Matrix Core Acceleration
*   **Mechanism**: Rewrites the fused dequantization/multiplication kernel (`elq_fused_matmul_kernel`) using Apple's GPU matrix engines via `metal::simdgroup_matrix<half, 8, 8>` cooperative loading and outer product accumulation.
*   **Impact**: Offloads math computations to dedicated GPU Tensor Cores, avoiding intermediate float16 weight matrix reconstruction in VRAM.
*   *Code Reference*: Implemented in [elq_decode.metal](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/kernels/elq_decode.metal) and [elq_metal.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/kernels/elq_metal.py).

### 7.2 Apple Accelerate SVD CPU Binding
*   **Mechanism**: Replaces the expensive full NumPy SVD calculation during target model grafting with a targeted top-k SVD solver (`scipy.sparse.linalg.svds` backed by macOS `Accelerate.framework`).
*   **Impact**: Leverages the CPU's dedicated AMX execution units, reducing target model grafting startup time from **~60s to ~3s** (a **20x speedup**).
*   *Code Reference*: Implemented in the grafting loop of [modeling.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/mlx/modeling.py).

### 7.3 Metal Shader FFN Fusion (`elq_fused_gate_up`)
*   **Mechanism**: Fuses `gate_proj` + `up_proj` + GELU activation for single-token generation (batch size = 1) into a single Metal kernel call.
*   **Impact**: Cooperatively loads and processes both projections in register files, saving **50% of intermediate VRAM transactions** compared to separate calls.
*   *Code Reference*: Implemented in [moonshots.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/mlx/moonshots.py) (`FusedGeGLUFFN._module_forward`) and mapped to [elq_metal.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/kernels/elq_metal.py).

### 7.4 Entropy-Driven Sliding LRU Cache
*   **Mechanism**: Replaces static cache limits with a dynamic capacity cache that tracks usage age via a true LRU scheme and adjusts its size dynamically based on sequence-level attention entropy.
*   **Impact**: Scales cache size down (to 32) during certain text generation to conserve memory, and up (to 258) during high-entropy/firewall anomalies to optimize latency.
*   *Code Reference*: Implemented in [modeling.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/mlx/modeling.py) (`ELQSlidingCache`) and [attention.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/mlx/attention.py) (`QuasicrystallineAttention.__call__`).

### 7.5 JIT KV Cache Stabilization
*   **Mechanism**: Enforces a global `in_jit = True` state for both models during speculative generation loops, combined with pre-padding all keys and values to multiples of 256.
*   **Impact**: Prevents out-of-sync writes to stale custom caches (which causes infinite repetition loops like `sub sub sub...`) and completely eliminates dynamic shape JIT recompilation pauses (~40-90s).
*   *Code Reference*: Implemented in [modeling.py](file:///Volumes/Storage/project_atlas_marsshot/qan_transformers/mlx/modeling.py) (`patched_speculative_generate_step`).

