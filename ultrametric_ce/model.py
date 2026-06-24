"""UCEModel: toy skeleton orchestrator.

embed (per-ball + leaf activation injection from previous) ->
UltrametricDiffusion stack ->
DigitHeads (per-depth, using ball states for conditional paths) ->
reconstruct distribution over registered leaves (via digit product).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import mmap
import os
from collections import OrderedDict
import numpy as np

import mlx.core as mx
import mlx.nn as nn

from ultrametric_ce.padic import address_to_digits
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.diffusion import UltrametricDiffusion
from ultrametric_ce.routing import DigitHeads


__all__ = ["UCEModel", "WeightManager", "PagedEmbedding"]


class WeightManager:
    """Manages LRU caching and dynamic paging of massive model weights from disk."""
    
    def __init__(
        self,
        weight_file_path: str,
        num_balls: int,
        dim: int,
        dtype: mx.Dtype = mx.float16,
        max_vram_bytes: int = 3 * 1024 * 1024 * 1024,  # Default: 3 GB VRAM ceiling
        use_mmap: bool = True
    ) -> None:
        self.weight_file_path = weight_file_path
        self.num_balls = num_balls
        self.dim = dim
        self.dtype = dtype
        self.max_vram_bytes = max_vram_bytes
        self.use_mmap = use_mmap
        
        # Parameter layout sizes
        self.itemsize = 2 if dtype == mx.float16 else 4
        self.embedding_size_bytes = self.dim * self.itemsize
        self.total_file_size = self.num_balls * self.embedding_size_bytes
        self.base_offset = 0
        
        # LRU cache: index (int) -> mx.array (shape: [dim])
        self.cache: OrderedDict[int, mx.array] = OrderedDict()
        self.current_vram_bytes = 0
        
        # Disk resources
        self.file_handle = None
        self.mmap_obj = None
        self._initialize_resources()

    def _initialize_resources(self) -> None:
        """Opens file handle, parses safetensors header, and maps it into address space."""
        self.file_handle = open(self.weight_file_path, "rb")
        
        # Parse safetensors header to find absolute tensor offset and dtype
        import struct
        import json
        self.file_handle.seek(0)
        h_size_bytes = self.file_handle.read(8)
        if len(h_size_bytes) == 8:
            try:
                h_size = struct.unpack("<Q", h_size_bytes)[0]
                if h_size > 10 * 1024 * 1024:  # Defensive size check to prevent MemoryError
                    raise ValueError("Safetensors header size is unreasonably large")
                header_bytes = self.file_handle.read(h_size)
                if len(header_bytes) == h_size:
                    header = json.loads(header_bytes.decode("utf-8"))
                    
                    # Find ball_embed.weight details
                    be_info = header.get("ball_embed.weight", {})
                    offsets = be_info.get("data_offsets", [0, 0])
                    self.base_offset = 8 + h_size + offsets[0]
                    
                    # Detect dtype from header
                    be_dtype_str = be_info.get("dtype", "F32")
                    if be_dtype_str == "F16":
                        self.dtype = mx.float16
                        self.itemsize = 2
                    else:
                        self.dtype = mx.float32
                        self.itemsize = 4
                    self.embedding_size_bytes = self.dim * self.itemsize
                else:
                    self.base_offset = 0
            except Exception:
                self.base_offset = 0
        else:
            self.base_offset = 0

            
        if self.use_mmap:
            self.mmap_obj = mmap.mmap(
                self.file_handle.fileno(), 
                0, 
                access=mmap.ACCESS_READ
            )

    def close(self) -> None:
        """Closes open files and releases mmap allocations."""
        if self.mmap_obj is not None:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def prefetch(self, indices: List[int]) -> None:
        """Loads a batch of indices from disk, sorting them to minimize seek latency."""
        # 1. Deduplicate indices and filter out already cached ones
        needed = [idx for idx in set(indices) if 0 <= idx < self.num_balls and idx not in self.cache]
        if not needed:
            return
            
        # 2. Sort to make disk access sequential (critical for mechanical/SSD seek times)
        needed.sort()
        
        # 3. Load the batch, evicting space per-element to strictly obey VRAM ceiling
        for idx in needed:
            self._evict_if_needed(self.embedding_size_bytes)
            arr = self._load_from_disk(idx)
            self.cache[idx] = arr
            self.current_vram_bytes += self.embedding_size_bytes

    def get_embedding(self, index: int) -> mx.array:
        """Returns the embedding for the index. Updates LRU access queue."""
        if index < 0 or index >= self.num_balls:
            raise IndexError(f"Index {index} out of bounds for num_balls {self.num_balls}")
            
        if index in self.cache:
            # Move to end of OrderedDict to mark as most recently used
            self.cache.move_to_end(index)
            return self.cache[index]
            
        # Cache miss fallback
        self._evict_if_needed(self.embedding_size_bytes)
        arr = self._load_from_disk(index)
        self.cache[index] = arr
        self.current_vram_bytes += self.embedding_size_bytes
        return arr

    def _load_from_disk(self, index: int) -> mx.array:
        """Loads a single vector from disk. Handles raw bytes -> mlx conversion."""
        offset = self.base_offset + index * self.embedding_size_bytes
        
        if self.use_mmap:
            # Retrieve memoryview slice directly from mapped memory without copies
            buf = memoryview(self.mmap_obj)[offset : offset + self.embedding_size_bytes]
            np_arr = np.frombuffer(buf, dtype=np.float16 if self.dtype == mx.float16 else np.float32).copy()
            del buf
        else:
            self.file_handle.seek(offset)
            raw_bytes = self.file_handle.read(self.embedding_size_bytes)
            np_arr = np.frombuffer(raw_bytes, dtype=np.float16 if self.dtype == mx.float16 else np.float32).copy()
            
        return mx.array(np_arr, dtype=self.dtype)

    def _evict_if_needed(self, required_bytes: int) -> None:
        """Evicts oldest entries if adding required_bytes violates max_vram_bytes."""
        while self.current_vram_bytes + required_bytes > self.max_vram_bytes and self.cache:
            # Pop oldest item (first item in OrderedDict)
            idx, arr = self.cache.popitem(last=False)
            self.current_vram_bytes -= self.embedding_size_bytes
            del arr  # Release reference for MLX GC


class PagedEmbedding(nn.Module):
    """Drop-in MLX module replacement for nn.Embedding mapping indices to paged storage."""
    
    def __init__(self, weight_manager: WeightManager, num_embeddings: int, dim: int):
        super().__init__()
        self.weight_manager = weight_manager
        self.num_embeddings = num_embeddings
        self.dim = dim

    def __call__(self, indices: mx.array) -> mx.array:
        """Retrieves embeddings. Emulates standard embedding forward pass."""
        if isinstance(indices, (int, float)):
            return self.weight_manager.get_embedding(int(indices))
        if isinstance(indices, mx.array) and indices.ndim == 0:
            return self.weight_manager.get_embedding(int(indices.item()))
            
        # General case (array, list, etc.)
        # Ensure we have flat list of ints
        if isinstance(indices, mx.array):
            flat_indices = np.array(indices).flatten().tolist()
        else:
            flat_indices = np.array(indices).flatten().tolist()
            
        if not flat_indices:
            new_shape = list(indices.shape if isinstance(indices, mx.array) else np.array(indices).shape) + [self.dim]
            return mx.zeros(new_shape)
            
        self.weight_manager.prefetch(flat_indices)
        vectors = [self.weight_manager.get_embedding(idx) for idx in flat_indices]
        stacked = mx.stack(vectors)
        
        if isinstance(indices, mx.array):
            new_shape = list(indices.shape) + [self.dim]
        else:
            new_shape = list(np.array(indices).shape) + [self.dim]
        return mx.reshape(stacked, new_shape)


class UCEModel(nn.Module):
    """Minimal runnable UCE skeleton for toy synthetic arithmetic (generic over tree).

    Takes any FiniteTree at construction (registered leaves determine output vocab).
    The hard-coded toy tree builder (p=3, depth=4, 21 leaves for synthetic arithmetic
    expressions/grammar) + VALID_TOY_EXPRS live only in tests/test_model_toy.py per scope.
    - Low dim ball features + leaf activation injection for 'previous' tokens.
    - 1-2 diffusion layers doing simple p-adic weighted intra/cross mixing.
    - Per-depth linear heads.
    - Forward on list of previous leaf addresses -> prob dist over leaves (sums to 1).
    - .sample() convenience for categorical draw.
    """

    def __init__(
        self,
        tree: FiniteTree,
        dim: int = 16,
        num_diff_layers: int = 1,
        alpha: float = 0.5,
        weight_manager: Optional[WeightManager] = None,
        schwarzschild_warp: bool = True,
        r_s: float = 0.5,
        wormhole_gate: bool = True,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(tree, FiniteTree):
            raise ValueError("tree must be FiniteTree")
        self.tree = tree
        self.p = tree.p
        self.depth = tree.depth
        self.dim = int(dim)

        # Ordered registered leaves (addresses) and their local indices for activation embed
        # Use public API on tree (leaf_addresses) to avoid private _addr_to_token coupling.
        self.leaf_addrs: List[int] = tree.leaf_addresses()
        self.num_leaves = len(self.leaf_addrs)
        self.addr_to_leaf_idx: Dict[int, int] = {
            addr: i for i, addr in enumerate(self.leaf_addrs)
        }

        # All possible balls for this (p,depth) tree: position embeddings
        self.all_balls: List[Tuple[int, int]] = []
        self.ball_to_idx: Dict[Tuple[int, int], int] = {}
        bidx = 0
        for d in range(self.depth + 1):
            max_pref = self.p ** d
            for pref in range(max_pref):
                # depth 0 has only prefix 0
                if d == 0 and pref != 0:
                    continue
                key = (d, pref)
                self.all_balls.append(key)
                self.ball_to_idx[key] = bidx
                bidx += 1
        self.num_balls = len(self.all_balls)

        # Learnable params (random for toy skeleton)
        self.weight_manager = weight_manager
        if weight_manager is not None:
            self.ball_embed = PagedEmbedding(weight_manager, self.num_balls, self.dim)
        else:
            self.ball_embed = nn.Embedding(self.num_balls, self.dim)
        self.leaf_activation = nn.Embedding(self.num_leaves, self.dim)

        # The diffusion skeleton
        self.diffusion = UltrametricDiffusion(
            p=self.p,
            depth=self.depth,
            dim=self.dim,
            num_layers=num_diff_layers,
            alpha=alpha,
            schwarzschild_warp=schwarzschild_warp,
            r_s=r_s,
            wormhole_gate=wormhole_gate,
            epsilon=epsilon,
        )

        # The routing heads skeleton
        self.heads = DigitHeads(p=self.p, depth=self.depth, dim=self.dim)

    def load_weights(self, path: str, strict: bool = True) -> None:
        if hasattr(self, "weight_manager") and self.weight_manager is not None:
            weights = mx.load(path)
            if "ball_embed.weight" in weights:
                del weights["ball_embed.weight"]
            from mlx.utils import tree_unflatten
            self.update(tree_unflatten(weights))
        else:
            super().load_weights(path, strict=strict)

    def _init_ball_states(
        self, active_balls: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[Tuple[int, int], mx.array]:
        """Initialize state dict for (all or active subset of) balls from position embedding.

        Uses simple dict lookup (ball_to_idx) for sparse gather of the relevant ball embeds.
        If active_balls provided, only those keys (that are valid for tree) are populated.
        This is the starting point for sparse active-path execution (later can optimize further).
        """
        states: Dict[Tuple[int, int], mx.array] = {}
        if active_balls:
            if isinstance(self.ball_embed, PagedEmbedding):
                bidxs = [self.ball_to_idx[key] for key in active_balls if key in self.ball_to_idx]
                self.ball_embed.weight_manager.prefetch(bidxs)
            for key in active_balls:
                if key in self.ball_to_idx:
                    bidx = self.ball_to_idx[key]
                    states[key] = self.ball_embed(mx.array(bidx))
        else:
            if isinstance(self.ball_embed, PagedEmbedding):
                bidxs = list(self.ball_to_idx.values())
                self.ball_embed.weight_manager.prefetch(bidxs)
            for key, bidx in self.ball_to_idx.items():
                # Embedding accepts python int or mx scalar
                states[key] = self.ball_embed(mx.array(bidx))
        return states

    def _inject_previous(self, states: Dict[Tuple[int, int], mx.array], previous_addresses: List[int]) -> None:
        """Add leaf-activation signal for previous tokens into their leaf ball states (ultrametric injection point)."""
        if not previous_addresses:
            return
        # Use a few recent for skeleton context (toy scale)
        recent = previous_addresses[-5:]
        for addr in recent:
            if addr not in self.addr_to_leaf_idx:
                continue  # ignore unknown for robustness
            lidx = self.addr_to_leaf_idx[addr]
            lkey = (self.depth, addr)
            if lkey in states:
                act = self.leaf_activation(mx.array(lidx))
                states[lkey] = states[lkey] + act

    def embed_and_diffuse(
        self,
        previous_addresses: List[int] | None = None,
        active_balls: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[Tuple[int, int], mx.array]:
        """Public helper: embed balls + inject previous activations + run diffusion.

        If active_balls list provided (sparse gather), only those balls' states are
        initialized via dict lookup in _init_ball_states and diffused (subset returned).
        Callers in inference use this for active path + p-siblings restriction at decision points.
        Full call (no active_balls) preserves original behavior for training/eval.
        """
        if previous_addresses is None:
            previous_addresses = []
        states = self._init_ball_states(active_balls)
        self._inject_previous(states, previous_addresses)
        diffused = self.diffusion(states)
        return diffused

    def forward(self, previous_addresses: List[int] | None = None) -> mx.array:
        """Run embed + diffuse + heads -> distribution over the tree's leaves (sums to ~1)."""
        if previous_addresses is None:
            previous_addresses = []
        # Use the shared embed+diffuse (for training/eval state access too)
        diffused = self.embed_and_diffuse(previous_addresses)

        # Vectorized path: for each depth d, gather all leaf ball states and predict
        logps_total = mx.zeros((self.num_leaves,))
        for d in range(self.depth):
            states_d_list = []
            for addr in self.leaf_addrs:
                key = (d, addr % (self.p ** d))
                states_d_list.append(diffused.get(key, mx.zeros((self.dim,))))
            states_d = mx.stack(states_d_list)  # shape: [num_leaves, dim]

            # Predict logits for choice at depth d
            digit_logits_d = self.heads(states_d, d)  # shape: [num_leaves, p]

            # log softmax
            log_probs_d = digit_logits_d - mx.logsumexp(digit_logits_d, axis=-1, keepdims=True)

            # Get target digits at depth d
            digits_d = [address_to_digits(addr, self.p, self.depth)[d] for addr in self.leaf_addrs]
            digits_d_arr = mx.array(digits_d, dtype=mx.int32)

            logp_d = log_probs_d[mx.arange(self.num_leaves), digits_d_arr]
            logps_total = logps_total + logp_d

        probs = mx.softmax(logps_total)
        return probs

    def __call__(self, previous_addresses: List[int] | None = None) -> mx.array:
        return self.forward(previous_addresses)

    def sample(self, previous_addresses: List[int] | None = None) -> int:
        """Draw one next leaf address according to the model's distribution."""
        probs = self.forward(previous_addresses)
        if probs.size == 0:
            # fallback: return first leaf
            return self.leaf_addrs[0]
        # categorical on log-probs
        logits = mx.log(probs + 1e-12)
        # mx.random.categorical expects shape (..., vocab) returns index along last
        idx = mx.random.categorical(logits)
        # idx is array, take scalar
        idx_int = int(idx.item()) if hasattr(idx, "item") else int(idx)
        return self.leaf_addrs[idx_int]
