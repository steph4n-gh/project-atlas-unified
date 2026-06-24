import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

import torch

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from transformers import PreTrainedTokenizerBase
    TokenizerBase = PreTrainedTokenizerBase
except ImportError:
    TokenizerBase = object

class KineticsTokenizer(TokenizerBase):
    """
    High-performance BPE tokenizer implementing Parallel Reaction Kinetics and GPU BlockBPE.
    Integrates with QAN-Atlas long-context workloads for Gemma4 models.
    """
    def __init__(self, model_path_or_id: str, use_rust: bool = True, use_mlx: bool = False, base_tokenizer = None, organism_db: Optional[str] = None, use_kinetics: bool = True, **kwargs):
        if TokenizerBase is not object:
            try:
                super().__init__(**kwargs)
            except Exception:
                pass
        self.model_path_or_id = model_path_or_id
        self.use_rust = use_rust
        self.use_mlx = use_mlx and HAS_MLX
        self.base_tokenizer = base_tokenizer
        self.use_kinetics = use_kinetics
        if organism_db is None:
            import os
            cache_dir = os.getenv("ATLAS_CACHE_DIR")
            if not cache_dir:
                if os.path.exists("/Volumes/Storage"):
                    cache_dir = "/Volumes/Storage/attention_cache"
                else:
                    cache_dir = ".attention_cache"
            organism_db = os.path.join(cache_dir, "organism_db.json")
        from qan_transformers.tokenizer.context_organism import DeterministicContextOrganism
        self.organism = DeterministicContextOrganism(organism_db)
        
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, bytes] = {}
        self.merge_ranks: Dict[Tuple[int, int], int] = {}
        self.merge_result: Dict[Tuple[int, int], int] = {}
        self.catalyst_ids: Set[int] = set()
        
        # Paths to search for tokenizer.json
        self.tokenizer_json_path = self._find_tokenizer_json(model_path_or_id)
        if self.tokenizer_json_path:
            self._load_tokenizer_json(self.tokenizer_json_path)
            
        # Locate Rust microtok-bench binary
        self.rust_bin_path = self._find_rust_binary()
        if not self.rust_bin_path:
            self.use_rust = False

        # If MLX is active, build MLX/Metal structures
        self.mlx_initialized = False
        if self.use_mlx:
            self._init_mlx_structures()

    def __getattr__(self, name):
        if self.base_tokenizer is not None:
            return getattr(self.base_tokenizer, name)
        raise AttributeError(f"'KineticsTokenizer' object has no attribute '{name}'")

    def __call__(self, *args, **kwargs):
        if self.base_tokenizer is not None:
            return self.base_tokenizer(*args, **kwargs)
        raise NotImplementedError("base_tokenizer is not set")

    def __len__(self) -> int:
        if self.base_tokenizer is not None:
            return len(self.base_tokenizer)
        return len(self.vocab)

    def __repr__(self) -> str:
        if self.base_tokenizer is not None:
            return repr(self.base_tokenizer)
        return f"KineticsTokenizer(vocab_size={len(self.vocab)})"

    def __str__(self) -> str:
        if self.base_tokenizer is not None:
            return str(self.base_tokenizer)
        return f"KineticsTokenizer(vocab_size={len(self.vocab)})"

    def __contains__(self, item) -> bool:
        if isinstance(item, (int, np.integer)):
            return int(item) in self.inv_vocab
        return item in self.vocab

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            return self.inv_vocab[int(item)].decode("utf-8", errors="replace")
        return self.vocab[item]

    def __iter__(self):
        return iter(self.vocab)

    def is_catalyst(self, token_id: int) -> bool:
        return token_id in self.catalyst_ids

    def save_attention_metadata(self, text: str, tensors: Dict[str, torch.Tensor], non_tensor_metadata: Optional[Dict[str, Any]] = None):
        self.organism.save_attention_metadata(text, tensors, non_tensor_metadata)

    def load_attention_metadata(self, text: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        return self.organism.load_attention_metadata(text)

    def _find_tokenizer_json(self, path_or_id: str) -> Optional[Path]:
        # 1. Check direct path
        p = Path(path_or_id)
        if (p / "tokenizer.json").exists():
            return p / "tokenizer.json"
        if p.exists() and p.name == "tokenizer.json":
            return p
            
        # 2. Check workspace benchmarks data (relative to this file)
        bench_data = Path(__file__).parent.parent.parent / "benchmarks/data"
        if (bench_data / "gemma4-E4B-tokenizer.json").exists():
            return bench_data / "gemma4-E4B-tokenizer.json"
            
        # 3. Check Hugging Face Cache
        hf_cache_str = os.environ.get("HF_HUB_CACHE")
        if not hf_cache_str:
            hf_cache_str = os.path.expanduser("~/.cache/huggingface/hub")
        hf_cache = Path(hf_cache_str)
        if hf_cache.exists():
            # Search for gemma-4 or tokenizer.json files
            for tokenizer_path in hf_cache.glob("**/tokenizer.json"):
                if "gemma-4" in str(tokenizer_path).lower() or "gemma4" in str(tokenizer_path).lower():
                    return tokenizer_path
        return None

    def _find_rust_binary(self) -> Optional[Path]:
        candidates = [
            Path(__file__).parent / "rust/target/release/microtok-bench",
            Path("/Volumes/Storage/grokTest/rust/target/release/microtok-bench"),
            Path("rust/target/release/microtok-bench"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_tokenizer_json(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model", {})
            if model.get("type") != "BPE":
                return
                
            self.vocab = model.get("vocab", {})
            
            # Setup reverse vocab for decoding
            for tok_str, tok_id in self.vocab.items():
                self.inv_vocab[tok_id] = tok_str.encode("utf-8")
                
                # Check for Gemma catalyst word-start characters: " " (SentencePiece space) or "▁"
                if tok_str.startswith(" ") or tok_str.startswith("▁"):
                    self.catalyst_ids.add(tok_id)

            merges = model.get("merges", [])
            for priority, merge_str in enumerate(merges):
                if isinstance(merge_str, str):
                    parts = merge_str.split()
                else:
                    parts = merge_str
                if len(parts) != 2:
                    continue
                left, right = parts[0], parts[1]
                left_id = self.vocab.get(left)
                right_id = self.vocab.get(right)
                
                # Construct merged string
                merged_str = left + right
                merged_id = self.vocab.get(merged_str)
                
                if left_id is not None and right_id is not None and merged_id is not None:
                    pair = (left_id, right_id)
                    self.merge_ranks[pair] = priority
                    self.merge_result[pair] = merged_id
        except Exception as e:
            print(f"[Tokenizer Warning] Failed to load tokenizer.json: {e}", file=sys.stderr)

    def _init_mlx_structures(self):
        if not HAS_MLX or not self.merge_ranks:
            return
            
        try:
            # Represent merges as a sorted 1D array of keys for binary search.
            # Key: (left_id << 32) | right_id (64-bit key)
            # Value: (parent_id << 32) | rank
            merge_keys = []
            merge_vals = []
            for (left, right), rank in self.merge_ranks.items():
                parent = self.merge_result[(left, right)]
                key = (left << 32) | right
                val = (parent << 32) | rank
                merge_keys.append(key)
                merge_vals.append(val)
                
            # Sort by key for binary search
            sorted_indices = np.argsort(merge_keys)
            self.mlx_merge_keys = mx.array(np.array(merge_keys)[sorted_indices], dtype=mx.uint64)
            self.mlx_merge_vals = mx.array(np.array(merge_vals)[sorted_indices], dtype=mx.uint64)
            
            # Catalyst set as a boolean mask array or lookup array
            self.mlx_catalysts = mx.zeros((len(self.vocab) + 1000,), dtype=mx.bool_)
            catalyst_indices = [idx for idx in self.catalyst_ids if idx < self.mlx_catalysts.shape[0]]
            if catalyst_indices:
                self.mlx_catalysts[mx.array(catalyst_indices)] = True
                
            # Generate the E8 coordinates map for all vocabulary IDs
            from qan_transformers.math.e8_projection import generate_e8_coordinates
            e8_roots = generate_e8_coordinates() # shape [240, 8]
            vocab_size = len(self.vocab) + 1000
            indices = np.arange(vocab_size) % 240
            vocab_coords_np = e8_roots[indices].astype(np.float32)
            self.mlx_vocab_e8_coords = mx.array(vocab_coords_np, dtype=mx.float32)

            # Load and compile custom Metal BPE merge kernel
            kernel_path = Path(__file__).parent.parent / "kernels" / "bpe_merge.metal"
            if kernel_path.exists():
                with open(kernel_path, "r", encoding="utf-8") as f:
                    source = f.read()
                
                header = ""
                if "inline uint32_t rand_hash" in source:
                    idx = source.find("inline uint32_t rand_hash")
                    brace_idx = source.find("}\n", idx)
                    if brace_idx == -1:
                        brace_idx = source.find("}\r\n", idx)
                    if brace_idx != -1:
                        end_idx = brace_idx + 1
                        header = source[idx:end_idx+1]
                        source = source[:idx] + source[end_idx+1:]

                self._mlx_kernel = mx.fast.metal_kernel(
                    name="bpe_merge_kernel",
                    input_names=["symbols", "merge_keys", "merge_vals", "catalysts", "vocab_e8_coords", "temperature"],
                    output_names=["output_symbols", "output_lengths", "output_coords"],
                    source=source,
                    header=header,
                )
                self.mlx_initialized = True
        except Exception as e:
            print(f"[Tokenizer Warning] Failed to initialize MLX Metal structures: {e}", file=sys.stderr)

    def gemma_normalize(self, text: str) -> str:
        # Standard Gemma4 SentencePiece normalization (replace ASCII space with lower five eighths block / '▁')
        return text.replace(" ", "▁")

    def gemma_pre_tokenize(self, normalized: str) -> List[str]:
        pieces = []
        start = 0
        for i, c in enumerate(normalized):
            if c == "▁":
                if i > 0 and normalized[i-1] == "▁":
                    continue
                if i > start:
                    pieces.append(normalized[start:i])
                start = i
        if start < len(normalized):
            pieces.append(normalized[start:])
        if not pieces:
            pieces.append(normalized)
        return pieces

    def _encode_rust(self, text: str) -> List[int]:
        if not self.rust_bin_path:
            return []
        try:
            cmd = [str(self.rust_bin_path), "--encode"]
            proc = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True)
            lines = proc.stdout.strip().splitlines()
            for line in lines:
                if not line.startswith("ENCODE_US"):
                    return [int(x) for x in line.split()]
        except Exception as e:
            print(f"[Tokenizer Warning] Rust subprocess encoding failed: {e}", file=sys.stderr)
        return []

    def _parallel_reaction_merge_python(self, symbols: List[int]) -> List[int]:
        """
        Pure Python implementation of the parallel reaction BPE merge.
        Fires non-overlapping merges in waves using rate-based prioritization.
        """
        passes = 0
        max_passes = 128
        
        while passes < max_passes:
            candidates = [] # List of (rate, pos, new_id)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    new_id = self.merge_result[pair]
                    rate = (1000000 - rank) * 100
                    # Catalyst boost
                    if symbols[i] in self.catalyst_ids or symbols[i+1] in self.catalyst_ids:
                        rate += 50
                    candidates.append((rate, i, new_id))
                    
            if not candidates:
                break
                
            # Sort by rate desc
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Apply merges greedily
            used = [False] * len(symbols)
            new_syms = []
            i = 0
            applied_any = False
            
            while i < len(symbols):
                applied = False
                for rate, pos, new_id in candidates:
                    if pos == i and not used[i] and not used[i+1]:
                        new_syms.append(new_id)
                        used[i] = True
                        used[i+1] = True
                        applied_any = True
                        i += 2
                        applied = True
                        break
                if not applied:
                    if not used[i]:
                        new_syms.append(symbols[i])
                    i += 1
                    
            if not applied_any:
                break
            symbols = new_syms
            passes += 1
            
        return symbols

    def _exact_merge_python(self, symbols: List[int]) -> List[int]:
        """
        Exact BPE greedy merge using priority queue and doubly-linked list.
        Guarantees 100% bit-identical parity with Hugging Face's tokenizer.
        """
        if len(symbols) <= 1:
            return symbols
            
        class Node:
            def __init__(self, val, pos, prev=None, next=None):
                self.val = val
                self.pos = pos
                self.prev = prev
                self.next = next
                
        nodes = [Node(val, i) for i, val in enumerate(symbols)]
        for i in range(len(nodes)):
            if i > 0:
                nodes[i].prev = nodes[i-1]
            if i < len(nodes) - 1:
                nodes[i].next = nodes[i+1]
                
        import heapq
        heap = []
        
        def push_pair(left_node, right_node):
            pair = (left_node.val, right_node.val)
            rank = self.merge_ranks.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, left_node.pos, left_node, right_node))
                
        for i in range(len(nodes) - 1):
            push_pair(nodes[i], nodes[i+1])
            
        merged_nodes = set()
        
        while heap:
            rank, pos, left_node, right_node = heapq.heappop(heap)
            if id(left_node) in merged_nodes or id(right_node) in merged_nodes:
                continue
            if left_node.next != right_node:
                continue
            pair = (left_node.val, right_node.val)
            if self.merge_ranks.get(pair) != rank:
                continue
                
            new_id = self.merge_result[pair]
            left_node.val = new_id
            
            next_node = right_node.next
            left_node.next = next_node
            if next_node is not None:
                next_node.prev = left_node
                
            merged_nodes.add(id(right_node))
            
            if left_node.prev is not None:
                push_pair(left_node.prev, left_node)
            if left_node.next is not None:
                push_pair(left_node, left_node.next)
                
        curr = nodes[0]
        res = []
        while curr is not None:
            if id(curr) not in merged_nodes:
                res.append(curr.val)
            curr = curr.next
        return res

    def _encode_gemma_python(self, text: str, exact: bool = True) -> List[int]:
        normalized = self.gemma_normalize(text)
        pieces = self.gemma_pre_tokenize(normalized)
        out = []
        for piece in pieces:
            # 1. Fast match: whole piece is in vocab
            if piece in self.vocab:
                out.append(self.vocab[piece])
                continue
                
            # 2. Split piece into characters
            initial = []
            for ch in piece:
                if ch in self.vocab:
                    initial.append(self.vocab[ch])
                    
            if initial:
                if exact:
                    merged = self._exact_merge_python(initial)
                else:
                    merged = self._parallel_reaction_merge_python(initial)
                out.extend(merged)
        return out

    def _encode_gemma_mlx(self, text: str) -> List[int]:
        """
        Gemma BPE tokenization accelerated via GPU/Metal BlockBPE kinetics kernel.
        """
        if not self.mlx_initialized:
            return self._encode_gemma_python(text, exact=False)

        normalized = self.gemma_normalize(text)
        pieces = self.gemma_pre_tokenize(normalized)
        
        piece_tokens = [None] * len(pieces)
        blocks = []
        block_indices = [] # list of (piece_index, chunk_length)
        
        for idx, piece in enumerate(pieces):
            if piece in self.vocab:
                piece_tokens[idx] = [self.vocab[piece]]
            else:
                initial = [self.vocab[ch] for ch in piece if ch in self.vocab]
                if not initial:
                    piece_tokens[idx] = []
                elif len(initial) == 1:
                    piece_tokens[idx] = initial
                else:
                    # Pad to 64 elements for threadgroup layout
                    for offset in range(0, len(initial), 64):
                        chunk = initial[offset:offset+64]
                        pad_len = 64 - len(chunk)
                        padded = chunk + [0xFFFFFFFF] * pad_len
                        blocks.append(padded)
                        block_indices.append((idx, len(chunk)))

        if blocks:
            num_blocks = len(blocks)
            symbols_arr = mx.array(np.array(blocks, dtype=np.uint32).reshape(-1))
            
            # Launch Metal GPU BPE merge kernel via graph_encode
            out_symbols, out_lengths, _ = self.graph_encode(symbols_arr)
            # Use zero-copy compaction on the GPU
            compacted = self.compact_token_ids(out_symbols, out_lengths)
            
            mx.eval(compacted, out_lengths)
            flat_mlx_tokens = compacted.tolist()
            out_lens_np = np.array(out_lengths)
            
            # Reconstruct piece tokens by slicing the flat list
            offset = 0
            block_idx = 0
            for idx, orig_len in block_indices:
                length = int(out_lens_np[block_idx])
                block_tokens = flat_mlx_tokens[offset : offset + length]
                if piece_tokens[idx] is None:
                    piece_tokens[idx] = block_tokens
                else:
                    piece_tokens[idx].extend(block_tokens)
                offset += length
                block_idx += 1
                
        # Flatten and return
        flat_tokens = []
        for pt in piece_tokens:
            if pt:
                flat_tokens.extend(pt)
        return flat_tokens

    def _encode_raw(self, text: str, add_special_tokens: bool = False, exact: bool = True) -> List[int]:
        result_ids = None
        # Autopilot logic:
        # For short prompts (len < 5000 chars), delegate directly to baseline tokenizer
        # to avoid GPU/subprocess overhead and preserve 100% template compatibility.
        if len(text) < 5000 and self.base_tokenizer is not None:
            try:
                # We can call the base tokenizer's encode directly
                if hasattr(self.base_tokenizer, "_original_encode"):
                    result_ids = self.base_tokenizer._original_encode(text, add_special_tokens=add_special_tokens)
                elif hasattr(self.base_tokenizer, "encode"):
                    result_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens)
                elif hasattr(self.base_tokenizer, "encode_text"):
                    result_ids = self.base_tokenizer.encode_text(text)
                
                if result_ids is not None and hasattr(result_ids, "ids"):
                    result_ids = result_ids.ids
            except Exception:
                pass

        if result_ids is None:
            # For large texts, run parallel kinetics on GPU or CPU
            # 1. GPU/Metal MLX path
            if self.use_mlx and self.mlx_initialized:
                # MLX BlockBPE uses parallel reaction waves (approx BPE)
                result_ids = self._encode_gemma_mlx(text)
            # 2. Rust Turbo path
            elif self.use_rust:
                ids = self._encode_rust(text)
                if ids:
                    result_ids = ids
                    
            # 3. Pure Python / Parallel Reaction Kinetics path
            if result_ids is None and self.vocab:
                result_ids = self._encode_gemma_python(text, exact=exact)
                
            # 4. Fallback mock tokenization (if no vocabulary is loaded)
            if result_ids is None:
                result_ids = [ord(c) % 256 for c in text]
        return result_ids

    def encode(self, text: str, add_special_tokens: bool = False, exact: Optional[bool] = None) -> List[int]:
        if exact is None:
            exact = not self.use_kinetics

        # Try Trie Context Caching first
        prefix_ids, suffix_text = self.organism.get_longest_prefix(text)
        if prefix_ids is not None:
            if not suffix_text:
                return prefix_ids
                
            # Search backward from the end of prefix_ids for a catalyst (word-start) token.
            found_catalyst_idx = -1
            for i in range(len(prefix_ids) - 1, -1, -1):
                if self.is_catalyst(prefix_ids[i]):
                    found_catalyst_idx = i
                    break
            
            if found_catalyst_idx != -1:
                prefix_ids_truncated = prefix_ids[:found_catalyst_idx]
            else:
                prefix_ids_truncated = []
                
            decoded_truncated = self.decode(prefix_ids_truncated) if prefix_ids_truncated else ""
            
            clean_text = text.replace("▁", " ").replace("Ġ", " ")
            clean_decoded = decoded_truncated.replace("▁", " ").replace("Ġ", " ")
            target = clean_decoded.strip()
            
            if not target:
                split_char_idx = 0
            else:
                idx = clean_text.find(target, 0)
                if idx != -1:
                    trailing_len = len(clean_decoded) - len(clean_decoded.rstrip())
                    split_char_idx = idx + len(target) + trailing_len
                else:
                    split_char_idx = len(decoded_truncated)
                    
            remainder_text = text[split_char_idx:]
            remainder_ids = self._encode_raw(remainder_text, add_special_tokens=False, exact=exact)
            full_ids = prefix_ids_truncated + remainder_ids
                
            self.organism.set(text, full_ids)
            return full_ids
            
        # If no prefix match, do full encoding
        result_ids = self._encode_raw(text, add_special_tokens=add_special_tokens, exact=exact)
        self.organism.set(text, result_ids)
        return result_ids

    def graph_encode(self, symbols_arr: mx.array, temperature: Optional[mx.array] = None) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Symbolic Graph-Fused tokenization entry point. Runs entirely in JIT compile trace.
        """
        if not self.mlx_initialized:
            raise RuntimeError("MLX GPU tokenizer not initialized")
        
        if temperature is None:
            # Dynamically scale temperature based on running entropy (Leap 0021)
            from qan_transformers.mlx.attention import QuasicrystallineAttention
            prev_entropy = getattr(QuasicrystallineAttention, "_shared_prev_entropy", None)
            if prev_entropy is not None:
                # Cool down temperature during high-confidence stretches to boost context organism cache hits
                if prev_entropy < 1.8:
                    temp_val = 0.0
                else:
                    temp_val = min(0.5, (prev_entropy - 1.8) * 0.25)
            else:
                temp_val = 0.0
            temperature = mx.array([temp_val], dtype=mx.float32)
        elif not isinstance(temperature, mx.array):
            temperature = mx.array([float(temperature)], dtype=mx.float32)
            
        num_blocks = symbols_arr.shape[0] // 64
        out_symbols, out_lengths, out_coords = self._mlx_kernel(
            inputs=[
                symbols_arr, 
                self.mlx_merge_keys, 
                self.mlx_merge_vals, 
                self.mlx_catalysts,
                self.mlx_vocab_e8_coords,
                temperature
            ],
            output_shapes=[[num_blocks * 64], [num_blocks], [num_blocks * 64, 8]],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32],
            template=[
                ("NUM_MERGES", len(self.merge_ranks)),
            ],
            grid=(num_blocks * 64, 1, 1),
            threadgroup=(64, 1, 1),
        )
        return out_symbols, out_lengths, out_coords

    def compact_coords(self, out_coords: mx.array, out_lengths: mx.array) -> mx.array:
        """
        Efficient, zero-copy compaction of block-padded coordinates on the GPU.
        Uses symbolic boolean masking.
        """
        num_blocks = out_lengths.shape[0]
        N = num_blocks * 64
        lanes = mx.arange(64)[None, :]
        mask = lanes < out_lengths[:, None]
        flat_mask = mx.reshape(mask, (N,))
        num_true = int(mx.sum(flat_mask).item())
        if num_true == 0:
            return mx.array([], dtype=out_coords.dtype).reshape(0, 8)
        keys = mx.where(flat_mask, mx.arange(N) - N, mx.arange(N) + N)
        sorted_idx = mx.argsort(keys)
        true_idx = sorted_idx[:num_true]
        return mx.take(out_coords, true_idx, axis=0)

    def compact_coords_static(self, out_coords: mx.array, out_lengths: mx.array, pad_val: float = 0.0) -> Tuple[mx.array, mx.array]:
        """
        Efficient, zero-copy static compaction of block-padded coordinates on the GPU.
        Maintains static shapes suitable for mx.compile by avoiding host synchronizations.
        Returns a tuple of (compacted_coords, total_valid_length).
        """
        num_blocks = out_lengths.shape[0]
        N = num_blocks * 64
        lanes = mx.arange(64)[None, :]
        mask = lanes < out_lengths[:, None]
        flat_mask = mx.reshape(mask, (N,))
        
        # Stable sorting of active vs inactive elements
        keys = mx.where(flat_mask, mx.arange(N) - N, mx.arange(N) + N)
        sorted_idx = mx.argsort(keys)
        sorted_coords = mx.take(out_coords, sorted_idx, axis=0)
        
        active_count = mx.sum(flat_mask.astype(mx.int32))
        idx_grid = mx.arange(N)[:, None]
        compacted = mx.where(idx_grid < active_count, sorted_coords, pad_val)
        return compacted, active_count

    def prepare_fused_input(self, text: str) -> Tuple[mx.array, int]:
        """
        Normalize and segment a text string on the CPU, preparing it as a
        block-padded character ID array for custom Metal kernel execution.
        """
        normalized = self.gemma_normalize(text)
        pieces = self.gemma_pre_tokenize(normalized)
        
        blocks = []
        for piece in pieces:
            if piece in self.vocab:
                # Whole piece is directly in vocabulary
                blocks.append([self.vocab[piece]] + [0xFFFFFFFF] * 63)
            else:
                initial = [self.vocab[ch] for ch in piece if ch in self.vocab]
                if not initial:
                    continue
                # Pad to blocks of 64 elements
                for offset in range(0, len(initial), 64):
                    chunk = initial[offset:offset+64]
                    pad_len = 64 - len(chunk)
                    padded = chunk + [0xFFFFFFFF] * pad_len
                    blocks.append(padded)

        if not blocks:
            return mx.array([], dtype=mx.uint32), 0
            
        symbols_arr = mx.array(np.array(blocks, dtype=np.uint32).reshape(-1))
        return symbols_arr, len(blocks)

    def compact_token_ids(self, out_symbols: mx.array, out_lengths: mx.array) -> mx.array:
        """
        Efficient, zero-copy compaction of block-padded token IDs on the GPU.
        Uses symbolic boolean masking.
        """
        num_blocks = out_lengths.shape[0]
        N = num_blocks * 64
        lanes = mx.arange(64)[None, :]
        mask = lanes < out_lengths[:, None]
        flat_symbols = mx.reshape(out_symbols, (N,))
        flat_mask = mx.reshape(mask, (N,))
        num_true = int(mx.sum(flat_mask).item())
        if num_true == 0:
            return mx.array([], dtype=out_symbols.dtype)
        keys = mx.where(flat_mask, mx.arange(N) - N, mx.arange(N) + N)
        sorted_idx = mx.argsort(keys)
        true_idx = sorted_idx[:num_true]
        return mx.take(flat_symbols, true_idx)

    def compact_token_ids_static(self, out_symbols: mx.array, out_lengths: mx.array, pad_val: int = 0xFFFFFFFF) -> Tuple[mx.array, mx.array]:
        """
        Efficient, zero-copy static compaction of block-padded token IDs on the GPU.
        Maintains static shapes suitable for mx.compile by avoiding host synchronizations.
        Returns a tuple of (compacted_symbols, total_valid_length).
        """
        num_blocks = out_lengths.shape[0]
        N = num_blocks * 64
        flat_symbols = mx.reshape(out_symbols, (N,))
        lanes = mx.arange(64)[None, :]
        mask = lanes < out_lengths[:, None]
        flat_mask = mx.reshape(mask, (N,))
        
        # Stable sorting of active vs inactive elements
        keys = mx.where(flat_mask, mx.arange(N) - N, mx.arange(N) + N)
        sorted_idx = mx.argsort(keys)
        sorted_symbols = mx.take(flat_symbols, sorted_idx)
        
        active_count = mx.sum(flat_mask.astype(mx.int32))
        compacted = mx.where(mx.arange(N) < active_count, sorted_symbols, pad_val)
        return compacted, active_count

    def save_trie_metadata(self, filepath: str):
        """
        Saves the Trie cache metadata to a specified file path.
        """
        save_data = {
            "cache": self.organism.cache,
            "text_keys": self.organism.text_keys,
            "metadata_refs": self.organism.metadata_refs,
            "non_tensor_metadata": self.organism.non_tensor_metadata
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f)

    def load_trie_metadata(self, filepath: str):
        """
        Loads the Trie cache metadata from a specified file path and rebuilds the Trie.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "cache" in data and "text_keys" in data:
            self.organism.cache = data["cache"]
            self.organism.text_keys = data["text_keys"]
            self.organism.metadata_refs = data.get("metadata_refs", {})
            self.organism.non_tensor_metadata = data.get("non_tensor_metadata", {})
        else:
            self.organism.cache = data
            self.organism.text_keys = {}
            self.organism.metadata_refs = {}
            self.organism.non_tensor_metadata = {}
        self.organism._rebuild_trie()

    def decode(self, tokens: List[int]) -> str:
        parts = []
        for t in tokens:
            if t in self.inv_vocab:
                parts.append(self.inv_vocab[t])
            else:
                parts.append(bytes([t % 256]))
        try:
            res = b"".join(parts).decode("utf-8", errors="replace")
            return res.replace("▁", " ")
        except Exception:
            return ""

def load_qan_tokenizer(model_path_or_id: str, base_tokenizer = None) -> KineticsTokenizer:
    """
    Helper function to load KineticsTokenizer with standard defaults matching Apple Silicon target.
    Delegates to baseline tokenizer automatically for short chat inputs.
    """
    if base_tokenizer is None:
        try:
            from transformers import AutoTokenizer as HFAutoTokenizer
            base_tokenizer = HFAutoTokenizer.from_pretrained(model_path_or_id)
        except Exception:
            pass
    # Enable both rust and mlx for Autopilot to choose the best available backend
    return KineticsTokenizer(
        model_path_or_id,
        use_rust=True,
        use_mlx=True,
        base_tokenizer=base_tokenizer
    )

def graft_tokenizer(base_tokenizer, use_rust: bool = True, use_mlx: bool = True, use_kinetics: bool = True, **kwargs):
    """
    Grafts KineticsTokenizer capabilities onto an existing AutoTokenizer instance in-place.
    """
    import types
    model_path_or_id = getattr(base_tokenizer, "name_or_path", "")
    
    kinetics = KineticsTokenizer(
        model_path_or_id=model_path_or_id,
        use_rust=use_rust,
        use_mlx=use_mlx,
        base_tokenizer=base_tokenizer,
        use_kinetics=use_kinetics,
        **kwargs
    )
    
    orig_class = base_tokenizer.__class__

    class KineticsGraftedTokenizer(orig_class):
        def __call__(self, text, text_pair=None, add_special_tokens=True, padding=False, truncation=False, max_length=None, return_tensors=None, **kwargs_call):
            if text_pair is not None or not isinstance(text, (str, list)) or kwargs_call.get("is_split_into_words", False):
                return super().__call__(
                    text, text_pair=text_pair, add_special_tokens=add_special_tokens,
                    padding=padding, truncation=truncation, max_length=max_length,
                    return_tensors=return_tensors, **kwargs_call
                )
                
            exact = kwargs_call.pop("exact", None)
            if isinstance(text, list):
                batch_ids = []
                for item in text:
                    ids = self.kinetics.encode(item, add_special_tokens=add_special_tokens, exact=exact)
                    if truncation and max_length is not None and len(ids) > max_length:
                        ids = ids[:max_length]
                    batch_ids.append({"input_ids": ids})
                return self.pad(batch_ids, padding=padding, max_length=max_length, return_tensors=return_tensors, **kwargs_call)
            else:
                ids = self.kinetics.encode(text, add_special_tokens=add_special_tokens, exact=exact)
                if truncation and max_length is not None and len(ids) > max_length:
                    ids = ids[:max_length]
                features = {"input_ids": ids}
                return self.pad(features, padding=padding, max_length=max_length, return_tensors=return_tensors, **kwargs_call)

        def encode(self, text, *args, **kwargs_enc):
            add_special_tokens = kwargs_enc.get("add_special_tokens", True)
            exact = kwargs_enc.get("exact", None)
            if len(args) == 0 and "text_pair" not in kwargs_enc:
                return self.kinetics.encode(text, add_special_tokens=add_special_tokens, exact=exact)
            return super().encode(text, *args, **kwargs_enc)

        def decode(self, token_ids, *args, **kwargs_dec):
            if not args and not kwargs_dec and isinstance(token_ids, list) and all(isinstance(x, (int, np.integer)) for x in token_ids):
                return self.kinetics.decode([int(x) for x in token_ids])
            return super().decode(token_ids, *args, **kwargs_dec)

        def graph_encode(self, *a, **k):
            return self.kinetics.graph_encode(*a, **k)

        def prepare_fused_input(self, *a, **k):
            return self.kinetics.prepare_fused_input(*a, **k)

        def compact_token_ids(self, *a, **k):
            return self.kinetics.compact_token_ids(*a, **k)

        def compact_token_ids_static(self, *a, **k):
            return self.kinetics.compact_token_ids_static(*a, **k)

        def compact_coords(self, *a, **k):
            return self.kinetics.compact_coords(*a, **k)

        def compact_coords_static(self, *a, **k):
            return self.kinetics.compact_coords_static(*a, **k)

        def save_trie_metadata(self, *a, **k):
            return self.kinetics.save_trie_metadata(*a, **k)

        def load_trie_metadata(self, *a, **k):
            return self.kinetics.load_trie_metadata(*a, **k)

        def save_attention_metadata(self, *a, **k):
            return self.kinetics.save_attention_metadata(*a, **k)

        def load_attention_metadata(self, *a, **k):
            return self.kinetics.load_attention_metadata(*a, **k)

        def __len__(self) -> int:
            try:
                return super().__len__()
            except Exception:
                return len(self.kinetics.vocab)

        def __repr__(self) -> str:
            try:
                orig_repr = super().__repr__()
            except Exception:
                orig_repr = "PreTrainedTokenizer"
            return f"KineticsGraftedTokenizer(wrapped={orig_repr})"

        def __str__(self) -> str:
            try:
                return super().__str__()
            except Exception:
                return f"KineticsGraftedTokenizer(vocab_size={len(self.kinetics.vocab)})"

        def __contains__(self, item) -> bool:
            return item in self.kinetics

        def __getitem__(self, item):
            return self.kinetics[item]

        def __iter__(self):
            return iter(self.kinetics)

    base_tokenizer.__class__ = KineticsGraftedTokenizer
    base_tokenizer.kinetics = kinetics
    base_tokenizer.organism = kinetics.organism

    if not hasattr(base_tokenizer, "_original_encode"):
        base_tokenizer._original_encode = types.MethodType(orig_class.encode, base_tokenizer)
    if not hasattr(base_tokenizer, "_original_decode"):
        base_tokenizer._original_decode = types.MethodType(orig_class.decode, base_tokenizer)

    for attr in ["vocab", "inv_vocab", "merge_ranks", "merge_result", "catalyst_ids", "use_kinetics", "use_rust", "use_mlx", "mlx_initialized"]:
        if hasattr(kinetics, attr):
            try:
                setattr(base_tokenizer, attr, getattr(kinetics, attr))
            except AttributeError:
                pass
            
    return base_tokenizer

class AutoTokenizer:
    """
    Transparent AutoTokenizer wrapper that automatically grafts KineticsTokenizer
    onto Hugging Face AutoTokenizer instances.
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from transformers import AutoTokenizer as HFAutoTokenizer
        base_tokenizer = HFAutoTokenizer.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return graft_tokenizer(base_tokenizer, **kwargs)

def patch_global_autotokenizer():
    try:
        from transformers import AutoTokenizer as HFAutoTokenizer
        if not hasattr(HFAutoTokenizer, "_original_from_pretrained"):
            HFAutoTokenizer._original_from_pretrained = HFAutoTokenizer.from_pretrained
            @classmethod
            def new_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
                base_tokenizer = cls._original_from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                return graft_tokenizer(base_tokenizer, **kwargs)
            HFAutoTokenizer.from_pretrained = new_from_pretrained
    except ImportError:
        pass

# Automatically patch transformers.AutoTokenizer globally on import
patch_global_autotokenizer()
