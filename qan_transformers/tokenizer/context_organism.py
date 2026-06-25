import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch
from safetensors.torch import save_file, load_file

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.token_ids: Optional[List[int]] = None
        self.metadata_ref: Optional[str] = None
        self.non_tensor_metadata: Dict[str, Any] = {}
        self.reversible_tape: Optional[List[Any]] = None
        self.lattice_coords: Optional[List[List[float]]] = None

class AttentionMetadataStore:
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            import os
            cache_dir = os.getenv("ATLAS_CACHE_DIR")
            if not cache_dir:
                if os.path.exists("/Volumes/Storage"):
                    cache_dir = "/Volumes/Storage/attention_cache"
                else:
                    cache_dir = ".attention_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata_id: str, tensors: Dict[str, torch.Tensor]) -> str:
        if not tensors:
            return ""
        file_path = self.cache_dir / f"{metadata_id}.safetensors"
        torch_tensors = {}
        for k, v in tensors.items():
            if not isinstance(v, torch.Tensor):
                torch_tensors[k] = torch.tensor(v)
            else:
                torch_tensors[k] = v
        save_file(torch_tensors, str(file_path))
        return str(file_path)

    def load_metadata(self, metadata_ref: str) -> Dict[str, torch.Tensor]:
        if not metadata_ref or not Path(metadata_ref).exists():
            return {}
        return load_file(metadata_ref)

class DeterministicContextOrganism:
    _active_saves = {}

    def __init__(self, db_path: Optional[str] = None, cache_dir: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else None
        self.cache: Dict[str, List[int]] = {}
        self.text_keys: Dict[str, str] = {}
        self.metadata_refs: Dict[str, str] = {}
        self.non_tensor_metadata: Dict[str, Dict[str, Any]] = {}
        self.reversible_tapes: Dict[str, List[Any]] = {}
        self.thermal_state: Dict[str, Any] = {"temperature": 0.0, "average_energy": 0.0}
        self.lattice_cache: Dict[int, Tuple[List[float], float]] = {}
        self.metadata_store = AttentionMetadataStore(cache_dir)
        self.last_metadata_ref = None
        self.last_non_tensor_metadata = {}
        self.last_reversible_tape = None
        self.trie_root = TrieNode()
        
        if self.db_path:
            abs_path = str(self.db_path.resolve())
            if abs_path in DeterministicContextOrganism._active_saves:
                thread = DeterministicContextOrganism._active_saves[abs_path]
                if thread.is_alive():
                    thread.join()
            if self.db_path.exists():
                self.restore()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_token_lattice_metadata(self, token_id: int, lattice: str = 'e8') -> Tuple[List[float], float]:
        """
        0019 Periodic Vocab Lattice Alignment:
        Maps token_id deterministically to an E8 (8D) or Leech (24D) root lattice point and returns its coordinates and energy.
        """
        cache_key = (token_id, lattice)
        if cache_key in self.lattice_cache:
            return self.lattice_cache[cache_key]
        
        try:
            if lattice == 'leech':
                from qan_transformers.math.leech_lattice import generate_leech_coordinates
                roots = generate_leech_coordinates(shell=1) # shape (196560, 24)
                base_coords = roots[token_id % len(roots)]
                scale = 1.0 + (token_id // len(roots)) * 0.05
                coords = (base_coords * scale).tolist()
                energy = float(np.sum(np.square(coords)))
            else:
                from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates
                roots = generate_dynamic_e8_coordinates(1) # shape (240, 8)
                base_coords = roots[token_id % len(roots)]
                scale = 1.0 + (token_id // len(roots)) * 0.05
                coords = (base_coords * scale).tolist()
                energy = float(np.sum(np.square(coords)))
        except Exception:
            # Fallback mock coordinates if projection fails or is uninitialized
            if lattice == 'leech':
                coords = [0.0] * 24
                coords[token_id % 24] = 1.0 + (token_id // 24) * 0.05
                energy = float(sum(c*c for c in coords))
            else:
                coords = [0.0] * 8
                coords[token_id % 8] = 1.0 + (token_id // 8) * 0.05
                energy = float(sum(c*c for c in coords))
            
        self.lattice_cache[cache_key] = (coords, energy)
        return coords, energy

    def get_sequence_lattice_metadata(self, token_ids: List[int], lattice: str = 'e8') -> Tuple[List[List[float]], List[float]]:
        coords_list = []
        energies_list = []
        for tid in token_ids:
            c, e = self.get_token_lattice_metadata(tid, lattice=lattice)
            coords_list.append(c)
            energies_list.append(e)
        return coords_list, energies_list

    def _insert_trie(self, text: str, token_ids: List[int], metadata_ref: Optional[str] = None, non_tensor_metadata: Optional[Dict[str, Any]] = None, reversible_tape: Optional[List[Any]] = None):
        current = self.trie_root
        for char in text:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.token_ids = token_ids
        current.metadata_ref = metadata_ref
        if non_tensor_metadata is not None:
            current.non_tensor_metadata = non_tensor_metadata
        if reversible_tape is not None:
            current.reversible_tape = reversible_tape
        
        # Prefetch lattice coordinates for trie node
        coords_list, _ = self.get_sequence_lattice_metadata(token_ids)
        current.lattice_coords = coords_list

    def get_longest_prefix(self, text: str) -> Tuple[Optional[List[int]], str]:
        current = self.trie_root
        longest_ids = None
        longest_len = 0
        longest_node = None
        for i, char in enumerate(text):
            if char in current.children:
                current = current.children[char]
                if current.token_ids is not None:
                    longest_ids = current.token_ids
                    longest_len = i + 1
                    longest_node = current
            else:
                break
        if longest_ids is not None:
            self.last_metadata_ref = longest_node.metadata_ref
            self.last_non_tensor_metadata = longest_node.non_tensor_metadata
            self.last_reversible_tape = longest_node.reversible_tape
            return longest_ids, text[longest_len:]
        self.last_metadata_ref = None
        self.last_non_tensor_metadata = {}
        self.last_reversible_tape = None
        return None, text

    def get(self, text: str) -> Optional[List[int]]:
        h = self._hash_text(text)
        return self.cache.get(h)

    def set(self, text: str, token_ids: List[int], metadata_ref: Optional[str] = None, non_tensor_metadata: Optional[Dict[str, Any]] = None, reversible_tape: Optional[List[Any]] = None):
        h = self._hash_text(text)
        self.cache[h] = token_ids
        self.text_keys[h] = text
        if metadata_ref is not None:
            self.metadata_refs[h] = metadata_ref
        if non_tensor_metadata is not None:
            self.non_tensor_metadata[h] = non_tensor_metadata
        if reversible_tape is not None:
            self.reversible_tapes[h] = reversible_tape
        self._insert_trie(text, token_ids, metadata_ref, non_tensor_metadata, reversible_tape)

    def _rebuild_trie(self):
        self.trie_root = TrieNode()
        for h, token_ids in self.cache.items():
            text = self.text_keys.get(h)
            if text is not None:
                metadata_ref = self.metadata_refs.get(h)
                non_tensor_metadata = self.non_tensor_metadata.get(h)
                reversible_tape = self.reversible_tapes.get(h)
                self._insert_trie(text, token_ids, metadata_ref, non_tensor_metadata, reversible_tape)

    def snapshot(self):
        if self.db_path:
            import threading
            abs_path = str(self.db_path.resolve())
            
            # If there's an active thread writing to this path, wait for it first
            if abs_path in DeterministicContextOrganism._active_saves:
                thread = DeterministicContextOrganism._active_saves[abs_path]
                if thread.is_alive():
                    thread.join()
                    
            # Copy dictionaries on the main thread to prevent thread-safety mutation exceptions during serialization
            save_data = {
                "cache": dict(self.cache),
                "text_keys": dict(self.text_keys),
                "metadata_refs": dict(self.metadata_refs),
                "non_tensor_metadata": dict(self.non_tensor_metadata),
                "reversible_tapes": dict(self.reversible_tapes),
                "thermal_state": dict(self.thermal_state)
            }
            
            def _async_save(path, data, path_str):
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    import json
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                except Exception as e:
                    import sys
                    print(f"[ContextOrganism Warning] Failed to save database asynchronously: {e}", file=sys.stderr)
                finally:
                    DeterministicContextOrganism._active_saves.pop(path_str, None)
            
            thread = threading.Thread(target=_async_save, args=(self.db_path, save_data, abs_path), daemon=True)
            DeterministicContextOrganism._active_saves[abs_path] = thread
            thread.start()

    def restore(self):
        if self.db_path:
            abs_path = str(self.db_path.resolve())
            if abs_path in DeterministicContextOrganism._active_saves:
                thread = DeterministicContextOrganism._active_saves[abs_path]
                if thread.is_alive():
                    thread.join()
                    
        if self.db_path and self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "cache" in data and "text_keys" in data:
                    self.cache = data["cache"]
                    self.text_keys = data["text_keys"]
                    self.metadata_refs = data.get("metadata_refs", {})
                    self.non_tensor_metadata = data.get("non_tensor_metadata", {})
                    self.reversible_tapes = data.get("reversible_tapes", {})
                    self.thermal_state = data.get("thermal_state", {"temperature": 0.0, "average_energy": 0.0})
                else:
                    self.cache = data
                    self.text_keys = {}
                    self.metadata_refs = {}
                    self.non_tensor_metadata = {}
                    self.reversible_tapes = {}
                    self.thermal_state = {"temperature": 0.0, "average_energy": 0.0}
                self._rebuild_trie()
            except Exception as e:
                import sys
                print(f"[ContextOrganism Warning] Failed to restore database: {e}", file=sys.stderr)

    def fork(self) -> "DeterministicContextOrganism":
        new_organism = DeterministicContextOrganism(self.db_path, self.metadata_store.cache_dir)
        new_organism.cache = dict(self.cache)
        new_organism.text_keys = dict(self.text_keys)
        new_organism.metadata_refs = dict(self.metadata_refs)
        new_organism.non_tensor_metadata = dict(self.non_tensor_metadata)
        new_organism.reversible_tapes = dict(self.reversible_tapes)
        new_organism.thermal_state = dict(self.thermal_state)
        new_organism._rebuild_trie()
        return new_organism

    def merge(self, other: "DeterministicContextOrganism"):
        self.cache.update(other.cache)
        self.text_keys.update(other.text_keys)
        self.metadata_refs.update(other.metadata_refs)
        self.non_tensor_metadata.update(other.non_tensor_metadata)
        self.reversible_tapes.update(other.reversible_tapes)
        self.thermal_state.update(other.thermal_state)
        self._rebuild_trie()

    def vaccinate(self, corpus_texts: List[str], tokenizer_fn):
        for text in corpus_texts:
            if text:
                token_ids = tokenizer_fn(text)
                self.set(text, token_ids)
        self.snapshot()

    def save_attention_metadata(self, text: str, tensors: Dict[str, torch.Tensor], non_tensor_metadata: Optional[Dict[str, Any]] = None):
        h = self._hash_text(text)
        metadata_ref = self.metadata_store.save_metadata(h, tensors)
        self.metadata_refs[h] = metadata_ref
        if non_tensor_metadata is not None:
            self.non_tensor_metadata[h] = non_tensor_metadata
        token_ids = self.cache.get(h)
        if token_ids is not None:
            reversible_tape = self.reversible_tapes.get(h)
            self._insert_trie(text, token_ids, metadata_ref, non_tensor_metadata, reversible_tape)

    def load_attention_metadata(self, text: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        h = self._hash_text(text)
        metadata_ref = self.metadata_refs.get(h)
        non_tensor_metadata = self.non_tensor_metadata.get(h, {})
        tensors = {}
        if metadata_ref:
            tensors = self.metadata_store.load_metadata(metadata_ref)
        return tensors, non_tensor_metadata

    def backtrack_sequence(self, token_ids: List[int], target_len: int, tokenizer = None) -> List[int]:
        """
        0022 Reversible Tapes:
        Backtracks a sequence of token IDs to a target base sequence length without expensive full re-tokenization.
        Uses the unmerge history BPE rules.
        """
        if not token_ids:
            return []
        if target_len <= 0:
            return []
        if target_len >= len(token_ids):
            return token_ids
            
        # Build parent_to_pair BPE rules dynamically if tokenizer is provided
        parent_to_pair = {}
        if tokenizer is not None and hasattr(tokenizer, "merge_result"):
            parent_to_pair = {v: k for k, v in tokenizer.merge_result.items()}
            
        def _unmerge(tid: int) -> List[int]:
            if tid in parent_to_pair:
                left, right = parent_to_pair[tid]
                return _unmerge(left) + _unmerge(right)
            return [tid]
            
        # Unmerge the token sequence to base character/byte level tokens
        base_ids = []
        for tid in token_ids:
            base_ids.extend(_unmerge(tid))
            
        # Truncate base sequence to target_len
        if target_len < len(base_ids):
            base_ids = base_ids[:target_len]
            
        # Re-merge the base_ids using tokenizer BPE rules without expensive string operations
        if tokenizer is not None and hasattr(tokenizer, "_exact_merge_python"):
            return tokenizer._exact_merge_python(base_ids)
            
        # Fallback to simple truncation
        return token_ids[:target_len]

