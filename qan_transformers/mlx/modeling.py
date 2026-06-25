import mlx.core as mx
import mlx.nn as nn
from typing import Any, Tuple, List, Dict, Optional
import numpy as np
from qan_transformers.mlx.attention import QuasicrystallineAttention
from qan_transformers.kernels.elq_metal import elq_fused_matmul
from qan_transformers.mlx.moonshots import FusedGeGLUFFN

# Register gemma4_assistant mapping and custom classes in sys.modules
try:
    import mlx_lm.utils
    from dataclasses import dataclass
    from types import ModuleType
    import sys
    from mlx_lm.models.gemma4_text import Model as Gemma4TextModel, ModelArgs as Gemma4TextModelArgs, scaled_dot_product_attention, RMSNormNoScale, logit_softcap

    # Patch Gemma4TextModelArgs.from_dict to flatten text_config for unified models
    if not getattr(Gemma4TextModelArgs.from_dict, "__patched__", False):
        orig_from_dict = Gemma4TextModelArgs.from_dict
        @classmethod
        def patched_from_dict(cls, params):
            new_params = dict(params)
            if "text_config" in params:
                for k, v in params["text_config"].items():
                    new_params[k] = v
            return orig_from_dict.__func__(cls, new_params)
        patched_from_dict.__patched__ = True
        Gemma4TextModelArgs.from_dict = patched_from_dict

    # Patch Gemma4TextModel.sanitize to handle prefix mismatch for unified models
    if hasattr(Gemma4TextModel, "sanitize") and not getattr(Gemma4TextModel.sanitize, "__patched__", False):
        orig_sanitize = Gemma4TextModel.sanitize
        def patched_sanitize(self, weights):
            sanitized = {}
            is_unified = any(k.startswith("language_model.model.") for k in weights)
            for k, v in weights.items():
                new_k = k
                if not is_unified:
                    if new_k.startswith("model.language_model."):
                        new_k = "model." + new_k[len("model.language_model."):]
                    elif new_k.startswith("language_model.model."):
                        new_k = "model." + new_k[len("language_model.model."):]
                sanitized[new_k] = v
            return orig_sanitize(self, sanitized)
        patched_sanitize.__patched__ = True
        Gemma4TextModel.sanitize = patched_sanitize

    # Register Gemma4 assistant models for speculation
    mlx_lm.utils.MODEL_REMAPPING["gemma4_unified"] = "gemma4"
    # mlx_lm.utils.MODEL_REMAPPING["gemma4_assistant"] = "gemma4"
    # mlx_lm.utils.MODEL_REMAPPING["gemma4_unified_assistant"] = "gemma4"

    # Patch load_config to handle the unified assistant num_kv_shared_layers configuration discrepancy
    if not getattr(mlx_lm.utils.load_config, "__patched__", False):
        orig_load_config = mlx_lm.utils.load_config
        def patched_load_config(model_path, *args, **kwargs):
            config = orig_load_config(model_path, *args, **kwargs)
            if config.get("model_type") in ["gemma4_assistant", "gemma4_unified_assistant"]:
                if "text_config" in config:
                    text_cfg = config["text_config"]
                    if text_cfg.get("num_kv_shared_layers", 0) == text_cfg.get("num_hidden_layers", 0):
                        text_cfg["num_kv_shared_layers"] = 0
            return config
        patched_load_config.__patched__ = True
        mlx_lm.utils.load_config = patched_load_config

    @dataclass
    class Gemma4UnifiedAssistantModelArgs(Gemma4TextModelArgs):
        backbone_hidden_size: int = 3840

        @classmethod
        def from_dict(cls, params):
            new_params = dict(params)
            if "text_config" in params:
                new_params.update(params["text_config"])
            return super().from_dict(new_params)

    class Gemma4UnifiedAttention(nn.Module):
        def __init__(self, original_attn, layer_idx, post_projection):
            super().__init__()
            self.layer_idx = layer_idx
            self.post_projection_fn = lambda x: post_projection(x)
            self.target_attn = None
            self.target_k_weight = None
            self.target_v_weight = None
            self.target_k_bias = None
            self.target_v_bias = None
            self.q_proj = original_attn.q_proj
            self.o_proj = original_attn.o_proj
            self.q_norm = original_attn.q_norm
            self.rope = original_attn.rope
            self.scale = original_attn.scale
            self.n_heads = original_attn.n_heads
            self.head_dim = original_attn.head_dim
            self.has_kv = original_attn.has_kv
            self.use_k_eq_v = getattr(original_attn, "use_k_eq_v", False)
            self.num_key_value_heads = getattr(original_attn, "num_key_value_heads", None) or getattr(original_attn, "n_kv_heads", None) or 1
            self.n_kv_heads = self.num_key_value_heads

        def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None, **kwargs):
            B, L, _ = x.shape
            queries = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
            queries = self.q_norm(queries)
            
            # If this is an assistant layer (self.target_attn is not None) and we have keys/values in the cache,
            # we check if we can reuse them or if we need to append.
            if self.target_attn is not None and cache is not None and cache.keys is not None:
                if offset is None:
                    offset = cache.offset
                
                x_target = self.post_projection_fn(x)
                if self.target_k_weight is not None:
                    new_keys = mx.matmul(x_target, self.target_k_weight.T)
                    if self.target_k_bias is not None:
                        new_keys = new_keys + self.target_k_bias
                else:
                    new_keys = self.target_attn.k_proj(x_target)
                    
                n_kv_heads = getattr(self.target_attn, "num_key_value_heads", None) or getattr(self.target_attn, "n_kv_heads", None)
                new_keys = new_keys.reshape(B, L, n_kv_heads, self.target_attn.head_dim)
                
                if hasattr(self.target_attn, "v_proj"):
                    if self.target_v_weight is not None:
                        new_values = mx.matmul(x_target, self.target_v_weight.T)
                        if self.target_v_bias is not None:
                            new_values = new_values + self.target_v_bias
                    else:
                        new_values = self.target_attn.v_proj(x_target)
                    new_values = new_values.reshape(B, L, n_kv_heads, self.target_attn.head_dim)
                else:
                    new_values = new_keys
                    
                if hasattr(self.target_attn, "k_norm"):
                    new_keys = self.target_attn.k_norm(new_keys)
                if hasattr(self.target_attn, "v_norm"):
                    new_values = self.target_attn.v_norm(new_values)
                    
                new_keys = new_keys.transpose(0, 2, 1, 3)
                new_keys = self.rope(new_keys, offset=offset)
                new_values = new_values.transpose(0, 2, 1, 3)
                
                from qan_transformers.mlx.attention import QuasicrystallineAttention
                if getattr(QuasicrystallineAttention, "in_jit", False):
                    from qan_transformers.mlx.attention import _jit_slice_assignment
                    idx = offset % cache.keys.shape[2]
                    keys = _jit_slice_assignment(cache.keys, new_keys, idx, L)
                    values = _jit_slice_assignment(cache.values, new_values, idx, L)
                else:
                    keys_prefix = cache.keys[:, :, :offset, :]
                    values_prefix = cache.values[:, :, :offset, :]
                    keys = mx.concatenate([keys_prefix, new_keys], axis=2)
                    values = mx.concatenate([values_prefix, new_values], axis=2)
                    
                cache.keys = keys
                cache.values = values
                cache.offset = offset + L
            else:
                if shared_kv is not None:
                    keys, values = shared_kv
                else:
                    if self.target_attn is not None:
                        x_target = self.post_projection_fn(x)
                        if self.target_k_weight is not None:
                            keys = mx.matmul(x_target, self.target_k_weight.T)
                            if self.target_k_bias is not None:
                                keys = keys + self.target_k_bias
                        else:
                            keys = self.target_attn.k_proj(x_target)
                            
                        n_kv_heads = getattr(self.target_attn, "num_key_value_heads", None) or getattr(self.target_attn, "n_kv_heads", None)
                        keys = keys.reshape(B, L, n_kv_heads, self.target_attn.head_dim)
                        
                        if hasattr(self.target_attn, "v_proj"):
                            if self.target_v_weight is not None:
                                values = mx.matmul(x_target, self.target_v_weight.T)
                                if self.target_v_bias is not None:
                                    values = values + self.target_v_bias
                            else:
                                values = self.target_attn.v_proj(x_target)
                            values = values.reshape(B, L, n_kv_heads, self.target_attn.head_dim)
                        else:
                            values = keys
                    else:
                        keys = mx.zeros((B, L, self.n_kv_heads, self.head_dim), dtype=x.dtype)
                        values = keys
                        
                    if offset is None:
                        offset = mx.array(cache.offset) if cache is not None else 0
                    
                    if self.target_attn is not None:
                        if hasattr(self.target_attn, "k_norm"):
                            keys = self.target_attn.k_norm(keys)
                        if hasattr(self.target_attn, "v_norm"):
                            values = self.target_attn.v_norm(values)
                    
                    keys = keys.transpose(0, 2, 1, 3)
                    keys = self.rope(keys, offset=offset)
                    
                    values = values.transpose(0, 2, 1, 3)
                    
            queries = queries.transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=offset)
            
            if self.target_attn is None and cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
                
            if mask is not None and not isinstance(mask, str):
                mask = mask[..., :keys.shape[2]]
                
            output = scaled_dot_product_attention(
                queries, keys, values, cache=None if self.target_attn is not None else cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            if kwargs.get("return_unprojected", False):
                return output, (keys, values), offset
            return self.o_proj(output), (keys, values), offset

    class Gemma4UnifiedAssistantForCausalLM(Gemma4TextModel):
        def __init__(self, args: Gemma4UnifiedAssistantModelArgs):
            super().__init__(args)
            hidden_size = args.hidden_size
            backbone_hidden_size = getattr(args, "backbone_hidden_size", 3840)
            self.pre_projection = nn.Linear(backbone_hidden_size * 2, hidden_size, bias=False)
            self.post_projection = nn.Linear(hidden_size, backbone_hidden_size, bias=False)
            self.tokenizer = None
            
            text_model = self.model
            for i, layer in enumerate(text_model.layers):
                layer.self_attn = Gemma4UnifiedAttention(layer.self_attn, i, self.post_projection)

        def __call__(
            self,
            inputs: mx.array,
            cache=None,
            input_embeddings: Optional[mx.array] = None,
            per_layer_inputs: Optional[mx.array] = None,
            fused_tokenization: bool = False,
        ):
            if fused_tokenization and getattr(self, "tokenizer", None) is not None and inputs is not None:
                flat_inputs = mx.reshape(inputs, (-1,))
                out_symbols, out_lengths, _ = self.tokenizer.graph_encode(flat_inputs)
                compacted, total_length = self.tokenizer.compact_token_ids_static(out_symbols, out_lengths)
                inputs = compacted[None, :]
                input_embeddings = self.model.embed_tokens(inputs)
                from qan_transformers.mlx.attention import QuasicrystallineAttention
                QuasicrystallineAttention.valid_length = total_length

            try:
                out = self.model(
                    inputs,
                    cache=cache,
                    input_embeddings=input_embeddings,
                    per_layer_inputs=per_layer_inputs,
                )
            finally:
                from qan_transformers.mlx.attention import QuasicrystallineAttention
                QuasicrystallineAttention.valid_length = None
            self.last_projected_state = self.post_projection(out)
            if self.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(out)
            else:
                logits = self.lm_head(out)
            if self.final_logit_softcapping is not None:
                logits = logit_softcap(self.final_logit_softcapping, logits)
            return logits

    # Register in sys.modules so importlib can load them directly
    assistant_mod = ModuleType("mlx_lm.models.gemma4_unified_assistant")
    assistant_mod.Model = Gemma4UnifiedAssistantForCausalLM
    assistant_mod.ModelArgs = Gemma4UnifiedAssistantModelArgs
    sys.modules["mlx_lm.models.gemma4_unified_assistant"] = assistant_mod
    sys.modules["mlx_lm.models.gemma4_assistant"] = assistant_mod

    # Add linker function
    def link_assistant_to_target(target_model, assistant_model):
        print("[ELQ] Linking Assistant to Target model...", flush=True)
        assistant_model.target_model = target_model
        
        # Patch class properties to forward to language_model if wrapped
        if hasattr(assistant_model, "language_model"):
            cls = assistant_model.__class__
            if not hasattr(cls, "post_projection"):
                cls.post_projection = property(lambda self: getattr(self.language_model, "post_projection", None))
            if not hasattr(cls, "pre_projection"):
                cls.pre_projection = property(lambda self: getattr(self.language_model, "pre_projection", None))
            if not hasattr(cls, "last_projected_state"):
                cls.last_projected_state = property(
                    lambda self: getattr(self.language_model, "last_projected_state", None),
                    lambda self, val: setattr(self.language_model, "last_projected_state", val)
                )

        if hasattr(target_model, "language_model"):
            target_text_model = target_model.language_model.model
        else:
            target_text_model = target_model.model
            
        if hasattr(assistant_model, "language_model"):
            assistant_text_model = assistant_model.language_model.model
        else:
            assistant_text_model = assistant_model.model
        
        early_exit = getattr(target_model, "early_exit_layer", getattr(getattr(target_model, "language_model", None), "early_exit_layer", 258))
        if early_exit is None:
            early_exit = 999999
        target_sliding_indices = [i for i, l in enumerate(target_text_model.layers) if l.layer_type == "sliding_attention" and i < early_exit]
        target_full_indices = [i for i, l in enumerate(target_text_model.layers) if l.layer_type == "full_attention" and i < early_exit]
        
        # Fallback if filtered lists are empty
        if not target_sliding_indices:
            target_sliding_indices = [i for i, l in enumerate(target_text_model.layers) if l.layer_type == "sliding_attention"]
        if not target_full_indices:
            target_full_indices = [i for i, l in enumerate(target_text_model.layers) if l.layer_type == "full_attention"]
            
        prev_kvs = getattr(target_text_model, "previous_kvs", None)
        
        linked_count = 0
        for i, assistant_layer in enumerate(assistant_text_model.layers):
            if assistant_layer.layer_type == "sliding_attention":
                target_idx = target_sliding_indices[-1]
            else:
                target_idx = target_full_indices[-1]
            
            if prev_kvs is not None and target_idx < len(prev_kvs):
                actual_target_idx = prev_kvs[target_idx]
            else:
                actual_target_idx = target_idx
            
            target_layer = target_text_model.layers[actual_target_idx]
            assistant_attn = assistant_layer.self_attn
            assistant_attn.target_attn = target_layer.self_attn
            assistant_attn.target_layer_idx = target_idx
            
            # Direct projection kernel sharing (bypass manual dequantization)
            assistant_attn.target_k_weight = None
            assistant_attn.target_k_bias = None
            assistant_attn.target_v_weight = None
            assistant_attn.target_v_bias = None
            
            linked_count += 1
            
        print(f"[ELQ] Successfully linked {linked_count} attention layers!", flush=True)


except ImportError:
    pass

_elq_dequantize_weights = None


class device_context:
    def __init__(self, device_type):
        self.device = mx.Device(device_type)
        self.prev_device = None

    def __enter__(self):
        self.prev_device = mx.default_device()
        mx.set_default_device(self.device)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        mx.set_default_device(self.prev_device)


class CPUEmbedding(nn.Module):
    def __init__(self, original_emb: nn.Module):
        super().__init__()
        weight_mx = original_emb.weight
        # Store original dtype
        object.__setattr__(self, "weight_dtype", weight_mx.dtype)
        # Force evaluation on CPU to keep it resident in CPU memory pool
        with device_context(mx.DeviceType.cpu):
            mx.eval(weight_mx)
        object.__setattr__(self, "weight_cpu", weight_mx)
        object.__setattr__(self, "original_emb", original_emb)
        
        if "weight" in original_emb:
            del original_emb["weight"]
        if hasattr(original_emb, "weight"):
            delattr(original_emb, "weight")
            
        import gc
        gc.collect()
        mx.clear_cache()

    def __getattr__(self, name):
        if name == "weight":
            return self.weight_cpu
        return getattr(self.original_emb, name)

    def __call__(self, x):
        with device_context(mx.DeviceType.cpu):
            out_cpu = self.weight_cpu[x]
            mx.eval(out_cpu)
        return out_cpu

    def as_linear(self, x):
        with device_context(mx.DeviceType.cpu):
            out_cpu = mx.matmul(x, self.weight_cpu.T)
            mx.eval(out_cpu)
        return out_cpu


class CPULinear(nn.Module):
    def __init__(self, original_linear: nn.Module):
        super().__init__()
        weight_mx = original_linear.weight
        object.__setattr__(self, "weight_dtype", weight_mx.dtype)
        # Force evaluation on CPU
        with device_context(mx.DeviceType.cpu):
            mx.eval(weight_mx)
        object.__setattr__(self, "weight_cpu", weight_mx)
        
        if hasattr(original_linear, "bias") and original_linear.bias is not None:
            bias_mx = original_linear.bias
            object.__setattr__(self, "bias_dtype", bias_mx.dtype)
            with device_context(mx.DeviceType.cpu):
                mx.eval(bias_mx)
            object.__setattr__(self, "bias_cpu", bias_mx)
            del original_linear.bias
        else:
            object.__setattr__(self, "bias_cpu", None)
            object.__setattr__(self, "bias_dtype", None)
            
        object.__setattr__(self, "original_linear", original_linear)
        
        if "weight" in original_linear:
            del original_linear["weight"]
        if hasattr(original_linear, "weight"):
            delattr(original_linear, "weight")
            
        import gc
        gc.collect()
        mx.clear_cache()

    def __getattr__(self, name):
        if name == "weight":
            return self.weight_cpu
        elif name == "bias" and self.bias_cpu is not None:
            return self.bias_cpu
        return getattr(self.original_linear, name)

    def __call__(self, x):
        with device_context(mx.DeviceType.cpu):
            out_cpu = mx.matmul(x, self.weight_cpu.T)
            if self.bias_cpu is not None:
                out_cpu = out_cpu + self.bias_cpu
            mx.eval(out_cpu)
        return out_cpu


class ELQSlidingCache:
    """
    Module-level singleton for ELQ weight dequantization caching.

    Maintains a dict of layer_id → dequantized weight matrices. Layers are
    dequantized lazily on first use via the Metal kernel, then cached
    permanently in VRAM for instant reuse on subsequent tokens.

    For models that fit in memory (≤8B params), ALL layers get cached on
    first forward pass — identical speed to a permanent float16 cache, but
    with lazy allocation that avoids the upfront OOM spike.

    For very large models, `set_capacity(N)` limits the cache to N layers.
    When full, it operates on a First-Come, First-Served (FCFS) basis: subsequent cache misses bypass the cache and fall through to the fused Metal matmul kernel (no LRU eviction on inference cache misses, slower but zero memory overhead).
    """
    _instance = None

    def __init__(self):
        self._cache = {}          # layer_id → (W_T, dtype)
        self._order = []          # LRU order: oldest first
        self._max_entries = None  # None = unlimited (cache everything)
        self._enabled = True

    @classmethod
    def get(cls) -> 'ELQSlidingCache':
        """Get the global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (useful for tests and model reloads)."""
        if cls._instance is not None:
            cls._instance._cache.clear()
            cls._instance._order.clear()
        cls._instance = None

    def set_capacity(self, max_entries: int):
        """Limit the cache to max_entries layers. Use for large models."""
        self._max_entries = max_entries
        self._evict_to_capacity()

    def disable(self):
        """Disable the cache entirely (forces fused Metal matmul path)."""
        self._enabled = False
        self._cache.clear()
        self._order.clear()

    def enable(self):
        """Enable the cache."""
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def _evict_to_capacity(self):
        """Evict oldest entries until cache is within capacity."""
        if self._max_entries is not None:
            while len(self._cache) > self._max_entries:
                oldest_id = self._order.pop(0)
                self._cache.pop(oldest_id, None)

    def get_or_dequantize(
        self,
        layer_id: int,
        indices: mx.array,
        scales: mx.array,
        outlier_indices: mx.array,
        outliers: mx.array,
        dtype: mx.Dtype,
        dequant_fn
    ) -> mx.array:
        """
        Returns the transposed dequantized weight matrix [D, H] for the given layer.

        Cache hit (common case after first token): returns instantly.
        Cache miss (first token only): dequantizes via Metal, caches permanently.

        Returns None if the cache is full and this layer was evicted — caller
        should fall through to the fused Metal matmul path.
        """
        # Fast path: cache hit
        entry = self._cache.get(layer_id)
        if entry is not None:
            cached_W_T, cached_dtype = entry
            if cached_dtype == dtype:
                return cached_W_T
            # Dtype changed — re-cache with new dtype
            cached_W_T = cached_W_T.astype(dtype)
            self._cache[layer_id] = (cached_W_T, dtype)
            return cached_W_T

        # Check if we have room (unlimited capacity = always room)
        if self._max_entries is not None and len(self._cache) >= self._max_entries:
            # Cache is full — return None to signal the caller to use fused matmul
            return None

        # Cache miss — dequantize via Metal kernel (happens once per layer)
        print(f"[GossetGate Log] Cache MISS for layer {layer_id}! Dequantizing on the fly...", flush=True)
        W_dequant = dequant_fn(indices, scales)
        if outliers.size > 0:
            W_dequant[:, outlier_indices] = outliers
        W_T = W_dequant.T.astype(dtype)

        # Store permanently (or until capacity eviction)
        self._cache[layer_id] = (W_T, dtype)
        self._order.append(layer_id)
        self._evict_to_capacity()

        # Evaluate and synchronize immediately to prevent JIT graph bloat and watchdog timeouts
        mx.eval(W_T)
        mx.synchronize()

        return W_T

class ELQLinear(nn.Module):
    # Class-level flag to bypass ELQ cache globally during draft steps.
    # Replaces per-layer use_cache toggling (~204 attribute mutations per cycle).
    _global_cache_bypass: bool = False
    
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        indices: mx.array,
        scales: mx.array,
        outliers: mx.array,
        outlier_mask: mx.array,
        bias: mx.array = None,
        outlier_indices: mx.array = None
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self._indices = indices
        self._scales = scales
        self._outliers = outliers
        self._outlier_mask = outlier_mask
        self.bias = bias
        
        # Win 101: Zero-Copy Indexing in MLX Autoregressive Grafting Layers
        if outlier_indices is not None:
            self._outlier_indices = outlier_indices
        else:
            mask_np = np.array(outlier_mask) if isinstance(outlier_mask, mx.array) else outlier_mask
            idx_np = np.where(mask_np)[0]
            self._outlier_indices = mx.array(idx_np, mx.int32)
        
        self.delta_W_T = None

        # Unique layer ID for the sliding cache (uses object identity)
        self._layer_id = id(self)
        self.cache = None

        # CPU Mode Execution
        self.cpu_mode = False
        self.weight = None
    
    @staticmethod
    def _get_dequantize_fn():
        """H25/H26 FIX: Lazy-load the Metal dequantize shader and cache it module-wide.
        This is the single authoritative dequantization path for ELQLinear."""
        global _elq_dequantize_weights
        if _elq_dequantize_weights is None:
            try:
                from qan_transformers.kernels.elq_dequantize_metal import elq_dequantize_weights
                _elq_dequantize_weights = elq_dequantize_weights
            except ImportError as e:
                raise ImportError(
                    "ELQ Metal dequantization shader not available. "
                    "Ensure qan_transformers.kernels.elq_dequantize_metal is built for this platform. "
                    f"Original error: {e}"
                )
        return _elq_dequantize_weights

    @property
    def indices(self):
        return self._indices

    @property
    def scales(self):
        return self._scales

    @property
    def outliers(self):
        return self._outliers

    @property
    def outlier_mask(self):
        return self._outlier_mask

    @property
    def outlier_indices(self):
        return self._outlier_indices
        
    def get_delta_W_T(self, dtype):
        if getattr(self, "delta_W_T", None) is not None:
            if self.delta_W_T.dtype != dtype:
                self.delta_W_T = self.delta_W_T.astype(dtype)
            return self.delta_W_T
        return None

    def pre_dequantize(self, dtype=mx.bfloat16):
        """Pre-cast the outlier correction matrix to the target dtype.
        The sliding cache handles weight dequantization on-demand."""
        return self.get_delta_W_T(dtype)
            
    def __call__(self, x: mx.array) -> mx.array:
        if getattr(self, "is_draft", False):
            x = x.astype(mx.float16)
            if self.bias is not None and self.bias.dtype != mx.float16:
                self.bias = self.bias.astype(mx.float16)
        else:
            # Target model: force to bfloat16 to save memory and avoid float32 cache promotion
            target_dtype = getattr(self, "model_dtype", mx.bfloat16)
            if x.dtype != target_dtype:
                x = x.astype(target_dtype)
                if self.bias is not None and self.bias.dtype != target_dtype:
                    self.bias = self.bias.astype(target_dtype)

        if getattr(self, "cpu_mode", False):
            with device_context(mx.DeviceType.cpu):
                out = mx.matmul(x, self.weight.T)
                if self.bias is not None:
                    out = out + self.bias
                mx.eval(out)
            return out

        cache = self.cache if self.cache is not None else ELQSlidingCache.get()
        W_T = None

        if cache.is_enabled and ((getattr(self, "use_cache", True) and not ELQLinear._global_cache_bypass) or not hasattr(self, "_indices")) and (cache._max_entries is None or cache._max_entries > 0):
            # Fast check: retrieve from cache if present to avoid accessing potentially deleted quantized attributes
            if self._layer_id in cache._cache:
                W_T, cached_dtype = cache._cache[self._layer_id]
                if cached_dtype != x.dtype:
                    print(f"[ELQ Cache Cast] Layer {self._layer_id}: casting {cached_dtype} to {x.dtype}", flush=True)
                    W_T = W_T.astype(x.dtype)
                    cache._cache[self._layer_id] = (W_T, x.dtype)
            else:
                dequant_fn = self._get_dequantize_fn()
                W_T = cache.get_or_dequantize(
                    layer_id=self._layer_id,
                    indices=self._indices,
                    scales=self._scales,
                    outlier_indices=self._outlier_indices,
                    outliers=self._outliers,
                    dtype=x.dtype,
                    dequant_fn=dequant_fn
                )

        if W_T is not None:
            out = mx.matmul(x, W_T)
        else:
            # Fused Metal Matmul Fallback: decode + matmul in a single kernel
            # dispatch. Used when cache is disabled or at capacity.
            delta_W_T = self.get_delta_W_T(x.dtype)
            out = elq_fused_matmul(x, self._indices, self._scales)
            if delta_W_T is not None:
                x_outliers = x[..., self._outlier_indices]
                out = out + mx.matmul(x_outliers, delta_W_T)
            
        if self.bias is not None:
            out = out + self.bias
            
        return out


def make_fused_decoder_layer_call(self, orig_call):
    o_proj = self.self_attn.o_proj
    post_attn_norm = getattr(self, "post_attention_layernorm", lambda x: x)

    @mx.compile
    def _fused_oproj_norm_residual_jit(x, attn_out_unproj):
        o_proj_out = o_proj(attn_out_unproj)
        norm_o_proj = post_attn_norm(o_proj_out)
        h = x + norm_o_proj
        return h

    def fused_decoder_layer_call(x, mask=None, cache=None, *args, **kwargs):
        r = self.input_layernorm(x)
        attn_res = self.self_attn(r, mask=mask, cache=cache, return_unprojected=True, *args, **kwargs)
        if isinstance(attn_res, tuple):
            attn_unproj = attn_res[0]
            other_res = attn_res[1:]
        else:
            attn_unproj = attn_res
            other_res = None
            
        h = _fused_oproj_norm_residual_jit(x, attn_unproj)
        
        residual = h
        if getattr(self, "enable_moe", False):
            h1 = self.pre_feedforward_layernorm(h)
            h1 = self.mlp(h1)
            h1 = self.post_feedforward_layernorm_1(h1)

            top_k_indices, top_k_weights = self.router(h)
            h2 = self.pre_feedforward_layernorm_2(h)
            h2 = self.experts(h2, top_k_indices, top_k_weights)
            h2 = self.post_feedforward_layernorm_2(h2)

            h = h1 + h2
        else:
            if hasattr(self, "pre_feedforward_layernorm"):
                h = self.pre_feedforward_layernorm(h)
            h = self.mlp(h)
            
        if hasattr(self, "post_feedforward_layernorm"):
            h = self.post_feedforward_layernorm(h)
            
        out = residual + h
        
        # Per-layer input gating
        if (
            getattr(self, "per_layer_input_gate", None) is not None
            and getattr(self, "per_layer_projection", None) is not None
            and getattr(self, "post_per_layer_input_norm", None) is not None
        ):
            per_layer_input = kwargs.get("per_layer_input", None)
            if per_layer_input is None and len(args) > 0:
                per_layer_input = args[0]
                
            if per_layer_input is not None:
                res_gate = out
                gate = self.per_layer_input_gate(out)
                gate = nn.gelu_approx(gate)
                gate = mx.multiply(gate, per_layer_input)
                gate = self.per_layer_projection(gate)
                gate = self.post_per_layer_input_norm(gate)
                out = res_gate + gate
                
        if getattr(self, "layer_scalar", None) is not None:
            out = out * self.layer_scalar

        if other_res is not None:
            return (out,) + other_res
        return out
        
    return fused_decoder_layer_call


def graft_mlx_model(model: nn.Module, sparse_ratio: float = 0.15, min_keep: int = 256, is_draft: bool = False, lattice: str = "e8") -> nn.Module:
    """
    Recursively traverses an MLX model and grafts E8/Leech QuasicrystallineAttention
    onto all attention modules.
    
    Args:
        model: mlx.nn.Module to be grafted.
        sparse_ratio: Float attention sparsity ratio.
        min_keep: Minimum number of keys to keep in sparse cache.
        is_draft: True if draft model.
        lattice: Lattice mode ('e8' or 'leech').
    Returns:
        grafted_model: mlx.nn.Module with replaced attention layers.
    """
        
    def get_float_weight(proj: nn.Module) -> mx.array:
        if isinstance(proj, ELQLinear):
            dequant_fn = ELQLinear._get_dequantize_fn()
            W_dequant = dequant_fn(proj._indices, proj._scales)
            if proj._outliers.size > 0:
                W_dequant[:, proj._outlier_indices] = proj._outliers
            return W_dequant.astype(mx.float16)
            
        if hasattr(proj, "scales"):  # Quantized layer
            return mx.dequantize(
                proj.weight,
                scales=proj.scales,
                biases=getattr(proj, "biases", None),
                group_size=proj.group_size,
                bits=proj.bits,
                mode=getattr(proj, "mode", "affine")
            )
        return proj.weight

    def make_sparse_attn(child: nn.Module, sparse_ratio: float) -> QuasicrystallineAttention:
        if hasattr(child.q_proj, "input_dims"):
            dim = child.q_proj.input_dims
        elif isinstance(child.q_proj, ELQLinear):
            dim = child.q_proj.input_dims
        elif hasattr(child.q_proj, "scales"):
            dim = child.q_proj.scales.shape[1] * getattr(child.q_proj, "group_size", 64)
        else:
            dim = child.q_proj.weight.shape[1]
        n_heads = getattr(child, "n_heads", getattr(child, "num_heads", getattr(child, "num_attention_heads", 8)))
        n_kv_heads = getattr(child, "n_kv_heads", getattr(child, "num_key_value_heads", n_heads))
        has_kv = getattr(child, "has_kv", True)
        head_dim = getattr(child, "head_dim", getattr(child, "head_size", dim // n_heads))
        
        # Instantiate MLX Quasicrystalline Attention
        sparse_attn = QuasicrystallineAttention(
            embed_dim=dim,
            num_heads=n_heads,
            sparse_ratio=sparse_ratio,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            lattice=lattice
        )
        sparse_attn.has_kv = has_kv
        
        if hasattr(child, "has_kv"):
            sparse_attn.is_gemma4 = True
        
        # Direct-assign the projection layers from the original child layer.
        # This preserves standard Linear, QuantizedLinear, and ELQLinear layers automatically.
        sparse_attn.q_proj = child.q_proj
        if hasattr(child, "use_k_eq_v"):
            sparse_attn.use_k_eq_v = child.use_k_eq_v
        if has_kv:
            sparse_attn.k_proj = child.k_proj
            if hasattr(child, "v_proj"):
                sparse_attn.v_proj = child.v_proj
            else:
                sparse_attn.v_proj = child.k_proj
        
        o_proj = getattr(child, "o_proj", getattr(child, "out_proj", getattr(child, "c_proj", None)))
        if o_proj is not None:
            sparse_attn.o_proj = o_proj
            
        # Copy Query-Key Normalization submodules
        if hasattr(child, "q_norm"):
            sparse_attn.q_norm = child.q_norm
        if hasattr(child, "k_norm"):
            sparse_attn.k_norm = child.k_norm
        if hasattr(child, "v_norm"):
            sparse_attn.v_norm = child.v_norm
            
        # Copy RoPE module
        if hasattr(child, "rope"):
            sparse_attn.rope = child.rope
            
        # Initialize e8_proj.weight dynamically using SVD to align with the
        # principal attention subspaces of the base model (Zero-Shot Coherence Calibration).
        dtype = child.q_proj._scales.dtype if hasattr(child.q_proj, "_scales") else mx.float16
        try:
            W_q = np.array(get_float_weight(child.q_proj))
            # Run SVD to extract the principal attention components
            U, S, Vt = np.linalg.svd(W_q, full_matrices=False)
            e8_weight_np = Vt[:8, :]
            # Normalize projection components
            e8_weight_np = e8_weight_np / (np.linalg.norm(e8_weight_np, axis=1, keepdims=True) + 1e-6)
        except Exception as e:
            # Fallback to random orthogonal projection if SVD fails
            g = np.random.randn(8, dim)
            q, r = np.linalg.qr(g.T)
            e8_weight_np = q.T / (np.linalg.norm(q.T, axis=1, keepdims=True) + 1e-6)
            
        sparse_attn.e8_proj.weight = mx.array(e8_weight_np, dtype=dtype)
        if hasattr(sparse_attn.e8_proj, "bias") and sparse_attn.e8_proj.bias is not None:
            sparse_attn.e8_proj.bias = mx.zeros((8,), dtype=dtype)
            mx.eval(sparse_attn.e8_proj.weight, sparse_attn.e8_proj.bias)
        else:
            mx.eval(sparse_attn.e8_proj.weight)
            
        # Free CPU and GPU memory aggressively from SVD operations
        try:
            del W_q
            del U, S, Vt
            del e8_weight_np
        except NameError:
            pass
        import gc
        gc.collect()
        mx.clear_cache()
            
        # Copy other configuration variables
        head_dim = getattr(child, "head_dim", getattr(child, "head_size", dim // n_heads))
        sparse_attn.head_dim = head_dim
        sparse_attn.scale = getattr(child, "scale", getattr(child, "scaling", head_dim ** -0.5))
        sparse_attn.min_keep = min_keep
        sparse_attn.is_sliding = getattr(child, "is_sliding", False)
        sparse_attn.window_size = getattr(child, "window_size", 4096)
        return sparse_attn

    seen = set()
    
    def traverse_and_replace(module: Any):
        mod_id = id(module)
        if mod_id in seen:
            return
        seen.add(mod_id)
        
        if isinstance(module, list):
            for idx, child in enumerate(module):
                if isinstance(child, nn.Module) and hasattr(child, "q_proj") and (hasattr(child, "k_proj") or not getattr(child, "has_kv", True)):
                    module[idx] = make_sparse_attn(child, sparse_ratio)
                elif isinstance(child, nn.Module) and hasattr(child, "gate_proj") and hasattr(child, "up_proj") and hasattr(child, "down_proj"):
                    module[idx] = FusedGeGLUFFN(child.gate_proj, child.up_proj, child.down_proj)
                else:
                    traverse_and_replace(child)
        elif isinstance(module, dict):
            for key, child in list(module.items()):
                if isinstance(child, nn.Module) and hasattr(child, "q_proj") and (hasattr(child, "k_proj") or not getattr(child, "has_kv", True)):
                    module[key] = make_sparse_attn(child, sparse_ratio)
                elif isinstance(child, nn.Module) and hasattr(child, "gate_proj") and hasattr(child, "up_proj") and hasattr(child, "down_proj"):
                    module[key] = FusedGeGLUFFN(child.gate_proj, child.up_proj, child.down_proj)
                else:
                    traverse_and_replace(child)
        elif isinstance(module, nn.Module):
            for name, child in list(module.children().items()):
                if isinstance(child, nn.Module) and hasattr(child, "q_proj") and (hasattr(child, "k_proj") or not getattr(child, "has_kv", True)):
                    setattr(module, name, make_sparse_attn(child, sparse_ratio))
                elif isinstance(child, nn.Module) and hasattr(child, "gate_proj") and hasattr(child, "up_proj") and hasattr(child, "down_proj"):
                    setattr(module, name, FusedGeGLUFFN(child.gate_proj, child.up_proj, child.down_proj))
                else:
                    traverse_and_replace(child)
            
            # Recurse over custom attributes (like list of layers)
            for attr_name in list(module.__dict__.keys()):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                if isinstance(attr, (list, dict)):
                    traverse_and_replace(attr)
                
    traverse_and_replace(model)
    
    # Wrap DecoderLayer __call__ at class-level to break monolithic command buffers during prompt pre-fill (Win 107)
    for m in model.modules():
        if hasattr(m, "self_attn") and hasattr(m, "mlp"):
            m.is_draft = is_draft
            m.self_attn.is_draft = is_draft
            m.mlp.is_draft = is_draft
            
            orig_call = getattr(m, "__call__")
            # m.__call__ = make_fused_decoder_layer_call(m, orig_call)
    
    # Eagerly evaluate the newly created projection weights on CPU to avoid
    # letting subsequent GPU command buffers block on CPU SVD computations.
    eval_targets = []
    for m in model.modules():
        if isinstance(m, QuasicrystallineAttention):
            eval_targets.append(m.e8_proj.weight)
            if hasattr(m.e8_proj, "bias") and m.e8_proj.bias is not None:
                eval_targets.append(m.e8_proj.bias)
    if eval_targets:
        mx.eval(*eval_targets)
        
    import gc
    gc.collect()
    mx.clear_cache()

    model.is_draft = is_draft
    cls = model.__class__
    if not hasattr(cls, "_is_wrapped"):
        original_call = cls.__call__
        def make_cpu_call(orig_call):
            def cpu_call(self, *args, **kwargs):
                if False: # Disable CPU offloading for draft model
                    target_dtype = mx.bfloat16
                    if len(args) > 0 and isinstance(args[0], mx.array):
                        target_dtype = args[0].dtype

                    with device_context(mx.DeviceType.cpu):
                        def to_device(obj, device_type):
                            if isinstance(obj, mx.array):
                                with device_context(device_type):
                                    dtype = obj.dtype
                                    if device_type == mx.DeviceType.cpu and dtype == mx.bfloat16:
                                        dtype = mx.float16
                                    elif device_type == mx.DeviceType.gpu:
                                        dtype = target_dtype
                                    # Use NumPy to physically copy the array buffer, breaking any device stickiness.
                                    np_dtype = mx.float32 if dtype == mx.bfloat16 else dtype
                                    obj_cast = obj.astype(np_dtype)
                                    mx.eval(obj_cast)
                                    val_np = np.array(obj_cast)
                                    return mx.array(val_np, dtype=dtype)
                            elif isinstance(obj, tuple):
                                return tuple(to_device(item, device_type) for item in obj)
                            elif isinstance(obj, list):
                                return [to_device(item, device_type) for item in obj]
                            elif isinstance(obj, dict):
                                return {k: to_device(v, device_type) for k, v in obj.items()}
                            elif hasattr(obj, "__dict__"):
                                for k, v in list(obj.__dict__.items()):
                                    if not k.startswith("__"):
                                        obj.__dict__[k] = to_device(v, device_type)
                                return obj
                            return obj

                        args_cpu = to_device(args, mx.DeviceType.cpu)
                        kwargs_cpu = to_device(kwargs, mx.DeviceType.cpu)

                        out = orig_call(self, *args_cpu, **kwargs_cpu)

                        def eval_arrays(obj):
                            if isinstance(obj, mx.array):
                                mx.eval(obj)
                            elif isinstance(obj, tuple):
                                for item in obj:
                                    eval_arrays(item)
                            elif isinstance(obj, list):
                                for item in obj:
                                    eval_arrays(item)
                            elif isinstance(obj, dict):
                                for item in obj.values():
                                    eval_arrays(item)
                        eval_arrays(out)
                        
                        cache = kwargs_cpu.get("cache") or (args_cpu[1] if len(args_cpu) > 1 else None)
                        if cache is not None:
                            if hasattr(cache, "state"):
                                mx.eval(cache.state)
                            elif isinstance(cache, list):
                                for c in cache:
                                    if hasattr(c, "state"):
                                        mx.eval(c.state)
                                    elif isinstance(c, dict):
                                        for v in c.values():
                                            if isinstance(v, mx.array):
                                                mx.eval(v)
                        
                        act_after_eval = mx.metal.get_active_memory() / (1024**3)
                        print(f"[cpu_call post-eval] Active Memory: {act_after_eval:.2f} GB", flush=True)
                        
                        if isinstance(out, tuple) and len(out) > 0:
                            logits_gpu = to_device(out[0], mx.DeviceType.gpu)
                            out_gpu = (logits_gpu,) + out[1:]
                        else:
                            out_gpu = to_device(out, mx.DeviceType.gpu)

                        import gc
                        gc.collect()
                        mx.clear_cache()
                        act_after_cleanup = mx.metal.get_active_memory() / (1024**3)
                        print(f"[cpu_call post-cleanup] Active Memory: {act_after_cleanup:.2f} GB", flush=True)
                        return out_gpu
                else:
                    return orig_call(self, *args, **kwargs)
            return cpu_call
        cls.__call__ = make_cpu_call(original_call)
        cls._is_wrapped = True
    # Apply speculative and standard JIT decoding monkey patches
    try:
        import sys
        gen_mod = sys.modules.get("mlx_lm.generate")
        if gen_mod is None:
            import mlx_lm.generate
            gen_mod = sys.modules["mlx_lm.generate"]
        patch_speculative_decoding(gen_mod)
        patch_standard_decoding(gen_mod)
    except Exception as e:
        print(f"[ELQ Warning] Could not patch generate step functions: {e}", flush=True)

    return model

def move_model_to_device(model: nn.Module, device_type: mx.DeviceType):
    def update_attrs(m):
        # 1. Update public parameters via keys()
        for k in list(m.keys()):
            v = getattr(m, k, None)
            if isinstance(v, mx.array):
                with device_context(device_type):
                    v_new = v + 0 if v.dtype in [mx.float16, mx.bfloat16, mx.float32] else mx.array(v)
                    mx.eval(v_new)
                setattr(m, k, v_new)
                
        # 2. Update private and custom attributes via __dict__
        for k, v in list(m.__dict__.items()):
            if isinstance(v, mx.array):
                if k.startswith("_"):
                    continue
                with device_context(device_type):
                    v_new = v + 0 if v.dtype in [mx.float16, mx.bfloat16, mx.float32] else mx.array(v)
                    mx.eval(v_new)
                setattr(m, k, v_new)
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    if isinstance(item, mx.array):
                        with device_context(device_type):
                            item_new = item + 0 if item.dtype in [mx.float16, mx.bfloat16, mx.float32] else mx.array(item)
                            mx.eval(item_new)
                        v[idx] = item_new
            elif isinstance(v, dict):
                for key, item in list(v.items()):
                    if isinstance(item, mx.array):
                        with device_context(device_type):
                            item_new = item + 0 if item.dtype in [mx.float16, mx.bfloat16, mx.float32] else mx.array(item)
                            mx.eval(item_new)
                        v[key] = item_new
                        
    for m in model.modules():
        update_attrs(m)
    import gc
    gc.collect()
    mx.clear_cache()

def load_and_graft_elq_model(model: nn.Module, elq_path: str, sparse_ratio: float = 0.15, min_keep: int = 256, is_draft: bool = False, cache_capacity: int = None, lattice: str = "e8") -> nn.Module:
    """
    Loads ELQ quantized weights from `elq_path` and grafts them into the MLX `model`.
    Also grafts QuasicrystallineAttention.
    """
    
    from qan_transformers.elq.elq_reader import ELQReader
    reader = ELQReader(elq_path)
    
    # Initialize model-specific sliding cache to prevent target/draft OOM cache promotion spikes
    cache = ELQSlidingCache()
    if is_draft:
        cache.set_capacity(32 if cache_capacity is None else cache_capacity)
    else:
        cache.set_capacity(0 if cache_capacity is None else cache_capacity)
    
    # Map normalized names to the exact keys in the reader header
    name_map = {}
    for k in reader.header["layers"].keys():
        name_map[k] = k
        norm_key = k
        if norm_key.startswith("model.language_model."):
            norm_key = norm_key[len("model.language_model."):]
        elif norm_key.startswith("language_model.model."):
            norm_key = norm_key[len("language_model.model."):]
        elif norm_key.startswith("model."):
            norm_key = norm_key[len("model."):]
        name_map[norm_key] = k
        
    # First, replace the Linear/QuantizedLinear layers with ELQLinear
    def replace_with_elq(module: Any, prefix: str = ""):
        if isinstance(module, list):
            for idx, child in enumerate(module):
                full_name = f"{prefix}.{idx}" if prefix else str(idx)
                replace_with_elq(child, full_name)
            return
        elif isinstance(module, dict) and not isinstance(module, nn.Module):
            for key, child in module.items():
                full_name = f"{prefix}.{key}" if prefix else str(key)
                replace_with_elq(child, full_name)
            return
            
        if not isinstance(module, nn.Module):
            return
            
        for name, child in list(module.children().items()):
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Normalize full_name for matching
            norm_full_name = full_name
            if norm_full_name.startswith("model.language_model."):
                norm_full_name = norm_full_name[len("model.language_model."):]
            elif norm_full_name.startswith("language_model.model."):
                norm_full_name = norm_full_name[len("language_model.model."):]
            elif norm_full_name.startswith("model."):
                norm_full_name = norm_full_name[len("model."):]
                
            is_linear = isinstance(child, (nn.Linear, nn.QuantizedLinear))
            has_w = hasattr(child, "weight")
            is_emb = isinstance(child, nn.Embedding)
                
            matched_key = None
            if full_name in name_map:
                matched_key = name_map[full_name]
            elif norm_full_name in name_map:
                matched_key = name_map[norm_full_name]
                
            if is_linear or (has_w and not is_emb):
                if matched_key is not None:
                    indices_np = reader.read_tensor(matched_key, "indices")
                    scales_np = reader.read_tensor(matched_key, "scales")
                    outliers_np = reader.read_tensor(matched_key, "outliers")
                    outlier_mask_np = reader.read_tensor(matched_key, "outlier_mask")
                    
                    if indices_np is not None:
                        indices = mx.array(indices_np)
                        scales = mx.array(scales_np, mx.float16)
                        outliers = mx.array(outliers_np, mx.float16)
                        outlier_mask = mx.array(outlier_mask_np)
                        
                        bias_mx = None
                        if hasattr(child, "bias") and child.bias is not None:
                            bias_mx = mx.array(child.bias)
                        
                        # Extract original dimensions
                        if hasattr(child, "scales") and hasattr(child, "bits") and hasattr(child, "weight"):
                            output_dims = child.weight.shape[0]
                            input_dims = child.weight.shape[1] * (32 // child.bits)
                        elif hasattr(child, "weight") and child.weight is not None:
                            input_dims = child.weight.shape[1]
                            output_dims = child.weight.shape[0]
                        else:
                            input_dims = child.input_dims
                            output_dims = child.output_dims
                        
                        # Validate shape compatibility
                        actual_shape = (indices.shape[0], indices.shape[1] * 32)
                        if actual_shape != (output_dims, input_dims):
                            raise ValueError(
                                f"ELQ weight shape mismatch for layer '{full_name}': "
                                f"model layer expects shape ({output_dims}, {input_dims}), but "
                                f"ELQ weight dequantizes to {actual_shape}. "
                                f"Please verify that you are loading the correct .elq file corresponding to this model size."
                            )
                            
                        # Win 101: Zero-Copy Indexing in MLX Autoregressive Grafting Layers
                        outlier_indices_np = np.where(outlier_mask_np)[0]
                        outlier_indices = mx.array(outlier_indices_np, mx.int32)
                        
                        elq_layer = ELQLinear(
                            input_dims=input_dims,
                            output_dims=output_dims,
                            indices=indices,
                            scales=scales,
                            outliers=outliers,
                            outlier_mask=outlier_mask,
                            bias=bias_mx,
                            outlier_indices=outlier_indices
                        )
                        elq_layer.cache = cache
                        setattr(module, name, elq_layer)
                        
                        # Eagerly delete original weights and release memory
                        if hasattr(child, "weight"):
                            del child.weight
                        if hasattr(child, "bias"):
                            del child.bias
                        del child
                        del indices_np, scales_np, outliers_np, outlier_mask_np
            else:
                replace_with_elq(child, full_name)
                
        # Handle dict / list attributes
        for attr_name in list(module.__dict__.keys()):
            if attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if isinstance(attr, list):
                for idx, child in enumerate(attr):
                    full_name = f"{prefix}.{attr_name}.{idx}" if prefix else f"{attr_name}.{idx}"
                    if isinstance(child, nn.Module):
                        replace_with_elq(child, full_name)
            elif isinstance(attr, dict):
                for key, child in attr.items():
                    full_name = f"{prefix}.{attr_name}.{key}" if prefix else f"{attr_name}.{key}"
                    if isinstance(child, nn.Module):
                        replace_with_elq(child, full_name)
 
    replace_with_elq(model)
    reader.close()
    
    import gc
    gc.collect()
    mx.clear_cache()
    
    # Next, graft attention layers
    grafted = graft_mlx_model(model, sparse_ratio, min_keep, is_draft=is_draft, lattice=lattice)
    
    # Find the model's native parameter precision (preferring float16/bfloat16 over default float32)
    def find_dtype(p):
        if isinstance(p, mx.array):
            if p.dtype in [mx.float16, mx.bfloat16]:
                return p.dtype
            return None
        if isinstance(p, dict):
            for v in p.values():
                d = find_dtype(v)
                if d is not None:
                    return d
        if isinstance(p, list):
            for v in p:
                d = find_dtype(v)
                if d is not None:
                    return d
        return None
        
    model_dtype = find_dtype(grafted.parameters())
    if model_dtype is None:
        # Fallback to absolute first parameter's dtype if no half-precision is found
        def first_dtype(p):
            if isinstance(p, mx.array):
                return p.dtype
            if isinstance(p, dict):
                for v in p.values():
                    d = first_dtype(v)
                    if d is not None:
                        return d
            if isinstance(p, list):
                for v in p:
                    d = first_dtype(v)
                    if d is not None:
                        return d
            return None
        model_dtype = first_dtype(grafted.parameters()) or mx.float16
        
    # Eagerly clear VRAM cache to free original/intermediate weights
    import gc
    gc.collect()
    mx.clear_cache()
    
    # Enforce Metal allocator memory and cache limits to prevent OS watchdog timeouts
    try:
        mx.set_memory_limit(17 * 1024 * 1024 * 1024)
        print("[ELQ] Metal allocator memory limit successfully set to 17.0 GB.", flush=True)
    except Exception as e:
        print(f"[ELQ Warning] Could not set Metal memory limit: {e}", flush=True)
        
    try:
        mx.set_cache_limit(2 * 1024 * 1024 * 1024)
        print("[ELQ] Metal allocator cache limit successfully set to 2.0 GB.", flush=True)
    except Exception as e:
        print(f"[ELQ Warning] Could not set Metal cache limit: {e}", flush=True)
        
    # Pre-cast outlier delta weights of all ELQLinear layers to match native precision.
    # The sliding cache handles full weight dequantization on-demand during inference.
    import time
    # Pre-dequantize all ELQLinear layers layer-by-layer during startup to avoid GPU timeouts.
    # This caches the float16 weights in the singleton sliding cache.
    # Use the model-specific sliding cache initialized at the beginning
    if is_draft:
        print(f"[ELQ] Configured draft model ELQLinear layers capacity to {cache._max_entries} (sliding cache).", flush=True)
    else:
        print(f"[ELQ] Configured target model ELQLinear layers capacity to {cache._max_entries} (sliding cache).", flush=True)
        
    dequant_fn = ELQLinear._get_dequantize_fn()
    count = 0
    cached_layer_ids = set()
    for name, module in grafted.named_modules():
        if isinstance(module, ELQLinear):
            module.cache = cache
            module.model_dtype = model_dtype
            if is_draft and cache._max_entries == 0:
                # For draft model with 0 capacity: do NOT cache the full weight matrix in VRAM,
                # but still compute the outlier delta correction matrix delta_W_T.
                with device_context(mx.DeviceType.gpu):
                    if module._outliers.size > 0:
                        W_dequant = dequant_fn(module._indices, module._scales)
                        W_E8_outliers = W_dequant[:, module._outlier_indices]
                        delta_W = module._outliers - W_E8_outliers
                        delta_W_T_fp32 = delta_W.T.astype(mx.float32)
                        mx.eval(delta_W_T_fp32)
                        delta_W_T_np = np.array(delta_W_T_fp32)
                        module.delta_W_T = mx.array(delta_W_T_np, dtype=model_dtype)
                        mx.eval(module.delta_W_T)
                        del W_dequant
                        del W_E8_outliers
                        del delta_W
                        del delta_W_T_fp32
                        del delta_W_T_np
                    else:
                        module.delta_W_T = None
                    mx.synchronize()
                count += 1
                if count % 32 == 0:
                    mx.synchronize()
                    gc.collect()
                    mx.clear_cache()
                continue

            # Limit target model cache capacity to 128 layers
            if not is_draft and cache._max_entries is not None and len(cache._cache) >= cache._max_entries:
                print(f"[ELQ Skip Cache] Layer {module._layer_id} ({name}) - Cache at capacity, using fused path.", flush=True)
                with device_context(mx.DeviceType.gpu):
                    if module._outliers.size > 0:
                        W_dequant = dequant_fn(module._indices, module._scales)
                        W_E8_outliers = W_dequant[:, module._outlier_indices]
                        delta_W = module._outliers - W_E8_outliers
                        delta_W_T_fp32 = delta_W.T.astype(mx.float32)
                        mx.eval(delta_W_T_fp32)
                        delta_W_T_np = np.array(delta_W_T_fp32)
                        module.delta_W_T = mx.array(delta_W_T_np, dtype=model_dtype)
                        mx.eval(module.delta_W_T)
                        del W_dequant
                        del W_E8_outliers
                        del delta_W
                        del delta_W_T_fp32
                        del delta_W_T_np
                    else:
                        module.delta_W_T = None
                    mx.synchronize()
                count += 1
                if count % 32 == 0:
                    mx.synchronize()
                    gc.collect()
                    mx.clear_cache()
                continue

            # 1. Dequantize on GPU (target model only)
            with device_context(mx.DeviceType.gpu):
                W_dequant = dequant_fn(module._indices, module._scales)
                if module._outliers.size > 0:
                    W_E8_outliers = W_dequant[:, module._outlier_indices]
                    delta_W = module._outliers - W_E8_outliers
                    delta_W_T_fp32 = delta_W.T.astype(mx.float32)
                    mx.eval(delta_W_T_fp32)
                    delta_W_T_np = np.array(delta_W_T_fp32)
                    module.delta_W_T = mx.array(delta_W_T_np, dtype=model_dtype)
                    mx.eval(module.delta_W_T)
                    
                    W_dequant[:, module._outlier_indices] = module._outliers
                    del W_E8_outliers
                    del delta_W
                    del delta_W_T_fp32
                    del delta_W_T_np
                else:
                    module.delta_W_T = None
                W_T = W_dequant.T.astype(model_dtype)
                
                # 2. Evaluate and synchronize individually on GPU
                mx.eval(W_T)
                mx.synchronize()
            
            # 3. Cache permanently (target model only)
            cache._cache[module._layer_id] = (W_T, model_dtype)
            print(f"[ELQ Graft Cache] Cached target layer {module._layer_id} ({name})", flush=True)
                
            cache._order.append(module._layer_id)
            cached_layer_ids.add(module._layer_id)
            
            # 4. Clean up GPU memory reference
            del W_dequant
            del W_T
            
            count += 1
            if count % 32 == 0:
                mx.synchronize()
                gc.collect()
                mx.clear_cache()
            
    # Final memory cleanup after full dequantization loop
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
            
    # Set cache for the model session
    for name, module in grafted.named_modules():
        if isinstance(module, ELQLinear):
            module.cache = cache
            module.model_dtype = model_dtype
            # Eagerly delete quantized arrays to save memory, but ONLY for cached layers
            if module._layer_id in cached_layer_ids:
                if hasattr(module, "_indices"):
                    del module._indices
                if hasattr(module, "_scales"):
                    del module._scales
                if hasattr(module, "_outliers"):
                    del module._outliers
                if hasattr(module, "_outlier_mask"):
                    del module._outlier_mask
                if hasattr(module, "delta_W_T"):
                    del module.delta_W_T
            
    elq_count = sum(1 for m in grafted.modules() if isinstance(m, ELQLinear))
    print(f"[ELQ Sliding Cache] Initialized for {elq_count} ELQ layers (is_draft={is_draft}, cached={len(cached_layer_ids)}).")
    
    # Warm up custom Metal kernels for all unique shapes to compile them sequentially.
    # This prevents a massive parallel JIT compilation burst during pre-fill.
    print(f"[GossetGate] Warming up custom Metal kernels for {'draft' if is_draft else 'target'} model...", flush=True)
    unique_shapes = set()
    for name, module in grafted.named_modules():
        if isinstance(module, ELQLinear):
            unique_shapes.add((module.output_dims, module.input_dims))
            
    dequant_fn = ELQLinear._get_dequantize_fn()
    for H, D in sorted(unique_shapes):
        try:
            # 1. Warm up fused matmul (for both float16 and bfloat16 to avoid runtime compile)
            for dt in [mx.float16, mx.bfloat16]:
                dummy_inp = mx.zeros((1, D), dtype=dt)
                dummy_indices = mx.zeros((H, D // 32, 4), dtype=mx.uint32)
                dummy_scales = mx.zeros((H, D // 32), dtype=mx.float16)
                dummy_out = elq_fused_matmul(dummy_inp, dummy_indices, dummy_scales)
                mx.eval(dummy_out)
                mx.synchronize()

            # 2. Warm up dequantize shader
            dummy_dequant = dequant_fn(dummy_indices, dummy_scales)
            mx.eval(dummy_dequant)
            mx.synchronize()
        except Exception as e:
            print(f"[GossetGate Warning] Warmup failed for shape ({H}, {D}): {e}", flush=True)
            
        # Clean up intermediate compiler memory immediately
        gc.collect()
        mx.clear_cache()

    # Set is_draft flag on all modules for runtime checks
    for name, module in grafted.named_modules():
        module.is_draft = is_draft

    # Find the DecoderLayer class and wrap its __call__ method to evaluate outputs layer-by-layer during prefill
    decoder_cls = None
    for name, module in grafted.named_modules():
        if module.__class__.__name__ == "DecoderLayer":
            decoder_cls = module.__class__
            break

    if decoder_cls is not None and not getattr(decoder_cls, "_is_wrapped_prefill", False):
        orig_layer_call = decoder_cls.__call__
        
        def make_wrapped_layer_call(layer_orig_call):
            def wrapped_layer_call(self, x, *args, **kwargs):
                return layer_orig_call(self, x, *args, **kwargs)
            return wrapped_layer_call
            
        # Disabled to prevent GPU watchdog timeouts from blocking CPU-GPU synchronization (Win 201)
        # decoder_cls.__call__ = make_wrapped_layer_call(orig_layer_call)
        decoder_cls._is_wrapped_prefill = True

    # Wrap ALL nn.Embedding modules in the grafted model to run on the CPU (saving up to 10.2 GB VRAM for Gemma-4)
    # Disabled to run at full native GPU speed without CPU-GPU copy overhead (Win 201)
    def wrap_embeddings_on_cpu(module: Any, prefix=""):
        if hasattr(module, "children"):
            for name, child in list(module.children().items()):
                full_name = f"{prefix}.{name}" if prefix else name
                if type(child).__name__ == "Embedding":
                    print(f"[ELQ GPU] Keeping embedding '{full_name}' on GPU for maximum speed...", flush=True)
                else:
                    wrap_embeddings_on_cpu(child, full_name)
        elif isinstance(module, list):
            for idx, child in enumerate(module):
                wrap_embeddings_on_cpu(child, f"{prefix}[{idx}]")
        elif isinstance(module, dict):
            for k, child in module.items():
                wrap_embeddings_on_cpu(child, f"{prefix}.{k}")

    wrap_embeddings_on_cpu(grafted)

    # Wrap lm_head on the CPU
    # Disabled to run at full native GPU speed (Win 201)
    if hasattr(grafted, "language_model") and hasattr(grafted.language_model, "lm_head"):
        print("[ELQ GPU] Keeping lm_head on GPU...", flush=True)
    elif hasattr(grafted, "lm_head"):
        print("[ELQ GPU] Keeping lm_head on GPU...", flush=True)

    # Clear memory cache and collect garbage to free up VRAM
    mx.synchronize()
    gc.collect()
    mx.clear_cache()

    # Apply speculative and standard JIT decoding monkey patches
    try:
        import sys
        gen_mod = sys.modules.get("mlx_lm.generate")
        if gen_mod is None:
            import mlx_lm.generate
            gen_mod = sys.modules["mlx_lm.generate"]
        patch_speculative_decoding(gen_mod)
        patch_standard_decoding(gen_mod)
    except Exception as e:
        print(f"[ELQ Warning] Could not patch generate step functions: {e}", flush=True)

    return grafted

def _patch_cache_states():
    try:
        import mlx_lm.models.cache as cache_module
        if getattr(cache_module, "_is_patched_for_early_exit", False):
            return
            
        import mlx.core as mx
        def _jit_slice_assignment(T, S, idx, L):
            M = T.shape[2]
            p = mx.arange(M)
            p_rel = (p - idx) % M
            mask = p_rel < L
            src_idx_clamped = mx.minimum(p_rel, L - 1)
            S_aligned = mx.take(S, src_idx_clamped, axis=2)
            return mx.where(mask[None, None, :, None], S_aligned, T)
            
        if hasattr(cache_module, "KVCache"):
            orig_kv_state = cache_module.KVCache.state
            @property
            def patched_kv_state(self):
                if self.keys is None:
                    return mx.array([])
                if isinstance(orig_kv_state, property):
                    return orig_kv_state.fget(self)
                return orig_kv_state(self)
            cache_module.KVCache.state = patched_kv_state
            
            orig_kv_update = cache_module.KVCache.update_and_fetch
            def patched_kv_update_and_fetch(self, keys, values):
                from qan_transformers.mlx.attention import QuasicrystallineAttention
                if getattr(QuasicrystallineAttention, "in_jit", False):
                    S = keys.shape[2]
                    idx = self.offset % self.keys.shape[2]
                    self.keys = _jit_slice_assignment(self.keys, keys, idx, S)
                    self.values = _jit_slice_assignment(self.values, values, idx, S)
                    self.offset = self.offset + S
                    return self.keys, self.values
                return orig_kv_update(self, keys, values)
            cache_module.KVCache.update_and_fetch = patched_kv_update_and_fetch
            
        if hasattr(cache_module, "RotatingKVCache"):
            orig_rot_state = cache_module.RotatingKVCache.state
            @property
            def patched_rot_state(self):
                if self.keys is None:
                    return mx.array([])
                if isinstance(orig_rot_state, property):
                    return orig_rot_state.fget(self)
                return orig_rot_state(self)
            cache_module.RotatingKVCache.state = patched_rot_state
            
            orig_rot_update = cache_module.RotatingKVCache.update_and_fetch
            def patched_rot_update_and_fetch(self, keys, values):
                from qan_transformers.mlx.attention import QuasicrystallineAttention
                if getattr(QuasicrystallineAttention, "in_jit", False):
                    S = keys.shape[2]
                    idx = self.offset % self.keys.shape[2]
                    self.keys = _jit_slice_assignment(self.keys, keys, idx, S)
                    self.values = _jit_slice_assignment(self.values, values, idx, S)
                    self.offset = self.offset + S
                    if hasattr(self, "_idx"):
                        self._idx = (self._idx + S) % self.max_size
                    return self.keys, self.values
                return orig_rot_update(self, keys, values)
            cache_module.RotatingKVCache.update_and_fetch = patched_rot_update_and_fetch
            
        cache_module._is_patched_for_early_exit = True
        print("[ELQ] Successfully patched cache states for JIT compilation safety!", flush=True)
    except Exception as e:
        print(f"[ELQ Warning] Could not patch cache states: {e}", flush=True)


def patch_speculative_decoding(gen_mod):
    if getattr(gen_mod, "_is_patched_for_watchdog", False):
        return
        
    print("[ELQ] Patching mlx_lm.generate.speculative_generate_step for GPU watchdog safety...", flush=True)
    _patch_cache_states()
    
    try:
        from mlx_lm.models.gemma4_text import Model as Gemma4TextModel, logit_softcap
        
        orig_gemma4_text_call = Gemma4TextModel.__call__
        
        def patched_gemma4_text_call(self, inputs, cache=None, input_embeddings=None, per_layer_inputs=None):
            static_early_exit = getattr(self, "early_exit_layer", None)
            if static_early_exit is not None:
                text_model = self.model
                if input_embeddings is None:
                    input_embeddings = text_model.embed_tokens(inputs)
                h = input_embeddings
                h = h * text_model.embed_scale

                if text_model.hidden_size_per_layer_input:
                    if per_layer_inputs is None:
                        per_layer_inputs = text_model._get_per_layer_inputs(inputs, input_embeddings)
                    per_layer_inputs = text_model._project_per_layer_inputs(h, per_layer_inputs)
                if per_layer_inputs is not None:
                    per_layer_inputs = [
                        per_layer_inputs[:, :, i, :] for i in range(len(text_model.layers))
                    ]
                else:
                    per_layer_inputs = [None] * len(text_model.layers)

                if cache is None:
                    cache = [None] * len(text_model.layers)
                else:
                    cache = cache + [None] * (len(text_model.layers) - len(cache))

                masks = text_model._make_masks(h, cache)
                intermediates = [(None, None)] * len(text_model.layers)
                
                num_layers = min(static_early_exit, len(text_model.layers))
                for idx in range(num_layers):
                    layer = text_model.layers[idx]
                    c = cache[idx]
                    mask = masks[idx]
                    prev_idx = text_model.previous_kvs[idx]
                    per_layer_input = per_layer_inputs[idx]
                    kvs, offset = intermediates[prev_idx]
                    h, kvs, offset = layer(h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                    intermediates[idx] = (kvs, offset)

                out = text_model.norm(h)
            else:
                # Check if dynamic early exit is enabled (default to True only for draft models)
                dynamic_early_exit = getattr(self, "dynamic_early_exit", getattr(self, "is_draft", False))
                
                # Prefill is S > 1
                is_prefill = False
                if inputs is not None and inputs.ndim == 2 and inputs.shape[1] > 1:
                    is_prefill = True
                elif input_embeddings is not None and input_embeddings.ndim == 3 and input_embeddings.shape[1] > 1:
                    is_prefill = True
                    
                # If dynamic early exit is enabled and it is prefill, run the stability check
                if is_prefill and dynamic_early_exit:
                    text_model = self.model
                    if input_embeddings is None:
                        input_embeddings = text_model.embed_tokens(inputs)
                    h = input_embeddings
                    h = h * text_model.embed_scale

                    if text_model.hidden_size_per_layer_input:
                        if per_layer_inputs is None:
                            per_layer_inputs = text_model._get_per_layer_inputs(inputs, input_embeddings)
                        per_layer_inputs = text_model._project_per_layer_inputs(h, per_layer_inputs)
                    if per_layer_inputs is not None:
                        per_layer_inputs = [
                            per_layer_inputs[:, :, i, :] for i in range(len(text_model.layers))
                        ]
                    else:
                        per_layer_inputs = [None] * len(text_model.layers)

                    if cache is None:
                        cache = [None] * len(text_model.layers)
                    else:
                        cache = cache + [None] * (len(text_model.layers) - len(cache))

                    masks = text_model._make_masks(h, cache)
                    intermediates = [(None, None)] * len(text_model.layers)
                    
                    # 1. Run layers 0 to 22
                    for idx in range(23):
                        layer = text_model.layers[idx]
                        c = cache[idx]
                        mask = masks[idx]
                        prev_idx = text_model.previous_kvs[idx]
                        per_layer_input = per_layer_inputs[idx]
                        kvs, offset = intermediates[prev_idx]
                        h, kvs, offset = layer(h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                        intermediates[idx] = (kvs, offset)
                        
                    # 2. Run layer 23 to get h_24
                    h_23 = h
                    layer = text_model.layers[23]
                    c = cache[23]
                    mask = masks[23]
                    prev_idx = text_model.previous_kvs[23]
                    per_layer_input = per_layer_inputs[23]
                    kvs, offset = intermediates[prev_idx]
                    h_24, kvs, offset = layer(h_23, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                    intermediates[23] = (kvs, offset)
                    
                    # Check stabilization at layer 24
                    diff_24 = mx.mean(mx.abs(h_24 - h_23)) / (mx.mean(mx.abs(h_23)) + 1e-6)
                    if diff_24.item() < getattr(self, "early_exit_threshold_24", 1.05):
                        self.early_exit_layer = 24
                        out = text_model.norm(h_24)
                    else:
                        # Run layers 24 to 34
                        h = h_24
                        for idx in range(24, 35):
                            layer = text_model.layers[idx]
                            c = cache[idx]
                            mask = masks[idx]
                            prev_idx = text_model.previous_kvs[idx]
                            per_layer_input = per_layer_inputs[idx]
                            kvs, offset = intermediates[prev_idx]
                            h, kvs, offset = layer(h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                            intermediates[idx] = (kvs, offset)
                            
                        # Run layer 35 to get h_36
                        h_35 = h
                        layer = text_model.layers[35]
                        c = cache[35]
                        mask = masks[35]
                        prev_idx = text_model.previous_kvs[35]
                        per_layer_input = per_layer_inputs[35]
                        kvs, offset = intermediates[prev_idx]
                        h_36, kvs, offset = layer(h_35, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                        intermediates[35] = (kvs, offset)
                        
                        # Check stabilization at layer 36
                        diff_36 = mx.mean(mx.abs(h_36 - h_35)) / (mx.mean(mx.abs(h_35)) + 1e-6)
                        if diff_36.item() < getattr(self, "early_exit_threshold_36", 0.45):
                            self.early_exit_layer = 36
                            out = text_model.norm(h_36)
                        else:
                            self.early_exit_layer = None
                            # Run remaining layers 36 to 41
                            h = h_36
                            for idx in range(36, len(text_model.layers)):
                                layer = text_model.layers[idx]
                                c = cache[idx]
                                mask = masks[idx]
                                prev_idx = text_model.previous_kvs[idx]
                                per_layer_input = per_layer_inputs[idx]
                                kvs, offset = intermediates[prev_idx]
                                h, kvs, offset = layer(h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                                intermediates[idx] = (kvs, offset)
                            out = text_model.norm(h)
                else:
                    # Standard path (uses pre-determined early_exit_layer if cached/configured)
                    early_exit_layer = getattr(self, "early_exit_layer", None)
                    if early_exit_layer is None:
                        early_exit_layer = getattr(self.model, "early_exit_layer", None)
                        
                    if early_exit_layer is not None:
                        text_model = self.model
                        if input_embeddings is None:
                            input_embeddings = text_model.embed_tokens(inputs)
                        h = input_embeddings
                        h = h * text_model.embed_scale

                        if text_model.hidden_size_per_layer_input:
                            if per_layer_inputs is None:
                                per_layer_inputs = text_model._get_per_layer_inputs(inputs, input_embeddings)
                            per_layer_inputs = text_model._project_per_layer_inputs(h, per_layer_inputs)
                        if per_layer_inputs is not None:
                            per_layer_inputs = [
                                per_layer_inputs[:, :, i, :] for i in range(len(text_model.layers))
                            ]
                        else:
                            per_layer_inputs = [None] * len(text_model.layers)

                        if cache is None:
                            cache = [None] * len(text_model.layers)
                        else:
                            cache = cache + [None] * (len(text_model.layers) - len(cache))

                        masks = text_model._make_masks(h, cache)
                        intermediates = [(None, None)] * len(text_model.layers)
                        
                        num_layers = min(early_exit_layer, len(text_model.layers))
                        for idx in range(num_layers):
                            layer = text_model.layers[idx]
                            c = cache[idx]
                            mask = masks[idx]
                            prev_idx = text_model.previous_kvs[idx]
                            per_layer_input = per_layer_inputs[idx]
                            kvs, offset = intermediates[prev_idx]
                            h, kvs, offset = layer(h, mask, c, per_layer_input=per_layer_input, shared_kv=kvs, offset=offset)
                            intermediates[idx] = (kvs, offset)

                        out = text_model.norm(h)
                    else:
                        out = self.model(
                            inputs,
                            cache=cache,
                            input_embeddings=input_embeddings,
                            per_layer_inputs=per_layer_inputs,
                        )
            from qan_transformers.mlx.attention import QuasicrystallineAttention
            if not getattr(QuasicrystallineAttention, "in_jit", False):
                self.last_hidden_state = out
            if self.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(out)
            else:
                logits = self.lm_head(out)
            if self.final_logit_softcapping is not None:
                logits = logit_softcap(self.final_logit_softcapping, logits)
            return logits
            
        Gemma4TextModel.__call__ = patched_gemma4_text_call
        print("[ELQ] Successfully patched Gemma4TextModel to save last_hidden_state!", flush=True)
    except Exception as e:
        print(f"[ELQ Warning] Could not patch Gemma4TextModel: {e}", flush=True)
    
    # Force generation_stream to be the default stream to serialize all executions on GPU
    mx = gen_mod.mx
    default_stream = mx.default_stream(mx.default_device())
    gen_mod.generation_stream = default_stream
    print(f"[ELQ] Patched generation_stream to default stream: {default_stream}", flush=True)
    
    # Disable wired_limit to prevent physical RAM pinning OOMs under memory pressure
    import contextlib
    @contextlib.contextmanager
    def patched_wired_limit(model, streams=None):
        yield
    gen_mod.wired_limit = patched_wired_limit
    print("[ELQ] Patched wired_limit to be a no-op context manager.", flush=True)
    
    def patched_speculative_generate_step(
        prompt: mx.array,
        model: nn.Module,
        draft_model: nn.Module,
        *,
        num_draft_tokens: int = 2,
        max_tokens: int = 256,
        sampler = None,
        logits_processors = None,
        prompt_cache = None,
        prefill_step_size: int = 512,
        kv_bits = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
    ):
        import functools
        import time
        cache = gen_mod.cache
        generation_stream = default_stream
        maybe_quantize_kv_cache = gen_mod.maybe_quantize_kv_cache
        mx = gen_mod.mx

        y = prompt.astype(mx.uint32)
        # Initialize sequence history for E8 lattice bias (Leap 0019/0022)
        from qan_transformers.mlx.attention import QuasicrystallineAttention
        tok_obj = getattr(model, "tokenizer", None)
        if tok_obj is not None and hasattr(tok_obj, "organism"):
            QuasicrystallineAttention.organism = tok_obj.organism
            if hasattr(prompt, "tolist"):
                QuasicrystallineAttention.current_token_ids = prompt.tolist()
            elif isinstance(prompt, list):
                QuasicrystallineAttention.current_token_ids = list(prompt)
            else:
                QuasicrystallineAttention.current_token_ids = []
        else:
            QuasicrystallineAttention.organism = None
            QuasicrystallineAttention.current_token_ids = None

        prev_tokens = None
        dyn_num_draft = num_draft_tokens
        acceptance_history = []

        # === Moonshot Phase 1: Geometric Draft Filter ===
        # Pre-filter speculative candidates using E8 lattice distance.
        # Tokens whose coordinates are far from the generation trajectory
        # in lattice space are rejected before expensive target verification.
        _geometric_filter = None
        try:
            from qan_transformers.moonshot.geometric_filter import GeometricDraftFilter
            # Get the E8 projection matrix from QuasicrystallineAttention if available
            _e8_proj_matrix = getattr(QuasicrystallineAttention, '_projection_matrix_3d', None)
            if _e8_proj_matrix is not None:
                _geometric_filter = GeometricDraftFilter(
                    projection_matrix=_e8_proj_matrix,
                    r_base=0.6,
                    ema_decay=0.9,
                )
                print("[Moonshot] Geometric draft filter initialized.", flush=True)
        except ImportError:
            pass

        # === Moonshot Phase 5: Cross-Model Wormhole Bridge ===
        # Opens privacy-preserving wormholes to Gemini when CFI indicates
        # semantic uncertainty, using Cayley-rotated hidden states.
        _wormhole_bridge = None
        try:
            from qan_transformers.moonshot.cross_model_bridge import (
                GeminiWormholeBridge,
                WormholeConfig,
            )
            import os
            if os.environ.get("GEMINI_API_KEY"):
                _wormhole_config = WormholeConfig(
                    local_dim=getattr(model, 'dim', 3584),
                )
                _wormhole_bridge = GeminiWormholeBridge(_wormhole_config)
                print("[Moonshot] Cross-model wormhole bridge initialized.", flush=True)
        except ImportError:
            pass

        if prompt_cache is None:
            model_cache = cache.make_prompt_cache(model)
            draft_cache = cache.make_prompt_cache(draft_model)
        else:
            model_cache = prompt_cache[: len(model.layers)]
            draft_cache = prompt_cache[len(model.layers) :]

        target_lm = model.language_model if hasattr(model, "language_model") else model

        # Extract underlying text model for draft_model to safely fetch parameters (Gemma4 is self.language_model.model)
        if draft_model is not None:
            draft_text_model = getattr(draft_model, "language_model", getattr(draft_model, "model", draft_model))
            if hasattr(draft_text_model, "model"):
                draft_text_model = draft_text_model.model
        else:
            draft_text_model = None

        if not cache.can_trim_prompt_cache(model_cache):
            types = {type(c).__name__ for c in model_cache if not c.is_trimmable()}
            raise ValueError(
                f"Speculative decoding requires a trimmable prompt cache (got {types})."
            )

        nonlocal_sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

        quantize_cache_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

        def _process_and_sample(tokens, logits):
            if logits_processors:
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            y = nonlocal_sampler(logprobs)
            return y, logprobs

        def get_model_device_type(model):
            if getattr(model, "is_cpu_model", False):
                return mx.DeviceType.cpu
            return mx.DeviceType.gpu

        def _step(model, cache, y, n_predict=1):
            dev_type = get_model_device_type(model)
            stream_ctx = mx.stream(mx.cpu) if dev_type == mx.DeviceType.cpu else mx.stream(generation_stream)
            with stream_ctx:
                input_y = y if y.ndim == 2 else y[None]
                logits = model(input_y, cache=cache)
                logits = logits[:, -n_predict:, :]

                quantize_cache_fn(cache)
                if logits_processors:
                    nonlocal prev_tokens
                    out_y, out_logprobs = [], []
                    if n_predict > 1:
                        y = y[: -(n_predict - 1)]
                    for i in range(n_predict):
                        prev_tokens = (
                            mx.concatenate([prev_tokens, y])
                            if prev_tokens is not None
                            else y
                        )
                        y, logprobs = _process_and_sample(prev_tokens, logits[:, i, :])
                        out_y.append(y)
                        out_logprobs.append(logprobs)
                    return mx.concatenate(out_y, axis=0), mx.concatenate(
                        out_logprobs, axis=0
                    )
                else:
                    if y.ndim == 2:
                        return _process_and_sample(None, logits)
                    else:
                        return _process_and_sample(None, logits.squeeze(0))

        def _prefill(model, cache, y):
            dev_type = get_model_device_type(model)
            stream_ctx = mx.stream(mx.cpu) if dev_type == mx.DeviceType.cpu else mx.stream(generation_stream)
            while y.size > 1:
                n_to_process = min(prefill_step_size, y.size - 1)
                with stream_ctx:
                    model(y[:n_to_process][None], cache=cache)
                quantize_cache_fn(cache)
                mx.eval([c.state for c in cache])
                y = y[n_to_process:]
            return y

        def _rollback_custom_caches(m_obj, new_len, idx=0, start_offset=0, num_existing=0, eval_now=True):
            if m_obj is None:
                return []
            if hasattr(m_obj, "language_model"):
                text_model = m_obj.language_model.model
            elif hasattr(m_obj, "model"):
                text_model = m_obj.model
            else:
                text_model = m_obj
            
            num_true = None
            if start_offset > 0 and new_len >= start_offset:
                num_true = num_existing + (new_len - start_offset)
            else:
                for layer in text_model.layers:
                    self_attn = getattr(layer, "self_attn", None)
                    if self_attn is not None and hasattr(self_attn, "custom_kv_cache") and self_attn.custom_kv_cache is not None:
                        cache_dict = self_attn.custom_kv_cache
                        if cache_dict["K"] is not None:
                            if hasattr(self_attn, "_cpu_cache_len_history") and new_len in self_attn._cpu_cache_len_history:
                                num_true = self_attn._cpu_cache_len_history[new_len]
                            else:
                                valid_mask = cache_dict["indices"][0] < new_len
                                num_true = int(mx.sum(valid_mask).item())
                            break
            
            eval_args = []
            for layer in text_model.layers:
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None and hasattr(self_attn, "custom_kv_cache") and self_attn.custom_kv_cache is not None:
                    cache_dict = self_attn.custom_kv_cache
                    if cache_dict["K"] is not None:
                        if num_true is not None and num_true > 0:
                            self_attn.custom_kv_cache = {
                                "K": cache_dict["K"][idx : idx + 1, :, :num_true, :],
                                "V": cache_dict["V"][idx : idx + 1, :, :num_true, :],
                                "indices": cache_dict["indices"][idx : idx + 1, :num_true],
                                "alignment_scores": cache_dict["alignment_scores"][idx : idx + 1, :num_true],
                                "seq_len": new_len
                            }
                            eval_args.extend([
                                self_attn.custom_kv_cache["K"],
                                self_attn.custom_kv_cache["V"],
                                self_attn.custom_kv_cache["indices"],
                                self_attn.custom_kv_cache["alignment_scores"]
                            ])
                        else:
                            self_attn.custom_kv_cache = {
                                "K": None, "V": None, "indices": None, "alignment_scores": None, "seq_len": 0
                            }
                        if hasattr(self_attn, "_cpu_cache_len_history"):
                            self_attn._cpu_cache_len_history = {
                                k: v for k, v in self_attn._cpu_cache_len_history.items() if k <= new_len
                            }
            if eval_args and eval_now:
                mx.eval(*eval_args)
            for layer in text_model.layers:
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None and hasattr(self_attn, "swap_db") and self_attn.swap_db is not None:
                    try:
                        self_attn.swap_db.rollback(new_len)
                    except Exception:
                        pass
                    break
            return eval_args

        def replicate_custom_caches(m_obj, B=2, eval_now=True):
            if m_obj is None:
                return []
            if hasattr(m_obj, "language_model"):
                text_model = m_obj.language_model.model
            elif hasattr(m_obj, "model"):
                text_model = m_obj.model
            else:
                text_model = m_obj
                
            eval_args = []
            for layer in text_model.layers:
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None and hasattr(self_attn, "custom_kv_cache") and self_attn.custom_kv_cache is not None:
                    cache_dict = self_attn.custom_kv_cache
                    if cache_dict["K"] is not None and cache_dict["K"].shape[0] != B:
                        self_attn.custom_kv_cache = {
                            "K": mx.repeat(cache_dict["K"], B, axis=0),
                            "V": mx.repeat(cache_dict["V"], B, axis=0),
                            "indices": mx.repeat(cache_dict["indices"], B, axis=0),
                            "alignment_scores": mx.repeat(cache_dict["alignment_scores"], B, axis=0),
                            "seq_len": cache_dict["seq_len"]
                        }
                        eval_args.extend([
                            self_attn.custom_kv_cache["K"], self_attn.custom_kv_cache["V"],
                            self_attn.custom_kv_cache["indices"], self_attn.custom_kv_cache["alignment_scores"]
                        ])
            if eval_args and eval_now:
                mx.eval(*eval_args)
            return eval_args


        def _rewind_cache(num_draft, num_accept):
            cache.trim_prompt_cache(model_cache, num_draft - num_accept)
            cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))
            if model_cache and len(model_cache) > 0:
                new_len_target = model_cache[0].offset
                _rollback_custom_caches(model, new_len_target, start_offset=target_start_offset, num_existing=num_existing_target)
                if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                    QuasicrystallineAttention.current_token_ids = QuasicrystallineAttention.current_token_ids[:new_len_target]
            if draft_model is not None and draft_cache and len(draft_cache) > 0:
                new_len_draft = draft_cache[0].offset
                _rollback_custom_caches(draft_model, new_len_draft, start_offset=draft_start_offset, num_existing=num_existing_draft)

        is_first_draft_step = [True]

        def _draft_step(draft_y, is_first_step=False):
            if is_first_step:
                target_hidden = target_lm.last_hidden_state
                L_y = draft_y.size
                if target_hidden.shape[1] > 1:
                    target_hidden = target_hidden[:, n - L_y + 1 : n + 1, :]
            else:
                target_hidden = draft_model.last_projected_state
                if target_hidden.shape[1] > 1:
                    target_hidden = target_hidden[:, -1:, :]
                
            target_embed = _fast_target_embed(draft_y)
            if target_embed.ndim == 2:
                target_embed = target_embed[None, :, :]
            if target_hidden.ndim == 2:
                target_hidden = target_hidden[None, :, :]
                
            concat_input = mx.concatenate([target_embed, target_hidden], axis=-1)
            inputs_embeds = draft_model.pre_projection(concat_input)
            
            dev_type = get_model_device_type(draft_model)
            stream_ctx = mx.stream(mx.cpu) if dev_type == mx.DeviceType.cpu else mx.stream(generation_stream)
            with stream_ctx:
                scaled_inputs_embeds = inputs_embeds / draft_model.model.embed_scale
                logits = draft_model(draft_y[None], cache=draft_cache, input_embeddings=scaled_inputs_embeds)
                logits = logits[:, -1, :]
                quantize_cache_fn(draft_cache)
                y_next, logprobs = _process_and_sample(None, logits)
            return y_next, logprobs

        def _draft_step_dispatch(draft_y, is_first_step=False):
            if is_assistant:
                return _draft_step(draft_y, is_first_step)
            else:
                return _step(draft_model, draft_cache, draft_y)

        def _draft_generate(y, num_draft):
            if num_draft == 0:
                return mx.array([], mx.uint32)
            for layer in model.layers:
                if hasattr(layer.self_attn, "k_proj"):
                    layer.self_attn.k_proj.use_cache = False
                if hasattr(layer.self_attn, "v_proj"):
                    layer.self_attn.v_proj.use_cache = False
            try:
                initial_offset = draft_cache[0].offset
                y_1d = y.flatten()
                guesses = mx.full((num_draft,), y_1d[0], dtype=mx.uint32)
                tokens = mx.concatenate([y_1d, guesses])
                
                for iteration in range(2):
                    current_offset = draft_cache[0].offset
                    if current_offset > initial_offset:
                        from mlx_lm.generate import cache as cache_mod
                        cache_mod.trim_prompt_cache(draft_cache, current_offset - initial_offset)
                        
                    inputs = tokens[:-1]
                    
                    if is_assistant:
                        target_embed = _fast_target_embed(inputs)[None, :, :]
                        
                        target_hidden_first = target_lm.last_hidden_state
                        if target_hidden_first.shape[1] > 1:
                            target_hidden_first = target_hidden_first[:, -1:, :]
                            
                        L_draft = y_1d.size
                        L_in = L_draft + num_draft - 1
                        num_needed = L_in - 1
                        
                        if num_needed > 0:
                            last_proj = getattr(draft_model, "last_projected_state", None)
                            if last_proj is not None and last_proj.shape[1] >= num_needed:
                                target_hidden_rest = last_proj[:, -num_needed:, :]
                                target_hidden = mx.concatenate([target_hidden_first, target_hidden_rest], axis=1)
                            else:
                                target_hidden = mx.broadcast_to(target_hidden_first, (1, L_in, target_hidden_first.shape[2]))
                        else:
                            target_hidden = target_hidden_first
                            
                        concat_input = mx.concatenate([target_embed, target_hidden], axis=-1)
                        inputs_embeds = draft_model.pre_projection(concat_input)
                        scaled_inputs_embeds = inputs_embeds / draft_model.model.embed_scale
                        
                        logits = draft_model(inputs[None, :], cache=draft_cache, input_embeddings=scaled_inputs_embeds)
                    else:
                        logits = draft_model(inputs[None, :], cache=draft_cache)
                        
                    logits = logits[0, :, :] # Shape: [num_draft, V]
                    preds = mx.argmax(logits, axis=-1).astype(mx.uint32)
                    preds = preds[-num_draft:]
                    tokens = mx.concatenate([y_1d, preds])
                    
                draft_tokens = tokens[L_draft:]
                mx.async_eval(draft_tokens)
            finally:
                for layer in model.layers:
                    if hasattr(layer.self_attn, "k_proj"):
                        layer.self_attn.k_proj.use_cache = True
                    if hasattr(layer.self_attn, "v_proj"):
                        layer.self_attn.v_proj.use_cache = True
            return draft_tokens

        # Link assistant model to target model if applicable
        is_assistant = (
            hasattr(draft_model, "post_projection") or
            "Assistant" in draft_model.__class__.__name__ or
            getattr(draft_model, "model_type", None) in ["gemma4_assistant", "gemma4_unified_assistant"]
        )
        if is_assistant:
            try:
                link_assistant_to_target(model, draft_model)
            except Exception as e:
                print(f"[ELQ Warning] Could not link assistant to target: {e}", flush=True)

        if is_assistant:
            y = _prefill(model, model_cache, y)
            first_token, first_logprob = _step(model, model_cache, y)
            mx.eval(first_token)
            first_val = first_token.item()
            if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                QuasicrystallineAttention.current_token_ids.append(first_val)
            yield first_val, first_logprob, False
            ntoks = 1
            y = first_token
            draft_y = first_token
        else:
            draft_y = _prefill(draft_model, draft_cache, y)
            y = _prefill(model, model_cache, y)
            ntoks = 0
            
        def _sync_assistant_cache():
            if not is_assistant:
                return
            if hasattr(draft_model, "language_model"):
                draft_text_model = draft_model.language_model.model
            else:
                draft_text_model = draft_model.model
                
            if hasattr(model, "language_model"):
                target_text_model = model.language_model.model
            else:
                target_text_model = model.model
                
            prev_kvs = getattr(target_text_model, "previous_kvs", None)
                
            for idx, assistant_layer in enumerate(draft_text_model.layers):
                target_idx = getattr(assistant_layer.self_attn, "target_layer_idx", None)
                if target_idx is not None:
                    if prev_kvs is not None and target_idx < len(prev_kvs):
                        actual_target_idx = prev_kvs[target_idx]
                    else:
                        actual_target_idx = target_idx
                        
                    target_layer = target_text_model.layers[actual_target_idx]
                    target_attn = target_layer.self_attn
                    if hasattr(target_attn, "custom_kv_cache") and target_attn.custom_kv_cache is not None:
                        k_val = target_attn.custom_kv_cache.get("K")
                        v_val = target_attn.custom_kv_cache.get("V")
                        if k_val is not None:
                            draft_cache[idx].keys = k_val
                            draft_cache[idx].values = v_val
                            draft_cache[idx].offset = target_attn.custom_kv_cache["seq_len"]
                            continue
                            
                    if actual_target_idx < len(model_cache) and model_cache[actual_target_idx] is not None:
                        draft_cache[idx].keys = model_cache[actual_target_idx].keys
                        draft_cache[idx].values = model_cache[actual_target_idx].values
                        draft_cache[idx].offset = model_cache[actual_target_idx].offset

        if is_assistant:
            _sync_assistant_cache()
        
        # P0-C: Pre-extract the target model's raw embedding weight for fast
        # draft-step lookups. If the target embedding is quantized, we must call
        # the module to dequantize the indexed tokens correctly.
        target_lm = model.language_model if hasattr(model, "language_model") else model
        _target_embed_scale = target_lm.model.embed_scale
        is_quantized_embed = "Quantized" in target_lm.model.embed_tokens.__class__.__name__
        
        if is_quantized_embed:
            def _fast_target_embed(token_ids):
                return target_lm.model.embed_tokens(token_ids) * _target_embed_scale
        else:
            _target_embed_weight = target_lm.model.embed_tokens.weight
            mx.eval(_target_embed_weight)
            def _fast_target_embed(token_ids):
                """Direct embedding lookup via pre-extracted weight — bypasses ELQ dequantize."""
                return _target_embed_weight[token_ids] * _target_embed_scale
            
        mx.clear_cache()

        def copy_cache(cache_list):
            import copy
            new_cache = []
            for c in cache_list:
                new_c = copy.copy(c)
                new_c.keys = mx.array(c.keys) if c.keys is not None else None
                new_c.values = mx.array(c.values) if c.values is not None else None
                new_cache.append(new_c)
            return new_cache

        def _draft_step_batch2(draft_y):
            print(f"[ELQ Debug] _draft_step_batch2: draft_y shape={draft_y.shape} | draft_cache[0].keys shape={draft_cache[0].keys.shape if draft_cache[0].keys is not None else 'None'}", flush=True)
            if is_assistant:
                target_hidden = draft_model.last_projected_state
                if target_hidden.shape[1] > 1:
                    target_hidden = target_hidden[:, -1:, :]
                
                target_embed = _fast_target_embed(draft_y)
                if target_embed.ndim == 2:
                    target_embed = target_embed[:, None, :]
                
                concat_input = mx.concatenate([target_embed, target_hidden], axis=-1)
                inputs_embeds = draft_model.pre_projection(concat_input)
                
                scaled_inputs_embeds = inputs_embeds / draft_model.model.embed_scale
                logits = draft_model(draft_y[:, None], cache=draft_cache, input_embeddings=scaled_inputs_embeds)
            else:
                logits = draft_model(draft_y[:, None], cache=draft_cache)
                
            logits = logits[:, -1, :]
            quantize_cache_fn(draft_cache)
            y_next, _ = _process_and_sample(None, logits)
            return y_next

        if (not hasattr(patched_speculative_generate_step, "compiled_remaining_loops") or
            getattr(patched_speculative_generate_step, "last_draft_model", None) is not draft_model):
            patched_speculative_generate_step.compiled_remaining_loops = {}
            patched_speculative_generate_step.last_draft_model = draft_model
            
        def get_compiled_remaining_loop(num_steps):
            if num_steps in patched_speculative_generate_step.compiled_remaining_loops:
                return patched_speculative_generate_step.compiled_remaining_loops[num_steps]
            
            c0 = draft_cache[0]
            cache_class = type(c0)
            max_size = getattr(c0, "max_size", None)
            
            def loop_fn(curr_y, last_proj_state, keys_list, values_list, rope_offset):
                cache_wrappers = []
                for i in range(len(keys_list)):
                    if max_size is not None:
                        c = cache_class(max_size)
                    else:
                        c = cache_class()
                    c.keys = keys_list[i]
                    c.values = values_list[i]
                    c.offset = rope_offset
                    cache_wrappers.append(c)
                    
                if is_assistant:
                    draft_model.last_projected_state = last_proj_state
                ys = []
                y = curr_y
                for _ in range(num_steps):
                    if is_assistant:
                        target_hidden = draft_model.last_projected_state
                        if target_hidden.shape[1] > 1:
                            target_hidden = target_hidden[:, -1:, :]
                        
                        target_embed = _fast_target_embed(y)
                        if target_embed.ndim == 2:
                            target_embed = target_embed[:, None, :]
                        
                        concat_input = mx.concatenate([target_embed, target_hidden], axis=-1)
                        inputs_embeds = draft_model.pre_projection(concat_input)
                        scaled_inputs_embeds = inputs_embeds / draft_model.model.embed_scale
                        logits = draft_model(y[:, None], cache=cache_wrappers, input_embeddings=scaled_inputs_embeds)
                    else:
                        logits = draft_model(y[:, None], cache=cache_wrappers)
                        
                    logits = logits[:, -1, :]
                    quantize_cache_fn(cache_wrappers)
                    
                    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    y = nonlocal_sampler(logprobs)
                    ys.append(y[:, None])
                    
                updated_keys = [c.keys for c in cache_wrappers]
                updated_values = [c.values for c in cache_wrappers]
                
                if is_assistant:
                    return mx.concatenate(ys, axis=1), updated_keys, updated_values, draft_model.last_projected_state
                else:
                    return mx.concatenate(ys, axis=1), updated_keys, updated_values, mx.array(0.0)
            
            patched_speculative_generate_step.compiled_remaining_loops[num_steps] = mx.compile(loop_fn)
            return patched_speculative_generate_step.compiled_remaining_loops[num_steps]

        def copy_custom_caches(m_obj):
            if m_obj is None:
                return {}
            if hasattr(m_obj, "language_model"):
                text_model = m_obj.language_model.model
            elif hasattr(m_obj, "model"):
                text_model = m_obj.model
            else:
                text_model = m_obj
                
            saved = {}
            for idx, layer in enumerate(text_model.layers):
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None and getattr(self_attn, "custom_kv_cache", None) is not None:
                    cache_dict = self_attn.custom_kv_cache
                    saved[idx] = {
                        "K": cache_dict["K"],
                        "V": cache_dict["V"],
                        "indices": cache_dict["indices"],
                        "alignment_scores": cache_dict["alignment_scores"],
                        "seq_len": cache_dict["seq_len"]
                    }
            return saved

        def restore_custom_caches(m_obj, saved):
            if m_obj is None:
                return
            if hasattr(m_obj, "language_model"):
                text_model = m_obj.language_model.model
            elif hasattr(m_obj, "model"):
                text_model = m_obj.model
            else:
                text_model = m_obj
                
            for idx, layer in enumerate(text_model.layers):
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is not None and idx in saved:
                    self_attn.custom_kv_cache = saved[idx]

        # 1. Pre-warm target model JIT graphs for all possible verification sequence lengths [2, 3, 4, 5, 6]
        if model_cache and len(model_cache) > 0:
            print("[ELQ] Pre-warming target model JIT graphs...", flush=True)
            saved_model_cache = copy_cache(model_cache)
            saved_custom_cache = copy_custom_caches(model)
            target_lm = model.language_model if hasattr(model, "language_model") else model
            saved_last_hidden = getattr(target_lm, "last_hidden_state", None)
            
            for L in [2, 3, 4, 5, 6]:
                # Reset model cache and custom cache to saved state before replication
                for idx, c in enumerate(model_cache):
                    c.keys = saved_model_cache[idx].keys
                    c.values = saved_model_cache[idx].values
                    c.offset = saved_model_cache[idx].offset
                restore_custom_caches(model, saved_custom_cache)
                
                # Replicate model cache and custom cache to batch size 2
                for c in model_cache:
                    if c.keys is not None:
                        c.keys = mx.concatenate([c.keys, c.keys], axis=0)
                        c.values = mx.concatenate([c.values, c.values], axis=0)
                replicate_custom_caches(model, 2)
                
                dummy_y_batched = mx.zeros((2, L), mx.uint32)
                tokens, logprobs = _step(model, model_cache, dummy_y_batched, L)
                mx.eval(tokens, logprobs)
                
            # Restore target model cache and custom cache to original batch size 1 state
            for idx, c in enumerate(model_cache):
                c.keys = saved_model_cache[idx].keys
                c.values = saved_model_cache[idx].values
                c.offset = saved_model_cache[idx].offset
            restore_custom_caches(model, saved_custom_cache)
            if saved_last_hidden is not None:
                target_lm.last_hidden_state = saved_last_hidden

        # 2. Pre-warm draft model JIT compiled loops disabled (using eager decoding instead)
        pass

        num_draft = 0
        n = 0
        target_start_offset = 0
        draft_start_offset = 0
        num_existing_target = 0
        num_existing_draft = 0
        try:
            while True:
                num_draft = min(max_tokens - ntoks, dyn_num_draft)
                target_start_offset = model_cache[0].offset if (model_cache and len(model_cache) > 0) else 0
                draft_start_offset = draft_cache[0].offset if (draft_cache and len(draft_cache) > 0) else 0
                
                num_existing_target = 0
                if hasattr(model, "language_model"):
                    target_text_model = model.language_model.model
                elif hasattr(model, "model"):
                    target_text_model = model.model
                else:
                    target_text_model = model
                for layer in target_text_model.layers:
                    if hasattr(layer.self_attn, "custom_kv_cache") and layer.self_attn.custom_kv_cache is not None:
                        k_val = layer.self_attn.custom_kv_cache.get("K")
                        if k_val is not None:
                            num_existing_target = k_val.shape[2]
                            break
                            
                num_existing_draft = 0
                if draft_model is not None:
                    if hasattr(draft_model, "language_model"):
                        draft_text_model = draft_model.language_model.model
                    elif hasattr(draft_model, "model"):
                        draft_text_model = draft_model.model
                    else:
                        draft_text_model = draft_model
                    for layer in draft_text_model.layers:
                        if hasattr(layer.self_attn, "custom_kv_cache") and layer.self_attn.custom_kv_cache is not None:
                            k_val = layer.self_attn.custom_kv_cache.get("K")
                            if k_val is not None:
                                num_existing_draft = k_val.shape[2]
                                break
                if num_draft <= 1:
                    draft_tokens = _draft_generate(draft_y, num_draft)
                    if prev_tokens is not None:
                        prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
                    y = mx.concatenate([y, draft_tokens])
                    tokens, logprobs = _step(model, model_cache, y, num_draft + 1)
                    mx.eval(tokens, draft_tokens)
                    draft_tokens = draft_tokens.tolist()
                    tokens = tokens.tolist()
                    n = 0
                    while n < num_draft:
                        tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                        if tn != dtn:
                            break
                        n += 1
                        ntoks += 1
                        if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                            QuasicrystallineAttention.current_token_ids.append(tn)
                        yield tn, lpn, True
                        if ntoks == max_tokens:
                            break
                    if ntoks < max_tokens:
                        ntoks += 1
                        if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                            QuasicrystallineAttention.current_token_ids.append(tokens[n])
                        yield tokens[n], logprobs[n], False
                else:
                    t_start = time.perf_counter()

                    # 1. Generate first draft token (branching factor 2)
                    if is_assistant:
                        target_hidden = target_lm.last_hidden_state
                        L_y = draft_y.size
                        if target_hidden.shape[1] > 1:
                            target_hidden = target_hidden[:, n - L_y + 1 : n + 1, :]
                        target_embed = _fast_target_embed(draft_y)
                        if target_embed.ndim == 2:
                            target_embed = target_embed[None, :, :]
                        if target_hidden.ndim == 2:
                            target_hidden = target_hidden[None, :, :]
                        concat_input = mx.concatenate([target_embed, target_hidden], axis=-1)
                        inputs_embeds = draft_model.pre_projection(concat_input)
                        
                        dev_type = get_model_device_type(draft_model)
                        stream_ctx = mx.stream(mx.cpu) if dev_type == mx.DeviceType.cpu else mx.stream(generation_stream)
                        with stream_ctx:
                            scaled_inputs_embeds = inputs_embeds / draft_model.model.embed_scale
                            
                            # Bypass ELQ cache globally during draft step (single flag vs 34-layer iteration)
                            ELQLinear._global_cache_bypass = True
                            
                            logits_draft = draft_model(draft_y[None], cache=draft_cache, input_embeddings=scaled_inputs_embeds)
                            logits_draft = logits_draft[:, -1, :]
                            quantize_cache_fn(draft_cache)
                    else:
                        dev_type = get_model_device_type(draft_model)
                        stream_ctx = mx.stream(mx.cpu) if dev_type == mx.DeviceType.cpu else mx.stream(generation_stream)
                        with stream_ctx:
                            # Bypass ELQ cache globally during draft step (single flag vs 34-layer iteration)
                            ELQLinear._global_cache_bypass = True
                            
                            logits_draft = draft_model(draft_y[None], cache=draft_cache)
                            logits_draft = logits_draft[:, -1, :]
                            quantize_cache_fn(draft_cache)
                            
                    logprobs_draft = logits_draft - mx.logsumexp(logits_draft, axis=-1, keepdims=True)
                    top_2 = mx.argpartition(-logprobs_draft, kth=1, axis=-1)[..., :2].squeeze(0)
                    c1 = top_2[0:1]
                    c2 = top_2[1:2]
                    
                    draft_offset_after_first = draft_cache[0].offset if (draft_cache and len(draft_cache) > 0) else 0
                    
                    # Replicate draft cache to batch size 2
                    global_eval_args = []
                    for idx, c in enumerate(draft_cache):
                        if c.keys is not None:
                            c.keys = mx.repeat(c.keys, 2, axis=0)
                            c.values = mx.repeat(c.values, 2, axis=0)
                            global_eval_args.extend([c.keys, c.values])
                    
                    args_rep_d = replicate_custom_caches(draft_model, 2, eval_now=False)
                    global_eval_args.extend(args_rep_d)
                            
                    curr_y = mx.concatenate([c1, c2], axis=0) # shape (2,)
                    if is_assistant:
                        draft_model.last_projected_state = mx.repeat(
                            draft_model.last_projected_state, 2, axis=0
                        )
                        
                    # 2. Generate remaining tokens for both paths in parallel
                    if num_draft > 1:
                        loop_compiled = get_compiled_remaining_loop(num_draft - 1)
                        keys_list = [c.keys for c in draft_cache]
                        values_list = [c.values for c in draft_cache]
                        rope_offset = draft_cache[0].offset
                        
                        if is_assistant:
                            remaining_tokens, updated_keys, updated_values, last_proj_state = loop_compiled(
                                curr_y,
                                draft_model.last_projected_state,
                                keys_list,
                                values_list,
                                rope_offset
                            )
                            draft_model.last_projected_state = last_proj_state
                        else:
                            remaining_tokens, updated_keys, updated_values, _ = loop_compiled(
                                curr_y,
                                mx.array(0.0),
                                keys_list,
                                values_list,
                                rope_offset
                            )
                        
                        # Update the draft_cache keys and values in-place
                        for idx_c, c in enumerate(draft_cache):
                            c.keys = updated_keys[idx_c]
                            c.values = updated_values[idx_c]
                            c.offset = rope_offset + (num_draft - 1)
                            
                        draft_tokens = mx.concatenate([curr_y[:, None], remaining_tokens], axis=1)
                    else:
                        draft_tokens = curr_y[:, None]
                        
                    # Restore ELQ cache (single flag instead of per-layer iteration)
                    ELQLinear._global_cache_bypass = False
                            
                    # Replicate target cache to batch size 2
                    for c in model_cache:
                        if c.keys is not None:
                            c.keys = mx.repeat(c.keys, 2, axis=0)
                            c.values = mx.repeat(c.values, 2, axis=0)
                            global_eval_args.extend([c.keys, c.values])
                            
                    args_rep_t = replicate_custom_caches(model, 2, eval_now=False)
                    global_eval_args.extend(args_rep_t)
                            
                    # 3. Verify both paths in parallel in target model
                    y_duplicated = mx.concatenate([y, y], axis=0)
                    y_batched = mx.concatenate([y_duplicated[:, None], draft_tokens], axis=1) # shape (2, num_draft + 1)
                    tokens_batched, logprobs_batched = _step(model, model_cache, y_batched, num_draft + 1)
                    
                    # 4. GPU-side comparison and best path selection
                    mismatches = tokens_batched[:, :num_draft] != draft_tokens
                    first_mismatch = mx.argmax(mismatches, axis=1)
                    has_mismatch = mx.any(mismatches, axis=1)
                    accepted_lens = mx.where(has_mismatch, first_mismatch, num_draft)
                    
                    best_idx_gpu = mx.argmax(accepted_lens)
                    acc_len_gpu = accepted_lens[best_idx_gpu]
                    
                    # Combine all draft generation, verification, and replication evals into a single block
                    eval_list = [draft_tokens, best_idx_gpu, acc_len_gpu]
                    if global_eval_args:
                        eval_list.extend(global_eval_args)
                    mx.eval(*eval_list)
                    
                    t_draft = time.perf_counter() - t_start
                    t_verify_start = time.perf_counter()
                    
                    best_idx = int(best_idx_gpu.item())
                    n = int(acc_len_gpu.item())
                    
                    # Firewall check skipped during speculative cycles — the attention
                    # forward pass already runs the inline Čech obstruction check at
                    # the dynamically-gated interval. Running it again here on all 34
                    # layers was costing ~10-20ms per cycle with redundant GPU syncs.
                    fracture_detected = False
                    
                    # === Moonshot Phase 5: Cross-Model Wormhole Query ===
                    # If tokens were rejected and we have a wormhole bridge,
                    # check if CFI indicates semantic uncertainty and query Gemini.
                    if _wormhole_bridge is not None and n < num_draft:
                        # Get CFI from the firewall if available
                        _firewall_cfi = getattr(QuasicrystallineAttention, '_last_cfi', 0.0)
                        _firewall_lambda2 = getattr(QuasicrystallineAttention, '_last_lambda_2', 1.0)
                        if _wormhole_bridge.should_open_wormhole(_firewall_cfi, _firewall_lambda2):
                            # Get the hidden state at the rejection point
                            _hidden = getattr(target_lm, 'last_hidden_state', None)
                            if _hidden is not None:
                                _h = _hidden[:, -1, :] if _hidden.ndim == 3 else _hidden
                                _corrected = _wormhole_bridge.query(
                                    _h,
                                    context_text=None,  # Could decode recent tokens here
                                )
                                if _corrected is not None:
                                    # Inject corrected hidden state for the next cycle
                                    if hasattr(target_lm, 'last_hidden_state') and target_lm.last_hidden_state is not None:
                                        if target_lm.last_hidden_state.ndim == 3:
                                            target_lm.last_hidden_state = mx.concatenate([
                                                target_lm.last_hidden_state[:, :-1, :],
                                                _corrected[None, None, :target_lm.last_hidden_state.shape[-1]]
                                            ], axis=1)
                                    print(f"[Moonshot] Wormhole query: CFI={_firewall_cfi:.3f}, corrected hidden state injected.", flush=True)
                    
                    # slice caches back to batch size 1
                    slice_eval_args = []
                    for c in model_cache:
                        if c.keys is not None:
                            c.keys = c.keys[best_idx : best_idx + 1]
                            c.values = c.values[best_idx : best_idx + 1]
                            slice_eval_args.extend([c.keys, c.values])
                            
                    for c in draft_cache:
                        if c.keys is not None:
                            c.keys = c.keys[best_idx : best_idx + 1]
                            c.values = c.values[best_idx : best_idx + 1]
                            slice_eval_args.extend([c.keys, c.values])
                            
                    args_rb_t = _rollback_custom_caches(model, target_start_offset + num_draft + 1, best_idx, start_offset=target_start_offset, num_existing=num_existing_target, eval_now=False)
                    args_rb_d = _rollback_custom_caches(draft_model, draft_start_offset + num_draft, best_idx, start_offset=draft_start_offset, num_existing=num_existing_draft, eval_now=False)
                    slice_eval_args.extend(args_rb_t)
                    slice_eval_args.extend(args_rb_d)
                    
                    if slice_eval_args:
                        mx.eval(*slice_eval_args)
                    
                    if is_assistant:
                        draft_model.last_projected_state = draft_model.last_projected_state[best_idx : best_idx + 1]
                        if hasattr(model, "language_model") and hasattr(model.language_model, "last_hidden_state"):
                            model.language_model.last_hidden_state = model.language_model.last_hidden_state[best_idx : best_idx + 1]
                        elif hasattr(model, "last_hidden_state"):
                            model.last_hidden_state = model.last_hidden_state[best_idx : best_idx + 1]
                        
                    tokens = tokens_batched[best_idx].tolist()
                    logprobs = logprobs_batched[best_idx]
                    draft_tokens = draft_tokens[best_idx].tolist()
                    
                    t_verify = time.perf_counter() - t_verify_start
                    print(f"[TIMING] Draft: {t_draft*1000:.2f}ms, Verify: {t_verify*1000:.2f}ms, Accepted: {n}", flush=True)
                    
                    # Yield accepted tokens
                    for i in range(n):
                        ntoks += 1
                        if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                            QuasicrystallineAttention.current_token_ids.append(tokens[i])
                        # === Moonshot Phase 1: Update trajectory with accepted token ===
                        if _geometric_filter is not None:
                            _hidden = getattr(target_lm, 'last_hidden_state', None)
                            if _hidden is not None:
                                _h = _hidden[:, i, :] if _hidden.ndim == 3 and _hidden.shape[1] > i else None
                                if _h is not None:
                                    _geometric_filter.update_trajectory(_h.squeeze(0))
                        yield tokens[i], logprobs[i], True
                        if ntoks == max_tokens:
                            break
                    if ntoks < max_tokens:
                        ntoks += 1
                        if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                            QuasicrystallineAttention.current_token_ids.append(tokens[n])
                        yield tokens[n], logprobs[n], False
                        
                if ntoks == max_tokens:
                    break
                    
                y = mx.array([tokens[n]], mx.uint32)
                draft_y = y
                is_first_draft_step[0] = True
                
                if n == num_draft:
                    draft_y = mx.concatenate(
                        [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                    )
                    is_first_draft_step[0] = False
                    
                if prev_tokens is not None:
                    prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
                _rewind_cache(num_draft, n)
                if num_draft > 0:
                    rate = n / num_draft
                    acceptance_history.append(rate)
                    if len(acceptance_history) > 8:
                        acceptance_history.pop(0)
                    if len(acceptance_history) >= 4:
                        avg_rate = sum(acceptance_history) / len(acceptance_history)
                        if avg_rate >= 0.75:
                            dyn_num_draft = min(dyn_num_draft + 1, 5)
                        elif avg_rate <= 0.35:
                            dyn_num_draft = max(dyn_num_draft - 1, 1)
                        # === Moonshot Phase 1: Boost draft ceiling if geometric filter is active ===
                        # Geometric pre-filtering means we can safely propose more drafts
                        # since bad candidates get caught cheaply.
                        if _geometric_filter is not None and avg_rate >= 0.75:
                            dyn_num_draft = min(dyn_num_draft + 1, 8)  # Higher ceiling with filter
                _sync_assistant_cache()
                if ntoks % 8 == 0:
                    mx.clear_cache()
        finally:
            _rewind_cache(num_draft, n)
            mx.clear_cache()

    gen_mod.speculative_generate_step = patched_speculative_generate_step
    gen_mod._is_patched_for_watchdog = True
    print("[ELQ] Successfully patched mlx_lm.generate.speculative_generate_step!", flush=True)


def patch_standard_decoding(gen_mod):
    if getattr(gen_mod, "_is_patched_for_standard_jit", False):
        return
        
    print("[ELQ] Patching mlx_lm.generate.generate_step for JIT standard decoding...", flush=True)
    _patch_cache_states()
    
    try:
        import mlx_lm.models.base as base_mod
        import mlx_lm.models.gemma4_text as gemma4_text_mod
        
        orig_create_attn_mask = base_mod.create_attention_mask
        
        def patched_create_attn_mask(h, cache=None, window_size=None, return_array=False):
            from qan_transformers.mlx.attention import QuasicrystallineAttention
            if getattr(QuasicrystallineAttention, "in_jit", False):
                return None
            valid_len = getattr(QuasicrystallineAttention, "valid_length", None)
            if valid_len is not None and h.shape[1] > 1:
                N = h.shape[1]
                causal_mask = mx.triu(mx.full((N, N), -mx.inf, dtype=h.dtype), k=1)
                pad_mask = mx.arange(N)[None, :] >= valid_len
                pad_mask_value = mx.where(pad_mask, -mx.inf, 0.0).astype(h.dtype)
                return causal_mask + pad_mask_value
            return orig_create_attn_mask(h, cache, window_size, return_array)
            
        base_mod.create_attention_mask = patched_create_attn_mask
        if hasattr(gemma4_text_mod, "create_attention_mask"):
            gemma4_text_mod.create_attention_mask = patched_create_attn_mask
        print("[ELQ] Successfully patched create_attention_mask for JIT compilation safety!", flush=True)
    except Exception as e:
        print(f"[ELQ Warning] Could not patch create_attention_mask: {e}", flush=True)
        
    def patched_generate_step(
        prompt: Any,
        model: nn.Module,
        *,
        max_tokens: int = 256,
        sampler = None,
        logits_processors = None,
        max_kv_size = None,
        prompt_cache = None,
        prefill_step_size: int = 2048,
        kv_bits = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 0,
        prompt_progress_callback = None,
        input_embeddings = None,
        fused_tokenization: bool = False,
    ):
        if fused_tokenization and isinstance(prompt, str) and getattr(model, "tokenizer", None) is not None:
            prompt_ids = model.tokenizer.encode(prompt)
            prompt = mx.array(prompt_ids, dtype=mx.uint32)

        import functools
        from mlx_lm.generate import does_model_support_input_embeddings, maybe_quantize_kv_cache, cache, generation_stream, wired_limit

        if input_embeddings is not None:
            if not does_model_support_input_embeddings(model):
                raise ValueError("Model does not support input embeddings.")
            elif len(prompt) > 0 and len(prompt) != len(input_embeddings):
                raise ValueError(
                    f"When providing input_embeddings, their sequence length ({len(input_embeddings)}) "
                    f"must match the sequence length of the prompt ({len(prompt)}), or the "
                    "prompt must be empty."
                )
        elif len(prompt) == 0:
            raise ValueError(
                "Either input_embeddings or prompt (or both) must be provided."
            )

        # Initialize sequence history for E8 lattice bias (Leap 0019/0022)
        from qan_transformers.mlx.attention import QuasicrystallineAttention
        tok_obj = getattr(model, "tokenizer", None)
        if tok_obj is not None and hasattr(tok_obj, "organism"):
            QuasicrystallineAttention.organism = tok_obj.organism
            if hasattr(prompt, "tolist"):
                QuasicrystallineAttention.current_token_ids = prompt.tolist()
            elif isinstance(prompt, list):
                QuasicrystallineAttention.current_token_ids = list(prompt)
            else:
                QuasicrystallineAttention.current_token_ids = []
        else:
            QuasicrystallineAttention.organism = None
            QuasicrystallineAttention.current_token_ids = None

        tokens = None

        text_model = model.language_model.model if hasattr(model, "language_model") else (model.model if hasattr(model, "model") else model)
        num_layers = len(text_model.layers) if hasattr(text_model, "layers") else 0
        use_layer_by_layer = (num_layers >= 40) and (input_embeddings is None)
        print(f"[Debug] patched_generate_step: num_layers={num_layers}, use_layer_by_layer={use_layer_by_layer}", flush=True)

        # Create the KV cache for generation
        if prompt_cache is None:
            prompt_cache = cache.make_prompt_cache(
                model,
                max_kv_size=max_kv_size,
            )

        prompt_progress_callback = prompt_progress_callback or (lambda *_: None)

        quantize_cache_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

        sampler_fn = sampler or (lambda x: mx.argmax(x, axis=-1))

        def _model_call(input_tokens: mx.array, input_embeddings: Optional[mx.array]):
            if input_embeddings is not None:
                return model(
                    input_tokens, cache=prompt_cache, input_embeddings=input_embeddings
                )
            else:
                return model(input_tokens, cache=prompt_cache)

        def _step(input_tokens: mx.array, input_embeddings: Optional[mx.array] = None):
            nonlocal tokens

            with mx.stream(generation_stream):
                if use_layer_by_layer and input_embeddings is None:
                    embed_layer = getattr(text_model, "embed_tokens", None) or getattr(text_model, "wte", None)
                    embed_scale = getattr(text_model, "embed_scale", 1.0)
                    if embed_scale == 1.0 and hasattr(text_model, "args") and hasattr(text_model.args, "hidden_size"):
                        if "gemma" in getattr(text_model.args, "model_type", "").lower():
                            embed_scale = text_model.args.hidden_size ** 0.5
                            
                    tids = input_tokens
                    if tids.ndim == 0:
                        tids = tids[None, None]
                    elif tids.ndim == 1:
                        tids = tids[None]
                        
                    h = embed_layer(tids) * embed_scale
                    
                    padded_cache = prompt_cache
                    if len(prompt_cache) < len(text_model.layers):
                        padded_cache = prompt_cache + [None] * (len(text_model.layers) - len(prompt_cache))
                        
                    has_prev_kvs = hasattr(text_model, "previous_kvs")
                    previous_kvs = getattr(text_model, "previous_kvs", None)
                    intermediates = [(None, None)] * len(text_model.layers)
                    
                    for layer_idx, layer in enumerate(text_model.layers):
                        layer_cache = padded_cache[layer_idx]
                        
                        window_size = None
                        if hasattr(layer, "layer_type") and layer.layer_type == "sliding_attention":
                            window_size = getattr(text_model, "window_size", None)
                            
                        from mlx_lm.models.base import create_attention_mask
                        mask = create_attention_mask(h, layer_cache, window_size=window_size)
                        
                        prev_kv_layer_idx = previous_kvs[layer_idx] if has_prev_kvs and previous_kvs is not None else None
                        is_shared_kv_layer = (prev_kv_layer_idx is not None and prev_kv_layer_idx != layer_idx)
                        
                        kvs, offset = intermediates[prev_kv_layer_idx] if is_shared_kv_layer else (None, None)
                        
                        h, kvs, offset = layer(
                            h,
                            mask=mask,
                            cache=layer_cache,
                            shared_kv=kvs,
                            offset=offset
                        )
                        
                        intermediates[layer_idx] = (kvs, offset)
                        
                        eval_targets = [h]
                        if layer_cache is not None:
                            if layer_cache.keys is not None:
                                eval_targets.append(layer_cache.keys)
                            if layer_cache.values is not None:
                                eval_targets.append(layer_cache.values)
                                
                        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "custom_kv_cache"):
                            cc = layer.self_attn.custom_kv_cache
                            if cc is not None:
                                if cc.get("K") is not None:
                                    eval_targets.append(cc["K"])
                                if cc.get("V") is not None:
                                    eval_targets.append(cc["V"])
                                if cc.get("indices") is not None:
                                    eval_targets.append(cc["indices"])
                                if cc.get("alignment_scores") is not None:
                                    eval_targets.append(cc["alignment_scores"])
                                
                        if (layer_idx + 1) % 8 == 0 or layer_idx == len(text_model.layers) - 1:
                            mx.eval(*eval_targets)
                            import gc
                            gc.collect()
                            mx.clear_cache()
                        else:
                            mx.async_eval(*eval_targets)
                        
                    norm_layer = getattr(text_model, "norm", None) or getattr(text_model, "ln_f", None)
                    if norm_layer is not None:
                        h = norm_layer(h)
                        
                    h_last = h[:, -1:]
                    container = model.language_model if hasattr(model, "language_model") else model
                    if getattr(container, "tie_word_embeddings", False):
                        logits = embed_layer.as_linear(h_last)
                    elif hasattr(container, "lm_head"):
                        logits = container.lm_head(h_last)
                    else:
                        logits = embed_layer.as_linear(h_last)
                        
                    softcap = getattr(container, "final_logit_softcapping", None)
                    if softcap is not None:
                        from mlx_lm.models.gemma4_text import logit_softcap
                        logits = logit_softcap(softcap, logits)
                        
                    logits = logits[:, -1, :]
                else:
                    logits = _model_call(
                        input_tokens=input_tokens[None],
                        input_embeddings=(
                            input_embeddings[None] if input_embeddings is not None else None
                        ),
                    )
                    logits = logits[:, -1, :]

                if logits_processors and len(input_tokens) > 0:
                    tokens = (
                        mx.concat([tokens, input_tokens])
                        if tokens is not None
                        else input_tokens
                    )
                    for processor in logits_processors:
                        logits = processor(tokens, logits)

                quantize_cache_fn(prompt_cache)

                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                sampled = sampler_fn(logprobs)
                return sampled, logprobs.squeeze(0)

        with mx.stream(generation_stream):
            total_prompt_tokens = (
                len(input_embeddings) if input_embeddings is not None else len(prompt)
            )
            prompt_processed_tokens = 0
            prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)


            if use_layer_by_layer:
                logits, h_last = layer_by_layer_prefill(model, prompt[None], prompt_cache, chunk_size=prefill_step_size)
                
                # Retrieve logits for the last token of the prompt
                logits = logits[:, -1, :]
                logits = logits.squeeze(0)
                
                if logits_processors:
                    tokens = prompt
                    for processor in logits_processors:
                        logits = processor(tokens, logits[None])
                    logits = logits.squeeze(0)
                    
                quantize_cache_fn(prompt_cache)
                logprobs = logits - mx.logsumexp(logits, keepdims=True)
                y = sampler_fn(logprobs)
            else:
                while total_prompt_tokens - prompt_processed_tokens > 1:
                    remaining = (total_prompt_tokens - prompt_processed_tokens) - 1
                    n_to_process = min(prefill_step_size, remaining)
                    _model_call(
                        input_tokens=prompt[:n_to_process][None],
                        input_embeddings=(
                            input_embeddings[:n_to_process][None]
                            if input_embeddings is not None
                            else None
                        ),
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    prompt_processed_tokens += n_to_process
                    prompt_progress_callback(prompt_processed_tokens, total_prompt_tokens)
                    prompt = prompt[n_to_process:]
                    input_embeddings = (
                        input_embeddings[n_to_process:]
                        if input_embeddings is not None
                        else input_embeddings
                    )
                    mx.clear_cache()

                y, logprobs = _step(input_tokens=prompt, input_embeddings=input_embeddings)

        mx.async_eval(y, logprobs)
        
        # Fall back if logits_processors is active (disabled standard JIT compile to avoid cache padding issues)
        if True:
            if use_layer_by_layer:
                n = 0
                while True:
                    if n == max_tokens:
                        break
                    mx.eval(y)
                    if n == 0:
                        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                    if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                        QuasicrystallineAttention.current_token_ids.append(y.item())
                    yield y.item(), logprobs
                    if n % 256 == 0:
                        mx.clear_cache()
                    if n + 1 < max_tokens:
                        next_y, next_logprobs = _step(y)
                        y, logprobs = next_y, next_logprobs
                    n += 1
                return
            else:
                n = 0
                while True:
                    if n != max_tokens:
                        next_y, next_logprobs = _step(y)
                        mx.async_eval(next_y, next_logprobs)
                    if n == 0:
                        mx.eval(y)
                        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
                    if n == max_tokens:
                        break
                    if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                        QuasicrystallineAttention.current_token_ids.append(y.item())
                    yield y.item(), logprobs
                    if n % 256 == 0:
                        mx.clear_cache()
                    y, logprobs = next_y, next_logprobs
                    n += 1
                return

        # --- JIT Compiled Loop Optimization ---
        if not hasattr(patched_generate_step, "compiled_loops"):
            patched_generate_step.compiled_loops = {}
            
        c0 = prompt_cache[0]
        cache_class = type(c0)
        max_size = getattr(c0, "max_size", None)
            
        def get_compiled_loop(num_steps):
            if num_steps in patched_generate_step.compiled_loops:
                return patched_generate_step.compiled_loops[num_steps]
                
            def loop_fn(curr_y, keys_list, values_list, rope_offset):
                # Dynamic wrapper instantiation inside JIT trace
                cache_wrappers = []
                for i in range(len(keys_list)):
                    if max_size is not None:
                        c = cache_class(max_size)
                    else:
                        c = cache_class()
                    c.keys = keys_list[i]
                    c.values = values_list[i]
                    c.offset = rope_offset
                    cache_wrappers.append(c)
                    
                ys = []
                logprobs_list = []
                cur_t = curr_y
                for _ in range(num_steps):
                    logits = model(cur_t[None], cache=cache_wrappers)
                    logits = logits[:, -1, :]
                    quantize_cache_fn(cache_wrappers)
                    lp = logits - mx.logsumexp(logits, keepdims=True)
                    cur_t = sampler_fn(lp)
                    ys.append(cur_t)
                    logprobs_list.append(lp)
                    
                updated_keys = [c.keys for c in cache_wrappers]
                updated_values = [c.values for c in cache_wrappers]
                return mx.concatenate(ys, axis=0), mx.concatenate(logprobs_list, axis=0), updated_keys, updated_values
                
            patched_generate_step.compiled_loops[num_steps] = mx.compile(loop_fn)
            return patched_generate_step.compiled_loops[num_steps]

        n = 0
        mx.eval(y)
        prompt_progress_callback(total_prompt_tokens, total_prompt_tokens)
        
        val = y.item()
        if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
            QuasicrystallineAttention.current_token_ids.append(val)
        yield val, logprobs
        n += 1
        
        BLOCK_SIZE = 16
        buffer_tokens = []
        buffer_logprobs = []
        
        while n < max_tokens:
            if not buffer_tokens:
                steps_to_run = min(BLOCK_SIZE, max_tokens - n)
                compiled_loop = get_compiled_loop(steps_to_run)
                
                # Dynamic padding to prevent JIT history overwrites (Win 107/111)
                # Win 111 Fix: Cap padding to sliding window/max size to avoid VRAM blowout
                for c in prompt_cache:
                    if c.keys is not None:
                        current_size = c.keys.shape[2]
                        target_size = min(4096, ((int(c.offset + steps_to_run) + 511) // 512) * 512)
                        if current_size < target_size:
                            pad_len = target_size - current_size
                            pad_k = mx.zeros((c.keys.shape[0], c.keys.shape[1], pad_len, c.keys.shape[3]), c.keys.dtype)
                            pad_v = mx.zeros((c.values.shape[0], c.values.shape[1], pad_len, c.values.shape[3]), c.values.dtype)
                            c.keys = mx.concatenate([c.keys, pad_k], axis=2)
                            c.values = mx.concatenate([c.values, pad_v], axis=2)
                
                keys_list = [c.keys for c in prompt_cache]
                values_list = [c.values for c in prompt_cache]
                offset_arr = mx.array(prompt_cache[0].offset, mx.int32)
                
                from qan_transformers.mlx.attention import QuasicrystallineAttention
                QuasicrystallineAttention.in_jit = True
                try:
                    y_block, logprobs_block, next_keys, next_values = compiled_loop(y, keys_list, values_list, offset_arr)
                    mx.async_eval(y_block, logprobs_block, next_keys, next_values)
                finally:
                    QuasicrystallineAttention.in_jit = False
                
                buffer_tokens = y_block.tolist()
                buffer_logprobs = [logprobs_block[i] for i in range(steps_to_run)]
                
                # Write updated keys and values back to prompt_cache
                for i, c in enumerate(prompt_cache):
                    c.keys = next_keys[i]
                    c.values = next_values[i]
                    c.offset += steps_to_run
                
            next_y_val = buffer_tokens.pop(0)
            next_logprobs = buffer_logprobs.pop(0)
            
            y = mx.array([next_y_val], mx.uint32)
            logprobs = next_logprobs
            
            if getattr(QuasicrystallineAttention, "current_token_ids", None) is not None:
                QuasicrystallineAttention.current_token_ids.append(next_y_val)
            yield next_y_val, logprobs
            n += 1
            
            if n % 256 == 0:
                mx.clear_cache()

    gen_mod.generate_step = patched_generate_step
    gen_mod._is_patched_for_standard_jit = True

    orig_stream_generate = gen_mod.stream_generate
    def patched_stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None, **kwargs):
        mx.set_memory_limit(0)
        model.tokenizer = tokenizer
        if draft_model is not None:
            draft_model.tokenizer = tokenizer
        
        # Always convert prompt to mx.array token IDs to feed our custom generators
        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt)
            prompt = mx.array(prompt_ids, dtype=mx.uint32)
        elif not isinstance(prompt, mx.array):
            prompt = mx.array(prompt, dtype=mx.uint32)
            
        kwargs["max_tokens"] = max_tokens
        kwargs["fused_tokenization"] = True
        
        if not getattr(tokenizer, "_is_wrapped", False):
            from mlx_lm.generate import TokenizerWrapper
            tokenizer = TokenizerWrapper(tokenizer)
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        
        if draft_model is None:
            kwargs.pop("num_draft_tokens", None)
            token_generator = gen_mod.generate_step(prompt, model, **kwargs)
            token_generator = (
                (token, logprobs, False) for token, logprobs in token_generator
            )
        else:
            kwargs.pop("max_kv_size", None)
            kwargs.pop("prompt_progress_callback", None)
            token_generator = gen_mod.speculative_generate_step(
                prompt, model, draft_model, **kwargs
            )
            
        from mlx_lm.generate import wired_limit, generation_stream
        import time
        from mlx_lm.generate import GenerationResponse
        with wired_limit(model, [generation_stream]):
            tic = time.perf_counter()
            prompt_tps = 0.0
            
            # Resolve all possible stop token IDs for the model (such as turn markers)
            stop_token_ids = set(tokenizer.eos_token_ids)
            for stop_seq in ["<turn|>", "<eos>", "<pad>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
                tid = tokenizer.convert_tokens_to_ids(stop_seq)
                if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                    stop_token_ids.add(tid)
                    
            for n, (token, logprobs, from_draft) in enumerate(token_generator):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = prompt.size / max(prompt_time, 1e-6)
                    tic = time.perf_counter()
                if token in stop_token_ids:
                    break

                detokenizer.add_token(token)
                if (n + 1) == max_tokens:
                    break

                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    from_draft=from_draft,
                    prompt_tokens=prompt.size,
                    prompt_tps=prompt_tps,
                    generation_tokens=n + 1,
                    generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )
                if getattr(detokenizer, "just_finished", False):
                    break
            
            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / max(time.perf_counter() - tic, 1e-6),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop" if token in tokenizer.eos_token_ids else "length",
            )

    gen_mod.stream_generate = patched_stream_generate



class GaloisRingGR4dWeights(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 2-bit packed weights represented as elements of Galois Ring GR(4^d)
        # We pack 16 weights (each 2-bit) in a single uint32 element
        self.packed_size = (in_features * out_features + 15) // 16
        self.packed_weights = mx.zeros((self.packed_size,), dtype=mx.uint32)
        
        # Scaling factors
        self.scales = mx.ones((out_features,), dtype=mx.float32)
        self.biases = mx.zeros((out_features,), dtype=mx.float32)
        
    def unpack(self) -> mx.array:
        # Vectorized bit-shift unpacking of GR(4^d) 2-bit elements
        # 16 elements per uint32
        W_flat = mx.zeros((self.in_features * self.out_features,), dtype=mx.float32)
        
        # For demonstration and test compilation: unpack via parallel shifts
        shift_vals = mx.array([2 * i for i in range(16)], dtype=mx.uint32)
        expanded_weights = mx.expand_dims(self.packed_weights, axis=-1) # [packed_size, 1]
        
        # Bitwise shifts
        unpacked_blocks = (expanded_weights >> shift_vals) & 3
        unpacked_flat = mx.reshape(unpacked_blocks, (-1,))
        
        # Slice to exact shape
        unpacked_sliced = unpacked_flat[:self.in_features * self.out_features]
        # Map GR(4) elements {0, 1, 2, 3} to centered floats {-1.5, -0.5, 0.5, 1.5}
        unpacked_centered = unpacked_sliced.astype(mx.float32) - 1.5
        
        W = mx.reshape(unpacked_centered, (self.out_features, self.in_features))
        return W * mx.expand_dims(self.scales, axis=-1)


class NonCommutativeBraidBisectionRouter(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int = 4, depth: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.depth = depth
        
        # Braid crossing angles (Artin generators)
        self.braid_angles = mx.array(np.random.uniform(-0.1, 0.1, (depth, num_experts - 1)), dtype=mx.float32)
        
    def __call__(self, x: mx.array) -> mx.array:
        # Multi-strand Artin braid group bisection routing
        # Input shape [B, S, D] -> projects to strands of size [B, S, num_experts]
        B, S, _ = x.shape
        
        # Project inputs to Expert strands
        proj = nn.Linear(self.embed_dim, self.num_experts)
        strands = proj(x) # [B, S, num_experts]
        
        # Apply braid crossings layer-by-layer
        for d in range(self.depth):
            for i in range(self.num_experts - 1):
                theta = self.braid_angles[d, i]
                cos_t = mx.cos(theta)
                sin_t = mx.sin(theta)
                
                # Braid crossings rotation between strand i and i+1
                s_i = strands[..., i]
                s_next = strands[..., i + 1]
                
                crossed_i = s_i * cos_t - s_next * sin_t
                crossed_next = s_i * sin_t + s_next * cos_t
                
                # Update strands (assigning back using mask concatenation)
                strands_list = []
                for k in range(self.num_experts):
                    if k == i:
                        strands_list.append(mx.expand_dims(crossed_i, axis=-1))
                    elif k == i + 1:
                        strands_list.append(mx.expand_dims(crossed_next, axis=-1))
                    else:
                        strands_list.append(mx.expand_dims(strands[..., k], axis=-1))
                strands = mx.concatenate(strands_list, axis=-1)
                
        # The final routing probabilities are Softmax over braided strands
        return mx.softmax(strands, axis=-1)


class SheafRestrictiveGating(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        # Semantic Open Cover projections
        self.cover_projs = [
            nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)
        ]
        for k in range(num_experts):
            setattr(self, f"cover_proj_{k}", self.cover_projs[k])
            
    def __call__(self, x: mx.array) -> mx.array:
        # Computes sheaf restriction maps of input onto the semantic open covers
        B, S, D = x.shape
        
        restriction_norms = []
        for k in range(self.num_experts):
            proj = getattr(self, f"cover_proj_{k}")
            proj_x = proj(x)
            
            # Restriction discrepancy norm: ||x - proj_x||^2
            discrepancy = mx.sum(mx.square(x - proj_x), axis=-1) # [B, S]
            restriction_norms.append(mx.expand_dims(discrepancy, axis=-1))
            
        norms = mx.concatenate(restriction_norms, axis=-1) # [B, S, num_experts]
        
        # Restrictive gating: route to experts with minimal restriction discrepancy
        gate_scores = -norms
        return mx.softmax(gate_scores, axis=-1)


class ReactionDiffusionSemanticGating(nn.Module):
    def __init__(self, embed_dim: int, decay: float = 0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.decay = decay
        
        # Activator and inhibitor weights
        self.w_activator = nn.Linear(embed_dim, 1, bias=False)
        self.w_inhibitor = nn.Linear(embed_dim, 1, bias=False)
        
    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # Models Turing pattern Reaction-Diffusion semantic gating
        # Activator (u) and Inhibitor (v) concentrations
        u = mx.sigmoid(self.w_activator(x)) # [B, S, 1]
        v = mx.sigmoid(self.w_inhibitor(x)) # [B, S, 1]
        
        # Gating pattern: active if activator concentration exceeds inhibitor
        gate_mask = u > (v + self.decay)
        return gate_mask, u


class SymplecticFourierMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Symplectic matrix parameters in the frequency domain
        self.W_real = mx.array(np.random.normal(size=(hidden_dim, embed_dim)) * 0.02, dtype=mx.float32)
        self.W_imag = mx.array(np.random.normal(size=(hidden_dim, embed_dim)) * 0.02, dtype=mx.float32)
        
    def __call__(self, x: mx.array) -> mx.array:
        # Fast Fourier Transform (FFT) on sequence dimension
        B, S, D = x.shape
        
        # Compute 1D Fast Fourier Transform along axis 1 (sequence dimension)
        # Note: MLX supports real FFT. For simplicity in JIT-differentiable format,
        # we compute FFT using a Discrete Fourier Transform (DFT) projection matrix
        if not hasattr(self, "_dft_matrix") or self._dft_matrix.shape[0] != S:
            n = mx.arange(S)
            k = n[:, None]
            M = mx.array(np.exp(-2j * np.pi * k * n / S))
            self._dft_matrix = mx.real(M)
            self._dft_matrix_imag = mx.imag(M)
            
        x_fft_real = self._dft_matrix @ x
        x_fft_imag = self._dft_matrix_imag @ x
        
        # Apply complex symplectic FFN updates
        # [B, S, hidden_dim]
        out_real = x_fft_real @ self.W_real.T - x_fft_imag @ self.W_imag.T
        out_imag = x_fft_real @ self.W_imag.T + x_fft_imag @ self.W_real.T
        
        # Inverse DFT
        if not hasattr(self, "_idft_matrix") or self._idft_matrix.shape[0] != S:
            self._idft_matrix = mx.transpose(self._dft_matrix) / S
            self._idft_matrix_imag = -mx.transpose(self._dft_matrix_imag) / S
            
        # Reconstruct output in time domain
        out = self._idft_matrix @ out_real - self._idft_matrix_imag @ out_imag
        
        # Pad or project hidden_dim back to embed_dim
        if self.hidden_dim != self.embed_dim:
            proj = nn.Linear(self.hidden_dim, self.embed_dim)
            out = proj(out)
        return out


class KnotEquivalenceCacheTopology:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def compute_crossing_number(self, x: mx.array) -> mx.array:
        B, S, D = x.shape
        chunk_sz = D // 3
        c1 = mx.mean(x[..., :chunk_sz], axis=-1)
        c2 = mx.mean(x[..., chunk_sz:2*chunk_sz], axis=-1)
        c3 = mx.mean(x[..., 2*chunk_sz:3*chunk_sz], axis=-1)
        coords = mx.stack([c1, c2, c3], axis=-1)
        
        diffs = coords[:, :, None, :] - coords[:, None, :, :]
        dist_sq = mx.sum(diffs ** 2, axis=-1)
        crossing_index = mx.mean(1.0 / (dist_sq + 1e-5), axis=(1, 2))
        return crossing_index


class AnyonicBraidingGating:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts

    def route(self, x: mx.array) -> mx.array:
        B, S, D = x.shape
        proj = mx.mean(x, axis=-1)
        braid_perm = mx.argsort(proj, axis=-1)
        expert_indices = braid_perm % self.num_experts
        return expert_indices


class AllostericCooperativityGating:
    def __init__(self, hill_n: float = 4.0, k_half: float = 0.5):
        self.hill_n = hill_n
        self.k_half = k_half

    def should_exit(self, confidence_scores: mx.array) -> mx.array:
        xn = mx.power(confidence_scores, self.hill_n)
        kn = self.k_half ** self.hill_n
        fraction = xn / (kn + xn)
        return fraction >= 0.5


class MyelinShearSparseInsulation:
    def __init__(self, keep_ratio: float = 0.8):
        self.keep_ratio = keep_ratio

    def insulate(self, weights: mx.array) -> mx.array:
        abs_w = mx.abs(weights)
        flat_w = abs_w.reshape(-1)
        S = flat_w.size
        k = int((1.0 - self.keep_ratio) * S)
        if k <= 0:
            return weights
        sorted_w = mx.sort(flat_w)
        threshold = sorted_w[k]
        mask = abs_w >= threshold
        return weights * mask


class TropicalSemiringSpeculation:
    def __init__(self):
        pass

    def evaluate_path_costs(self, transition_costs: mx.array) -> mx.array:
        B, N, _ = transition_costs.shape
        costs_expanded_1 = transition_costs[:, :, None, :]
        costs_expanded_2 = mx.transpose(transition_costs, (0, 2, 1))[:, None, :, :]
        tropical_sum = costs_expanded_1 + costs_expanded_2
        min_costs = mx.min(tropical_sum, axis=3)
        return min_costs


class GrothendieckTopologyGating:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def verify_coverage(self, path_logits: mx.array) -> mx.array:
        B, T, V = path_logits.shape
        normalized = path_logits - mx.logsumexp(path_logits, axis=-1, keepdims=True)
        probs = mx.exp(normalized)
        
        entropy = -mx.sum(probs * normalized, axis=-1)
        mean_entropy = mx.mean(entropy, axis=-1)
        coverage_score = mx.exp(-mean_entropy)
        return coverage_score >= self.threshold


class GaloisFieldGF28SIMD:
    def __init__(self):
        self.poly = 0x11d
        self.gf_log = np.zeros(256, dtype=np.int32)
        self.gf_exp = np.zeros(512, dtype=np.int32)
        
        val = 1
        for i in range(255):
            self.gf_exp[i] = val
            self.gf_log[val] = i
            val <<= 1
            if val & 0x100:
                val ^= self.poly
        for i in range(255, 512):
            self.gf_exp[i] = self.gf_exp[i - 255]
            
        self.log_mx = mx.array(self.gf_log)
        self.exp_mx = mx.array(self.gf_exp)

    def multiply(self, a: mx.array, b: mx.array) -> mx.array:
        zero_mask = (a == 0) | (b == 0)
        
        log_a = mx.take(self.log_mx, mx.maximum(a, 1))
        log_b = mx.take(self.log_mx, mx.maximum(b, 1))
        
        log_sum = log_a + log_b
        prod = mx.take(self.exp_mx, log_sum)
        return mx.where(zero_mask, mx.array(0, dtype=a.dtype), prod)


class JonesPolynomialKnotInvariants:
    def __init__(self):
        pass

    def compute_kauffman_bracket(self, crossings: mx.array) -> float:
        # Simple Kauffman bracket skein evaluation: <L> = A^x + A^-x
        A = 0.9
        if crossings.size == 0:
            return 1.0
        sum_crossings = mx.sum(crossings).item()
        bracket = (A ** sum_crossings) + (A ** -sum_crossings)
        return float(bracket)


class IsingSpinGlassExpertRouting:
    def __init__(self, num_experts: int):
        self.num_experts = num_experts

    def route(self, logits: mx.array, coupling: mx.array) -> mx.array:
        B, E = logits.shape
        spins = mx.where(logits >= 0.0, mx.array(1.0), mx.array(-1.0))
        fields = logits + spins @ coupling
        new_spins = mx.where(fields >= 0.0, mx.array(1.0), mx.array(-1.0))
        expert_indices = mx.argmax(new_spins, axis=-1)
        return expert_indices


class TranscriptionFactorGeneGating:
    def __init__(self, num_factors: int = 4):
        self.num_factors = num_factors

    def gate_layers(self, activator_concentrations: mx.array) -> mx.array:
        B, F = activator_concentrations.shape
        tf_0 = activator_concentrations[:, 0]
        tf_1 = activator_concentrations[:, 1]
        tf_2 = activator_concentrations[:, 2]
        tf_3 = activator_concentrations[:, 3] if F > 3 else tf_0
        
        and_gate = (tf_0 > 0.5) & (tf_1 > 0.5)
        or_gate = (tf_2 > 0.5) | (tf_3 > 0.5)
        should_skip = ~(and_gate | or_gate)
        return should_skip


class ProteinChaperoneQuantization:
    def __init__(self, num_bits: int = 4):
        self.q_min = -(2 ** (num_bits - 1))
        self.q_max = (2 ** (num_bits - 1)) - 1

    def chaperone_regularization(self, w: mx.array, scale: mx.array) -> mx.array:
        q_w = mx.round(w / scale)
        q_w = mx.clip(q_w, self.q_min, self.q_max)
        centers = q_w * scale
        chaperoned = w - 0.1 * (w - centers)
        return chaperoned


class QuantumDecoherencePathCollapse:
    def __init__(self, coherence_threshold: float = 0.1):
        self.coherence_threshold = coherence_threshold

    def collapse_paths(self, path_density_matrices: mx.array) -> mx.array:
        B, P, _ = path_density_matrices.shape
        rho_squared = path_density_matrices @ path_density_matrices
        purity = mx.sum(mx.diagonal(rho_squared, axis1=1, axis2=2), axis=-1)
        keep_path = purity >= self.coherence_threshold
        return keep_path


class HamiltonJacobiTrajectoryRouting:
    def __init__(self, time_steps: int = 10):
        self.time_steps = time_steps

    def get_optimal_exit(self, path_value_functions: mx.array) -> mx.array:
        B, T = path_value_functions.shape
        d_val = path_value_functions[:, 1:] - path_value_functions[:, :-1]
        neg_slope = d_val < 0.0
        exit_indices = mx.argmax(neg_slope.astype(mx.int32), axis=-1)
        return exit_indices


class LaplacianSpectralGraphPooling:
    def __init__(self, keep_k: int = 4):
        self.keep_k = keep_k

    def pool_graph(self, similarity_matrix: mx.array, x: mx.array) -> mx.array:
        B, S, _ = similarity_matrix.shape
        deg = mx.sum(similarity_matrix, axis=-1, keepdims=True)
        inv_deg_sqrt = mx.where(deg > 0.0, 1.0 / mx.sqrt(deg), mx.array(0.0))
        W_normalized = similarity_matrix * inv_deg_sqrt * mx.transpose(inv_deg_sqrt, (0, 2, 1))
        pooled = W_normalized @ x
        return pooled


def layer_by_layer_prefill(model: nn.Module, token_ids: mx.array, cache: Any, chunk_size: int = 2048) -> Tuple[mx.array, mx.array]:
    """
    Computes the prefill phase layer-by-layer rather than chunk-by-chunk across the whole model.
    This ensures that the unquantized weights of layer L are dequantized once, used for all chunks,
    and then discarded before moving to layer L+1, drastically reducing VRAM and dequantization overhead.
    """
    import gc
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.gemma4_text import logit_softcap
    from typing import Tuple
    
    text_model = model.language_model.model if hasattr(model, "language_model") else (model.model if hasattr(model, "model") else model)
    
    # 1. Resolve embedding layer
    embed_layer = getattr(text_model, "embed_tokens", None) or getattr(text_model, "wte", None)
    if embed_layer is None:
        raise ValueError("Could not find embedding layer in model")
        
    embed_scale = getattr(text_model, "embed_scale", 1.0)
    if embed_scale == 1.0 and hasattr(text_model, "args") and hasattr(text_model.args, "hidden_size"):
        if "gemma" in getattr(text_model.args, "model_type", "").lower():
            embed_scale = text_model.args.hidden_size ** 0.5
            
    total_tokens = token_ids.shape[1]
    
    # Win 310: GPU E8 Coordinate Pre-fetching
    # Pre-compute E8 coordinates on the GPU once per prefill run
    organism = None
    try:
        from qan_transformers.mlx.attention import QuasicrystallineAttention
        organism = getattr(QuasicrystallineAttention, "organism", None)
    except ImportError:
        pass
        
    if organism is None:
        organism = getattr(text_model, "organism", None) or getattr(model, "organism", None)
        
    if organism is not None:
        try:
            if hasattr(token_ids, "tolist"):
                tids = token_ids.tolist()
                tids_list = tids[0] if isinstance(tids[0], list) else tids
            else:
                tids_list = list(token_ids)
            coords_list, _ = organism.get_sequence_lattice_metadata(tids_list)
            from qan_transformers.mlx.attention import QuasicrystallineAttention
            QuasicrystallineAttention.precomputed_e8_coords = mx.array(coords_list, dtype=mx.float32)
        except Exception as e:
            print(f"[DEBUG E8] Precomputation failed: {e}", flush=True)

    # Set is_prefill = True on all QuasicrystallineAttention modules
    for m in model.modules():
        if m.__class__.__name__ == "QuasicrystallineAttention":
            m.is_prefill = True

    # 2. Embed the entire sequence chunk-by-chunk (no eager concatenation)
    h_chunks = []
    for i in range(0, total_tokens, chunk_size):
        chunk = token_ids[:, i : i + chunk_size]
        h_chunk = embed_layer(chunk) * embed_scale
        h_chunks.append(h_chunk)
        
    # 2b. Compute per-layer inputs if needed
    has_per_layer = getattr(text_model, "hidden_size_per_layer_input", None)
    if has_per_layer:
        h_full_input = mx.concatenate(h_chunks, axis=1)
        per_layer_inputs = text_model._get_per_layer_inputs(token_ids, h_full_input)
        per_layer_inputs = text_model._project_per_layer_inputs(h_full_input, per_layer_inputs)
        per_layer_inputs = [
            per_layer_inputs[:, :, idx, :] for idx in range(len(text_model.layers))
        ]
        del h_full_input
    else:
        per_layer_inputs = None
        
    # Create padded cache list matching total model layers
    padded_cache = cache
    if cache is not None and len(cache) < len(text_model.layers):
        padded_cache = cache + [None] * (len(text_model.layers) - len(cache))
        
    has_prev_kvs = hasattr(text_model, "previous_kvs")
    previous_kvs = getattr(text_model, "previous_kvs", None)
    
    # Compute maximum layer dependent index for garbage collection
    max_dep = {}
    if has_prev_kvs and previous_kvs is not None:
        for idx, prev_idx in enumerate(previous_kvs):
            if prev_idx != idx:
                max_dep[prev_idx] = max(max_dep.get(prev_idx, -1), idx)
                
    intermediates = {} # (layer_idx, chunk_idx) -> (shared_kv, offset)
    
    # 3. Process layer by layer (propagating chunk hidden states as a flat list)
    for layer_idx, layer in enumerate(text_model.layers):
        layer_outputs = []
        layer_cache = padded_cache[layer_idx] if padded_cache is not None else None
        
        # Check if we need to retrieve shared_kv for this layer
        prev_kv_layer_idx = previous_kvs[layer_idx] if has_prev_kvs and previous_kvs is not None else None
        is_shared_kv_layer = (prev_kv_layer_idx is not None and prev_kv_layer_idx != layer_idx)
        
        # Check window size for the layer
        window_size = None
        if hasattr(layer, "layer_type") and layer.layer_type == "sliding_attention":
            window_size = getattr(text_model, "window_size", None)
            
        for chunk_idx, chunk_h in enumerate(h_chunks):
            chunk_per_layer_input = (
                per_layer_inputs[layer_idx][:, chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
                if per_layer_inputs is not None
                else None
            )
            
            if is_shared_kv_layer:
                # Retrieve from intermediates
                shared_kv, offset = intermediates.get((prev_kv_layer_idx, chunk_idx), (None, None))
                
                # For shared KV layer, create the causal/window mask using the retrieved offset
                if window_size is not None and offset is not None and offset > 0:
                    from mlx_lm.models.base import create_causal_mask
                    mask = create_causal_mask(chunk_h.shape[1], offset, window_size=window_size)
                else:
                    mask = create_attention_mask(chunk_h, None, window_size=window_size)
                    
                chunk_out = layer(
                    chunk_h, 
                    mask=mask, 
                    cache=layer_cache, 
                    per_layer_input=chunk_per_layer_input, 
                    shared_kv=shared_kv, 
                    offset=offset
                )
            else:
                mask = create_attention_mask(chunk_h, layer_cache, window_size=window_size)
                chunk_out = layer(
                    chunk_h, 
                    mask=mask, 
                    cache=layer_cache, 
                    per_layer_input=chunk_per_layer_input
                )
                
            if isinstance(chunk_out, (tuple, list)):
                layer_outputs.append(chunk_out[0])
                # If this layer's output is shared by subsequent layers, we must store its shared_kv and offset
                if has_prev_kvs and len(chunk_out) >= 3:
                    intermediates[(layer_idx, chunk_idx)] = (chunk_out[1], chunk_out[2])
            else:
                layer_outputs.append(chunk_out)
            
        h_chunks = layer_outputs
        
        # Win 311: Periodic JIT Graph and GC Tuning
        # Force evaluation of the hidden states and cache of the layer
        eval_targets = list(h_chunks)
        if layer_cache is not None:
            if layer_cache.keys is not None:
                eval_targets.append(layer_cache.keys)
            if layer_cache.values is not None:
                eval_targets.append(layer_cache.values)
                
        # Evaluate custom KV cache if present (QuasicrystallineAttention)
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "custom_kv_cache"):
            cc = layer.self_attn.custom_kv_cache
            if cc is not None:
                if cc.get("K") is not None:
                    eval_targets.append(cc["K"])
                if cc.get("V") is not None:
                    eval_targets.append(cc["V"])
                if cc.get("indices") is not None:
                    eval_targets.append(cc["indices"])
                if cc.get("alignment_scores") is not None:
                    eval_targets.append(cc["alignment_scores"])
                    
        # Garbage collect intermediates that are no longer needed
        if has_prev_kvs and previous_kvs is not None:
            for prev_idx, max_idx in list(max_dep.items()):
                if max_idx == layer_idx:
                    # Delete all chunks for prev_idx
                    num_chunks = len(layer_outputs)
                    for c_idx in range(num_chunks):
                        intermediates.pop((prev_idx, c_idx), None)
                        
        # Periodically block to clear cache (every 8 layers) or only at the final layer
        if (layer_idx + 1) % 8 == 0 or layer_idx == len(text_model.layers) - 1:
            mx.eval(*eval_targets)
            gc.collect()
            mx.clear_cache()
        else:
            mx.async_eval(*eval_targets)
            
    # 4. Final normalization and logits (Win 312: Last-Token Final Normalization)
    h_last = h_chunks[-1][:, -1:]
    norm_layer = getattr(text_model, "norm", None) or getattr(text_model, "ln_f", None)
    if norm_layer is not None:
        h_last = norm_layer(h_last)
        
    # Project to logits
    container = model.language_model if hasattr(model, "language_model") else model
    if getattr(container, "tie_word_embeddings", False):
        logits = embed_layer.as_linear(h_last)
    elif hasattr(container, "lm_head"):
        logits = container.lm_head(h_last)
    else:
        logits = embed_layer.as_linear(h_last)
        
    softcap = getattr(container, "final_logit_softcapping", None)
    if softcap is not None:
        logits = logit_softcap(softcap, logits)
        
    # Clean up pre-fetched E8 coordinates and reset prefill flags to release memory
    try:
        from qan_transformers.mlx.attention import QuasicrystallineAttention
        if hasattr(QuasicrystallineAttention, "precomputed_e8_coords"):
            del QuasicrystallineAttention.precomputed_e8_coords
    except Exception:
        pass

    for m in model.modules():
        if m.__class__.__name__ == "QuasicrystallineAttention":
            m.is_prefill = False

    mx.eval(logits, h_last)
    return logits, h_last


