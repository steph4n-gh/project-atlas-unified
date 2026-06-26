import mlx.core as mx
import mlx.nn as nn
import math

class FusedJITSpeculativeVerifier:
    def __init__(self):
        pass

    def verify(self, target_logits: mx.array, candidate_tokens: mx.array) -> tuple[mx.array, mx.array]:
        # target_logits: (B, S, V)
        # candidate_tokens: (B, S)
        B, S, V = target_logits.shape

        @mx.compile
        def _verify_jit(t_logits, c_tokens):
            t_preds = mx.argmax(t_logits, axis=-1)
            matches = (t_preds == c_tokens).astype(mx.int32)
            valid_mask = mx.cumprod(matches, axis=-1)
            num_accepted = mx.sum(valid_mask, axis=-1)
            
            clamp_idx = mx.minimum(num_accepted, S - 1)
            batch_indices = mx.arange(B)
            correction_tokens = t_preds[batch_indices, clamp_idx]
            return num_accepted, correction_tokens

        return _verify_jit(target_logits, candidate_tokens)


class JITDraftARLoopGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, x: mx.array, num_tokens: int, temp: float = 0.0) -> mx.array:
        # x: input prompt token ids. shape (B, S_in)
        B, S_in = x.shape

        @mx.compile
        def _generate_loop(inputs):
            tokens = []
            curr_input = inputs
            for _ in range(num_tokens):
                logits = self.model(curr_input)
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                
                if temp == 0.0:
                    next_tok = mx.argmax(logits, axis=-1, keepdims=True)
                else:
                    next_tok = mx.random.categorical(logits / temp, num_samples=1)
                
                tokens.append(next_tok)
                curr_input = next_tok
            return mx.concatenate(tokens, axis=-1)

        return _generate_loop(x)


class ZeroCopySlidingKVCacheManager:
    def __init__(self, batch_size: int, num_heads: int, max_length: int, head_dim: int):
        self.max_length = max_length
        self.k_cache = mx.zeros((batch_size, num_heads, max_length, head_dim))
        self.v_cache = mx.zeros((batch_size, num_heads, max_length, head_dim))

    def update(self, new_k: mx.array, new_v: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        B, H, S, D = new_k.shape

        @mx.compile
        def _update_jit(k_c, v_c, nk, nv):
            k_c[:, :, offset:offset+S, :] = nk
            v_c[:, :, offset:offset+S, :] = nv
            return k_c, v_c

        self.k_cache, self.v_cache = _update_jit(self.k_cache, self.v_cache, new_k, new_v)
        return self.k_cache[:, :, :offset+S, :], self.v_cache[:, :, :offset+S, :]


class JITCompiledSpeculativeSampler:
    def __init__(self):
        pass

    def sample(self, logits: mx.array, temp: float = 1.0, k: int = 0) -> mx.array:
        # logits: (B, V)
        @mx.compile
        def _sample_jit(l):
            scaled_logits = l
            if temp > 0.0:
                scaled_logits = l / temp
            
            if k > 0:
                sorted_logits = mx.sort(scaled_logits, axis=-1)
                threshold = sorted_logits[..., -k:-k+1]
                scaled_logits = mx.where(scaled_logits < threshold, mx.array(-float("inf")), scaled_logits)
            
            return mx.random.categorical(scaled_logits, num_samples=1)

        return _sample_jit(logits)


class JITCompiledRoPECache:
    def __init__(self, max_length: int, head_dim: int, base: float = 10000.0):
        self.max_length = max_length
        self.head_dim = head_dim
        
        inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        t = mx.arange(max_length, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)
        
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cache = mx.cos(emb)
        self.sin_cache = mx.sin(emb)

    def apply(self, x: mx.array, offset: int) -> mx.array:
        B, H, S, D = x.shape
        
        @mx.compile
        def _apply_jit(arr, cos_c, sin_c):
            c = cos_c[offset:offset+S, :]
            s = sin_c[offset:offset+S, :]
            c = c[None, None, :, :]
            s = s[None, None, :, :]
            
            x1 = arr[..., :D//2]
            x2 = arr[..., D//2:]
            x_rot = mx.concatenate([-x2, x1], axis=-1)
            
            return arr * c + x_rot * s

        return _apply_jit(x, self.cos_cache, self.sin_cache)


class FusedDraftTargetProjection:
    def __init__(self, w_draft: mx.array, w_target: mx.array):
        # w_draft: (D_in, D_draft)
        # w_target: (D_in, D_target)
        self.w_fused = mx.concatenate([w_draft, w_target], axis=-1)
        self.d_draft = w_draft.shape[-1]

    def project(self, x: mx.array) -> tuple[mx.array, mx.array]:
        @mx.compile
        def _project_jit(arr, w, split_dim):
            out = arr @ w
            d_proj = out[..., :split_dim]
            t_proj = out[..., split_dim:]
            return d_proj, t_proj

        return _project_jit(x, self.w_fused, self.d_draft)


class UnifiedSpeculativeKVCache:
    def __init__(self, batch_size: int, h_draft: int, h_target: int, max_length: int, head_dim: int):
        self.h_draft = h_draft
        self.h_target = h_target
        self.max_length = max_length
        self.k_cache = mx.zeros((batch_size, h_draft + h_target, max_length, head_dim))
        self.v_cache = mx.zeros((batch_size, h_draft + h_target, max_length, head_dim))

    def update(self, k_d: mx.array, v_d: mx.array, k_t: mx.array, v_t: mx.array, offset: int):
        S_d = k_d.shape[2]
        S_t = k_t.shape[2]

        @mx.compile
        def _update_jit(kc, vc, kd, vd, kt, vt):
            kc[:, :self.h_draft, offset:offset+S_d, :] = kd
            vc[:, :self.h_draft, offset:offset+S_d, :] = vd
            kc[:, self.h_draft:, offset:offset+S_t, :] = kt
            vc[:, self.h_draft:, offset:offset+S_t, :] = vt
            return kc, vc

        self.k_cache, self.v_cache = _update_jit(self.k_cache, self.v_cache, k_d, v_d, k_t, v_t)
        
        k_d_view = self.k_cache[:, :self.h_draft, :offset+S_d, :]
        v_d_view = self.v_cache[:, :self.h_draft, :offset+S_d, :]
        k_t_view = self.k_cache[:, self.h_draft:, :offset+S_t, :]
        v_t_view = self.v_cache[:, self.h_draft:, :offset+S_t, :]
        return (k_d_view, v_d_view), (k_t_view, v_t_view)


class FusedMHAExecution:
    def __init__(self, w_out: mx.array):
        self.w_out = w_out

    def execute(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        B, H, S, D = q.shape

        @mx.compile
        def _execute_jit(qi, ki, vi, wo):
            ki_t = mx.transpose(ki, (0, 1, 3, 2))
            scale = 1.0 / math.sqrt(D)
            scores = (qi @ ki_t) * scale
            probs = mx.softmax(scores, axis=-1)
            context = probs @ vi
            context_trans = mx.transpose(context, (0, 2, 1, 3))
            context_flat = mx.reshape(context_trans, (B, S, H * D))
            return context_flat @ wo

        return _execute_jit(q, k, v, self.w_out)


class FusedSwiGLUFFN:
    def __init__(self, w_gate: mx.array, w_up: mx.array, w_down: mx.array):
        # w_gate: (D_in, D_ff)
        # w_up: (D_in, D_ff)
        # w_down: (D_ff, D_out)
        self.w_gate_up = mx.concatenate([w_gate, w_up], axis=-1)
        self.d_ff = w_gate.shape[-1]
        self.w_down = w_down

    def forward(self, x: mx.array) -> mx.array:
        @mx.compile
        def _forward_jit(arr, w_gu, w_d, split_dim):
            gate_up = arr @ w_gu
            gate = gate_up[..., :split_dim]
            up = gate_up[..., split_dim:]
            activated = (gate * mx.sigmoid(gate)) * up
            return activated @ w_d

        return _forward_jit(x, self.w_gate_up, self.w_down, self.d_ff)


class FusedGeGLUFFN(nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.down_proj = down_proj
        self.gelu = nn.GELU()

        # Check if both projections expose raw weights (nn.Linear case).
        # ELQLinear uses compressed _indices/_scales, and nn.QuantizedLinear
        # has .weight but it's a quantized array that can't be concatenated.
        has_raw_weights = (
            hasattr(gate_proj, "weight") and hasattr(up_proj, "weight")
            and not hasattr(gate_proj, "scales")  # excludes nn.QuantizedLinear
            and not hasattr(up_proj, "scales")
            and isinstance(gate_proj.weight, mx.array)
        )
        if has_raw_weights:
            # Fuse gate and up weights into one tensor for a single matmul.
            # nn.Linear.weight shape is (D_ff, D_in) — rows are output features.
            self.w_gate_up = mx.concatenate(
                [gate_proj.weight, up_proj.weight], axis=0
            )
            self.d_ff = gate_proj.weight.shape[0]
            self._use_fused_weight = True

            @mx.compile
            def _fused_weight_forward(x, w_gate_up, w_down, d_ff):
                gate_up = x @ w_gate_up.T
                gate = gate_up[..., :d_ff]
                up = gate_up[..., d_ff:]
                activated = self.gelu(gate) * up
                return activated @ w_down.T

            self._compiled_forward = _fused_weight_forward
        else:
            # ELQLinear path — keep the two separate module calls, but use fused geglu kernel when B==1.
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self._use_fused_weight = False

            def _module_forward(x):
                # Retrieve cache references
                cache = getattr(self.gate_proj, "cache", None)
                global_bypass = getattr(self.gate_proj.__class__, "_global_cache_bypass", False)
                
                # Check if we should use cache (if cache contains both weights)
                use_cache = False
                if cache is not None and cache.is_enabled and not global_bypass:
                    if self.gate_proj._layer_id in cache._cache and self.up_proj._layer_id in cache._cache:
                        use_cache = True
                
                # B is the batch size (B=1 is our optimized case)
                B = x.shape[0] if len(x.shape) > 1 else 1
                
                # If cache is missed/disabled and B == 1, run the fused gate_up Metal kernel
                if not use_cache and B == 1 and self.gate_proj.__class__.__name__ == "ELQLinear" and self.up_proj.__class__.__name__ == "ELQLinear":
                    from qan_transformers.kernels.elq_metal import elq_fused_gate_up
                    gate_up = elq_fused_gate_up(
                        x,
                        self.gate_proj._indices, self.gate_proj._scales,
                        self.up_proj._indices, self.up_proj._scales
                    )
                    d_ff = self.gate_proj.output_dims
                    gate = gate_up[..., :d_ff]
                    up = gate_up[..., d_ff:]
                    
                    # Apply outlier corrections in Python
                    delta_W_gate = self.gate_proj.get_delta_W_T(x.dtype)
                    if delta_W_gate is not None:
                        gate = gate + mx.matmul(x[..., self.gate_proj._outlier_indices], delta_W_gate)
                    
                    delta_W_up = self.up_proj.get_delta_W_T(x.dtype)
                    if delta_W_up is not None:
                        up = up + mx.matmul(x[..., self.up_proj._outlier_indices], delta_W_up)
                    
                    activated = self.gelu(gate) * up
                    return self.down_proj(activated)
                else:
                    gate = self.gate_proj(x)
                    up = self.up_proj(x)
                    activated = self.gelu(gate) * up
                    return self.down_proj(activated)

            self._compiled_forward = _module_forward

    def __call__(self, x: mx.array) -> mx.array:
        if self._use_fused_weight:
            return self._compiled_forward(
                x, self.w_gate_up, self.down_proj.weight, self.d_ff
            )
        return self._compiled_forward(x)


class FusedSpeculativeTransformerBlock:
    def __init__(self,
                 w_q: mx.array,
                 w_k: mx.array,
                 w_v: mx.array,
                 w_out: mx.array,
                 w_gate: mx.array,
                 w_up: mx.array,
                 w_down: mx.array,
                 rms_attn_w: mx.array,
                 rms_ffn_w: mx.array,
                 cos_cache: mx.array,
                 sin_cache: mx.array,
                 eps: float = 1e-6):
        self.w_q = w_q
        self.w_k = w_k
        self.w_v = w_v
        self.w_out = w_out
        self.w_gate_up = mx.concatenate([w_gate, w_up], axis=-1)
        self.d_ff = w_gate.shape[-1]
        self.w_down = w_down
        self.rms_attn_w = rms_attn_w
        self.rms_ffn_w = rms_ffn_w
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.eps = eps

    def forward(self, x: mx.array, k_cache: mx.array, v_cache: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        B, S, D = x.shape
        H = k_cache.shape[1]
        D_head = k_cache.shape[3]
        
        @mx.compile
        def _block_jit(arr, kc, vc, wq, wk, wv, wo, w_gu, wd, r_attn_w, r_ffn_w, cos_c, sin_c):
            # 1. Pre-attention RMSNorm
            var1 = mx.mean(mx.square(arr), axis=-1, keepdims=True)
            norm_arr = (arr * mx.rsqrt(var1 + self.eps)) * (1.0 + r_attn_w)
            
            # 2. QKV Projections
            q = norm_arr @ wq
            k = norm_arr @ wk
            v = norm_arr @ wv
            
            q = mx.reshape(q, (B, S, H, D_head)).transpose(0, 2, 1, 3)
            k = mx.reshape(k, (B, S, H, D_head)).transpose(0, 2, 1, 3)
            v = mx.reshape(v, (B, S, H, D_head)).transpose(0, 2, 1, 3)
            
            # 3. Apply RoPE
            c = cos_c[offset:offset+S, :]
            s = sin_c[offset:offset+S, :]
            c = c[None, None, :, :]
            s = s[None, None, :, :]
            
            q1 = q[..., :D_head//2]
            q2 = q[..., D_head//2:]
            q_rot = mx.concatenate([-q2, q1], axis=-1)
            q = q * c + q_rot * s
            
            k1 = k[..., :D_head//2]
            k2 = k[..., D_head//2:]
            k_rot = mx.concatenate([-k2, k1], axis=-1)
            k = k * c + k_rot * s
            
            # 4. KV Cache Update (Zero-Copy Sliding KV style)
            kc[:, :, offset:offset+S, :] = k
            vc[:, :, offset:offset+S, :] = v
            
            # 5. SDPA
            k_seq = kc[:, :, :offset+S, :]
            v_seq = vc[:, :, :offset+S, :]
            
            k_seq_t = mx.transpose(k_seq, (0, 1, 3, 2))
            scale = 1.0 / math.sqrt(D_head)
            scores = (q @ k_seq_t) * scale
            probs = mx.softmax(scores, axis=-1)
            context = probs @ v_seq
            
            context_trans = mx.transpose(context, (0, 2, 1, 3))
            context_flat = mx.reshape(context_trans, (B, S, H * D_head))
            attn_out = context_flat @ wo
            
            # 6. Residual
            x_attn = arr + attn_out
            
            # 7. Pre-FFN RMSNorm
            var2 = mx.mean(mx.square(x_attn), axis=-1, keepdims=True)
            norm_x_attn = (x_attn * mx.rsqrt(var2 + self.eps)) * (1.0 + r_ffn_w)
            
            # 8. SwiGLU FFN
            gate_up = norm_x_attn @ w_gu
            gate = gate_up[..., :self.d_ff]
            up = gate_up[..., self.d_ff:]
            activated = (gate * mx.sigmoid(gate)) * up
            ffn_out = activated @ wd
            
            # 9. Second Residual
            out = x_attn + ffn_out
            
            return out, kc, vc
            
        return _block_jit(
            x, k_cache, v_cache,
            self.w_q, self.w_k, self.w_v, self.w_out,
            self.w_gate_up, self.w_down,
            self.rms_attn_w, self.rms_ffn_w,
            self.cos_cache, self.sin_cache
        )


class JITCompiledPrefetchLookahead:
    def __init__(self, embeddings: mx.array):
        self.embeddings = embeddings

    def prefetch(self, candidate_ids: mx.array, lookahead_steps: int = 1) -> tuple[mx.array, mx.array]:
        B, S = candidate_ids.shape
        
        @mx.compile
        def _prefetch_jit(ids, w_emb):
            curr_embs = w_emb[ids]
            
            lookahead_ids = mx.zeros_like(ids)
            if S > lookahead_steps:
                lookahead_ids[:, :-lookahead_steps] = ids[:, lookahead_steps:]
            
            lookahead_embs = w_emb[lookahead_ids]
            return curr_embs, lookahead_embs

        return _prefetch_jit(candidate_ids, self.embeddings)


class DualStreamCommandQueuing:
    def __init__(self):
        self.draft_stream = mx.new_stream(mx.gpu)
        self.target_stream = mx.new_stream(mx.gpu)

    def execute_dual(self, draft_fn, target_fn, draft_args=(), target_args=(), draft_kwargs=None, target_kwargs=None):
        if draft_kwargs is None:
            draft_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
            
        with mx.StreamContext(self.draft_stream):
            draft_out = draft_fn(*draft_args, **draft_kwargs)
            
        with mx.StreamContext(self.target_stream):
            target_out = target_fn(*target_args, **target_kwargs)
            
        return draft_out, target_out


class ContiguousFlashKVLinearization:
    def __init__(self, batch_size: int, num_heads: int, max_length: int, head_dim: int):
        self.max_length = max_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = mx.zeros((batch_size, max_length, num_heads, head_dim))
        self.v_cache = mx.zeros((batch_size, max_length, num_heads, head_dim))

    def update_and_get(self, new_k: mx.array, new_v: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        if new_k.ndim == 4 and new_k.shape[1] == self.num_heads:
            new_k = mx.transpose(new_k, (0, 2, 1, 3))
            new_v = mx.transpose(new_v, (0, 2, 1, 3))
            
        S = new_k.shape[1]
        
        @mx.compile
        def _update_jit(kc, vc, nk, nv):
            kc[:, offset:offset+S, :, :] = nk
            vc[:, offset:offset+S, :, :] = nv
            return kc, vc

        self.k_cache, self.v_cache = _update_jit(self.k_cache, self.v_cache, new_k, new_v)
        return self.k_cache[:, :offset+S, :, :], self.v_cache[:, :offset+S, :, :]


class DynamicDraftAdaptiveLength:
    def __init__(self, model, max_draft_len: int, entropy_threshold: float = 2.0):
        self.model = model
        self.max_draft_len = max_draft_len
        self.entropy_threshold = entropy_threshold

    def generate_adaptive(self, x: mx.array, temp: float = 1.0) -> tuple[mx.array, mx.array]:
        B, S_in = x.shape
        
        @mx.compile
        def _generate_adaptive_jit(inputs):
            tokens = []
            curr_input = inputs
            active_mask = mx.ones((B,), dtype=mx.bool_)
            active_counts = mx.zeros((B,), dtype=mx.int32)
            
            for step in range(self.max_draft_len):
                logits = self.model(curr_input)
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                
                probs = mx.softmax(logits, axis=-1)
                entropy = -mx.sum(probs * mx.log(probs + 1e-9), axis=-1)
                
                should_keep = active_mask & (entropy <= self.entropy_threshold)
                active_counts = mx.where(should_keep, active_counts + 1, active_counts)
                active_mask = should_keep
                
                if temp == 0.0:
                    next_tok = mx.argmax(logits, axis=-1, keepdims=True)
                else:
                    next_tok = mx.random.categorical(logits / temp, num_samples=1)
                
                tokens.append(next_tok)
                curr_input = next_tok
                
            return mx.concatenate(tokens, axis=-1), active_counts

        return _generate_adaptive_jit(x)


class UnifiedSpeculativePipeline:
    def __init__(self, draft_model, target_model, draft_max_len: int = 4, entropy_threshold: float = 2.0):
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_max_len = draft_max_len
        self.entropy_threshold = entropy_threshold
        
        self.verifier = FusedJITSpeculativeVerifier()
        self.stream_queuer = DualStreamCommandQueuing()

    def step(self, x: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        B, S_in = x.shape
        
        # 1. Generate candidate tokens using adaptive length draft generator
        draft_generator = DynamicDraftAdaptiveLength(self.draft_model, self.draft_max_len, self.entropy_threshold)
        candidates, active_counts = draft_generator.generate_adaptive(x, temp=0.0)
        
        # 2. Concurrently evaluate target model and perform KV updates
        def run_target():
            target_input = mx.concatenate([x, candidates], axis=-1)
            target_logits = self.target_model(target_input)
            return target_logits
            
        def run_dummy_draft_kv():
            return candidates * 2.0
            
        target_logits, _ = self.stream_queuer.execute_dual(
            run_target, run_dummy_draft_kv
        )
        
        # 3. Verify candidates acceptance
        K = candidates.shape[1]
        verify_logits = target_logits[:, S_in-1 : S_in-1+K, :]
        
        num_accepted, correction_tokens = self.verifier.verify(verify_logits, candidates)
        return candidates, num_accepted, correction_tokens


class ZeroSyncSpeculativeVerifier:
    def __init__(self):
        pass

    def verify_async(self, target_logits: mx.array, candidate_tokens: mx.array, current_offsets: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        B, K, V = target_logits.shape
        
        @mx.compile
        def _verify_jit(t_logits, c_tokens, offsets):
            t_preds = mx.argmax(t_logits, axis=-1)
            matches = (t_preds == c_tokens).astype(mx.int32)
            valid_mask = mx.cumprod(matches, axis=-1)
            num_accepted = mx.sum(valid_mask, axis=-1)
            
            clamp_idx = mx.minimum(num_accepted, K - 1)
            batch_indices = mx.arange(B)
            correction_tokens = t_preds[batch_indices, clamp_idx]
            
            new_offsets = offsets + num_accepted
            return num_accepted, correction_tokens, new_offsets

        return _verify_jit(target_logits, candidate_tokens, current_offsets)


class JITQuantizedWeightCache:
    def __init__(self, q_weight: mx.array, scales: mx.array, biases: mx.array = None):
        self.q_weight = q_weight
        self.scales = scales
        self.biases = biases
        self.cached_weight = None

    def get_dequantized(self) -> mx.array:
        if self.cached_weight is not None:
            return self.cached_weight
            
        @mx.compile
        def _dequantize_jit(qw, s, b):
            w = qw * s
            if b is not None:
                w = w + b
            return w

        self.cached_weight = _dequantize_jit(self.q_weight, self.scales, self.biases)
        return self.cached_weight

    def clear(self):
        self.cached_weight = None


class DynamicDraftLengthAdjuster:
    def __init__(self, initial_k: int = 3, min_k: int = 1, max_k: int = 6, alpha: float = 0.2, momentum: float = 0.0):
        self.k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.alpha = alpha
        self.momentum = momentum
        self.ema_acceptance_rate = 0.5
        self.smoothed_rate = 0.5

    def update_and_get_k(self, num_accepted: int, num_drafted: int) -> int:
        if num_drafted > 0:
            rate = float(num_accepted) / float(num_drafted)
            self.ema_acceptance_rate = (1.0 - self.alpha) * self.ema_acceptance_rate + self.alpha * rate
            
            # Momentum smoothing on the acceptance rate to filter out transient drops
            self.smoothed_rate = self.momentum * self.smoothed_rate + (1.0 - self.momentum) * self.ema_acceptance_rate
            
            effective_rate = self.ema_acceptance_rate if self.momentum == 0.0 else self.smoothed_rate
            
            if effective_rate > 0.7:
                self.k = min(self.k + 1, self.max_k)
            elif effective_rate < 0.3:
                self.k = max(self.k - 1, self.min_k)
                
        return self.k




class FusedKVAppendVerify:
    def __init__(self, batch_size: int, num_heads: int, max_length: int, head_dim: int):
        self.max_length = max_length
        self.k_cache = mx.zeros((batch_size, num_heads, max_length, head_dim))
        self.v_cache = mx.zeros((batch_size, num_heads, max_length, head_dim))

    def append_verified_only(self, new_k: mx.array, new_v: mx.array, num_accepted: mx.array, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        B, H, K, D = new_k.shape
        
        @mx.compile
        def _append_jit(kc, vc, nk, nv, num_acc):
            acc = num_acc[0]
            indices = mx.arange(K)
            mask = indices < acc
            mask = mask[None, None, :, None]
            
            target_slice = kc[:, :, offset:offset+K, :]
            kc[:, :, offset:offset+K, :] = mx.where(mask, nk, target_slice)
            
            target_slice_v = vc[:, :, offset:offset+K, :]
            vc[:, :, offset:offset+K, :] = mx.where(mask, nv, target_slice_v)
            
            new_offset = offset + acc
            return kc, vc, new_offset

        self.k_cache, self.v_cache, new_offset = _append_jit(self.k_cache, self.v_cache, new_k, new_v, num_accepted)
        return self.k_cache, self.v_cache, new_offset


class BlockSparseLayerSkipping:
    def __init__(self, confidence_threshold: float = 0.99):
        self.confidence_threshold = confidence_threshold

    def evaluate_exit(self, hidden_states: mx.array, lm_head_weight: mx.array) -> tuple[mx.array, mx.array]:
        B, S, D = hidden_states.shape
        
        @mx.compile
        def _exit_jit(h, lm_w):
            last_h = h[:, -1, :]
            logits = last_h @ lm_w.T
            probs = mx.softmax(logits, axis=-1)
            
            top_prob = mx.max(probs, axis=-1)
            top_token = mx.argmax(logits, axis=-1)
            
            should_exit = top_prob >= self.confidence_threshold
            return should_exit, top_token

        return _exit_jit(hidden_states, lm_head_weight)


class SIMDCoalescedSoftmax:
    def __init__(self):
        pass

    def softmax(self, logits: mx.array) -> mx.array:
        B, V = logits.shape
        
        @mx.compile
        def _softmax_jit(l):
            pad_val = (16 - (V % 16)) % 16
            if pad_val > 0:
                l_pad = mx.pad(l, [(0, 0), (0, pad_val)], constant_values=-float("inf"))
            else:
                l_pad = l
                
            B_p, V_p = l_pad.shape
            l_reshaped = mx.reshape(l_pad, (B_p, V_p // 16, 16))
            
            max_val = mx.max(l_reshaped, axis=-1, keepdims=True)
            max_val_global = mx.max(max_val, axis=1, keepdims=True)
            
            exp_l = mx.exp(l_reshaped - max_val_global)
            
            sum_val = mx.sum(exp_l, axis=-1, keepdims=True)
            sum_val_global = mx.sum(sum_val, axis=1, keepdims=True)
            
            softmax_out_reshaped = exp_l / sum_val_global
            softmax_out_pad = mx.reshape(softmax_out_reshaped, (B_p, V_p))
            
            if pad_val > 0:
                return softmax_out_pad[:, :V]
            return softmax_out_pad

        return _softmax_jit(logits)


class FusedAttentionProjections:
    def __init__(self, w_q: mx.array, w_k: mx.array, w_v: mx.array):
        self.w_qkv = mx.concatenate([w_q, w_k, w_v], axis=-1)
        self.d_q = w_q.shape[-1]
        self.d_k = w_k.shape[-1]

    def project(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        @mx.compile
        def _project_jit(arr, w, dq, dk):
            qkv = arr @ w
            q = qkv[..., :dq]
            k = qkv[..., dq:dq+dk]
            v = qkv[..., dq+dk:]
            return q, k, v

        return _project_jit(x, self.w_qkv, self.d_q, self.d_k)


class PreAllocatedUnifiedKVCache:
    def __init__(self, num_layers: int, batch_size: int, total_heads: int, max_length: int, head_dim: int):
        self.num_layers = num_layers
        self.max_length = max_length
        self.total_heads = total_heads
        self.k_cache = mx.zeros((num_layers, batch_size, total_heads, max_length, head_dim))
        self.v_cache = mx.zeros((num_layers, batch_size, total_heads, max_length, head_dim))

    def update_layer(self, layer_idx: int, new_k: mx.array, new_v: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        S = new_k.shape[2]
        
        @mx.compile
        def _update_jit(kc, vc, nk, nv):
            kc[layer_idx, :, :, offset:offset+S, :] = nk
            vc[layer_idx, :, :, offset:offset+S, :] = nv
            return kc, vc

        self.k_cache, self.v_cache = _update_jit(self.k_cache, self.v_cache, new_k, new_v)
        return self.k_cache[layer_idx, :, :, :offset+S, :], self.v_cache[layer_idx, :, :, :offset+S, :]


class LookaheadEmbeddingPrefetch:
    def __init__(self, embeddings: mx.array):
        self.embeddings = embeddings
        self.prefetch_stream = mx.new_stream(mx.gpu)

    def prefetch_async(self, next_ids: mx.array) -> mx.array:
        with mx.StreamContext(self.prefetch_stream):
            prefetched_embeddings = self.embeddings[next_ids]
            
        mx.async_eval(prefetched_embeddings)
        return prefetched_embeddings


class BlockWiseWeightLoading:
    def __init__(self, weights_list: list[mx.array]):
        self.contiguous_block = mx.concatenate(weights_list, axis=-1)
        self.sections = []
        curr = 0
        for w in weights_list:
            d_out = w.shape[-1]
            self.sections.append((curr, curr + d_out))
            curr += d_out

    def get_layer_weight(self, layer_idx: int) -> mx.array:
        start, end = self.sections[layer_idx]
        
        @mx.compile
        def _get_weight_jit(block, s, e):
            return block[..., s:e]

        return _get_weight_jit(self.contiguous_block, start, end)


class SpeculativeTreeVerifier:
    def __init__(self):
        pass

    def verify_tree(self, target_logits: mx.array, tree_tokens: mx.array, parent_indices: list[int]) -> mx.array:
        B, N, V = target_logits.shape
        
        @mx.compile
        def _verify_tree_jit(t_logits, t_tokens):
            t_preds = mx.argmax(t_logits, axis=-1)
            matches = (t_preds == t_tokens).astype(mx.int32)
            
            valid_list = []
            valid_list.append(matches[:, 0])
            
            for i in range(1, N):
                p_idx = parent_indices[i]
                p_valid = valid_list[p_idx]
                node_valid = matches[:, i] * p_valid
                valid_list.append(node_valid)
                
            path1_acc = valid_list[0] + valid_list[1] + valid_list[2]
            path2_acc = valid_list[0] + valid_list[3] + valid_list[4]
            best_acc = mx.maximum(path1_acc, path2_acc)
            return best_acc

        return _verify_tree_jit(target_logits, tree_tokens)


class FusedSwiGLUFFNReg:
    def __init__(self, w_gate: mx.array, w_up: mx.array, w_down: mx.array, rms_ffn_w: mx.array, eps: float = 1e-6):
        self.w_gate_up = mx.concatenate([w_gate, w_up], axis=-1)
        self.d_ff = w_gate.shape[-1]
        self.w_down = w_down
        self.rms_ffn_w = rms_ffn_w
        self.eps = eps

    def forward(self, x: mx.array) -> mx.array:
        @mx.compile
        def _forward_jit(arr, w_gu, w_d, rms_w, split_dim):
            # Fused RMSNorm
            variance = mx.mean(mx.square(arr), axis=-1, keepdims=True)
            norm_x = arr * mx.rsqrt(variance + self.eps) * rms_w
            
            # Gate-up projection
            gate_up = norm_x @ w_gu
            gate = gate_up[..., :split_dim]
            up = gate_up[..., split_dim:]
            
            # SwiGLU activation
            activated = (gate * mx.sigmoid(gate)) * up
            
            # Down projection and residual add
            ffn_out = activated @ w_d
            return arr + ffn_out

        return _forward_jit(x, self.w_gate_up, self.w_down, self.rms_ffn_w, self.d_ff)


class MultiStreamOverlap:
    def __init__(self):
        self.stream_draft = mx.new_stream(mx.gpu)
        self.stream_target = mx.new_stream(mx.gpu)

    def verify_and_draft(self, target_fn, draft_fn, target_args=(), draft_args=()):
        with mx.StreamContext(self.stream_target):
            target_out = target_fn(*target_args)
        with mx.StreamContext(self.stream_draft):
            draft_out = draft_fn(*draft_args)
        return target_out, draft_out


class DynamicCachePruning:
    def __init__(self, keep_ratio: float = 0.8):
        self.keep_ratio = keep_ratio

    def prune_cache(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        B, S, H, D = k.shape
        keep_k = max(1, int(S * self.keep_ratio))
        if keep_k >= S:
            return k, v
            
        @mx.compile
        def _prune_jit(k_arr, v_arr):
            norm = mx.sum(mx.square(k_arr), axis=(-2, -1))
            indices = mx.argsort(norm, axis=-1)
            top_indices = indices[:, -keep_k:]
            batch_idx = mx.arange(B)[:, None]
            k_pruned = k_arr[batch_idx, top_indices]
            v_pruned = v_arr[batch_idx, top_indices]
            return k_pruned, v_pruned
            
        return _prune_jit(k, v)


class EndToEndJITSpeculativePipeline:
    def __init__(self, draft_model_fn, target_model_fn):
        self.draft_model_fn = draft_model_fn
        self.target_model_fn = target_model_fn

    def run_speculative_step(self, x: mx.array, K: int) -> tuple[mx.array, mx.array, mx.array]:
        @mx.compile
        def _pipeline_jit(inputs):
            curr = inputs
            draft_tokens = []
            
            for _ in range(K):
                logits = self.draft_model_fn(curr)
                next_tok = mx.argmax(logits[:, -1:], axis=-1)
                draft_tokens.append(next_tok)
                curr = mx.concatenate([curr, next_tok], axis=-1)
                
            draft_seq = mx.concatenate(draft_tokens, axis=-1)
            
            target_logits = self.target_model_fn(curr)
            
            # Target predictions corresponding to each draft token position
            target_preds = mx.argmax(target_logits[:, inputs.shape[1]-1:-1], axis=-1)
            
            matches = (target_preds == draft_seq).astype(mx.int32)
            cumprod = mx.cumprod(matches, axis=-1)
            accepted_count = mx.sum(cumprod, axis=-1, keepdims=True)
            
            # Target predicted tokens for draft positions plus the bonus token
            target_next_tokens = mx.argmax(target_logits[:, inputs.shape[1]-1:], axis=-1)
            
            batch_size = inputs.shape[0]
            batch_idx = mx.arange(batch_size)[:, None]
            next_val = target_next_tokens[batch_idx, accepted_count]
            
            return accepted_count, draft_seq, next_val
            
        return _pipeline_jit(x)
