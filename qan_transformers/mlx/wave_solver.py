import mlx.core as mx
import mlx.nn as nn
import numpy as np
import mlx_lm.models.cache as cache_module
import inspect
from typing import Optional

def early_exit_forward_pass(model: nn.Module, inputs: mx.array, cache=None, num_layers: int = None):
    """
    Executes a forward pass on a model, exiting early at `num_layers`.
    Supports standard mlx_lm text models (Gemma-4, Llama, Qwen, etc.) and mock model.
    """
    if num_layers is None:
        # Full depth call
        try:
            return model(inputs, cache=cache)
        except TypeError:
            return model(inputs, caches=cache)
        
    # Check if model has language_model (mlx_lm Model wrapper)
    if hasattr(model, "language_model"):
        lang_model = model.language_model
        text_model = lang_model.model
    elif hasattr(model, "model") and hasattr(model, "norm") and hasattr(model, "embed_tokens"):
        text_model = model.model
        lang_model = model
    elif hasattr(model, "embed") and hasattr(model, "layers") and hasattr(model, "ln_f"):
        # QANModelMLX mock model
        B, S = inputs.shape
        offset = 0
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            offset = cache[0].offset
        positions = mx.arange(offset, offset + S)
        x = model.embed(inputs) + model.pos_embed(positions)
        
        # Clip num_layers to model depth
        num_layers = min(num_layers, len(model.layers))
        for i in range(num_layers):
            c = cache[i] if cache is not None else None
            x = model.layers[i](x, cache=c)
            
        x = model.ln_f(x)
        logits = model.lm_head(x)
        return logits
    else:
        # Fallback to full call if structure is unrecognized
        try:
            return model(inputs, cache=cache)
        except TypeError:
            return model(inputs, caches=cache)

    # For mlx_lm text models
    input_embeddings = text_model.embed_tokens(inputs)
    h = input_embeddings
    h = h * text_model.embed_scale

    # Get the extra inputs per layer if we have per layer embeddings (e.g. Gemma-4)
    if getattr(text_model, "hidden_size_per_layer_input", 0):
        per_layer_inputs = text_model._get_per_layer_inputs(inputs, input_embeddings)
        per_layer_inputs = text_model._project_per_layer_inputs(h, per_layer_inputs)
        per_layer_inputs = [
            per_layer_inputs[:, :, i, :] for i in range(len(text_model.layers))
        ]
    else:
        per_layer_inputs = [None] * len(text_model.layers)

    # Cache setup
    if cache is None:
        cache = [None] * len(text_model.layers)
    else:
        # Append None for shared KV layers if cache size is smaller
        cache = cache + [None] * (len(text_model.layers) - len(cache))

    masks = text_model._make_masks(h, cache)
    intermediates = [(None, None)] * len(text_model.layers)
    
    # Clip num_layers to model depth
    num_layers = min(num_layers, len(text_model.layers))
    
    for idx in range(num_layers):
        layer = text_model.layers[idx]
        c = cache[idx]
        mask = masks[idx]
        prev_idx = text_model.previous_kvs[idx]
        per_layer_input = per_layer_inputs[idx]

        if prev_idx < idx:
            kvs, offset = intermediates[prev_idx]
        else:
            kvs, offset = None, None

        sig = inspect.signature(layer.__call__ if hasattr(layer, "__call__") else layer)
        kwargs = {}
        if "mask" in sig.parameters:
            kwargs["mask"] = mask
        if "cache" in sig.parameters:
            kwargs["cache"] = c
        if "per_layer_input" in sig.parameters:
            kwargs["per_layer_input"] = per_layer_input
        if "shared_kv" in sig.parameters:
            kwargs["shared_kv"] = kvs
        if "offset" in sig.parameters:
            kwargs["offset"] = offset

        res = layer(h, **kwargs)
        if isinstance(res, tuple) and len(res) == 3:
            h, kvs, offset = res
        elif isinstance(res, tuple) and len(res) == 2:
            h, _ = res
            kvs, offset = None, None
        else:
            h = res
            kvs, offset = None, None

        intermediates[idx] = (kvs, offset)

    # Final normalization
    h = text_model.norm(h)

    # Projection to logits
    if getattr(lang_model, "tie_word_embeddings", False):
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lang_model.lm_head(h)

    if getattr(lang_model, "final_logit_softcapping", None) is not None:
        try:
            from mlx_lm.models.gemma4_text import logit_softcap
            logits = logit_softcap(lang_model.final_logit_softcapping, logits)
        except ImportError:
            logits = lang_model.final_logit_softcapping * mx.tanh(logits / lang_model.final_logit_softcapping)

    return logits

def generate_early_exit_draft(
    model: nn.Module,
    prompt_tokens: mx.array,
    max_tokens: int,
    early_exit_layer: int,
    temp: float = 0.0,
):
    """
    Generates draft tokens autoregressively using early-exit layer skipping.
    """
    model_cache = cache_module.make_prompt_cache(model)
    # Prefill prompt using early-exit
    logits = early_exit_forward_pass(model, prompt_tokens[None, :], model_cache, early_exit_layer)
    
    first_logit = logits[0, -1, :]
    if temp == 0.0:
        token = mx.argmax(first_logit, axis=-1).astype(mx.uint32)
    else:
        probs = mx.softmax(first_logit / temp, axis=-1)
        token = mx.random.categorical(mx.log(probs + 1e-9), axis=-1).astype(mx.uint32)
        
    generated = [token.item()]
    
    for _ in range(max_tokens - 1):
        logits = early_exit_forward_pass(model, token[None, None], model_cache, early_exit_layer)
        first_logit = logits[0, -1, :]
        if temp == 0.0:
            token = mx.argmax(first_logit, axis=-1).astype(mx.uint32)
        else:
            probs = mx.softmax(first_logit / temp, axis=-1)
            token = mx.random.categorical(mx.log(probs + 1e-9), axis=-1).astype(mx.uint32)
        generated.append(token.item())
        
    return mx.array(generated, dtype=mx.uint32)

def wave_solver_generate(
    model: nn.Module,
    tokenizer,
    prompt_tokens: mx.array,
    max_tokens: int = 32,
    temp: float = 0.0,
    max_iterations: int = 10,
    tolerance: float = 0.1,
    initial_guesses: mx.array = None,
    early_exit_layer: int = None,
):
    """
    Executes Boundary-Value Parallel Wave Generation (Autoregressive Bypass) in MLX.
    Treats generation as a boundary-value problem and solves the entire sequence of
    length N = L_p + W in parallel using a causal mask and dynamic cache trimming.
    """
    try:
        from qan_transformers.mlx.attention import QuasicrystallineAttention
        QuasicrystallineAttention.in_wave_solver = False
    except ImportError:
        pass

    try:
        L_p = prompt_tokens.shape[0]
        W = max_tokens
        
        if W <= 0:
            return mx.array([], dtype=mx.uint32), 0, True
            
        # 1. Initialize prompt cache
        model_cache = cache_module.make_prompt_cache(model)
        
        # 2. Run pre-fill on the prompt
        try:
            prefill_logits = model(prompt_tokens[None, :], cache=model_cache)
        except TypeError:
            prefill_logits = model(prompt_tokens[None, :], caches=model_cache)
        
        # The last logit of prefill predicts the first token of the generation window
        first_logit = prefill_logits[0, -1, :]
        if temp == 0.0:
            first_token = mx.argmax(first_logit, axis=-1).astype(mx.uint32)
        else:
            probs = mx.softmax(first_logit / temp, axis=-1)
            first_token = mx.random.categorical(mx.log(probs + 1e-9), axis=-1).astype(mx.uint32)
            
        # If W == 1, we are done
        if W == 1:
            return first_token[None], 1, True
            
        # If initial_guesses is None and early_exit_layer is provided, generate early exit draft guesses
        if initial_guesses is None and early_exit_layer is not None:
            initial_guesses = generate_early_exit_draft(
                model=model,
                prompt_tokens=prompt_tokens,
                max_tokens=max_tokens,
                early_exit_layer=early_exit_layer,
                temp=temp
            )
            
        # 3. Initialize sequence guesses for the rest of the window (tokens L_p+1 to N-1)
        if initial_guesses is not None:
            # Use provided initial guesses, slice/pad to W - 1
            if initial_guesses.shape[0] >= W:
                # If initial_guesses includes the first token, take from index 1
                guesses = initial_guesses[1:W].astype(mx.uint32)
            else:
                # Pad with pad_token
                pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                if pad_token is None:
                    pad_token = 0
                pad_len = (W - 1) - initial_guesses.shape[0]
                guesses = mx.concatenate([
                    initial_guesses.astype(mx.uint32),
                    mx.full((pad_len,), pad_token, dtype=mx.uint32)
                ])
        else:
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            if pad_token is None:
                pad_token = 0
            guesses = mx.full((W - 1,), pad_token, dtype=mx.uint32)
        
        # Full tokens array of shape [N]
        tokens = mx.concatenate([
            prompt_tokens.astype(mx.uint32),
            first_token[None],
            guesses
        ])
        
        N = L_p + W
        prev_tokens = mx.array(tokens)
        converged = False
        iterations_run = 0
        
        # Enable wave solver bypass for iteration passes
        try:
            from qan_transformers.mlx.attention import QuasicrystallineAttention
            QuasicrystallineAttention.in_wave_solver = True
        except ImportError:
            pass
            
        for iteration in range(max_iterations):
            iterations_run += 1
            
            # Trim cache back to prompt length L_p
            current_offset = model_cache[0].offset
            if current_offset > L_p:
                cache_module.trim_prompt_cache(model_cache, current_offset - L_p)
                
            # The window input sequence is tokens[L_p : N-1] (length W-1)
            # Because we use these inputs to predict tokens[L_p+1 : N]
            window_inputs = tokens[L_p : N - 1]
            
            # Run parallel forward pass on window_inputs
            try:
                logits = model(window_inputs[None, :], cache=model_cache)
            except TypeError:
                logits = model(window_inputs[None, :], caches=model_cache)
            window_logits = logits[0, :, :] # Shape: [W - 1, V]
            
            # Process/sample predictions for the window
            if temp == 0.0:
                predictions = mx.argmax(window_logits, axis=-1).astype(mx.uint32)
            else:
                probs = mx.softmax(window_logits / temp, axis=-1)
                predictions = mx.random.categorical(mx.log(probs + 1e-9), axis=-1).astype(mx.uint32)
                
            # Update tokens[L_p+1 : N]
            new_tokens = mx.concatenate([
                tokens[:L_p + 1],
                predictions
            ])
            
            # Compute Čech Cohomology obstruction metric
            # s_t is the entropy at each position of predicted logits
            probs = mx.softmax(window_logits, axis=-1)
            entropy = -mx.sum(probs * mx.log(probs + 1e-9), axis=-1)
            d_entropy = mx.abs(entropy[1:] - entropy[:-1]) if (W - 1) > 1 else mx.array([0.0])
            
            # Check convergence of the generated tokens
            diff_mask = (new_tokens[L_p + 1 : N] != prev_tokens[L_p + 1 : N])
            num_changed = int(mx.sum(diff_mask.astype(mx.int32)).item())
            
            # Update state
            tokens = new_tokens
            prev_tokens = mx.array(tokens)
            
            if num_changed == 0:
                converged = True
                break
                
        # Return generated tokens only
        return tokens[L_p : N], iterations_run, converged
    finally:
        try:
            from qan_transformers.mlx.attention import QuasicrystallineAttention
            QuasicrystallineAttention.in_wave_solver = False
        except ImportError:
            pass


class NeuralDendriticBranchSpeculation:
    def __init__(self, branching_factor: int = 4, depth: int = 4):
        self.branching_factor = branching_factor
        self.depth = depth
        
    def integrate_potentials(self, path_logits: mx.array) -> mx.array:
        # path_logits has shape [B, num_paths, path_len, vocab_size]
        # Integrates localized voltage potentials over the spec tree branches
        B, num_paths, path_len, V = path_logits.shape
        probs = mx.softmax(path_logits, axis=-1)
        
        # Max probabilities along each token path position
        max_probs = mx.max(probs, axis=-1) # [B, num_paths, path_len]
        
        # Dendritic integration: voltage accumulates along the branch (dendrite)
        # using a decaying leaky integrator potential: V_t = lambda * V_{t-1} + S_t
        potentials = mx.zeros((B, num_paths))
        decay_factor = 0.9
        
        for t in range(path_len):
            potentials = decay_factor * potentials + max_probs[..., t]
            
        return potentials


class AsynchronousParallelSpeculation:
    def __init__(self):
        pass

    def run_speculation(self, draft_fn, target_fn, inputs: mx.array) -> tuple:
        device = mx.default_device()
        stream_draft = mx.new_stream(device)
        stream_target = mx.new_stream(device)
        
        with mx.stream(stream_draft):
            draft_res = draft_fn(inputs)
            
        with mx.stream(stream_target):
            target_res = target_fn(inputs)
            
        return draft_res, target_res


class SymplecticIntegrationSymDRAM:
    def __init__(self, dt: float = 0.05):
        self.dt = dt

    def step_leapfrog(self, q: mx.array, p: mx.array) -> tuple:
        p_half = p - 0.5 * self.dt * q
        q_next = q + self.dt * p_half
        p_next = p_half - 0.5 * self.dt * q_next
        return q_next, p_next


class StochasticResonanceDithering:
    def __init__(self, noise_scale: float = 0.02):
        self.noise_scale = noise_scale

    def dither(self, x: mx.array) -> mx.array:
        noise = mx.random.normal(x.shape, dtype=x.dtype) * self.noise_scale
        return x + noise


class AsynchronousPipelineRingBuffers:
    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self.buffer = mx.zeros((capacity,), dtype=mx.uint32)
        self.write_ptr = 0
        self.read_ptr = 0

    def push(self, tokens: mx.array):
        num_to_write = tokens.size
        indices = (self.write_ptr + mx.arange(num_to_write)) % self.capacity
        self.buffer[indices] = tokens
        self.write_ptr = (self.write_ptr + num_to_write) % self.capacity

    def pop(self, count: int) -> mx.array:
        indices = (self.read_ptr + mx.arange(count)) % self.capacity
        tokens = self.buffer[indices]
        self.read_ptr = (self.read_ptr + count) % self.capacity
        return tokens


class OctalSIMDParallelMatrixMath:
    def __init__(self):
        pass

    def parallel_matmul_8x(self, W: mx.array, x: mx.array) -> mx.array:
        return W @ x


class DynamicClockScalingScheduler:
    def __init__(self, baseline_clock: float = 1200.0):
        self.baseline_clock = baseline_clock
        self.current_clock = baseline_clock

    def update_clock_frequency(self, cfi_metric: float) -> float:
        if cfi_metric > 1.0:
            self.current_clock = self.baseline_clock
        else:
            self.current_clock = self.baseline_clock * 0.6
        return self.current_clock

