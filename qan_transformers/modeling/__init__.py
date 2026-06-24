import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qan_transformers.modeling.attention import DenseAttention, QuasicrystallineAttention
from qan_transformers.modeling.gemma import GemmaAttention
from qan_transformers.modeling.gpt_oss import GPTOSSAttention
from qan_transformers.modeling.qwen import QwenAttention
from qan_transformers.modeling.auto import AutoQANGraftModel
def make_superposition_mlp_forward(original_forward, module):
    def forward(x_or_self, x=None, *args, **kwargs):
        if x is None:
            real_x = x_or_self
            real_self = module
        else:
            real_x = x
            real_self = x_or_self
            
        if real_x.dim() == 3:
            return original_forward(real_x, *args, **kwargs)
            
        B, C, S, D = real_x.shape
        x0 = real_x[:, 0, :, :]
        dx = real_x - x0.unsqueeze(1)
        g0 = real_self.gate_proj(x0)
        u0 = real_self.up_proj(x0)
        sig0 = torch.sigmoid(g0)
        dsilu_g0 = sig0 * (1.0 + g0 * (1.0 - sig0))
        silu_g0 = g0 * sig0
        
        dg = real_self.gate_proj(dx.view(B * C, S, D)).view(B, C, S, -1)
        if getattr(real_self.gate_proj, "bias", None) is not None:
            dg = dg - real_self.gate_proj.bias.view(1, 1, 1, -1)
            
        du = real_self.up_proj(dx.view(B * C, S, D)).view(B, C, S, -1)
        if getattr(real_self.up_proj, "bias", None) is not None:
            du = du - real_self.up_proj.bias.view(1, 1, 1, -1)
            
        dh = (dsilu_g0 * u0).unsqueeze(1) * dg + (silu_g0).unsqueeze(1) * du
        out_ref = real_self.down_proj(silu_g0 * u0)
        
        out_dev = real_self.down_proj(dh.view(B * C, S, -1)).view(B, C, S, D)
        if getattr(real_self.down_proj, "bias", None) is not None:
            out_dev = out_dev - real_self.down_proj.bias.view(1, 1, 1, -1)
            
        return out_ref.unsqueeze(1) + out_dev
    return forward

def make_superposition_sequential_mlp_forward(original_forward, module):
    def forward(x_or_self, x=None, *args, **kwargs):
        if x is None:
            real_x = x_or_self
            real_self = module
        else:
            real_x = x
            real_self = x_or_self
            
        if real_x.dim() == 3:
            return original_forward(real_x, *args, **kwargs)
            
        B, C, S, D = real_x.shape
        x_flat = real_x.view(B * C, S, D)
        out_flat = original_forward(x_flat, *args, **kwargs)
        D_out = out_flat.shape[-1]
        return out_flat.view(B, C, S, D_out)
    return forward


MODEL_CONFIGS = {
    "google/gemma-4-e2b": {"vocab_size": 256000, "embed_dim": 2048, "num_heads": 8, "num_layers": 18, "sparse_ratio": 0.15},
    "google/gemma-4-e4b": {"vocab_size": 256000, "embed_dim": 3072, "num_heads": 12, "num_layers": 26, "sparse_ratio": 0.15},
    "openai/gpt-oss-20b": {"vocab_size": 50257, "embed_dim": 5120, "num_heads": 40, "num_layers": 44, "sparse_ratio": 0.15},
    "Qwen/Qwen3.6-27B": {"vocab_size": 151936, "embed_dim": 5120, "num_heads": 40, "num_layers": 64, "sparse_ratio": 0.15},
    "Qwen/Qwen3.6-35B-A3B": {"vocab_size": 151936, "embed_dim": 6144, "num_heads": 48, "num_layers": 80, "sparse_ratio": 0.15},
    "meta-llama/Llama-3-8B-Instruct": {"vocab_size": 128256, "embed_dim": 4096, "num_heads": 32, "num_layers": 32, "sparse_ratio": 0.15},
}

def extract_linear_properties(module):
    """
    Robustly extracts projection features and parameters from any projection module,
    including standard nn.Linear and custom wrappers (e.g. Gemma4ClippableLinear).
    """
    if module is None:
        return None, None, None, None
        
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    
    # 1. Check if the module wraps an underlying linear layer or exposes it as an attribute
    for attr_name in ["linear", "proj", "base_layer", "wrapped", "module", "_proj"]:
        if hasattr(module, attr_name):
            wrapped = getattr(module, attr_name)
            if wrapped is not None:
                if out_features is None:
                    out_features = getattr(wrapped, "out_features", None)
                if in_features is None:
                    in_features = getattr(wrapped, "in_features", None)
                if weight is None:
                    weight = getattr(wrapped, "weight", None)
                if bias is None:
                    bias = getattr(wrapped, "bias", None)
                    
    # 2. Scan children recursively to find the first linear-like layer or nn.Linear
    if out_features is None or in_features is None or weight is None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) or hasattr(child, "weight"):
                if out_features is None:
                    out_features = getattr(child, "out_features", None)
                if in_features is None:
                    in_features = getattr(child, "in_features", None)
                if weight is None:
                    weight = getattr(child, "weight", None)
                if bias is None:
                    bias = getattr(child, "bias", None)
                break
                
    # 3. If we have weight but features are still None, infer from weight shape
    if weight is not None and hasattr(weight, "shape"):
        if out_features is None:
            out_features = weight.shape[0]
        if in_features is None:
            in_features = weight.shape[1]
            
    # 4. Try parent parameter scan if weight is still None
    if weight is None:
        for p_name, p in module.named_parameters():
            if "weight" in p_name:
                weight = p
                if out_features is None:
                    out_features = p.shape[0]
                if in_features is None:
                    in_features = p.shape[1]
                break
        for p_name, p in module.named_parameters():
            if "bias" in p_name:
                bias = p
                break
                
    return out_features, in_features, weight, bias

def patch_projection_module(module):
    """
    Extracts features robustly from a projection layer and dynamically attaches
    them to the module to prevent downstream AttributeErrors.
    """
    if module is None:
        return
    out_features, in_features, weight, bias = extract_linear_properties(module)
    if out_features is not None and not hasattr(module, "out_features"):
        module.out_features = out_features
    if in_features is not None and not hasattr(module, "in_features"):
        module.in_features = in_features
    if weight is not None and not hasattr(module, "weight"):
        module.weight = weight
    if not hasattr(module, "bias"):
        module.bias = bias

def make_quasicrystalline(model: nn.Module) -> nn.Module:
    """
    Recursively traverses a PyTorch model and replaces standard dense self-attention layers
    with QuasicrystallineAttention instances.
    """
    for name, child in model.named_children():
        if any(k in name.lower() for k in ["audio", "vision", "image", "video", "multimodal"]):
            continue
        if type(child).__name__ in ("DenseAttention", "GemmaAttention", "GPTOSSAttention", "QwenAttention") or "Attention" in type(child).__name__:
            # Patch child projection modules robustly
            patch_projection_module(getattr(child, "q_proj", None))
            patch_projection_module(getattr(child, "k_proj", None))
            patch_projection_module(getattr(child, "v_proj", None))
            patch_projection_module(getattr(child, "out_proj", None))
            patch_projection_module(getattr(child, "o_proj", None))
            patch_projection_module(getattr(child, "c_proj", None))
            
            embed_dim = getattr(child, "embed_dim", None)
            if embed_dim is None:
                embed_dim = getattr(child, "hidden_size", None)
            if embed_dim is None:
                embed_dim = getattr(child, "config", None) and getattr(child.config, "hidden_size", None)
            if embed_dim is None:
                embed_dim = getattr(child, "config", None) and getattr(child.config, "embed_dim", None)
            if embed_dim is None:
                embed_dim = getattr(model, "config", None) and getattr(model.config, "hidden_size", None)
            if embed_dim is None:
                embed_dim = getattr(model, "config", None) and getattr(model.config, "embed_dim", None)
                
            num_heads = getattr(child, "num_heads", None)
            if num_heads is None:
                num_heads = getattr(child, "num_attention_heads", None)
            if num_heads is None:
                num_heads = getattr(child, "config", None) and getattr(child.config, "num_attention_heads", None)
            if num_heads is None:
                num_heads = getattr(child, "config", None) and getattr(child.config, "num_heads", None)
            if num_heads is None:
                num_heads = getattr(model, "config", None) and getattr(model.config, "num_attention_heads", None)
            if num_heads is None:
                num_heads = getattr(model, "config", None) and getattr(model.config, "num_heads", None)
                
            if embed_dim is None or num_heads is None or getattr(child, "k_proj", None) is None:
                import warnings
                warnings.warn(f"Skipping grafting for child module {name} (type: {type(child).__name__}) "
                              f"because embed_dim ({embed_dim}), num_heads ({num_heads}), or k_proj is None.")
                make_quasicrystalline(child)
                continue
                
            head_dim = embed_dim // num_heads
            num_key_value_heads = getattr(child, "num_key_value_heads", None)
            if num_key_value_heads is None:
                num_key_value_heads = getattr(child, "config", None) and getattr(child.config, "num_key_value_heads", None)
            if num_key_value_heads is None:
                num_key_value_heads = getattr(model, "config", None) and getattr(model.config, "num_key_value_heads", None)
            if num_key_value_heads is None:
                if getattr(child, "k_proj", None) is not None:
                    patch_projection_module(child.k_proj)
                    num_key_value_heads = child.k_proj.out_features // head_dim
                else:
                    num_key_value_heads = num_heads

            sparse_ratio = getattr(child, "sparse_ratio", 0.15)
            
            sparse_attn = QuasicrystallineAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sparse_ratio=sparse_ratio,
                num_key_value_heads=num_key_value_heads
            )
            
            device = child.q_proj.weight.device
            dtype = child.q_proj.weight.dtype
            sparse_attn = sparse_attn.to(device=device, dtype=dtype)
            sparse_attn.scaling = getattr(child, "scaling", getattr(child, "scale", getattr(child, "attn_scale", 1.0 / np.sqrt(head_dim))))
            
            # Copy Query-Key-Value Normalization submodules if present (e.g. Gemma4)
            if hasattr(child, "q_norm"):
                sparse_attn.q_norm = child.q_norm
            if hasattr(child, "k_norm"):
                sparse_attn.k_norm = child.k_norm
            if hasattr(child, "v_norm"):
                sparse_attn.v_norm = child.v_norm
            
            with torch.no_grad():
                sparse_attn.q_proj.weight.copy_(child.q_proj.weight)
                if child.q_proj.bias is None:
                    sparse_attn.q_proj.bias = None
                elif sparse_attn.q_proj.bias is not None and sparse_attn.q_proj.bias is not None:
                    sparse_attn.q_proj.bias.copy_(child.q_proj.bias)

                sparse_attn.k_proj.weight.copy_(child.k_proj.weight)
                if child.k_proj.bias is None:
                    sparse_attn.k_proj.bias = None
                elif child.k_proj.bias is not None and sparse_attn.k_proj.bias is not None:
                    sparse_attn.k_proj.bias.copy_(child.k_proj.bias)

                sparse_attn.v_proj.weight.copy_(child.v_proj.weight)
                if child.v_proj.bias is None:
                    sparse_attn.v_proj.bias = None
                elif child.v_proj.bias is not None and sparse_attn.v_proj.bias is not None:
                    sparse_attn.v_proj.bias.copy_(child.v_proj.bias)
                    
                if hasattr(child, "out_proj"):
                    sparse_attn.out_proj.weight.copy_(child.out_proj.weight)
                    if child.out_proj.bias is None:
                        sparse_attn.out_proj.bias = None
                    elif child.out_proj.bias is not None and sparse_attn.out_proj.bias is not None:
                        sparse_attn.out_proj.bias.copy_(child.out_proj.bias)
                elif hasattr(child, "o_proj"):
                    sparse_attn.o_proj = sparse_attn.out_proj
                    sparse_attn.o_proj.weight.copy_(child.o_proj.weight)
                    if child.o_proj.bias is None:
                        sparse_attn.o_proj.bias = None
                    elif child.o_proj.bias is not None and sparse_attn.o_proj.bias is not None:
                        sparse_attn.o_proj.bias.copy_(child.o_proj.bias)
                elif hasattr(child, "c_proj"):
                    sparse_attn.c_proj = sparse_attn.out_proj
                    sparse_attn.c_proj.weight.copy_(child.c_proj.weight)
                    if child.c_proj.bias is None:
                        sparse_attn.c_proj.bias = None
                    elif child.c_proj.bias is not None and sparse_attn.c_proj.bias is not None:
                        sparse_attn.c_proj.bias.copy_(child.c_proj.bias)
                
                # Perform SVD-based initialization for the e8_proj linear layer during grafting
                _, _, q_weight, _ = extract_linear_properties(child.q_proj)
                _, _, k_weight, _ = extract_linear_properties(child.k_proj)
                if q_weight is not None and k_weight is not None:
                    W_q = q_weight.detach().cpu().to(torch.float32)
                    W_k = k_weight.detach().cpu().to(torch.float32)
                    W_stacked = torch.cat([W_q, W_k], dim=0)
                    U, S, Vh = torch.linalg.svd(W_stacked, full_matrices=False)
                    Vt = Vh
                    
                    _, _, e8_weight, e8_bias = extract_linear_properties(sparse_attn.e8_proj)
                    if e8_weight is not None:
                        e8_weight.zero_()
                        num_vectors = Vt.shape[0]
                        copy_rows = min(8, num_vectors)
                        e8_weight[:copy_rows, :].copy_(Vt[:copy_rows, :].to(device=e8_weight.device, dtype=e8_weight.dtype))
                    if e8_bias is not None:
                        e8_bias.zero_()
                        
            setattr(model, name, sparse_attn)
        else:
            make_quasicrystalline(child)
            
    def set_review_mode(self, mode: bool):
        for m in self.modules():
            if m.__class__.__name__ == "QuasicrystallineAttention":
                m.review_mode = mode
    import types
    model.set_review_mode = types.MethodType(set_review_mode, model)

    # Patch FFN modules for SwiGLU superposition
    for name, m in model.named_modules():
        if hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj"):
            if not hasattr(m, "_is_superposition_patched"):
                original_forward = m.forward
                m.forward = types.MethodType(make_superposition_mlp_forward(original_forward, m), m)
                m._is_superposition_patched = True
                
        if hasattr(m, "mlp") and isinstance(m.mlp, nn.Sequential):
            if not hasattr(m.mlp, "_is_superposition_patched"):
                original_forward = m.mlp.forward
                m.mlp.forward = types.MethodType(make_superposition_sequential_mlp_forward(original_forward, m.mlp), m.mlp)
                m.mlp._is_superposition_patched = True
    
    return model

class QANBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, sparse_ratio=0.15, use_dense=False, attention_class=None):
        super().__init__()
        if attention_class is not None:
            self.attn = attention_class(embed_dim, num_heads, sparse_ratio=sparse_ratio)
        elif use_dense:
            self.attn = DenseAttention(embed_dim, num_heads, sparse_ratio=sparse_ratio)
        else:
            self.attn = QuasicrystallineAttention(embed_dim, num_heads, sparse_ratio=sparse_ratio)
            
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x, kv_cache=None, attn_mask=None):
        if kv_cache is not None:
            attn_out, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, attn_mask=attn_mask)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x, kv_cache
        else:
            attn_out = self.attn(self.ln1(x), attn_mask=attn_mask)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x

class QANModel(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, num_heads=4, num_layers=2, sparse_ratio=0.15, max_seq_len=2048, use_dense=False, attention_class=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sparse_ratio = sparse_ratio
        self.max_seq_len = max_seq_len
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        
        self.layers = nn.ModuleList([
            QANBlock(embed_dim, num_heads, sparse_ratio=sparse_ratio, use_dense=use_dense, attention_class=attention_class)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, input_ids, kv_caches=None, attn_mask=None):
        if input_ids.dim() == 3:
            B, C, S = input_ids.shape
        else:
            B, S = input_ids.shape
        device = input_ids.device
        
        if S == 0:
            raise ValueError("Sequence length S must be greater than 0")
            
        # Position embedding (wrap around if S exceeds max_seq_len to prevent index out of bounds)
        positions = torch.arange(0, S, device=device).unsqueeze(0).expand(B, -1)
        positions = positions % self.max_seq_len
        pos_emb = self.pos_embed(positions)
        if input_ids.dim() == 3:
            pos_emb = pos_emb.unsqueeze(1)
        x = self.embed(input_ids) + pos_emb
        
        new_kv_caches = [] if kv_caches is not None else None
        
        for i, layer in enumerate(self.layers):
            if kv_caches is not None:
                cache = kv_caches[i] if len(kv_caches) > i else {}
                x, new_cache = layer(x, kv_cache=cache, attn_mask=attn_mask)
                new_kv_caches.append(new_cache)
            else:
                x = layer(x, attn_mask=attn_mask)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if kv_caches is not None:
            return logits, new_kv_caches
            
        return logits

    def generate(self, input_ids, max_new_tokens=20, temperature=1.0, top_k=50, firewall=None, rollback_limit=3):
        """
        Generates tokens auto-regressively with rollback support when firewall triggers.
        """
        device = input_ids.device
        B, S = input_ids.shape
        
        # Helper to clone kv caches
        def clone_kv_caches(caches):
            if caches is None:
                return None
            cloned = []
            for cache in caches:
                cloned_cache = {}
                for k, v in cache.items():
                    if isinstance(v, torch.Tensor):
                        cloned_cache[k] = v.clone()
                    else:
                        cloned_cache[k] = v
                cloned.append(cloned_cache)
            return cloned

        # Set the firewall on the layers' attention blocks if a custom one is passed
        if firewall is not None:
            for layer in self.layers:
                if hasattr(layer, "attn") and layer.attn is not None:
                    layer.attn.firewall = firewall

        # Initialize kv_caches
        kv_caches = [{} for _ in range(len(self.layers))]
        
        # Prefill / initial forward pass
        logits, kv_caches = self.forward(input_ids, kv_caches=kv_caches)
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Back up kv_caches and generated_ids in case of rollback
            backup_kv_caches = clone_kv_caches(kv_caches)
            
            # Predict next token logits
            next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1) # Shape [B, 1]
            
            # Run one forward step (only with the newly generated token)
            step_logits, new_kv_caches = self.forward(next_token, kv_caches=clone_kv_caches(backup_kv_caches))
            
            # Initialize tried tokens tracker
            tried_tokens = {b: [int(next_token[b, 0].item())] for b in range(B)}
            
            rollback_count = 0
            while rollback_count < rollback_limit:
                # 1. Gather fracture status and alt indices for each batch item
                batch_fractured = [False] * B
                alt_idx_batch = [[] for _ in range(B)]
                
                for cache in new_kv_caches:
                    is_frac = cache.get("is_fractured", False)
                    alts = cache.get("alt_idx", [])
                    
                    if B == 1:
                        is_frac_list = [is_frac]
                        alts_list = [alts]
                    else:
                        if isinstance(is_frac, bool):
                            is_frac_list = [is_frac] * B
                            alts_list = [alts] * B
                        else:
                            is_frac_list = is_frac
                            alts_list = alts
                            
                    for b in range(B):
                        if b < len(is_frac_list) and is_frac_list[b]:
                            batch_fractured[b] = True
                            if b < len(alts_list) and alts_list[b]:
                                alt_idx_batch[b].extend(alts_list[b])
                
                # Check if any batch item is fractured. If none, we are done!
                if not any(batch_fractured):
                    break
                    
                rollback_count += 1
                
                # 2. For each fractured batch item, find alternative or sample new
                for b in range(B):
                    if batch_fractured[b]:
                        alt_token = None
                        while len(alt_idx_batch[b]) > 0:
                            candidate = alt_idx_batch[b].pop(0)
                            if candidate not in tried_tokens[b]:
                                alt_token = candidate
                                break
                        
                        if alt_token is not None:
                            next_token[b, 0] = alt_token
                            tried_tokens[b].append(alt_token)
                        else:
                            # mask the probability of the current token for that batch item, re-normalize, and sample a new token
                            probs_b = probs[b].clone()
                            for t in tried_tokens[b]:
                                if t < len(probs_b):
                                    probs_b[t] = 0.0
                            probs_b_sum = probs_b.sum()
                            if probs_b_sum > 0:
                                probs_b /= probs_b_sum
                            else:
                                probs_b = torch.ones_like(probs_b) / probs_b.shape[-1]
                            new_sampled = torch.multinomial(probs_b, num_samples=1)
                            next_token[b, 0] = new_sampled.item()
                            tried_tokens[b].append(int(next_token[b, 0].item()))
                
                # Re-run forward step with the updated next_token and clean backup_kv_caches
                step_logits, new_kv_caches = self.forward(next_token, kv_caches=clone_kv_caches(backup_kv_caches))
            
            # Accept the token (either because it's not fractured, or we hit rollback limit)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            kv_caches = new_kv_caches
            logits = step_logits
            
        return generated_ids


def get_lightweight_config(model_name):
    """
    Returns a scaled-down lightweight version of standard configurations to avoid
    massive resource requirements during tests/local training.
    """
    base_config = MODEL_CONFIGS.get(model_name, {
        "vocab_size": 32000, "embed_dim": 128, "num_heads": 4, "num_layers": 2, "sparse_ratio": 0.15
    })
    
    # Scale down embedding, heads, layers for extremely lightweight testing
    config = {
        "vocab_size": min(base_config["vocab_size"], 1000), # Cap vocab size at 1000 for tests
        "embed_dim": 64,
        "num_heads": 2,
        "num_layers": 2,
        "sparse_ratio": base_config["sparse_ratio"],
        "max_seq_len": 2048,
        "model_name": model_name
    }
    return config

def graft_model(model_name, lightweight=True, vocab_size=None):
    """
    Creates a new model instance preloaded with grafted E8 Quasicrystalline attention.
    Grafting is performed genuinely by instantiating a dense model and applying make_quasicrystalline.
    """
    if lightweight:
        config = get_lightweight_config(model_name)
    else:
        config = MODEL_CONFIGS.get(model_name, {
            "vocab_size": 32000, "embed_dim": 128, "num_heads": 4, "num_layers": 2, "sparse_ratio": 0.15
        })
        
    if vocab_size is not None:
        config["vocab_size"] = vocab_size
        
    # Select attention class based on model family
    attn_class = DenseAttention
    if "gemma" in model_name.lower():
        attn_class = GemmaAttention
    elif "gpt-oss" in model_name.lower():
        attn_class = GPTOSSAttention
    elif "qwen" in model_name.lower() or "llama" in model_name.lower():
        attn_class = QwenAttention
        
    # 1. Instantiate pure standard dense model
    dense_model = QANModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        sparse_ratio=config["sparse_ratio"],
        max_seq_len=config.get("max_seq_len", 2048),
        attention_class=attn_class
    )
    
    # 2. Genuinely graft the Quasicrystalline attention
    grafted_model = make_quasicrystalline(dense_model)
    return grafted_model
