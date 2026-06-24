import os
from typing import Any, Dict, Optional, Union

class AutoQANGraftModel:
    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        sparse_ratio: float = 0.15,
        framework: str = "pt",
        **kwargs
    ) -> Any:
        """
        Dynamically loads a standard transformer model and grafts QAN
        coordinate-sparse attention onto all self-attention layers in-place.
        
        Args:
            model_id_or_path: Hugging Face model repository ID or local path.
            sparse_ratio: Float attention sparsity ratio (default: 0.15).
            framework: "pt" (PyTorch) or "mlx" (Apple MLX).
            **kwargs: Forwarded to AutoModelForCausalLM.from_pretrained or mlx_lm.load.
        """
        if framework == "mlx":
            try:
                from mlx_lm import load
                from qan_transformers.mlx.modeling import graft_mlx_model
            except ImportError:
                raise ImportError(
                    "MLX framework or mlx_lm library not installed. "
                    "Please install mlx and mlx-lm to run on Apple Silicon."
                )
            
            # mlx_lm.load returns (model, tokenizer)
            model, tokenizer = load(model_id_or_path, **kwargs)
            grafted_model = graft_mlx_model(model, sparse_ratio=sparse_ratio)
            # Attach tokenizer to the model for convenience
            setattr(grafted_model, "tokenizer", tokenizer)
            return grafted_model
            
        elif framework == "pt":
            # Extract lightweight flag for tests/local mock runs
            lightweight = kwargs.pop("lightweight", False)
            if lightweight:
                from qan_transformers.modeling import graft_model
                return graft_model(model_id_or_path, lightweight=True)
                
            from transformers import AutoModelForCausalLM
            from qan_transformers.modeling import make_quasicrystalline
            
            model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
            grafted_model = make_quasicrystalline(model)
            return grafted_model
            
        else:
            raise ValueError(f"Unsupported framework: {framework}. Must be 'pt' or 'mlx'.")


def wrap_rotary_embeddings(model):
    """
    Wraps the forward method of each layer's rotary embedding to enforce modulo-wrapping
    of position IDs, preventing out-of-bounds errors on large codebases.
    """
    from qan_transformers.math.context_builder import wrap_position_ids
    print("[RoPE Wrapping] Intercepting rotary embedding forward passes for position-wrapping...")
    wrap_count = 0
    for m in model.modules():
        if hasattr(m, "rotary_emb") and m.rotary_emb is not None:
            original_forward = m.rotary_emb.forward
            
            def make_wrapped_forward(orig, module_ref):
                def wrapped_forward(x, position_ids, **kwargs):
                    max_pos = getattr(module_ref, "max_seq_len_cached", None)
                    if max_pos is None:
                        max_pos = getattr(model.config, "max_position_embeddings", 4096)
                    wrapped_pos = wrap_position_ids(position_ids, max_pos)
                    return orig(x, wrapped_pos, **kwargs)
                return wrapped_forward
                
            m.rotary_emb.forward = make_wrapped_forward(original_forward, m.rotary_emb)
            wrap_count += 1
            
    print(f"[RoPE Wrapping] Wrapped {wrap_count} rotary embedding modules successfully.")

