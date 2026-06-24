#!/usr/bin/env python
"""Initialize student configuration and model weights for the 85M E8 Regex Wizard.

Creates a standard transformer config.json and saves randomly initialized
weights in safetensors format to act as the base student model before distillation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

# Insert src directory
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialize 85M student configuration and weight structures.")
    parser.add_argument("--out-dir", type=str, default="/Volumes/Storage/project_atlas/scratch/student_85m", help="Output directory for student configuration and weights.")
    args = parser.parse_args(argv)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Define the 85M configuration (Gemma-like topology)
    config = {
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "vocab_size": 256000,
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 2048,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "initializer_range": 0.02,
        "attention_bias": False,
        "torch_dtype": "float16",
        "transformers_version": "4.42.0"
    }

    # Save config.json
    config_file = out_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved student configuration to {config_file}")

    # 2. Build randomly initialized weights matching this architecture (Float16)
    # Using Gemma-2 standard tensor keys
    print("Generating randomly initialized student weights (float16)...")
    weights = {}

    # Embeddings
    weights["model.embed_tokens.weight"] = mx.random.normal((config["vocab_size"], config["hidden_size"]), dtype=mx.float16) * config["initializer_range"]

    # Layer weights
    h = config["hidden_size"]
    inter = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    for i in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{i}."
        
        # Attention projections
        weights[prefix + "self_attn.q_proj.weight"] = mx.random.normal((num_heads * head_dim, h), dtype=mx.float16) * config["initializer_range"]
        weights[prefix + "self_attn.k_proj.weight"] = mx.random.normal((num_kv_heads * head_dim, h), dtype=mx.float16) * config["initializer_range"]
        weights[prefix + "self_attn.v_proj.weight"] = mx.random.normal((num_kv_heads * head_dim, h), dtype=mx.float16) * config["initializer_range"]
        weights[prefix + "self_attn.o_proj.weight"] = mx.random.normal((h, num_heads * head_dim), dtype=mx.float16) * config["initializer_range"]
        
        # MLP projections
        weights[prefix + "mlp.gate_proj.weight"] = mx.random.normal((inter, h), dtype=mx.float16) * config["initializer_range"]
        weights[prefix + "mlp.up_proj.weight"] = mx.random.normal((inter, h), dtype=mx.float16) * config["initializer_range"]
        weights[prefix + "mlp.down_proj.weight"] = mx.random.normal((h, inter), dtype=mx.float16) * config["initializer_range"]
        
        # Norm weights
        weights[prefix + "input_layernorm.weight"] = mx.ones((h,), dtype=mx.float16)
        weights[prefix + "post_attention_layernorm.weight"] = mx.ones((h,), dtype=mx.float16)

    # Output projection & Final Norm
    weights["model.norm.weight"] = mx.ones((h,), dtype=mx.float16)
    weights["lm_head.weight"] = weights["model.embed_tokens.weight"] # Tied weights

    # Save to safetensors
    weight_file = out_path / "model.safetensors"
    mx.save_safetensors(str(weight_file), weights)
    print(f"Saved initialized student weights to {weight_file}")
    print("Student initialization completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
