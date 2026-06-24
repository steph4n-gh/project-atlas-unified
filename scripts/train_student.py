#!/usr/bin/env python
"""Local student distillation training script for the E8 Regex Wizard.

Loads the initialized student model configuration and weights,
loads the serialized teacher distillation cache, and runs the training loop
using E8 geometric routing losses (KL divergence + hidden alignment MSE + hierarchical CE).
"""

import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Insert src directory
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.model import UCEModel
from ultrametric_ce.distillation import (
    build_toy_arithmetic_tree,
    load_dataset_cache,
    run_distillation_phase,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train student model on pre-computed distillation cache.")
    parser.add_argument("--student-dir", type=str, default="/Volumes/Storage/project_atlas/scratch/student_85m", help="Path to initialized student checkpoint directory.")
    parser.add_argument("--cache", type=str, default="/Volumes/Storage/project_atlas/scratch/teacher_distill_cache.pkl", help="Path to serialized teacher distillation cache.")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--out-dir", type=str, default="/Volumes/Storage/project_atlas/scratch/student_trained", help="Directory to save trained weights.")
    args = parser.parse_args(argv)

    print("=== Regex Wizard Distillation Trainer ===")
    print(f"Student Directory: {args.student_dir}")
    print(f"Distillation Cache: {args.cache}")
    print(f"Steps: {args.steps} | LR: {args.lr}")

    # 1. Load the Tree and Student
    # For local validation, load the toy tree structure
    print("Loading tree structure...")
    tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()

    # Instantiate UCEModel
    print("Initializing student model...")
    model = UCEModel(tree, dim=16)

    # 2. Load the serialized distillation cache
    if not os.path.exists(args.cache):
        print(f"Error: Cache file {args.cache} does not exist. Run scripts/generate_teacher_cache.py first.")
        return 1
        
    print(f"Loading distillation cache: {args.cache}...")
    batches = load_dataset_cache(args.cache)
    print(f"Loaded {len(batches)} batches from cache.")

    # 3. Run distillation loop (Phase 1: freeze routing heads, train diffusion)
    print("Running Phase 1 Distillation (attuning student attention/diffusion)...")
    model, log = run_distillation_phase(
        model=model,
        teacher=None, # Teacher logits/hidden states are pre-loaded from batch cache
        tree=tree,
        sym_to_token=sym_to_token,
        phase=1,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        precomputed_batches=batches
    )

    # Save trained student weights
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    weights_file = out_path / "model.safetensors"
    
    # Save the parameters using MLX save_weights
    from mlx.utils import tree_flatten
    flat_params = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(weights_file), flat_params)

    print(f"Successfully saved distilled student weights to: {weights_file}")
    
    # Save a config.json helper
    import shutil
    src_cfg = Path(args.student_dir) / "config.json"
    if src_cfg.exists():
        shutil.copy2(src_cfg, out_path / "config.json")
        
    print("Student training session completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
