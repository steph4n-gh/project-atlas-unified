#!/usr/bin/env python
"""Orchestrator script to compile and train all 8 developer experts.

This script loops over the 8 experts, compiling their trees, precomputing their caches,
and distilling them offline using local Gemma-4 teacher models on the storage drive.
It supports skipping already completed checkpoints to allow resuming.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Workspace settings
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
TMP_MOE_DIR = ROOT_DIR / "tmp" / "moe"

# Ensure output directory exists
TMP_MOE_DIR.mkdir(parents=True, exist_ok=True)

# Experts configuration
EXPERTS = {
    "python_coder": {
        "model": "google/gemma-4-e4b-it",
        "seed_prompt": "def class self __init__ import from as return if elif else for in while try except raise with pass def __repr__ classmethod staticmethod yield lambda",
        "corpus_file": "tmp/corpora/corpus_python_coder.txt",
    },
    "web_stack": {
        "model": "google/gemma-4-e4b-it",
        "seed_prompt": "const let function import export React component state effect render props HTML CSS JS typescript div class",
        "corpus_file": "tmp/corpora/corpus_web_stack.txt",
    },
    "rust_systems": {
        "model": "google/gemma-4-e4b-it",
        "seed_prompt": "fn struct impl enum trait match lifetime use unsafe mut let pub crate Cargo std core borrow",
        "corpus_file": "tmp/corpora/corpus_rust_systems.txt",
    },
    "database_sql": {
        "model": "google/gemma-4-e4b-it",
        "seed_prompt": "SELECT FROM WHERE JOIN ON GROUP BY HAVING ORDER BY LIMIT INNER LEFT RIGHT OUTER CROSS UNION ALL CTE WITH INSERT INTO UPDATE DELETE CREATE TABLE",
        "corpus_file": "tmp/corpora/corpus_database_sql.txt",
    },
    "devops_infra": {
        "model": "google/gemma-4-E2B-it",
        "seed_prompt": "docker run build exec ps images rm rmi volume network kubectl get apply delete describe logs expose port-forward namespace configmap secret deployment pod service",
        "corpus_file": "tmp/corpora/corpus_devops_infra.txt",
    },
    "ml_tensors": {
        "model": "google/gemma-4-12B-it",
        "seed_prompt": "reshape transpose permute squeeze unsqueeze view expand repeat stride contiguous squeeze_dims split concat stack gather scatter slice broadcast",
        "corpus_file": "tmp/corpora/corpus_ml_tensors.txt",
    },
    "markdown_config": {
        "model": "google/gemma-4-E2B-it",
        "seed_prompt": '{ } [ ] "name" "id" "value" "type" "properties" "required" "items" "string" "integer" "boolean" "object" "array" "description" "version" "metadata" "tags"',
        "corpus_file": "tmp/corpora/corpus_markdown_config.txt",
    },
    "gateway_router": {
        "model": "google/gemma-4-E2B-it",
        "seed_prompt": "route gateway router query domain classify classification confidence expert select logit vector",
        "corpus_file": "tmp/corpora/corpus_gateway_router.txt",
    },
}

def run_command(cmd, name="command"):
    print(f"[{name}] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["HF_HOME"] = "/Volumes/Storage/huggingface_cache"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(SRC_DIR)
    
    start_time = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if res.returncode != 0:
        print(f"[{name}] FAILED with code {res.returncode} in {duration:.1f}s")
        print("=== STDOUT ===")
        print(res.stdout)
        print("=== STDERR ===")
        print(res.stderr)
        raise RuntimeError(f"Command failed: {name}")
    else:
        print(f"[{name}] Succeeded in {duration:.1f}s")
        # Print a short summary of the output if any
        lines = [line for line in res.stdout.splitlines() if line.strip()]
        if lines:
            print(f"[{name}] Last output line: {lines[-1]}")
    return res

def main():
    print(f"=== Starting UCE MoE Compilation and Training Orchestrator ===")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"TMP_MOE_DIR: {TMP_MOE_DIR}")
    
    for idx, (name, cfg) in enumerate(EXPERTS.items(), 1):
        print(f"\n[{idx}/8] Processing expert: {name} (Model: {cfg['model']})")
        
        # Target checkpoint path
        ckpt_path = TMP_MOE_DIR / f"uce_{name}.safetensors"
        
        # Check if already compiled/trained
        if ckpt_path.exists():
            print(f"Checkpoint for '{name}' already exists at {ckpt_path}. Skipping.")
            continue
            
        # File paths for intermediate artifacts
        tree_json = TMP_MOE_DIR / f"tree_{name}.json"
        cache_pkl = TMP_MOE_DIR / f"cache_{name}.pkl"
        corpus_path = ROOT_DIR / cfg["corpus_file"]
        
        # 1. Compile Tree
        print(f"--- Step 1: Compiling tree for '{name}' ---")
        build_tree_cmd = [
            sys.executable, str(ROOT_DIR / "scripts" / "build_tree_from_gemma.py"),
            "--gemma-model", cfg["model"],
            "--p", "8",
            "--depth", "4",
            "--max-tokens", "2048",
            "--seed-prompt", cfg["seed_prompt"],
            "--out", str(tree_json)
        ]
        run_command(build_tree_cmd, f"{name}-build-tree")
        
        # 2. Precompute Cache
        print(f"--- Step 2: Precomputing dataset cache for '{name}' ---")
        precompute_cmd = [
            sys.executable, str(ROOT_DIR / "scripts" / "precompute_dataset.py"),
            "--tree-config", str(tree_json),
            "--gemma-model", cfg["model"],
            "--num-tokens", "2000",
            "--out", str(cache_pkl),
            "--corpus-file", str(corpus_path)
        ]
        run_command(precompute_cmd, f"{name}-precompute")
        
        # 3. Distillation
        print(f"--- Step 3: Running distillation for '{name}' ---")
        distill_cmd = [
            sys.executable, str(ROOT_DIR / "scripts" / "run_distillation.py"),
            "--tree-config", str(tree_json),
            "--cached-dataset", str(cache_pkl),
            "--steps", "60",
            "--out", str(ckpt_path)
        ]
        run_command(distill_cmd, f"{name}-distill")
        
        print(f"Successfully finished training expert: {name}!")
        
    print(f"\n=== All 8 UCE experts compiled and trained successfully! ===")

if __name__ == "__main__":
    main()
