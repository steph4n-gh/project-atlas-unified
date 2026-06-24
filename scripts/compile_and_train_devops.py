#!/usr/bin/env python
"""Orchestrator script to compile and train the four DevOps SysAdmin experts.

Loops over SQL, Cron, Git/CLI, and YAML Config, compiling their trees, precomputing
their caches, and distilling them offline using local Gemma-4 teacher models.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Workspace settings
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DEVOPS_OUT_DIR = ROOT_DIR / "tmp" / "devops"

# Ensure output directory exists
DEVOPS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Experts configuration
EXPERTS = {
    "database_sql": {
        "model": "mlx-community/gemma-4-E4B-it-4bit",
        "seed_prompt": "[POSTGRES] [MYSQL] [SQLITE] SELECT FROM WHERE JOIN ON GROUP BY HAVING ORDER BY LIMIT INNER LEFT RIGHT INSERT UPDATE DELETE CREATE TABLE COUNT SUM AVG",
        "dataset_file": "scratch/devops_data/sql_dataset.jsonl",
    },
    "cron_scheduler": {
        "model": "mlx-community/gemma-4-E4B-it-4bit",
        "seed_prompt": "[CRON] * / - , 0 1 2 3 4 5 6 7 8 9 /scripts/backup.sh rm -rf /tmp/ pg_dumpall logrotate healthcheck apt-get systemctl clear redis python vacuum",
        "dataset_file": "scratch/devops_data/cron_dataset.jsonl",
    },
    "git_cli": {
        "model": "mlx-community/gemma-4-E4B-it-4bit",
        "seed_prompt": "[GIT] [CLI] git reset --hard --soft HEAD branch -d feature push origin force cherry-pick add checkout stash pull rebase log find grep rm delete count wc sed free top sort uniq du kill",
        "dataset_file": "scratch/devops_data/git_cli_dataset.jsonl",
    },
    "yaml_config": {
        "model": "mlx-community/gemma-4-E4B-it-4bit",
        "seed_prompt": "[K8S] [DOCKER] [GHA] apiVersion v1 kind Service Pod Deployment name web spec ports port targetPort selector app image alpine docker-compose version redis github-actions jobs build runs-on steps run",
        "dataset_file": "scratch/devops_data/yaml_dataset.jsonl",
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
        lines = [line for line in res.stdout.splitlines() if line.strip()]
        if lines:
            print(f"[{name}] Last output line: {lines[-1]}")
    return res

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compile and train the 4 DevOps expert models.")
    parser.add_argument("--steps", type=int, default=60, help="Number of distillation training steps.")
    parser.add_argument("--num-tokens", type=int, default=2000, help="Number of tokens/samples to precompute in cache.")
    args = parser.parse_args()

    print(f"=== Starting DevOps SysAdmin Toolbox Orchestrator ===")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"DEVOPS_OUT_DIR: {DEVOPS_OUT_DIR}")
    
    for idx, (name, cfg) in enumerate(EXPERTS.items(), 1):
        print(f"\n[{idx}/4] Processing expert: {name} (Model: {cfg['model']})")
        
        # Target checkpoint path
        ckpt_path = DEVOPS_OUT_DIR / f"uce_{name}.safetensors"
        
        # Check if already compiled/trained
        if ckpt_path.exists():
            print(f"Checkpoint for '{name}' already exists at {ckpt_path}. Skipping.")
            continue
            
        # File paths for intermediate artifacts
        tree_json = DEVOPS_OUT_DIR / f"tree_{name}.json"
        cache_pkl = DEVOPS_OUT_DIR / f"cache_{name}.pkl"
        dataset_path = ROOT_DIR / cfg["dataset_file"]
        
        if not dataset_path.exists():
            # Try absolute path or project_atlas path
            dataset_path = Path("/Volumes/Storage/project_atlas") / cfg["dataset_file"]
            
        # 1. Compile Tree
        print(f"--- Step 1: Compiling tree for '{name}' ---")
        build_tree_cmd = [
            sys.executable, str(ROOT_DIR / "scripts" / "build_tree_from_gemma.py"),
            "--gemma-model", cfg["model"],
            "--p", "8",
            "--depth", "2",
            "--max-tokens", "64",
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
            "--num-tokens", str(args.num_tokens),
            "--out", str(cache_pkl),
            "--corpus-file", str(dataset_path)
        ]
        run_command(precompute_cmd, f"{name}-precompute")
        
        # 3. Distillation
        print(f"--- Step 3: Running distillation for '{name}' ---")
        distill_cmd = [
            sys.executable, str(ROOT_DIR / "scripts" / "run_distillation.py"),
            "--tree-config", str(tree_json),
            "--cached-dataset", str(cache_pkl),
            "--steps", str(args.steps),
            "--dim", "16",
            "--out", str(ckpt_path)
        ]
        run_command(distill_cmd, f"{name}-distill")
        
        print(f"Successfully finished training expert: {name}!")
        
    print(f"\n=== All 4 DevOps experts compiled and trained successfully! ===")

if __name__ == "__main__":
    main()
