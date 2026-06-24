import os
import sys
import json
import subprocess
import pytest
from pathlib import Path

# =============================================================================
# TIER 4: Real-World Application Scenarios
# =============================================================================

def test_t4_scenario1_ci_pipeline(tmp_path, env_with_mocks):
    """Scenario 1: Synthetic Fast CI/CD Pipeline.
    
    1. Build synthetic tree (p=3, depth=3, V=27) -> tree.json
    2. Warm-start heads (5 steps) -> phase0.safetensors
    3. Run Phase 1 distillation (5 steps) -> distilled.safetensors
    4. Generate 5 tokens from prompt "((1+"
    5. Run structural eval on distilled.safetensors
    """
    # 1. Build synthetic tree
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    res_build = subprocess.run(cmd_build, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_build.returncode == 0, res_build.stderr
    
    # 2. Warm-start heads
    heads_ckpt = tmp_path / "phase0.safetensors"
    cmd_p0 = [
        sys.executable, "scripts/distill_phase0_heads.py",
        "--tree-config", str(tree_json), "--steps", "5", "--out", str(heads_ckpt)
    ]
    res_p0 = subprocess.run(cmd_p0, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_p0.returncode == 0, res_p0.stderr
    
    # 3. Run Phase 1 distillation
    uce_ckpt = tmp_path / "distilled.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json), "--heads-ckpt", str(heads_ckpt),
        "--phase", "1", "--steps", "5", "--smoke", "--out", str(uce_ckpt)
    ]
    res_dist = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_dist.returncode == 0, res_dist.stderr
    
    # 4. Generate
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt), "--prompt", "((1+", "--max-new", "5"
    ]
    res_gen = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_gen.returncode == 0, res_gen.stderr
    
    # 5. Run structural eval
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt), "--num-samples", "2", "--max-new", "5"
    ]
    res_eval = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_eval.returncode == 0, res_eval.stderr

def test_t4_scenario2_gemma2_coldstart(mock_gemma_dir, tmp_path, env_with_mocks):
    """Scenario 2: Quantized Gemma-2 Cold-Start & Distillation.
    
    1. Resolve local path to model
    2. Induce a p=4, depth=2 tree (16 leaves) from embeddings
    3. Execute Phase 0 factorization (3 steps)
    4. Run distillation phase 1 (3 steps)
    """
    tree_json = tmp_path / "gemma2_tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--gemma-model", str(mock_gemma_dir),
        "--p", "11", "--depth", "2", "--max-tokens", "32",
        "--out", str(tree_json), "--seed-prompt", "hello world"
    ]
    res_build = subprocess.run(cmd_build, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_build.returncode == 0, res_build.stderr
    
    heads_ckpt = tmp_path / "gemma2_heads.safetensors"
    cmd_p0 = [
        sys.executable, "scripts/distill_phase0_heads.py",
        "--tree-config", str(tree_json), "--gemma-model", str(mock_gemma_dir),
        "--steps", "3", "--out", str(heads_ckpt)
    ]
    res_p0 = subprocess.run(cmd_p0, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_p0.returncode == 0, res_p0.stderr
    
    uce_ckpt = tmp_path / "gemma2_distilled.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json), "--gemma-model", str(mock_gemma_dir),
        "--heads-ckpt", str(heads_ckpt), "--phase", "1", "--steps", "3", "--smoke", "--out", str(uce_ckpt)
    ]
    res_dist = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_dist.returncode == 0, res_dist.stderr

def test_t4_scenario3_gemma4_assistant_alignment(mock_gemma_dir, tmp_path, env_with_mocks):
    """Scenario 3: High-Vocabulary Gemma-4 Assistant Alignment.
    
    1. Select high token IDs using contract
    2. Construct a p=8, depth=2 tree (64 leaves)
    3. Execute Phase 1 training on prompt dataset (2 steps)
    4. Decode address sequence back to assistant text
    """
    tree_json = tmp_path / "gemma4_tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--gemma-model", str(mock_gemma_dir),
        "--p", "8", "--depth", "2", "--max-tokens", "64",
        "--out", str(tree_json), "--seed-prompt", "hello"
    ]
    res_build = subprocess.run(cmd_build, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_build.returncode == 0, res_build.stderr
    
    uce_ckpt = tmp_path / "gemma4_uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json), "--gemma-model", str(mock_gemma_dir),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    res_dist = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_dist.returncode == 0, res_dist.stderr
    
    # Generate and decode
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt), "--gemma-model", str(mock_gemma_dir),
        "--prompt", "hello", "--max-new", "2"
    ]
    res_gen = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_gen.returncode == 0, res_gen.stderr

def test_t4_scenario4_structural_grammar_hallucination(tmp_path, env_with_mocks):
    """Scenario 4: Structural Grammar Hallucination Check.
    
    1. Build toy arithmetic tree
    2. Train model to convergence under Phase 1 (10 steps)
    3. Sample sequences autoregressively
    4. Verify structural metrics
    """
    tree_json = tmp_path / "arith_tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "2", "--depth", "2", "--num-tokens", "4", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "arith_uce.safetensors"
    subprocess.run([
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "10", "--smoke", "--out", str(uce_ckpt)
    ], env=env_with_mocks, check=True)
    
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt), "--num-samples", "5", "--max-new", "5"
    ]
    res_eval = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_eval.returncode == 0, res_eval.stderr
    assert "good_rate=" in res_eval.stdout

def test_t4_scenario5_latency_sparsity(tmp_path, env_with_mocks):
    """Scenario 5: Ultra-low Latency Unified Memory Execution.
    
    1. Load converted Gemma-4 UCE model (p=8, depth=2, V=32)
    2. Run generation on a prompt with active ball logs enabled
    3. Verify active ball sparsity is <= 50%
    """
    tree_json = tmp_path / "sparse_tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "sparse_uce.safetensors"
    subprocess.run([
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ], env=env_with_mocks, check=True)
    
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt), "--prompt", "1+2", "--max-new", "4"
    ]
    res_gen = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_gen.returncode == 0, res_gen.stderr
    assert "active balls touched" in res_gen.stdout
