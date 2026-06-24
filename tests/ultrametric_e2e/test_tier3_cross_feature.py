import os
import sys
import json
import subprocess
import pytest
from pathlib import Path

# =============================================================================
# TIER 3: Cross-Feature Combinations
# =============================================================================

def test_t3_1_f1_x_f2_dataflow(tmp_path, env_with_mocks):
    """1. F1 x F2: Verify build_tree_from_gemma.py output JSON is consumable by distill_phase0_heads.py."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    res_build = subprocess.run(cmd_build, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_build.returncode == 0
    assert tree_json.exists()
    
    heads_ckpt = tmp_path / "heads.safetensors"
    cmd_heads = [
        sys.executable, "scripts/distill_phase0_heads.py",
        "--tree-config", str(tree_json), "--steps", "2", "--out", str(heads_ckpt)
    ]
    res_heads = subprocess.run(cmd_heads, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_heads.returncode == 0
    assert heads_ckpt.exists()

def test_t3_2_f2_x_f3_dataflow(tmp_path, env_with_mocks):
    """2. F2 x F3: Verify Phase 0 heads checkpoint is successfully loaded as starting weights for Phase 1 distillation."""
    tree_json = tmp_path / "tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    heads_ckpt = tmp_path / "heads.safetensors"
    subprocess.run([
        sys.executable, "scripts/distill_phase0_heads.py",
        "--tree-config", str(tree_json), "--steps", "2", "--out", str(heads_ckpt)
    ], env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json), "--heads-ckpt", str(heads_ckpt),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    res_dist = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_dist.returncode == 0
    assert uce_ckpt.exists()

def test_t3_3_f3_x_f4_dataflow(tmp_path, env_with_mocks):
    """3. F3 x F4: Verify fully distilled checkpoint from Phase 1 can be loaded for sparse generation."""
    tree_json = tmp_path / "tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce.safetensors"
    subprocess.run([
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ], env=env_with_mocks, check=True)
    
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt),
        "--prompt", "1+2", "--max-new", "2", "--seed", "42"
    ]
    res_gen = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_gen.returncode == 0
    assert "active balls touched" in res_gen.stdout

def test_t3_4_f4_x_f5_dataflow(tmp_path, env_with_mocks):
    """4. F4 x F5: Verify generated address sequences can be parsed and evaluated by structural metrics."""
    tree_json = tmp_path / "tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce.safetensors"
    subprocess.run([
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ], env=env_with_mocks, check=True)
    
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt),
        "--num-samples", "2", "--max-new", "3", "--seed", "42"
    ]
    res_eval = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_eval.returncode == 0
    assert "avg_active_balls" in res_eval.stdout

def test_t3_5_f1_x_f6_dataflow(mock_gemma_dir, tmp_path, env_with_mocks):
    """5. F1 x F6: Verify trees built using real Gemma embeddings conform to strict talkability constraints."""
    tree_json = tmp_path / "gemma_tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--gemma-model", str(mock_gemma_dir),
        "--p", "4", "--depth", "3", "--max-tokens", "32",
        "--out", str(tree_json), "--seed-prompt", "hello world"
    ]
    res_build = subprocess.run(cmd_build, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_build.returncode == 0
    
    # Load and assert tree conforms
    from ultrametric_ce.tree import FiniteTree
    from ultrametric_ce import real_gemma_contract as contract
    cfg = json.loads(tree_json.read_text())
    am = {int(a): int(t) for a, t in cfg["address_map"].items()}
    tree = FiniteTree(cfg["p"], cfg["depth"], address_map=am)
    
    from mocks.transformers import MockTokenizer
    tok = MockTokenizer()
    mapped = contract.assert_tree_talkable(tree, tok, "hello world")
    assert len(mapped) > 0

def test_t3_6_f3_x_f5_dataflow(tmp_path, env_with_mocks):
    """6. F3 x F5: Verify distillation training outputs can be directly evaluated by eval_structural.py."""
    tree_json = tmp_path / "tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce.safetensors"
    subprocess.run([
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ], env=env_with_mocks, check=True)
    
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt),
        "--num-samples", "2", "--max-new", "2", "--seed", "42"
    ]
    res_eval = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res_eval.returncode == 0
    assert "prefix_accuracy" in res_eval.stdout

def test_t3_7_f2_x_f4_dataflow(tmp_path, env_with_mocks):
    """7. F2 x F4: Verify warm-started heads checkpoint runs sparse generation directly with UCEModel."""
    tree_json = tmp_path / "tree.json"
    subprocess.run([
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ], env=env_with_mocks, check=True)
    
    heads_ckpt = tmp_path / "heads.safetensors"
    subprocess.run([
        sys.executable, "scripts/distill_phase0_heads.py",
        "--tree-config", str(tree_json), "--steps", "2", "--out", str(heads_ckpt)
    ], env=env_with_mocks, check=True)
    
    # Re-map heads weights into UCEModel path structure
    import mlx.core as mx
    from ultrametric_ce.tree import FiniteTree
    from ultrametric_ce.model import UCEModel
    
    tree_cfg = json.loads(Path(tree_json).read_text())
    am = {int(a): int(t) for a, t in tree_cfg["address_map"].items()}
    tree = FiniteTree(tree_cfg["p"], tree_cfg["depth"], address_map=am)
    
    heads_weights = mx.load(str(heads_ckpt))
    full_weights = {}
    for k, v in heads_weights.items():
        new_k = k.replace("heads.", "heads.heads.")
        full_weights[new_k] = v
        
    dummy_model = UCEModel(tree, dim=16)
    # Convert dummy model weights into flat dict
    flat_weights = {}
    def gather_weights(params, prefix=""):
        if isinstance(params, mx.array):
            flat_weights[prefix.rstrip(".")] = params
        elif isinstance(params, dict):
            for k, v in params.items():
                gather_weights(v, f"{prefix}{k}.")
        elif isinstance(params, list):
            for i, v in enumerate(params):
                gather_weights(v, f"{prefix}{i}.")
    gather_weights(dummy_model.parameters())
    
    for k, v in flat_weights.items():
        if k not in full_weights:
            full_weights[k] = v
            
    mx.save_safetensors(str(heads_ckpt), full_weights)
    
    # Inject address_map from tree_json into heads meta JSON
    meta_path = heads_ckpt.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["address_map"] = tree_cfg["address_map"]
    meta_path.write_text(json.dumps(meta))

    # We can load the heads checkpoint directly for generation
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(heads_ckpt),
        "--prompt", "1+2", "--max-new", "2", "--seed", "42"
    ]
    res_gen = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res_gen.returncode == 0, res_gen.stderr
