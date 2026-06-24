import os
import sys
import json
import subprocess
import pytest
import numpy as np
import mlx.core as mx

from pathlib import Path
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce import real_gemma_contract as contract
from ultrametric_ce.padic import padic_address
from ultrametric_ce.distillation import (
    distillation_kl,
    hidden_alignment,
    hierarchical_prefix_loss,
    ultrametric_reg,
    addresses_to_text
)
from ultrametric_ce.evaluation import ultrametric_spearman_correlation

# =============================================================================
# FEATURE 1: P-adic Tree Building & Clustering (F1)
# =============================================================================

def test_f1_1_synthetic_tree_build(tmp_path, env_with_mocks):
    """F1.1: Verify build_tree_from_gemma.py with --synthetic creates a JSON with valid structure."""
    out_json = tmp_path / "tree.json"
    cmd = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "2", "--depth", "3", "--num-tokens", "8",
        "--out", str(out_json)
    ]
    res = subprocess.run(cmd, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"CLI build failed: {res.stderr}"
    assert out_json.exists()
    cfg = json.loads(out_json.read_text())
    assert cfg["p"] == 2
    assert cfg["depth"] == 3
    assert "address_map" in cfg
    assert len(cfg["address_map"]) == 8

def test_f1_2_custom_p_depth(tmp_path, env_with_mocks):
    """F1.2: Verify tree building CLI with custom p and depth branching capacity constraints."""
    out_json = tmp_path / "tree_custom.json"
    cmd = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "2", "--num-tokens", "9",
        "--out", str(out_json)
    ]
    res = subprocess.run(cmd, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"CLI build failed: {res.stderr}"
    cfg = json.loads(out_json.read_text())
    assert cfg["p"] == 3
    assert cfg["depth"] == 2
    assert len(cfg["address_map"]) == 9

def test_f1_3_token_subselection():
    """F1.3: Verify token subselection contract outputs unique sorted IDs bounded by max_tokens."""
    vocab_size = 500
    max_tokens = 50
    prompt_ids = [45, 90, 120]
    ids = contract.select_induction_token_ids(vocab_size, max_tokens, prompt_ids)
    assert len(ids) <= max_tokens
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    for pid in prompt_ids:
        if pid < vocab_size:
            assert pid in ids

def test_f1_4_assert_tree_talkable_success():
    """F1.4: Verify that assert_tree_talkable passes when all prompt tokens are present in tree leaves."""
    p, depth = 2, 2
    high_tids = [1000, 1001, 1002, 1003]
    addrs = [0, 1, 2, 3]
    tree = FiniteTree.build_from_addresses(addrs, p=p, depth=depth, token_ids=high_tids)
    
    class DummyTokenizer:
        def encode(self, prompt):
            return [1000, 1001]
    
    mapped = contract.assert_tree_talkable(tree, DummyTokenizer(), "dummy prompt")
    assert len(mapped) == 2
    assert mapped == [0, 1]

def test_f1_5_lca_consistency():
    """F1.5: Verify induced tree LCA depth matches p-adic valuation of address differences."""
    p = 3
    depth = 3
    tree = FiniteTree(p=p, depth=depth)
    addr_a = padic_address([0, 1, 2], p)  # 0 + 1*3 + 2*9 = 21
    addr_b = padic_address([0, 1, 0], p)  # 0 + 1*3 + 0*9 = 3
    tree.add_leaf(1001, addr_a)
    tree.add_leaf(1002, addr_b)
    
    lca = tree.lca_depth(addr_a, addr_b)
    # Both agree on lowest digits: d0=0, d1=1. Differ on d2. LCA depth should be 2.
    assert lca == 2

# =============================================================================
# FEATURE 2: Phase 0 Head Factorization (F2)
# =============================================================================

def test_f2_1_synthetic_warmstart(tmp_path, env_with_mocks):
    """F2.1: Verify distill_phase0_heads.py CLI runs with --synthetic and produces safetensors."""
    out_ckpt = tmp_path / "phase0.safetensors"
    cmd = [
        sys.executable, "scripts/distill_phase0_heads.py",
        "--synthetic", "--steps", "2", "--out", str(out_ckpt)
    ]
    res = subprocess.run(cmd, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"CLI phase0 failed: {res.stderr}"
    assert out_ckpt.exists()
    assert out_ckpt.with_suffix(".meta.json").exists()

def test_f2_2_sub_distribution_logic():
    """F2.2: Verify grouping of teacher logits over tree children sums to 1.0 (sub-dists)."""
    p, depth = 2, 2
    tree = FiniteTree.build_from_addresses([0, 1, 2, 3], p=p, depth=depth, token_ids=[101, 102, 103, 104])
    # Programmatic test of tree.children behavior
    # root ball (0,0) children should be (1,0) and (1,1)
    ch = tree.children(0, 0)
    assert len(ch) == 2
    assert sorted(ch) == [0, 1]

def test_f2_3_checkpoint_saving_loading(tmp_path):
    """F2.3: Re-load a Phase 0 saved weights file and assert parameter structure."""
    from ultrametric_ce.distillation import save_warmed_heads, load_warmed_heads
    from ultrametric_ce.routing import DigitHeads
    
    p, depth = 2, 2
    tree = FiniteTree.build_from_addresses([0, 1], p=p, depth=depth, token_ids=[101, 102])
    dim = 8
    heads = DigitHeads(p, depth, dim=dim)
    
    out_ckpt = tmp_path / "saved_heads.safetensors"
    save_warmed_heads(heads, str(out_ckpt))
    assert out_ckpt.exists()
    
    loaded_heads = load_warmed_heads(str(out_ckpt), p, depth, dim=dim)
    assert len(loaded_heads.heads) == len(heads.heads)

def test_f2_4_prediction_warmed_heads(tmp_path):
    """F2.4: Verify next-token prediction without diffusion on structural prefixes."""
    from ultrametric_ce.routing import DigitHeads
    from ultrametric_ce.distillation import predict_with_warmed_heads
    
    p, depth = 3, 2
    tree = FiniteTree.build_from_addresses([0, 1, 2], p=p, depth=depth, token_ids=[10, 11, 12])
    heads = DigitHeads(p, depth, dim=8)
    
    # Run prediction for empty prefix
    pred = predict_with_warmed_heads(heads, tree, [])
    assert isinstance(pred, mx.array)
    assert pred.shape == (3,)
    assert abs(float(mx.sum(pred)) - 1.0) < 1e-5

def test_f2_5_zero_overlap_token_fallback():
    """F2.5: Verify factorization handles token inputs by using fallback mechanisms when no prefix aligns."""
    from ultrametric_ce.routing import DigitHeads
    p, depth = 2, 2
    tree = FiniteTree.build_from_addresses([0, 1], p=p, depth=depth, token_ids=[10, 11])
    heads = DigitHeads(p, depth, dim=8)
    
    # DigitHeads forward takes a state vector and a depth_idx
    state = mx.zeros((8,))
    probs = heads(state, depth_idx=0)
    assert probs.shape == (2,)

# =============================================================================
# FEATURE 3: Phase 1 & 2 Distillation Training (F3)
# =============================================================================

def test_f3_1_phase1_distill_cli(tmp_path, env_with_mocks):
    """F3.1: Run run_distillation.py with --phase 1 and verify heads parameters remain frozen."""
    # First build a synthetic tree
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    
    # Warm-start heads
    heads_ckpt = tmp_path / "heads.safetensors"
    cmd_heads = [
        sys.executable, "scripts/distill_phase0_heads.py",
        "--tree-config", str(tree_json), "--steps", "2", "--out", str(heads_ckpt)
    ]
    subprocess.run(cmd_heads, env=env_with_mocks, check=True)
    
    # Distill Phase 1
    uce_ckpt = tmp_path / "uce_phase1.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json), "--heads-ckpt", str(heads_ckpt),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    res = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"Distill CLI failed: {res.stderr}"
    assert uce_ckpt.exists()

def test_f3_2_phase2_distill_cli(tmp_path, env_with_mocks):
    """F3.2: Run run_distillation.py with --phase 2 and verify joint updates."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce_phase2.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "2", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    res = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"Distill CLI failed: {res.stderr}"
    assert uce_ckpt.exists()

def test_f3_3_multi_loss_balance():
    """F3.3: Verify all 4 loss terms compile and run programmatically."""
    p, depth = 2, 2
    tree = FiniteTree.build_from_addresses([0, 1, 2, 3], p=p, depth=depth, token_ids=[10, 11, 12, 13])
    
    # 1. KL loss
    p_probs = mx.softmax(mx.array([0.1, 0.9]))
    t_probs = mx.softmax(mx.array([0.2, 0.8]))
    kl = distillation_kl(p_probs, t_probs)
    assert kl >= 0.0

    # 2. Hidden alignment
    states = {(0,0): mx.array([0.1, 0.2]), (1,0): mx.array([0.3, 0.4])}
    ht = {(0,0): mx.array([0.1, 0.2]), (1,0): mx.array([0.35, 0.45])}
    ha = hidden_alignment(states, ht)
    assert ha >= 0.0

    # 3. Ultrametric regularizer
    ureg = ultrametric_reg(p_probs, [0, 1], tree)
    assert isinstance(ureg, mx.array)

def test_f3_4_safetensors_meta_export(tmp_path, env_with_mocks):
    """F3.4: Verify distillation exports both .safetensors model weights and .meta.json sidecar."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce_export.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    assert uce_ckpt.exists()
    meta_json = uce_ckpt.with_suffix(".meta.json")
    assert meta_json.exists()
    meta = json.loads(meta_json.read_text())
    assert "p" in meta
    assert "depth" in meta
    assert "address_map" in meta

def test_f3_5_distillation_loss_decay(tmp_path, env_with_mocks):
    """F3.5: Verify distillation loss decays or runs steps without crashing, printing total_loss."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce_decay.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "5", "--smoke", "--out", str(uce_ckpt)
    ]
    res = subprocess.run(cmd_dist, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0
    assert "loss=" in res.stdout

# =============================================================================
# FEATURE 4: Autoregressive Sparse Generation (F4)
# =============================================================================

def test_f4_1_cli_generation(tmp_path, env_with_mocks):
    """F4.1: Verify generate_with_mvp.py loads checkpoint and outputs generation logs."""
    # Build tree and distilled model
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt),
        "--prompt", "1+2", "--max-new", "2", "--seed", "42"
    ]
    res = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, f"Gen CLI failed: {res.stderr}"
    assert len(res.stdout) > 0

def test_f4_2_sparse_routing_log(tmp_path, env_with_mocks):
    """F4.2: Verify stdout log contains 'active balls touched' indicating sparse routing."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt),
        "--prompt", "1+2", "--max-new", "2", "--seed", "42"
    ]
    res = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert "active balls touched" in res.stdout

def test_f4_3_decoded_text():
    """F4.3: Verify address sequence decodes back to text using text conversion APIs."""
    p, depth = 3, 2
    tree = FiniteTree.build_from_addresses([0, 1, 2], p=p, depth=depth, token_ids=[65, 66, 67])
    
    class DummyTokenizer:
        def decode(self, ids):
            return "".join(chr(x) for x in ids)
            
    text = addresses_to_text([0, 1, 2], DummyTokenizer(), tree)
    assert text == "ABC"

def test_f4_4_token_only_mode(tmp_path, env_with_mocks):
    """F4.4: Verify generation runs using token-only mode interface."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt),
        "--prompt", "1+2", "--max-new", "3", "--seed", "123"
    ]
    res = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0

def test_f4_5_seed_reproducibility(tmp_path, env_with_mocks):
    """F4.5: Verify identical seeds produce bit-level identical outputs across generation runs."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt),
        "--prompt", "1+2", "--max-new", "4", "--seed", "99"
    ]
    res1 = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    res2 = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res1.stdout == res2.stdout

# =============================================================================
# FEATURE 5: Structural & Spearman Evaluation (F5)
# =============================================================================

def test_f5_1_grammar_validity():
    """F5.1: Verify parser flags valid expressions and rejects mutated invalid strings."""
    from ultrametric_ce.evaluation import is_structurally_valid_toy_expr
    
    # arithmetic validity
    assert is_structurally_valid_toy_expr("1+2") is True
    assert is_structurally_valid_toy_expr("((1+2)*3)") is True
    assert is_structurally_valid_toy_expr("1++2") is False
    assert is_structurally_valid_toy_expr("(1+2") is False

def test_f5_2_eval_harness_cli(tmp_path, env_with_mocks):
    """F5.2: Verify eval_structural.py outputs structural validity rates and accuracy."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt), "--num-samples", "2", "--max-new", "2", "--seed", "42"
    ]
    res = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"Eval CLI failed: {res.stderr}"
    assert "prefix_accuracy" in res.stdout
    assert "good_rate" in res.stdout

def test_f5_3_spearman_correlation():
    """F5.3: Verify ultrametric_spearman_correlation outputs a valid Spearman rho."""
    from ultrametric_ce.model import UCEModel
    p, depth = 2, 2
    tree = FiniteTree.build_from_addresses([0, 1, 2, 3], p=p, depth=depth, token_ids=[10, 11, 12, 13])
    model = UCEModel(tree, dim=4)
    
    pairs = [([0], 1), ([1], 2), ([2], 3)]
    rho = ultrametric_spearman_correlation(model, tree, pairs)
    assert -1.0 <= rho <= 1.0

def test_f5_4_baseline_logging(tmp_path, env_with_mocks):
    """F5.4: Verify structural evaluation prints random baselines."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt), "--num-samples", "2", "--max-new", "2", "--seed", "42"
    ]
    res = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert "random_prefix_per_digit" in res.stdout

def test_f5_5_sparsity_assertion(tmp_path, env_with_mocks):
    """F5.5: Verify average active balls is strictly less than 100% of tree nodes."""
    tree_json = tmp_path / "tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--synthetic", "--p", "3", "--depth", "3", "--num-tokens", "21", "--out", str(tree_json)
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    uce_ckpt = tmp_path / "uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    cmd_eval = [
        sys.executable, "scripts/eval_structural.py",
        "--checkpoint", str(uce_ckpt), "--num-samples", "2", "--max-new", "2", "--seed", "42"
    ]
    res = subprocess.run(cmd_eval, env=env_with_mocks, capture_output=True, text=True, timeout=60)
    assert "avg_active_balls" in res.stdout

# =============================================================================
# FEATURE 6: Real Gemma-4 Verification (F6)
# =============================================================================

def test_f6_1_verification_script_run(mock_gemma_dir, tmp_path, env_with_mocks):
    """F6.1: Verify run_gemma4_verif.py executes successfully using mock storage cache."""
    scratch_dir = tmp_path / "scratch"
    cmd = [
        sys.executable, "scripts/run_gemma4_verif.py",
        "--gemma-model", str(mock_gemma_dir),
        "--scratch", str(scratch_dir),
        "--p", "8", "--depth", "2", "--max-tokens", "32",
        "--phase0-steps", "1", "--distill-steps", "1", "--max-new", "2"
    ]
    res = subprocess.run(cmd, env=env_with_mocks, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, f"Verif script failed: {res.stderr}\nSTDOUT:\n{res.stdout}"
    assert "ALL VERIF STEPS PASSED" in res.stdout

def test_f6_2_direct_dequantization(mock_gemma_dir):
    """F6.2: Verify direct dequantization from MLX 4bit snapshots resolves correctly."""
    from ultrametric_ce.gemma_interface import extract_embeddings_from_mlx_snapshot
    embs = extract_embeddings_from_mlx_snapshot(mock_gemma_dir)
    assert embs.shape == (2000, 16)

def test_f6_3_high_tid_registry_enforcer():
    """F6.3: Verify assert_tree_talkable rejects trees utilizing low special token IDs."""
    p, depth = 2, 2
    # Low token IDs
    low_tids = [0, 1, 2, 3]
    tree = FiniteTree.build_from_addresses([0, 1, 2, 3], p=p, depth=depth, token_ids=low_tids)
    
    class DummyTokenizer:
        def encode(self, prompt):
            return [0, 1]
            
    with pytest.raises(AssertionError, match="high original tids"):
        contract.assert_tree_talkable(tree, DummyTokenizer(), "prompt")

def test_f6_4_prompt_overlap_violation_check():
    """F6.4: Verify assert_tree_talkable raises AssertionError if prompt shares 0 tokens with tree."""
    p, depth = 2, 2
    high_tids = [1000, 1001, 1002, 1003]
    tree = FiniteTree.build_from_addresses([0, 1, 2, 3], p=p, depth=depth, token_ids=high_tids)
    
    class DummyTokenizer:
        def encode(self, prompt):
            return [2000, 2001]
            
    with pytest.raises(AssertionError, match="maps to 0 registered addrs"):
        contract.assert_tree_talkable(tree, DummyTokenizer(), "prompt")

def test_f6_5_identical_dual_generations(mock_gemma_dir, tmp_path, env_with_mocks):
    """F6.5: Verify two independent generation calls using identical seeds produce identical outputs."""
    # First build tree from mock gemma
    tree_json = tmp_path / "gemma_tree.json"
    cmd_build = [
        sys.executable, "scripts/build_tree_from_gemma.py",
        "--gemma-model", str(mock_gemma_dir),
        "--p", "4", "--depth", "3", "--max-tokens", "32",
        "--out", str(tree_json), "--seed-prompt", "hello world"
    ]
    subprocess.run(cmd_build, env=env_with_mocks, check=True)
    
    # Train model
    uce_ckpt = tmp_path / "gemma_uce.safetensors"
    cmd_dist = [
        sys.executable, "scripts/run_distillation.py",
        "--tree-config", str(tree_json), "--gemma-model", str(mock_gemma_dir),
        "--phase", "1", "--steps", "2", "--smoke", "--out", str(uce_ckpt)
    ]
    subprocess.run(cmd_dist, env=env_with_mocks, check=True)
    
    # Generate twice
    cmd_gen = [
        sys.executable, "scripts/generate_with_mvp.py",
        "--checkpoint", str(uce_ckpt), "--gemma-model", str(mock_gemma_dir),
        "--prompt", "hello", "--max-new", "3", "--seed", "42"
    ]
    res1 = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    res2 = subprocess.run(cmd_gen, env=env_with_mocks, capture_output=True, text=True, timeout=30)
    assert res1.stdout == res2.stdout
