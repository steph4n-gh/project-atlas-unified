import pytest
from unittest.mock import MagicMock
import mlx.core as mx

from ultrametric_ce.moe import UCEMoeRouter
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.model import UCEModel

def test_moe_topk_routing_and_blending():
    # 1. Setup mock tokenizer
    mock_tokenizer = MagicMock()
    # Mock token encoding: map words to mock token IDs
    mock_tokenizer.encode.side_effect = lambda text: [1000, 1001, 1002]
    # Mock decoding
    mock_tokenizer.decode.side_effect = lambda tokens: "".join(chr(min(max(t - 1000 + 97, 0), 255)) for t in tokens)

    p = 3
    depth = 3
    dim = 16
    num_leaves = p ** depth  # 27 leaves

    # Define mock address map (27 leaf coordinates -> token IDs)
    address_map = {i: i + 1000 for i in range(num_leaves)}

    # Helper to instantiate a mock UCE expert
    def create_mock_expert_instance(domain_name):
        tree = FiniteTree(p=p, depth=depth, address_map=address_map)
        model = UCEModel(tree, dim=dim)
        
        # Mock forward to return a deterministic distribution where index matching
        # the domain predictions are slightly higher to test routing divergence
        probs_array = [0.01] * num_leaves
        # Accentuate specific leaves to favor specific domains
        # e.g., domain 0 (python_coder), domain 1 (web_stack), domain 2 (rust_systems)
        if domain_name == "gateway_router":
            probs_array[0] = 0.5  # maps to 0 -> python_coder
            probs_array[1] = 0.3  # maps to 1 -> web_stack
            probs_array[2] = 0.1  # maps to 2 -> rust_systems
        else:
            # For expert models, set high probability for token ID 1000 (leaf 0) and 1001 (leaf 1)
            probs_array[0] = 0.8
            probs_array[1] = 0.15
            
        sum_p = sum(probs_array)
        probs_normalized = [px / sum_p for px in probs_array]
        
        model.forward = MagicMock(return_value=mx.array(probs_normalized))
        return tree, model

    # Mock load_expert cache dictionary
    mock_experts = {}
    def mock_load_expert_func(self_instance, name):
        if name not in mock_experts:
            mock_experts[name] = create_mock_expert_instance(name)
        tree, model = mock_experts[name]
        self_instance.experts[name] = (tree, model)
        return tree, model

    # Monkey patch UCEMoeRouter._load_expert dynamically to intercept disk loads
    original_load_expert = UCEMoeRouter._load_expert
    try:
        UCEMoeRouter._load_expert = mock_load_expert_func

        # Instantiate router with k=3
        router = UCEMoeRouter(moe_dir="/tmp/moe_mock", active_experts_k=3)
        router.tokenizer = mock_tokenizer

        # --- Test 1: Gating & Top-K Gating Weights ---
        # Query containing no direct keywords to force gateway model evaluation
        prompt = "evaluate complex algebraic logic"
        
        # Route with k=3 returning weights
        expert_weights = router.route_prompt(prompt, k=3, return_weights=True)
        
        assert len(expert_weights) == 3
        # Weights should sum to exactly 1.0
        total_w = sum(w for _, w in expert_weights)
        assert pytest.approx(total_w) == 1.0

        # Verify python_coder, web_stack, rust_systems are top 3
        expert_names = [name for name, _ in expert_weights]
        assert "python_coder" in expert_names
        assert "web_stack" in expert_names
        assert "rust_systems" in expert_names

        # --- Test 2: Multi-Expert Blended Generation ---
        decoded_text = router.generate(prompt, max_new_tokens=4, k=3, verbose=True)
        
        # Since tokenizer decodes token IDs offset by 1000 + 97,
        # token ID 1000 -> 'a', 1001 -> 'b', 1002 -> 'c'
        # Verification that we got a valid non-empty string decoded from blended tokens
        assert isinstance(decoded_text, str)
        assert len(decoded_text) > 0
        
        # Check that active models' forward calls were invoked during autoregressive loop
        for name in ["python_coder", "web_stack", "rust_systems"]:
            tree, model = mock_experts[name]
            assert model.forward.call_count > 0

    finally:
        UCEMoeRouter._load_expert = original_load_expert
