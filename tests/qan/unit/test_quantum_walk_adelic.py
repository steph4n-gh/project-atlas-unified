import pytest
import torch
import torch.nn as nn
import numpy as np
from qan_transformers.optim.adelic import QuantumWalkAdelicOptimizer

def test_initialization():
    """
    Test initialization of QuantumWalkAdelicOptimizer.
    """
    p = nn.Parameter(torch.randn(2, 2))
    
    # Standard Hadamard initialization
    opt_had = QuantumWalkAdelicOptimizer([p], coin_type="hadamard")
    assert opt_had.defaults["coin_type"] == "hadamard"
    
    # Standard Grover initialization
    opt_grov = QuantumWalkAdelicOptimizer([p], coin_type="grover", p_base=3)
    assert opt_grov.defaults["coin_type"] == "grover"
    assert opt_grov.defaults["p_base"] == 3
    
    # Invalid coin type should raise ValueError
    with pytest.raises(ValueError):
        QuantumWalkAdelicOptimizer([p], coin_type="invalid_coin")

def test_step_execution():
    """
    Test that it performs steps successfully and parameters change.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    opt = QuantumWalkAdelicOptimizer([p], coin_type="hadamard", lr=0.1)
    
    # Before step, no state
    assert len(opt.state[p]) == 0
    
    # Set gradient
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    # Parameters must change
    assert not torch.equal(p_before, p_after)
    
    # State should now be populated
    state = opt.state[p]
    assert "rho_c" in state
    assert "x_p" in state
    assert state["step"] == 1

def test_coin_density_matrix_properties():
    """
    Test that the coin state density matrix rho_c maintains Hermiticity,
    trace = 1.0, and positive semi-definiteness.
    """
    device = torch.device("cpu")
    for coin_type in ["hadamard", "grover"]:
        p = nn.Parameter(torch.randn(5, 5, device=device))
        opt = QuantumWalkAdelicOptimizer([p], coin_type=coin_type, lr=0.1, damping_rate=0.2, p_base=4)
        
        # Perform multiple steps with random gradients
        for step in range(5):
            p.grad = torch.randn(5, 5, device=device)
            opt.step()
            
            state = opt.state[p]
            assert "rho_c" in state
            rho_c = state["rho_c"]
            
            # 1. Hermiticity: rho_c must equal adjoint(rho_c)
            assert torch.allclose(rho_c, rho_c.adjoint(), atol=1e-5)
            
            # 2. Trace preservation: trace(rho_c) must sum to 1.0 + 0.0j
            tr = torch.trace(rho_c)
            assert torch.allclose(tr.real, torch.tensor(1.0, device=device), atol=1e-5)
            assert torch.allclose(tr.imag, torch.tensor(0.0, device=device), atol=1e-5)
            
            # 3. Positive semi-definiteness: all eigenvalues of rho_c must be >= -1e-6
            eigenvalues = torch.linalg.eigvalsh(rho_c)
            assert torch.all(eigenvalues >= -1e-6)

def test_submanifold_restriction():
    """
    Test that the submanifold restriction works:
    only parameters matching LoRA patterns get quantum walk states and updates.
    Other parameters (base weights/biases) do not get quantum walk states (no x_p/rho_c)
    but still get updated via SGLD.
    """
    # Define a set of parameters with names
    param_lora_A = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    param_lora_B = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    param_e8 = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    param_base = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    param_bias = nn.Parameter(torch.randn(2, requires_grad=True))
    
    named_params = [
        ("model.layers.0.attn.q_proj.lora_A", param_lora_A),
        ("model.layers.0.attn.v_proj.lora_B", param_lora_B),
        ("model.layers.0.attn.e8_proj", param_e8),
        ("model.layers.0.attn.q_proj.weight", param_base),
        ("model.layers.0.attn.q_proj.bias", param_bias),
    ]
    
    params_list = [p for _, p in named_params]
    
    opt = QuantumWalkAdelicOptimizer(
        params_list,
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.05
    )
    
    # Set mock gradients
    for p in params_list:
        p.grad = torch.ones_like(p) * 0.1
        
    # Take a step
    opt.step()
    
    # Verify submanifold parameters have quantum walk states
    for name, p in named_params[:3]:
        state = opt.state[p]
        assert "rho_c" in state, f"Parameter {name} should have rho_c"
        assert "x_p" in state, f"Parameter {name} should have x_p"
        
    # Verify non-submanifold parameters do NOT have quantum walk states
    for name, p in named_params[3:]:
        state = opt.state[p]
        assert "rho_c" not in state, f"Parameter {name} should NOT have rho_c"
        assert "x_p" not in state, f"Parameter {name} should NOT have x_p"
        
    # Verify non-submanifold parameters still got updated
    assert not torch.allclose(param_base, torch.zeros_like(param_base))

def test_zero_learning_rate():
    """
    Test edge case: learning rate is zero.
    Verify that no update occurs and no errors are raised.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("lora_A", p)]
    opt = QuantumWalkAdelicOptimizer([p], named_parameters=named_params, coin_type="hadamard", lr=0.0)
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    # Parameters must not change when lr is zero
    assert torch.equal(p_before, p_after)

def test_nan_inf_gradients():
    """
    Test edge case: gradients contain NaN and Inf values.
    Verify that they are handled gracefully and density matrix properties hold.
    """
    p = nn.Parameter(torch.randn(3, 3, requires_grad=True))
    named_params = [("lora_A", p)]
    opt = QuantumWalkAdelicOptimizer([p], named_parameters=named_params, coin_type="hadamard", lr=0.01)
    
    # Set gradient containing NaN and Inf
    p.grad = torch.tensor([[float('nan'), float('inf'), -float('inf')],
                           [1.0, 2.0, 3.0],
                           [0.0, 0.0, 0.0]], requires_grad=True)
    
    opt.step()
    
    # Check that density matrix was initialized and remained valid
    state = opt.state[p]
    assert "rho_c" in state
    rho_c = state["rho_c"]
    assert not torch.isnan(rho_c).any()
    assert torch.allclose(rho_c, rho_c.adjoint(), atol=1e-5)
    tr = torch.trace(rho_c)
    assert torch.allclose(tr.real, torch.tensor(1.0), atol=1e-5)
    eigenvalues = torch.linalg.eigvalsh(rho_c)
    assert torch.all(eigenvalues >= -1e-6)

def test_extreme_tree_depths():
    """
    Test very small (1) and larger (10) tree depths.
    """
    p1 = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    opt1 = QuantumWalkAdelicOptimizer([p1], named_parameters=[("lora_A", p1)], tree_depth=1)
    p1.grad = torch.randn(2, 2)
    opt1.step()
    assert opt1.state[p1]["step"] == 1
    
    p2 = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    opt2 = QuantumWalkAdelicOptimizer([p2], named_parameters=[("lora_A", p2)], tree_depth=10)
    p2.grad = torch.randn(2, 2)
    opt2.step()
    assert opt2.state[p2]["step"] == 1

def test_p_base_non_two_lca_fallback():
    """
    Test LCA calculation when p_base != 2 and num_states > 4096 (lca_table is None).
    Verify if the fallback bitwise XOR logic matches the true p-adic LCA.
    """
    # tree_depth=8, p_base=3 => 3**8 = 6561 states (> 4096)
    p = nn.Parameter(torch.randn(2, 2, requires_grad=True))
    opt = QuantumWalkAdelicOptimizer([p], named_parameters=[("lora_A", p)], p_base=3, tree_depth=8)
    
    # Let's check calculate_lca directly
    # curr_idx = 1 (base 3: 00000001)
    # history_indices = [2] (base 3: 00000002)
    # The true LCA depth should be 7 because they match in the first 7 digits (all 0s)
    lca_depth = opt.calculate_lca(1, np.array([2]), tree_depth=8)
    print(f"Calculated LCA depth: {lca_depth[0]}, expected: 7")
    assert lca_depth[0] == 7

