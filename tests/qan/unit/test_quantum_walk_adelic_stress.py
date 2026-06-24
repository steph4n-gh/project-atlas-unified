import pytest
import torch
import torch.nn as nn
import numpy as np
from qan_transformers.optim.adelic import QuantumWalkAdelicOptimizer

def test_nan_gradients_submanifold():
    """
    Test that NaN gradients on submanifold parameters are cleaned (replaced by 0.0)
    and do not cause parameters or optimizer state to become NaN.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("model.layers.0.attn.q_proj.lora_A", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.1
    )
    
    # Set gradient containing NaN
    p.grad = torch.tensor([[float('nan'), -0.5], [1.0, float('nan')]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    # Verify that gradients were cleaned to 0.0 in-place
    assert torch.isnan(p.grad).sum() == 0
    assert torch.allclose(p.grad, torch.tensor([[0.0, -0.5], [1.0, 0.0]]))
    
    # Verify parameter values are not NaN and not infinite
    assert torch.isnan(p_after).sum() == 0
    assert torch.isinf(p_after).sum() == 0
    
    # Verify that updates occurred or remained stable
    assert not torch.equal(p_before, p_after)
    
    # Verify state variables are not NaN/Inf
    state = opt.state[p]
    assert "rho_c" in state
    assert torch.isnan(state["rho_c"]).sum() == 0
    assert not np.isnan(state["r"])
    assert not np.isnan(state["history_energies"]).any()

def test_nan_gradients_non_submanifold():
    """
    Test that NaN gradients on non-submanifold parameters are cleaned
    and do not cause parameters to become NaN.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    # No named parameters, but we can set up a dummy name that doesn't match submanifold patterns
    named_params = [("model.layers.0.attn.q_proj.weight", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.1
    )
    
    p.grad = torch.tensor([[float('nan'), -0.5], [1.0, float('nan')]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    assert torch.isnan(p.grad).sum() == 0
    assert torch.isnan(p_after).sum() == 0
    assert not torch.equal(p_before, p_after)

def test_infinite_gradients_submanifold():
    """
    Test that positive/negative infinite gradients on submanifold parameters are cleaned (replaced by 0.0)
    and do not cause parameters or optimizer state to become NaN/Inf.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("model.layers.0.attn.q_proj.lora_A", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.1
    )
    
    # Set gradient containing infinite values
    p.grad = torch.tensor([[float('inf'), -0.5], [1.0, float('-inf')]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    # Verify that gradients were cleaned to 0.0 in-place
    assert torch.isinf(p.grad).sum() == 0
    assert torch.allclose(p.grad, torch.tensor([[0.0, -0.5], [1.0, 0.0]]))
    
    # Verify parameter values are not NaN and not infinite
    assert torch.isnan(p_after).sum() == 0
    assert torch.isinf(p_after).sum() == 0
    
    # Verify that updates occurred or remained stable
    assert not torch.equal(p_before, p_after)

def test_infinite_gradients_non_submanifold():
    """
    Test that infinite gradients on non-submanifold parameters are cleaned
    and do not cause parameters to become NaN/Inf.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("model.layers.0.attn.q_proj.weight", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.1
    )
    
    p.grad = torch.tensor([[float('inf'), -0.5], [1.0, float('-inf')]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    assert torch.isinf(p.grad).sum() == 0
    assert torch.isnan(p_after).sum() == 0
    assert torch.isinf(p_after).sum() == 0
    assert not torch.equal(p_before, p_after)

def test_zero_learning_rate_submanifold():
    """
    Test that a learning rate of 0.0 prevents updates, and does not cause
    parameters to change or become NaN, even in the presence of gradients.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("model.layers.0.attn.q_proj.lora_A", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.0
    )
    
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    # Verify that parameter is completely unchanged
    assert torch.equal(p_before, p_after)
    
    # Since lr=0.0, step count should not be incremented and state should remain empty
    assert len(opt.state[p]) == 0

def test_zero_learning_rate_non_submanifold():
    """
    Test that a learning rate of 0.0 prevents updates on non-submanifold parameters.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("model.layers.0.attn.q_proj.weight", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.0
    )
    
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    assert torch.equal(p_before, p_after)
    assert len(opt.state[p]) == 0

def test_dynamic_zero_learning_rate():
    """
    Test that dynamically setting learning rate to 0.0 in the param_groups
    stops updates and leaves state intact/stable.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    named_params = [("model.layers.0.attn.q_proj.lora_A", p)]
    
    opt = QuantumWalkAdelicOptimizer(
        [p],
        named_parameters=named_params,
        coin_type="hadamard",
        lr=0.1
    )
    
    # Step 1: Normal step with lr=0.1
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    opt.step()
    
    # Verify state was populated
    state = opt.state[p]
    assert state["step"] == 1
    p_step1 = p.clone().detach()
    
    # Step 2: Set lr to 0.0 dynamically
    for group in opt.param_groups:
        group['lr'] = 0.0
        
    p.grad = torch.tensor([[0.2, -0.2], [0.4, -0.4]])
    opt.step()
    
    p_step2 = p.clone().detach()
    
    # Verify that parameter is completely unchanged between step 1 and step 2
    assert torch.equal(p_step1, p_step2)
    
    # Verify that the state was not modified/incremented when lr was 0.0
    assert opt.state[p]["step"] == 1
