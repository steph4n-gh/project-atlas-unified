import pytest
import torch
import torch.nn as nn
from qan_transformers.optim.adelic import SymplecticPhaseSpaceOptimizer

def test_symplectic_optimizer_init_and_step():
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    opt = SymplecticPhaseSpaceOptimizer([p], lr=0.1, damping=0.2)
    
    assert opt.defaults["lr"] == 0.1
    assert opt.defaults["damping"] == 0.2
    assert opt.defaults["project_noether"] is True
    
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    p_before = p.clone().detach()
    
    opt.step()
    
    p_after = p.clone().detach()
    assert not torch.equal(p_before, p_after)
    
    state = opt.state[p]
    assert "momentum" in state
    assert state["step"] == 1

def test_noether_charge_orthogonal_projection():
    # Large weights to check orthogonality
    p = nn.Parameter(torch.tensor([3.0, 4.0], requires_grad=True))
    opt = SymplecticPhaseSpaceOptimizer([p], lr=0.1, damping=0.0, project_noether=True)
    
    # Gradient in direction of p (scaling direction)
    p.grad = torch.tensor([3.0, 4.0])
    
    q_before = p.clone().detach()
    opt.step()
    
    # Momentum should be projected orthogonal to q_before
    momentum = opt.state[p]["momentum"]
    dot_prod = torch.sum(q_before * momentum)
    
    # Dot product must be extremely close to 0
    assert abs(dot_prod.item()) < 1e-6
    
    # Norm of coordinate: ||q_new||^2 should equal ||q_old||^2 + ||p||^2
    q_after = p.clone().detach()
    norm_before = torch.sum(q_before ** 2).item()
    norm_after = torch.sum(q_after ** 2).item()
    momentum_norm = torch.sum(momentum ** 2).item()
    
    assert abs(norm_after - (norm_before + momentum_norm)) < 1e-5

def test_no_noether_projection():
    p = nn.Parameter(torch.tensor([3.0, 4.0], requires_grad=True))
    opt = SymplecticPhaseSpaceOptimizer([p], lr=0.1, damping=0.0, project_noether=False)
    
    p.grad = torch.tensor([3.0, 4.0])
    q_before = p.clone().detach()
    opt.step()
    
    momentum = opt.state[p]["momentum"]
    dot_prod = torch.sum(q_before * momentum)
    
    # Without projection, momentum is in direction of gradient (-p.grad = -[3, 4])
    # so dot product should be non-zero (specifically -0.1 * (3*3 + 4*4) = -2.5)
    assert abs(dot_prod.item() + 2.5) < 1e-5

def test_invalid_arguments():
    p = nn.Parameter(torch.randn(2, 2))
    with pytest.raises(ValueError):
         SymplecticPhaseSpaceOptimizer([p], lr=-0.1)
    with pytest.raises(ValueError):
         SymplecticPhaseSpaceOptimizer([p], damping=-0.1)
    with pytest.raises(ValueError):
         SymplecticPhaseSpaceOptimizer([p], damping=1.5)
