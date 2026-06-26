import torch
import pytest
from qan_transformers.modeling.attention.symplectic import SymplecticAttention

def test_symplectic_initialization():
    model = SymplecticAttention(num_steps=3, dt=0.1)
    assert model.num_steps == 3
    assert model.dt == 0.1
    assert model.roots.shape == (240, 8)
    assert model.inv_mass.shape == (8,)


def test_time_reversibility():
    model = SymplecticAttention(num_steps=4, dt=0.1)
    
    # [B, H, S, 8]
    q0 = torch.randn(2, 2, 4, 8)
    p0 = torch.randn(2, 2, 4, 8)
    
    # Forward integration
    q1, p1 = model(q0, p0)
    
    # Backward integration: reverse momentum, run forward, reverse momentum again
    # This is equivalent to integrating backward in time
    q2, p2 = model(q1, -p1)
    p2 = -p2
    
    # Check that we recovered the initial state
    assert torch.allclose(q0, q2, rtol=1e-4, atol=1e-4)
    assert torch.allclose(p0, p2, rtol=1e-4, atol=1e-4)


def test_energy_conservation():
    model = SymplecticAttention(num_steps=10, dt=0.05)
    
    q0 = torch.randn(1, 1, 1, 8)
    p0 = torch.randn(1, 1, 1, 8)
    
    # Compute initial energy
    h0 = model.compute_hamiltonian(q0, p0)
    
    # Evolve
    q1, p1 = model(q0, p0)
    
    # Compute final energy
    h1 = model.compute_hamiltonian(q1, p1)
    
    # Energy should be conserved to a high accuracy (typical of symplectic integrators)
    assert torch.allclose(h0, h1, rtol=1e-3, atol=1e-3)


def test_symplectic_structure_and_volume_preservation():
    model = SymplecticAttention(num_steps=2, dt=0.1)
    
    # Symplectic matrix Omega
    Omega = torch.zeros(16, 16)
    Omega[:8, 8:] = torch.eye(8)
    Omega[8:, :8] = -torch.eye(8)
    
    # Input vector [16]
    qp = torch.randn(16, requires_grad=True)
    
    def evolve_qp(x):
        q = x[:8].view(1, 1, 1, 8)
        p = x[8:].view(1, 1, 1, 8)
        q_next, p_next = model(q, p)
        return torch.cat([q_next.view(8), p_next.view(8)])
        
    # Compute Jacobian J [16, 16]
    J = torch.autograd.functional.jacobian(evolve_qp, qp)
    
    # Verify J^T * Omega * J = Omega
    sym_check = torch.matmul(torch.matmul(J.t(), Omega), J)
    assert torch.allclose(sym_check, Omega, rtol=1e-3, atol=1e-3)
    
    # Verify volume preservation: det(J) = 1.0
    det_J = torch.linalg.det(J)
    assert torch.allclose(torch.abs(det_J), torch.tensor(1.0), rtol=1e-3, atol=1e-3)
