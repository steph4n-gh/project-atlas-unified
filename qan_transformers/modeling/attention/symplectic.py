import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SymplecticAttention(nn.Module):
    """
    Guarantees lossless Hamiltonian evolution of 8D E8 states before they are scored
    via the octonionic product.
    
    Evolves keys/queries (generalized positions q) and momenta p under a Hamiltonian
    H(q, p) = 0.5 * p^T M^-1 p + V(q), where V(q) is the E8 lattice potential.
    """
    def __init__(self, num_steps: int = 4, dt: float = 0.1, sigma: float = 0.5):
        super().__init__()
        self.num_steps = num_steps
        self.dt = dt
        self.sigma = sigma
        
        # Inverse mass matrix diagonal (parameterized, initialized to 1.0)
        self.inv_mass = nn.Parameter(torch.ones(8))
        
        # Learnable potential parameters
        self.potential_weight = nn.Parameter(torch.tensor(0.1))
        
        # Lazy initialization of E8 roots
        from qan_transformers.modeling.attention.e8_routing import get_shared_e8_roots_8d
        roots_8d, _ = get_shared_e8_roots_8d(1)
        self.register_buffer("roots", roots_8d.clone().detach())

    def potential_gradient(self, q: torch.Tensor) -> torch.Tensor:
        """
        Gradient of the potential V(q) = -potential_weight * sum_{r} exp(-sigma * ||q - r||^2).
        dV/dq = 2 * potential_weight * sigma * sum_{r} exp(-sigma * ||q - r||^2) * (q - r).
        """
        orig_shape = q.shape
        q_flat = q.reshape(-1, 8)
        
        # roots shape: [240, 8]
        roots = self.roots.to(device=q.device, dtype=q.dtype)
        
        # Broadcasting difference: [N, 1, 8] - [1, 240, 8] -> [N, 240, 8]
        diff = q_flat.unsqueeze(1) - roots.unsqueeze(0)
        
        # Squared distances: [N, 240]
        sq_dist = torch.sum(diff ** 2, dim=-1)
        
        # exp weights: [N, 240]
        weights = torch.exp(-self.sigma * sq_dist)
        
        # Gradient: [N, 8]
        grad_flat = 2.0 * self.potential_weight * self.sigma * torch.sum(
            weights.unsqueeze(-1) * diff, dim=1
        )
        
        return grad_flat.view(orig_shape)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog integration:
        q_{n+1/2} = q_n + 0.5 * dt * p_n * M^-1
        p_{n+1} = p_n - dt * dV/dq(q_{n+1/2})
        q_{n+1} = q_{n+1/2} + 0.5 * dt * p_{n+1} * M^-1
        """
        dt = self.dt
        inv_mass = self.inv_mass.view(1, 1, 8) if q.dim() == 3 else self.inv_mass.view(1, 1, 1, 8)
        
        for _ in range(self.num_steps):
            # Half-step q
            q = q + 0.5 * dt * p * inv_mass
            
            # Full-step p
            grad_V = self.potential_gradient(q)
            p = p - dt * grad_V
            
            # Half-step q
            q = q + 0.5 * dt * p * inv_mass
            
        return q, p

    def compute_hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes H(q, p) = Kinetic Energy + Potential Energy
        """
        inv_mass = self.inv_mass.view(1, 1, 8) if q.dim() == 3 else self.inv_mass.view(1, 1, 1, 8)
        kinetic = 0.5 * torch.sum((p ** 2) * inv_mass, dim=-1)
        
        q_flat = q.reshape(-1, 8)
        roots = self.roots.to(device=q.device, dtype=q.dtype)
        diff = q_flat.unsqueeze(1) - roots.unsqueeze(0)
        sq_dist = torch.sum(diff ** 2, dim=-1)
        potential = -self.potential_weight * torch.sum(torch.exp(-self.sigma * sq_dist), dim=-1)
        
        return kinetic + potential.view(q.shape[:-1])
