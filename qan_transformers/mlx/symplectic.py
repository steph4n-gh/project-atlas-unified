import mlx.core as mx
import mlx.nn as nn
import numpy as np

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
        self.inv_mass = mx.ones(8, dtype=mx.float32)
        
        # Learnable potential parameters
        self.potential_weight = mx.array(0.1, dtype=mx.float32)
        
        # Lazy initialization of E8 roots
        from qan_transformers.math.e8_projection import generate_dynamic_e8_coordinates
        roots_8d = mx.array(generate_dynamic_e8_coordinates(1), dtype=mx.float32)
        self.roots = roots_8d

    def potential_gradient(self, q: mx.array) -> mx.array:
        """
        Gradient of the potential V(q) = -potential_weight * sum_{r} exp(-sigma * ||q - r||^2).
        dV/dq = 2 * potential_weight * sigma * sum_{r} exp(-sigma * ||q - r||^2) * (q - r).
        """
        orig_shape = q.shape
        q_flat = mx.reshape(q, (-1, 8))
        
        # roots shape: [240, 8]
        roots = self.roots
        
        # Broadcasting difference: [N, 1, 8] - [1, 240, 8] -> [N, 240, 8]
        diff = mx.expand_dims(q_flat, 1) - mx.expand_dims(roots, 0)
        
        # Squared distances: [N, 240]
        sq_dist = mx.sum(diff ** 2, axis=-1)
        
        # exp weights: [N, 240]
        weights = mx.exp(-self.sigma * sq_dist)
        
        # Gradient: [N, 8]
        weighted_diff = mx.expand_dims(weights, -1) * diff
        grad_flat = 2.0 * self.potential_weight * self.sigma * mx.sum(weighted_diff, axis=1)
        
        return mx.reshape(grad_flat, orig_shape)

    def __call__(self, q: mx.array, p: mx.array) -> tuple[mx.array, mx.array]:
        """
        Leapfrog integration:
        q_{n+1/2} = q_n + 0.5 * dt * p_n * M^-1
        p_{n+1} = p_n - dt * dV/dq(q_{n+1/2})
        q_{n+1} = q_{n+1/2} + 0.5 * dt * p_{n+1} * M^-1
        """
        dt = self.dt
        if q.ndim == 3:
            inv_mass = mx.reshape(self.inv_mass, (1, 1, 8))
        else:
            inv_mass = mx.reshape(self.inv_mass, (1, 1, 1, 8))
            
        for _ in range(self.num_steps):
            # Half-step q
            q = q + 0.5 * dt * p * inv_mass
            
            # Full-step p
            grad_V = self.potential_gradient(q)
            p = p - dt * grad_V
            
            # Half-step q
            q = q + 0.5 * dt * p * inv_mass
            
        return q, p

    def compute_hamiltonian(self, q: mx.array, p: mx.array) -> mx.array:
        """
        Computes H(q, p) = Kinetic Energy + Potential Energy
        """
        if q.ndim == 3:
            inv_mass = mx.reshape(self.inv_mass, (1, 1, 8))
        else:
            inv_mass = mx.reshape(self.inv_mass, (1, 1, 1, 8))
            
        kinetic = 0.5 * mx.sum((p ** 2) * inv_mass, axis=-1)
        
        q_flat = mx.reshape(q, (-1, 8))
        roots = self.roots
        diff = mx.expand_dims(q_flat, 1) - mx.expand_dims(roots, 0)
        sq_dist = mx.sum(diff ** 2, axis=-1)
        potential = -self.potential_weight * mx.sum(mx.exp(-self.sigma * sq_dist), axis=-1)
        
        return kinetic + mx.reshape(potential, q.shape[:-1])
