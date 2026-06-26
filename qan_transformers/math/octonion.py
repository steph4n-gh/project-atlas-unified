import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OctonionAlgebra:
    """
    Full octonionic algebra implementation vectorized for PyTorch.
    
    Basis: e0=1 (real), e1,...,e7 (imaginary units).
    Multiplication: e_i · e_j = -δ_{ij} + ε_{ijk} · e_k
    
    Uses standard Fano plane triples.
    """
    FANO_TRIPLES = [
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 7),
        (1, 5, 6),
        (2, 6, 7),
        (1, 3, 7),
    ]
    
    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        
        # Build structure constants tensor: gamma[i, j, k] is coefficient of e_k in e_i * e_j
        gamma = torch.zeros(8, 8, 8, device=device, dtype=dtype)
        
        # e_0 is the identity
        for i in range(8):
            gamma[0, i, i] = 1.0
            gamma[i, 0, i] = 1.0
            
        # e_i * e_i = -e_0 for i > 0
        for i in range(1, 8):
            gamma[i, i, 0] = -1.0
            
        # Fano plane triples and their cyclic permutations
        for (i, j, k) in self.FANO_TRIPLES:
            # Cyclic permutations give positive signs
            gamma[i, j, k] = 1.0
            gamma[j, k, i] = 1.0
            gamma[k, i, j] = 1.0
            # Anti-cyclic permutations give negative signs
            gamma[j, i, k] = -1.0
            gamma[k, j, i] = -1.0
            gamma[i, k, j] = -1.0
            
        self.gamma = gamma

    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Multiply two batches of octonions.
        a, b: Tensors of shape [..., 8]
        Returns: Tensor of shape [..., 8]
        """
        # Dynamically cast gamma if input device/dtype changed
        if self.gamma.device != a.device or self.gamma.dtype != a.dtype:
            self.gamma = self.gamma.to(device=a.device, dtype=a.dtype)
            
        # Compute outer product along components: shape [..., 8, 8]
        outer = a.unsqueeze(-1) * b.unsqueeze(-2)
        
        # Contract with structure constants: result_k = sum_{i,j} gamma[i,j,k] * a_i * b_j
        result = torch.einsum('ijk,...ij->...k', self.gamma, outer)
        return result

    def conjugate(self, a: torch.Tensor) -> torch.Tensor:
        """
        Conjugation: ā = a_0 - a_1*e_1 - ... - a_7*e_7
        """
        conj = a.clone()
        # In-place negate imaginary components to avoid allocations where possible
        conj[..., 1:] = -conj[..., 1:]
        return conj

    def norm_squared(self, a: torch.Tensor) -> torch.Tensor:
        """
        Norm squared: |a|^2 = sum(a_i^2)
        """
        return torch.sum(a ** 2, dim=-1)

    def norm(self, a: torch.Tensor) -> torch.Tensor:
        """
        Norm: |a| = sqrt(sum(a_i^2))
        """
        return torch.sqrt(self.norm_squared(a) + 1e-12)

    def real_part(self, a: torch.Tensor) -> torch.Tensor:
        """
        Extract the real (e0) component.
        """
        return a[..., 0]

    def imaginary_channels(self, a: torch.Tensor) -> torch.Tensor:
        """
        Extract the 7 imaginary components [..., 7].
        """
        return a[..., 1:]


class CayleyIntegerProjector:
    """
    Projector mapping 8D E8 root vectors ↔ Cayley integers (represented as 8D octonions).
    Since E8 coordinates are isomorphic to Cayley integers, this is a clean type-preserving mapping.
    """
    def to_octonion(self, e8_coords: torch.Tensor) -> torch.Tensor:
        """
        Map 8D E8 coordinates to octonion representation [..., 8].
        """
        return e8_coords.clone()

    def to_e8(self, octonions: torch.Tensor) -> torch.Tensor:
        """
        Map octonions [..., 8] back to E8 coordinates.
        """
        return octonions.clone()
