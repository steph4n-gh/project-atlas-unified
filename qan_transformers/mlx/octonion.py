import mlx.core as mx
import numpy as np

class OctonionAlgebra:
    """
    Full octonionic algebra implementation vectorized for MLX.
    
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
    
    def __init__(self):
        # Build structure constants tensor: gamma[i, j, k] is coefficient of e_k in e_i * e_j
        gamma = np.zeros((8, 8, 8), dtype=np.float32)
        
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
            
        self.gamma = mx.array(gamma)

    def multiply(self, a: mx.array, b: mx.array) -> mx.array:
        """
        Multiply two batches of octonions.
        a, b: Tensors of shape [..., 8]
        Returns: Tensor of shape [..., 8]
        """
        # Compute outer product along components: shape [..., 8, 8]
        outer = mx.expand_dims(a, -1) * mx.expand_dims(b, -2)
        
        # Contract with structure constants: result_k = sum_{i,j} gamma[i,j,k] * a_i * b_j
        result = mx.einsum('ijk,...ij->...k', self.gamma, outer)
        return result

    def conjugate(self, a: mx.array) -> mx.array:
        """
        Conjugation: ā = a_0 - a_1*e_1 - ... - a_7*e_7
        """
        mask = mx.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=a.dtype)
        return a * mask

    def norm_squared(self, a: mx.array) -> mx.array:
        """
        Norm squared: |a|^2 = sum(a_i^2)
        """
        return mx.sum(a ** 2, axis=-1)

    def norm(self, a: mx.array) -> mx.array:
        """
        Norm: |a| = sqrt(sum(a_i^2))
        """
        return mx.sqrt(self.norm_squared(a) + 1e-12)

    def real_part(self, a: mx.array) -> mx.array:
        """
        Extract the real (e0) component.
        """
        return a[..., 0]

    def imaginary_channels(self, a: mx.array) -> mx.array:
        """
        Extract the 7 imaginary components [..., 7].
        """
        return a[..., 1:]


class CayleyIntegerProjector:
    """
    Projector mapping 8D E8 root vectors ↔ Cayley integers (represented as 8D octonions).
    """
    def to_octonion(self, e8_coords: mx.array) -> mx.array:
        return e8_coords

    def to_e8(self, octonions: mx.array) -> mx.array:
        return octonions
