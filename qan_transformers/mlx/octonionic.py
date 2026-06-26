import mlx.core as mx
import mlx.nn as nn
import numpy as np
from qan_transformers.mlx.octonion import OctonionAlgebra

class OctonionicAttentionMode(nn.Module):
    """
    Computes E8 root alignment scores natively in 8D using octonionic algebra.
    Avoids lossy 3D projection, and mixes 7 imaginary geometric channels.
    """
    def __init__(self):
        super().__init__()
        self.algebra = OctonionAlgebra()
        # Mixer for imaginary channels [B, S, N, 7] -> [B, S, N, 1]
        self.mixer = nn.Linear(7, 1, bias=False)
        # Initialize to small weights so it starts close to real-part dot-product
        self.mixer.weight = mx.random.normal(self.mixer.weight.shape) * 0.01

    def __call__(self, seq_8d_norm: mx.array, roots_8d_norm: mx.array) -> mx.array:
        """
        seq_8d_norm: [B, S, 8] query projections normalized
        roots_8d_norm: [N, 8] E8 roots in 8D normalized
        Returns: [B, S, N] alignment scores
        """
        B, S, _ = seq_8d_norm.shape
        N, _ = roots_8d_norm.shape
        
        # Conjugate query projections
        Q_conj = self.algebra.conjugate(seq_8d_norm)  # [B, S, 8]
        
        # Expand for batch product via broadcasting
        Q_conj_exp = mx.expand_dims(Q_conj, 2)  # [B, S, 1, 8]
        roots_exp = mx.reshape(roots_8d_norm, (1, 1, N, 8))  # [1, 1, N, 8]
        
        # Compute octonionic product (broadcasting handles the [B, S, N, 8] structure)
        product = self.algebra.multiply(Q_conj_exp, roots_exp)  # [B, S, N, 8]
        
        # Real part = standard dot product
        real = self.algebra.real_part(product)  # [B, S, N]
        
        # 7 imaginary components
        imag = self.algebra.imaginary_channels(product)  # [B, S, N, 7]
        
        # Mix imaginary channels to produce geometric routing bias
        bias = mx.squeeze(self.mixer(imag), -1)  # [B, S, N]
        
        # Total score = real + bias
        scores = real + bias
        return scores
