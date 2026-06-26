import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qan_transformers.math.octonion import OctonionAlgebra

class OctonionicAttentionMode(nn.Module):
    """
    Computes E8 root alignment scores natively in 8D using octonionic algebra.
    Avoids lossy 3D projection, and mixes 7 imaginary geometric channels.
    """
    def __init__(self, device='cpu', dtype=torch.float32):
        super().__init__()
        self.algebra = OctonionAlgebra(device=device, dtype=dtype)
        # Mixer for imaginary channels [B, S, N, 7] -> [B, S, N, 1]
        self.mixer = nn.Linear(7, 1, bias=False, device=device, dtype=dtype)
        # Initialize to small weights so it starts close to real-part dot-product
        nn.init.normal_(self.mixer.weight, std=0.01)

    def forward(self, seq_8d_norm: torch.Tensor, roots_8d_norm: torch.Tensor) -> torch.Tensor:
        """
        seq_8d_norm: [B, S, 8] query projections normalized
        roots_8d_norm: [N, 8] E8 roots in 8D normalized
        Returns: [B, S, N] alignment scores
        """
        B, S, _ = seq_8d_norm.shape
        N, _ = roots_8d_norm.shape
        
        # Conjugate query projections
        Q_conj = self.algebra.conjugate(seq_8d_norm)  # [B, S, 8]
        
        # Expand for batch product
        Q_conj_exp = Q_conj.unsqueeze(2).expand(-1, -1, N, -1)  # [B, S, N, 8]
        roots_exp = roots_8d_norm.view(1, 1, N, 8).expand(B, S, -1, -1)  # [B, S, N, 8]
        
        # Compute octonionic product
        product = self.algebra.multiply(Q_conj_exp, roots_exp)  # [B, S, N, 8]
        
        # Real part = standard dot product
        real = self.algebra.real_part(product)  # [B, S, N]
        
        # 7 imaginary components
        imag = self.algebra.imaginary_channels(product)  # [B, S, N, 7]
        
        # Mix imaginary channels to produce geometric routing bias
        bias = self.mixer(imag).squeeze(-1)  # [B, S, N]
        
        # Total score = real + bias
        scores = real + bias
        return scores
