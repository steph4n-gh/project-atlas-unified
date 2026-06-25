import torch
from qan_transformers.math.e8_projection import ConwaySloaneE8Decoder

class ConwaySloaneLeechDecoder:
    """
    Vectorized Leech-of-E8s lattice decoder in PyTorch.
    Decomposes the 24D coordinate space into three 8D E8 blocks,
    decodes each block using the branch-free E8 Conway-Sloane solver,
    and reconstructs the closest 24D coordinate vector.
    """
    def __init__(self, device=None):
        self.device = device
        self.e8_decoder = ConwaySloaneE8Decoder(device=device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decomposes 24D space to 3x 8D blocks, runs vectorized branch-free E8 decoding,
        and reassembles.
        Args:
            x: torch.Tensor of shape [..., 24]
        Returns:
            nearest_points: torch.Tensor of shape [..., 24]
        """
        orig_shape = x.shape
        # Flatten batches
        x_flat = x.reshape(-1, 24)
        
        # Decompose 24D into three 8D components
        block1 = x_flat[:, 0:8]
        block2 = x_flat[:, 8:16]
        block3 = x_flat[:, 16:24]
        
        # Decode each E8 block
        dec1 = self.e8_decoder.decode(block1)
        dec2 = self.e8_decoder.decode(block2)
        dec3 = self.e8_decoder.decode(block3)
        
        # Reassemble to 24D
        nearest = torch.cat([dec1, dec2, dec3], dim=-1)
        
        return nearest.reshape(orig_shape)
