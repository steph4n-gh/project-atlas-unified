import mlx.core as mx
from qan_transformers.mlx.e8_swap import ConwaySloaneE8DecoderMLX

class ConwaySloaneLeechDecoderMLX:
    """
    Vectorized Leech-of-E8s lattice decoder in MLX.
    Decomposes the 24D coordinate space into three 8D E8 blocks,
    decodes them in parallel via a single vectorized E8 decoder call,
    and reconstructs the closest 24D coordinate vector on the GPU.
    """
    def __init__(self):
        self.e8_decoder = ConwaySloaneE8DecoderMLX()

    def decode(self, x: mx.array) -> mx.array:
        """
        Decomposes 24D array to 3x 8D blocks, decodes them in parallel, and reassembles.
        Args:
            x: mlx.core.array of shape [..., 24]
        Returns:
            nearest_points: mlx.core.array of shape [..., 24]
        """
        orig_shape = x.shape
        x_flat = mx.reshape(x, (-1, 24))
        
        # Decompose 24D into three 8D components
        block1 = x_flat[:, 0:8]
        block2 = x_flat[:, 8:16]
        block3 = x_flat[:, 16:24]
        
        # Stack blocks to decode in a single vectorized pass
        stacked = mx.stack([block1, block2, block3], axis=0)  # shape (3, N, 8)
        stacked_flat = mx.reshape(stacked, (-1, 8))
        
        # Decode E8 components
        decoded_flat = self.e8_decoder.decode(stacked_flat)
        decoded_stacked = mx.reshape(decoded_flat, (3, -1, 8))
        
        # Reassemble to 24D
        nearest = mx.concatenate([decoded_stacked[0], decoded_stacked[1], decoded_stacked[2]], axis=-1)
        
        return mx.reshape(nearest, orig_shape)
