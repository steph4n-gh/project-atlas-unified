import mlx.core as mx
import mlx.nn as nn

class SheafFirewall(nn.Module):
    """
    Sheaf-Theoretic Consistency Firewall.
    Checks gluing consistency (cohomological obstruction) between consecutive hidden layers
    inline during the forward pass.
    """
    def __init__(self, embed_dim: int, threshold: float = 0.25):
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = threshold
        
        # Connection operator/restriction map initialized to identity
        self.W_sheaf = mx.eye(embed_dim, dtype=mx.float32)
        
    def check_consistency(self, x_l: mx.array, x_next: mx.array):
        """
        Computes the local coboundary d^0(x)_l = x_{l+1} - x_l @ W_sheaf
        and returns the normalized sheaf consistency obstruction metric E_l.
        """
        # Reshape to [N_tokens, embed_dim]
        x_l_flat = mx.reshape(x_l, (-1, self.embed_dim))
        x_next_flat = mx.reshape(x_next, (-1, self.embed_dim))
        
        # Compute coboundary
        d = x_next_flat - x_l_flat @ self.W_sheaf.astype(x_l_flat.dtype)
        
        # Normalized obstruction metric
        E_l = mx.sum(mx.square(d)) / (mx.sum(mx.square(x_l_flat)) + 1e-9)
        
        is_fractured = E_l > self.threshold
        return is_fractured, E_l
        
    def update_connection(self, x_l: mx.array, x_next: mx.array, lr: float = 0.01):
        """
        Updates the local sheaf connection W_sheaf via online Procrustes/gradient descent.
        """
        x_l_flat = mx.reshape(x_l, (-1, self.embed_dim))
        x_next_flat = mx.reshape(x_next, (-1, self.embed_dim))
        N = x_l_flat.shape[0]
        
        if N == 0:
            return
            
        # Compute gradient of ||x_next - x_l @ W_sheaf||^2 w.r.t W_sheaf
        # W_sheaf: [D, D]
        # x_l: [N, D]
        # x_next: [N, D]
        pred = x_l_flat @ self.W_sheaf.astype(x_l_flat.dtype)
        d = pred - x_next_flat # [N, D]
        
        grad = (x_l_flat.T @ d) / N # [D, D]
        
        # Update connection weights
        self.W_sheaf = self.W_sheaf - lr * grad.astype(self.W_sheaf.dtype)
