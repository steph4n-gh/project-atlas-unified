import torch
from qan_transformers.modeling.attention import cayley_orthogonal_adapter

def verify_math():
    torch.manual_seed(42)
    d = 8
    r = 2
    
    A = torch.randn(d, r, dtype=torch.float64) * 0.5
    B = torch.randn(d, r, dtype=torch.float64) * 0.5
    X = torch.randn(5, d, dtype=torch.float64)
    
    # 1. Compute S, W, and W^T directly
    S = torch.matmul(A, B.t()) - torch.matmul(B, A.t())
    I_d = torch.eye(d, dtype=torch.float64)
    
    # Cayley transform: W = (I - S)(I + S)^{-1}
    W = torch.matmul(I_d - S, torch.linalg.inv(I_d + S))
    W_T = W.t()
    
    # 2. Compute using the PyTorch adapter snippet (with double precision for comparison)
    X_adapted = cayley_orthogonal_adapter(X, A, B)
    
    # 3. Check what X_adapted matches: X @ W or X @ W_T
    matches_W = torch.allclose(X_adapted, torch.matmul(X, W), atol=1e-12)
    matches_WT = torch.allclose(X_adapted, torch.matmul(X, W_T), atol=1e-12)
    
    print(f"X_adapted matches X @ W (direct): {matches_W}")
    print(f"X_adapted matches X @ W_T (transposed): {matches_WT}")
    
    # 4. Check if U and V are packed in the correct order
    # Let's compute: W_L = I - 2 U (I + V^T U)^{-1} V^T
    U = torch.cat([A, -B], dim=1)
    V = torch.cat([B, A], dim=1)
    
    I_2r = torch.eye(2 * r, dtype=torch.float64)
    core = torch.linalg.inv(I_2r + torch.matmul(V.t(), U))
    W_woodbury = I_d - 2.0 * torch.matmul(torch.matmul(U, core), V.t())
    
    matches_woodbury = torch.allclose(W, W_woodbury, atol=1e-12)
    print(f"W matches W_woodbury: {matches_woodbury}")

if __name__ == "__main__":
    verify_math()
