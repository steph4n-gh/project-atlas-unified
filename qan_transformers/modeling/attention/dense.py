import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparse_ratio=0.15):
        """
        Standard Dense Attention Layer.
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparse_ratio = sparse_ratio
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / np.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, kv_cache=None, attn_mask=None):
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        if kv_cache is not None:
            if "K" in kv_cache and kv_cache["K"] is not None:
                K = torch.cat([kv_cache["K"], K], dim=2)
                V = torch.cat([kv_cache["V"], V], dim=2)
            kv_cache["K"] = K
            kv_cache["V"] = V
            
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape[0] == S and attn_mask.shape[1] == S:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
                else:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores + attn_mask
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        
        if kv_cache is not None:
            return out, kv_cache
        return out
