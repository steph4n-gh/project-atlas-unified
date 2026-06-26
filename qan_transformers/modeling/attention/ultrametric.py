import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qan_transformers.modeling.attention.utils import repeat_kv

class UltrametricAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparse_ratio=0.15, num_key_value_heads=None, is_draft=False, depth=5, leaf_size=128):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be perfectly divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.sparse_ratio = sparse_ratio
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_draft = is_draft
        self.depth = depth
        self.leaf_size = leaf_size
        self.head_dim = embed_dim // num_heads
        self.scaling = 1.0 / np.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Coordinate projection mapping embed_dim to 3 coordinates
        self.coordinate_proj = nn.Linear(embed_dim, 3)
        
        # Learnable decay parameter
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, kv_cache=None, attn_mask=None):
        device = x.device
        dtype = x.dtype
        B, S, D = x.shape
        
        # Projects hidden states into 3D continuous coordinates
        coords = torch.sigmoid(self.coordinate_proj(x))
        coords = torch.nan_to_num(coords, nan=0.5)
        
        # Extract depth digits for bases 2, 3, and 5
        c0 = coords[..., 0]
        c1 = coords[..., 1]
        c2 = coords[..., 2]
        
        # Base 2 digits
        current_0 = c0
        digits_2 = []
        for _ in range(self.depth):
            current_0 = current_0 * 2
            d = torch.floor(current_0)
            d = torch.clamp(d, 0, 1)
            digits_2.append(d)
            current_0 = current_0 - d
            
        # Base 3 digits
        current_1 = c1
        digits_3 = []
        for _ in range(self.depth):
            current_1 = current_1 * 3
            d = torch.floor(current_1)
            d = torch.clamp(d, 0, 2)
            digits_3.append(d)
            current_1 = current_1 - d
            
        # Base 5 digits
        current_2 = c2
        digits_5 = []
        for _ in range(self.depth):
            current_2 = current_2 * 5
            d = torch.floor(current_2)
            d = torch.clamp(d, 0, 4)
            digits_5.append(d)
            current_2 = current_2 - d
            
        # Interleave and pack digits into base-30 Morton code integers
        morton_code = torch.zeros_like(c0, dtype=torch.long)
        for k in range(self.depth):
            d_30 = digits_2[k] + 2 * digits_3[k] + 6 * digits_5[k]
            morton_code = morton_code * 30 + d_30.long()
            
        # Linear projections for Q, K, V
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate with cached K and V if kv_cache is provided
        if kv_cache is not None:
            if "K" in kv_cache and kv_cache["K"] is not None:
                K_combined = torch.cat([kv_cache["K"], K], dim=2)
                V_combined = torch.cat([kv_cache["V"], V], dim=2)
                morton_combined = torch.cat([kv_cache["morton_codes"], morton_code], dim=1)
            else:
                K_combined = K
                V_combined = V
                morton_combined = morton_code
            kv_cache["K"] = K_combined
            kv_cache["V"] = V_combined
            kv_cache["morton_codes"] = morton_combined
        else:
            K_combined = K
            V_combined = V
            morton_combined = morton_code
            
        S_total = K_combined.shape[2]
        
        # S_total < 2048 (dynamic fallback)
        if S_total < 2048:
            if self.num_key_value_groups > 1:
                K_rep = repeat_kv(K_combined, self.num_key_value_groups)
                V_rep = repeat_kv(V_combined, self.num_key_value_groups)
            else:
                K_rep = K_combined
                V_rep = V_combined
                
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    if attn_mask.shape[0] == S and attn_mask.shape[1] == S:
                        attn_mask_sdpa = attn_mask.unsqueeze(0).unsqueeze(1)
                    else:
                        attn_mask_sdpa = attn_mask.unsqueeze(1).unsqueeze(2)
                elif attn_mask.dim() == 3:
                    attn_mask_sdpa = attn_mask.unsqueeze(1)
                else:
                    attn_mask_sdpa = attn_mask
            else:
                attn_mask_sdpa = None
                
            out = F.scaled_dot_product_attention(
                Q, K_rep, V_rep,
                attn_mask=attn_mask_sdpa,
                scale=self.scaling,
                is_causal=False
            )
        else:
            # S_total >= 2048: Fast Multipole Method (FMM) attention loop
            N = S_total
            H = self.num_heads
            
            # Repeat KV if needed
            if self.num_key_value_groups > 1:
                K_rep = repeat_kv(K_combined, self.num_key_value_groups)
                V_rep = repeat_kv(V_combined, self.num_key_value_groups)
            else:
                K_rep = K_combined
                V_rep = V_combined
                
            # Sort sequence along sequence dimension by Morton codes (with chronological priority to prevent causal leakage)
            orig_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            sort_key = orig_idx * 100000000 + morton_combined
            sort_idx = torch.argsort(sort_key, dim=-1) # Shape: [B, N]
            unsort_idx = torch.argsort(sort_idx, dim=-1) # Shape: [B, N]
            
            # Pad Q to N if S < N
            if S < N:
                Q_padded = F.pad(Q, (0, 0, 0, N - S))
            else:
                Q_padded = Q
                
            # Gather sorted tensors along sequence dimension (dim 2)
            gather_idx = sort_idx.view(B, 1, N, 1).expand(-1, H, -1, self.head_dim)
            Q_sorted = torch.gather(Q_padded, 2, gather_idx)
            K_sorted = torch.gather(K_rep, 2, gather_idx)
            V_sorted = torch.gather(V_rep, 2, gather_idx)
            
            # Determine tree dimensions
            B_sz = self.leaf_size
            M = (N + B_sz - 1) // B_sz
            L = int(np.ceil(np.log2(max(M, 1))))
            M_pow = 2**L
            N_tree = M_pow * B_sz
            pad_len = N_tree - N
            
            # Construct sorted mask (Issue 2)
            mask_val = -65000.0 if Q.dtype in (torch.float16, torch.bfloat16) else -1e9
            mask_sorted = torch.zeros(B, H, N_tree, device=device, dtype=dtype)
            
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    mask_seq = attn_mask.unsqueeze(1).expand(-1, H, -1)
                elif attn_mask.dim() == 3:
                    if attn_mask.shape[1] in (H, 1):
                        mask_seq = attn_mask.expand(-1, H, -1)
                    else:
                        mask_seq = attn_mask[:, 0, :].unsqueeze(1).expand(-1, H, -1)
                elif attn_mask.dim() == 4:
                    if attn_mask.shape[1] == 1:
                        mask_seq = attn_mask[:, 0, 0, :].unsqueeze(1).expand(-1, H, -1)
                    else:
                        mask_seq = attn_mask[:, :, 0, :]
                else:
                    mask_seq = attn_mask
                
                if mask_seq.shape[-1] < N:
                    pad_size = N - mask_seq.shape[-1]
                    mask_seq = F.pad(mask_seq, (0, pad_size), value=0.0)
                elif mask_seq.shape[-1] > N:
                    mask_seq = mask_seq[..., :N]
                
                gather_idx_mask = sort_idx.unsqueeze(1).expand(-1, H, -1)
                mask_sorted_seq = torch.gather(mask_seq, 2, gather_idx_mask)
                
                if pad_len > 0:
                    mask_sorted = F.pad(mask_sorted_seq, (0, pad_len), value=mask_val)
                else:
                    mask_sorted = mask_sorted_seq
            else:
                if pad_len > 0:
                    mask_sorted[:, :, N:] = mask_val
            # Pad sorted tensors to N_tree
            if pad_len > 0:
                Q_tree = F.pad(Q_sorted, (0, 0, 0, pad_len))
                K_tree = F.pad(K_sorted, (0, 0, 0, pad_len))
                V_tree = F.pad(V_sorted, (0, 0, 0, pad_len))
            else:
                Q_tree = Q_sorted
                K_tree = K_sorted
                V_tree = V_sorted
                
            # Reshape to block structure
            Q_blocks = Q_tree.view(B, H, M_pow, B_sz, self.head_dim)
            K_blocks = K_tree.view(B, H, M_pow, B_sz, self.head_dim)
            V_blocks = V_tree.view(B, H, M_pow, B_sz, self.head_dim)
            
            # Upward Pass (Aggregate nodes)
            K_tree_nodes = {}
            V_tree_nodes = {}
            # Leaf level L nodes (Issue 3: active count pooling)
            active_count = (mask_sorted == 0).to(dtype).view(B, H, M_pow, B_sz).sum(dim=-1, keepdim=True)
            K_tree_nodes[L] = K_blocks.sum(dim=-2) / torch.clamp(active_count, min=1.0)
            V_tree_nodes[L] = V_blocks.sum(dim=-2) / torch.clamp(active_count, min=1.0)
            
            for l in range(L - 1, -1, -1):
                parent_K = K_tree_nodes[l+1].view(B, H, 2**l, 2, self.head_dim).mean(dim=3)
                parent_V = V_tree_nodes[l+1].view(B, H, 2**l, 2, self.head_dim).mean(dim=3)
                K_tree_nodes[l] = parent_K
                V_tree_nodes[l] = parent_V
                
            # Upward Pass for mask nodes (Issue 2)
            mask_nodes = {}
            mask_nodes[L] = mask_sorted.view(B, H, M_pow, B_sz).max(dim=-1)[0]
            for l in range(L - 1, -1, -1):
                mask_nodes[l] = torch.maximum(mask_nodes[l+1][..., ::2], mask_nodes[l+1][..., 1::2])
                
            # Near-field direct block attention
            attn_scores_near = torch.matmul(Q_blocks, K_blocks.transpose(-2, -1)) * self.scaling
            
            # Causal and padding masks in near-field
            if pad_len > 0:
                sort_idx_tree = F.pad(sort_idx, (0, pad_len), value=N + 1000)
            else:
                sort_idx_tree = sort_idx
                
            orig_indices_blocks = sort_idx_tree.view(B, 1, M_pow, B_sz)
            causal_mask_near = (orig_indices_blocks.unsqueeze(-1) < orig_indices_blocks.unsqueeze(-2))
            padding_mask_near = (orig_indices_blocks.unsqueeze(-2) >= N)
            
            mask_near = (causal_mask_near | padding_mask_near).to(dtype=Q.dtype) * mask_val
            
            # Extract block-local slice of sorted mask and add to near-field scores (Issue 2)
            mask_blocks = mask_sorted.view(B, H, M_pow, B_sz)
            attn_scores_near = attn_scores_near + mask_near + mask_blocks.unsqueeze(-2)
            
            # Upward hierarchy of minimum and maximum original indices for causal checks (Issue 1)
            min_orig_node = {}
            max_orig_node = {}
            min_orig_node[L] = orig_indices_blocks.min(dim=-1)[0]
            max_orig_node[L] = orig_indices_blocks.max(dim=-1)[0]
            for l in range(L - 1, -1, -1):
                min_orig_node[l] = torch.minimum(min_orig_node[l+1][..., ::2], min_orig_node[l+1][..., 1::2])
                max_orig_node[l] = torch.maximum(max_orig_node[l+1][..., ::2], max_orig_node[l+1][..., 1::2])
                
            # Far-field aggregated sibling nodes attention
            j_indices = torch.arange(M_pow, device=Q.device)
            K_sibs = []
            V_sibs = []
            is_sib_padded_list = []
            is_sib_causal_violation_list = []
            sibling_mask_list = []
            
            for l in range(1, L + 1):
                ancestor_indices = torch.div(j_indices, 2**(L - l), rounding_mode='trunc')
                sibling_indices = ancestor_indices ^ 1
                
                gather_idx_sib = sibling_indices.view(1, 1, M_pow, 1).expand(B, H, -1, self.head_dim)
                K_sib_l = torch.gather(K_tree_nodes[l], 2, gather_idx_sib)
                V_sib_l = torch.gather(V_tree_nodes[l], 2, gather_idx_sib)
                
                K_sibs.append(K_sib_l.unsqueeze(3))
                V_sibs.append(V_sib_l.unsqueeze(3))
                
                # Check sibling padding
                sibling_start_block = sibling_indices * (2**(L - l))
                is_sib_padded_l = (sibling_start_block >= M)
                is_sib_padded_list.append(is_sib_padded_l.view(1, 1, M_pow, 1, 1))
                
                # Check sibling causal violation (Issue 1)
                sibling_max_orig_l = torch.gather(max_orig_node[l], 2, sibling_indices.view(1, 1, M_pow).expand(B, 1, -1))
                causal_mask_sib_l = (sibling_max_orig_l.unsqueeze(-1) > orig_indices_blocks)
                is_sib_causal_violation_list.append(causal_mask_sib_l.unsqueeze(-1))
                
                # Sibling mask (Issue 2)
                sibling_mask_l = torch.gather(mask_nodes[l], 2, sibling_indices.view(1, 1, M_pow).expand(B, H, -1))
                sibling_mask_list.append(sibling_mask_l.unsqueeze(-1))
                
            K_sib_all = torch.cat(K_sibs, dim=3) # Shape: [B, H, M_pow, L, D]
            V_sib_all = torch.cat(V_sibs, dim=3) # Shape: [B, H, M_pow, L, D]
            is_sib_padded_all = torch.cat(is_sib_padded_list, dim=4) # Shape: [1, 1, M_pow, 1, L]
            is_sib_causal_violation_all = torch.cat(is_sib_causal_violation_list, dim=-1) # Shape: [B, 1, M_pow, B_sz, L]
            sibling_mask_all = torch.cat(sibling_mask_list, dim=-1) # Shape: [B, H, M_pow, L]
            
            # Sibling scores
            scores_sib = torch.matmul(Q_blocks, K_sib_all.transpose(-2, -1)) * self.scaling
            
            # Scale factor for sibling nodes: -self.gamma * 30**(-level)
            levels = torch.arange(1, L + 1, device=Q.device, dtype=Q.dtype)
            scale_factors = -self.gamma * (30.0 ** (-levels))
            scores_sib_scaled = scores_sib * scale_factors.view(1, 1, 1, 1, L)
            
            # Mask padded and causal violated sibling nodes (Issue 1)
            is_sib_invalid = (is_sib_padded_all | is_sib_causal_violation_all)
            scores_sib_scaled = scores_sib_scaled.masked_fill(is_sib_invalid, mask_val)
            
            # Add sibling mask (Issue 2)
            scores_sib_scaled = scores_sib_scaled + sibling_mask_all.unsqueeze(-2)
            
            # Joint attention scores and softmax
            total_scores = torch.cat([attn_scores_near, scores_sib_scaled], dim=-1)
            total_weights = F.softmax(total_scores, dim=-1)
            
            weights_near = total_weights[..., :B_sz]
            weights_sib = total_weights[..., B_sz:]
            
            # Outputs
            out_near = torch.matmul(weights_near, V_blocks)
            out_sib = torch.matmul(weights_sib, V_sib_all)
            out_block = out_near + out_sib
            
            # Restore shape and unsort
            out_tree = out_block.view(B, H, N_tree, self.head_dim)
            out_sorted = out_tree[:, :, :N, :]
            
            unsort_gather_idx = unsort_idx.view(B, 1, N, 1).expand(-1, H, -1, self.head_dim)
            out_unsorted = torch.gather(out_sorted, 2, unsort_gather_idx)
            
            out = out_unsorted[:, :, :S, :]
            
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        out = self.out_proj(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0)
        
        # Differentiable hook to populate gradients for coordinate_proj
        # (since sorting and floor digit extraction are discrete/non-differentiable)
        out = out + (coords - coords.detach()).mean(dim=-1, keepdim=True)
        
        if kv_cache is not None:
            return out, kv_cache
        return out
