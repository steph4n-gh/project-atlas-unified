import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _triton_block_sparse_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_maskb, stride_maskh, stride_masks, stride_maskk,
        stride_ob, stride_oh, stride_os, stride_od,
        scale,
        B, H, S, K_sparse, d,
        has_mask: tl.constexpr,
        BLOCK_SIZE_S: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)
        
        offs_s = pid_s * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_d = tl.arange(0, BLOCK_D)
        
        q_ptrs = Q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_s[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(offs_s[:, None] < S) & (offs_d[None, :] < d), other=0.0)
        
        # Online softmax states
        m_i = tl.zeros((BLOCK_SIZE_S,), dtype=tl.float32) - 1e38
        l_i = tl.zeros((BLOCK_SIZE_S,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE_S, BLOCK_D), dtype=tl.float32)
        
        for start_k in range(0, K_sparse, BLOCK_SIZE_K):
            cur_offs_k = start_k + offs_k
            k_ptrs = K_ptr + pid_b * stride_kb + pid_h * stride_kh + cur_offs_k[None, :] * stride_ks + offs_d[:, None] * stride_kd
            v_ptrs = V_ptr + pid_b * stride_vb + pid_h * stride_vh + cur_offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
            
            k = tl.load(k_ptrs, mask=(cur_offs_k[None, :] < K_sparse) & (offs_d[:, None] < d), other=0.0)
            v = tl.load(v_ptrs, mask=(cur_offs_k[:, None] < K_sparse) & (offs_d[None, :] < d), other=0.0)
            
            # Dot product
            scores = tl.dot(q, k) * scale
            
            # Mask out-of-bounds keys in this block
            scores = tl.where(cur_offs_k[None, :] < K_sparse, scores, -1e38)
            
            if has_mask:
                mask_ptrs = Mask_ptr + pid_b * stride_maskb + pid_h * stride_maskh + offs_s[:, None] * stride_masks + cur_offs_k[None, :] * stride_maskk
                mask_val = tl.load(mask_ptrs, mask=(offs_s[:, None] < S) & (cur_offs_k[None, :] < K_sparse), other=0.0)
                scores = scores + mask_val
                
            # Online softmax update
            row_max = tl.max(scores, axis=1)
            m_next = tl.maximum(m_i, row_max)
            alpha = tl.math.exp(m_i - m_next)
            exp_scores = tl.math.exp(scores - m_next[:, None])
            sum_exp = tl.sum(exp_scores, axis=1)
            l_next = l_i * alpha + sum_exp
            
            acc = acc * alpha[:, None]
            acc += tl.dot(exp_scores.to(v.dtype), v)
            
            m_i = m_next
            l_i = l_next
            
        out = acc / l_i[:, None]
        out_ptrs = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_s[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out, mask=(offs_s[:, None] < S) & (offs_d[None, :] < d))


class TritonBlockSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask=None):
        ctx.save_for_backward(q, k, v, attn_mask)
        
        B, H, S, d = q.shape
        _, _, K_sparse, _ = k.shape
        
        out = torch.empty_like(q)
        scale = 1.0 / (d ** 0.5)
        
        BLOCK_SIZE_S = 16
        BLOCK_SIZE_K = 16
        import math
        BLOCK_D = int(2 ** math.ceil(math.log2(d)))
        grid = (B, H, triton.cdiv(S, BLOCK_SIZE_S))
        
        if attn_mask is not None:
            has_mask = True
            mask_stride_b = attn_mask.stride(0)
            mask_stride_h = attn_mask.stride(1)
            mask_stride_s = attn_mask.stride(2)
            mask_stride_k = attn_mask.stride(3)
            mask_ptr = attn_mask
        else:
            has_mask = False
            mask_stride_b = 0
            mask_stride_h = 0
            mask_stride_s = 0
            mask_stride_k = 0
            mask_ptr = q # Dummy pointer
            
        _triton_block_sparse_attn_fwd_kernel[grid](
            q, k, v, mask_ptr, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            mask_stride_b, mask_stride_h, mask_stride_s, mask_stride_k,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale,
            B, H, S, K_sparse, d,
            has_mask=has_mask,
            BLOCK_SIZE_S=BLOCK_SIZE_S,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_D=BLOCK_D,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attn_mask = ctx.saved_tensors
        
        # Detach inputs and enable gradients to compute standard backward pass using PyTorch autograd
        q_detached = q.detach().requires_grad_(True)
        k_detached = k.detach().requires_grad_(True)
        v_detached = v.detach().requires_grad_(True)
        
        if attn_mask is not None:
            attn_mask_detached = attn_mask.detach().requires_grad_(attn_mask.requires_grad)
        else:
            attn_mask_detached = None
            
        # Re-run standard PyTorch sparse attention locally
        head_dim = q_detached.shape[-1]
        attn_scores = torch.matmul(q_detached, k_detached.transpose(-2, -1)) / (head_dim ** 0.5)
        
        if attn_mask_detached is not None:
            attn_scores = attn_scores + attn_mask_detached
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v_detached)
        
        # Backpropagate grad_output
        out.backward(grad_output)
        
        # Return gradients
        grad_q = q_detached.grad
        grad_k = k_detached.grad
        grad_v = v_detached.grad
        grad_mask = attn_mask_detached.grad if attn_mask_detached is not None else None
        
        return grad_q, grad_k, grad_v, grad_mask


def triton_block_sparse_attention(q, k, v, attn_mask=None):
    """
    NVIDIA Triton block-sparse attention implementation with PyTorch autograd wrapper.
    """
    device = q.device
    if device.type == "cuda" and HAS_TRITON:
        return TritonBlockSparseAttention.apply(q, k, v, attn_mask)
    else:
        # Fallback to standard PyTorch attention
        head_dim = q.shape[-1]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return out
