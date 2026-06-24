import torch
import warnings

class MPSCoordinateGatherScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v, indices):
        ctx.k_shape = k.shape
        ctx.v_shape = v.shape
        
        B, H, S, d = k.shape
        device = k.device
        
        if device.type == "mps" and k.dtype == torch.float16:
            warnings.warn(
                "MPSCoordinateGatherScatter: float16 on MPS may have reduced precision. "
                "Consider using float32 or bfloat16.",
                UserWarning, stacklevel=2
            )
            
        # Cast indices to target type and ensure contiguous memory layout
        if device.type == "mps":
            idx_type = torch.int32
        else:
            idx_type = torch.long
            
        # Win 39: Cast indices and check contiguity only if needed to avoid redundant copy overhead
        if indices.device != device or indices.dtype != idx_type:
            indices = indices.to(device=device, dtype=idx_type)
        if not indices.is_contiguous():
            indices = indices.contiguous()

        if device.type == "mps":
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()
                
        # Determine index sharing statically without host-device sync
        is_shared = False
        if indices.dim() == 1:
            is_shared = True
            idx = indices
        elif indices.size(0) == 1:
            is_shared = True
            idx = indices[0]
        else:
            is_shared = False
            idx = None
            
        ctx.is_shared = is_shared
        
        if is_shared:
            ctx.save_for_backward(idx)
            # Optimize with 3D views to avoid 4D overhead (Win 99: reshape for safety)
            k_3d = k.reshape(-1, S, d)
            v_3d = v.reshape(-1, S, d)
            K_sparse_3d = torch.index_select(k_3d, 1, idx)
            V_sparse_3d = torch.index_select(v_3d, 1, idx)
            K_sparse = K_sparse_3d.reshape(B, H, -1, d)
            V_sparse = V_sparse_3d.reshape(B, H, -1, d)
        else:
            if B == 1:
                # Win 100: B=1 fast-path bypass
                ctx.save_for_backward(indices)
                K_sparse = torch.index_select(k[0], 1, indices[0]).unsqueeze(0)
                V_sparse = torch.index_select(v[0], 1, indices[0]).unsqueeze(0)
            else:
                # Vectorized gather for B > 1
                K_len = indices.shape[-1]
                offsets = torch.arange(B * H, device=device).view(B, H, 1) * S
                flat_indices = (indices.unsqueeze(1) + offsets).view(-1)
                ctx.save_for_backward(flat_indices)
                
                K_sparse = torch.index_select(k.reshape(-1, d), 0, flat_indices).view(B, H, K_len, d)
                V_sparse = torch.index_select(v.reshape(-1, d), 0, flat_indices).view(B, H, K_len, d)
            
        return K_sparse, V_sparse
 
    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_k_sparse, grad_v_sparse):
        saved_tensor, = ctx.saved_tensors
        B, H, S, d = ctx.k_shape
        device = grad_k_sparse.device
        
        if device.type == "mps":
            if not grad_k_sparse.is_contiguous():
                grad_k_sparse = grad_k_sparse.contiguous()
            if not grad_v_sparse.is_contiguous():
                grad_v_sparse = grad_v_sparse.contiguous()
                
        is_shared = ctx.is_shared
        
        if is_shared:
            grad_k = torch.zeros(B, H, S, d, dtype=grad_k_sparse.dtype, device=device)
            grad_v = torch.zeros(B, H, S, d, dtype=grad_v_sparse.dtype, device=device)
            idx = saved_tensor
            grad_k_3d = grad_k.reshape(-1, S, d)
            grad_v_3d = grad_v.reshape(-1, S, d)
            grad_k_sparse_3d = grad_k_sparse.reshape(-1, idx.size(0), d)
            grad_v_sparse_3d = grad_v_sparse.reshape(-1, idx.size(0), d)
            
            grad_k_3d.index_add_(1, idx, grad_k_sparse_3d)
            grad_v_3d.index_add_(1, idx, grad_v_sparse_3d)
        else:
            if B == 1:
                indices = saved_tensor
                grad_k = torch.zeros(B, H, S, d, dtype=grad_k_sparse.dtype, device=device)
                grad_v = torch.zeros(B, H, S, d, dtype=grad_v_sparse.dtype, device=device)
                grad_k[0].index_add_(1, indices[0], grad_k_sparse[0])
                grad_v[0].index_add_(1, indices[0], grad_v_sparse[0])
            else:
                flat_indices = saved_tensor
                grad_k_flat = torch.zeros(B * H * S, d, dtype=grad_k_sparse.dtype, device=device)
                grad_v_flat = torch.zeros(B * H * S, d, dtype=grad_v_sparse.dtype, device=device)
                
                grad_k_flat.index_add_(0, flat_indices, grad_k_sparse.reshape(-1, d))
                grad_v_flat.index_add_(0, flat_indices, grad_v_sparse.reshape(-1, d))
                
                grad_k = grad_k_flat.view(B, H, S, d)
                grad_v = grad_v_flat.view(B, H, S, d)
            
        return grad_k, grad_v, None
 
def mps_coordinate_gather_scatter(q, k, v, indices):
    device = k.device
    if device.type == "mps":
        idx_type = torch.int32
    else:
        idx_type = torch.long
        
    # Win 39: Cast indices and check contiguity only if needed to avoid redundant copy overhead
    if indices.device != device or indices.dtype != idx_type:
        indices = indices.to(device=device, dtype=idx_type)
    if not indices.is_contiguous():
        indices = indices.contiguous()

    if not torch.is_grad_enabled():
        B, H, S, d = k.shape
        
        if device.type == "mps":
            if not k.is_contiguous():
                k = k.contiguous()
            if not v.is_contiguous():
                v = v.contiguous()
                
        is_shared = False
        if indices.dim() == 1:
            is_shared = True
            idx = indices
        elif indices.size(0) == 1:
            is_shared = True
            idx = indices[0]
        else:
            is_shared = False
            
        if is_shared:
            k_3d = k.reshape(-1, S, d)
            v_3d = v.reshape(-1, S, d)
            K_sparse_3d = torch.index_select(k_3d, 1, idx)
            V_sparse_3d = torch.index_select(v_3d, 1, idx)
            return K_sparse_3d.reshape(B, H, -1, d), V_sparse_3d.reshape(B, H, -1, d)
        else:
            if B == 1:
                # Win 100: B=1 fast-path bypass
                K_sparse = torch.index_select(k[0], 1, indices[0]).unsqueeze(0)
                V_sparse = torch.index_select(v[0], 1, indices[0]).unsqueeze(0)
                return K_sparse, V_sparse
            else:
                K_len = indices.shape[-1]
                offsets = torch.arange(B * H, device=device).view(B, H, 1) * S
                flat_indices = (indices.unsqueeze(1) + offsets).view(-1)
                K_sparse = torch.index_select(k.reshape(-1, d), 0, flat_indices).view(B, H, K_len, d)
                V_sparse = torch.index_select(v.reshape(-1, d), 0, flat_indices).view(B, H, K_len, d)
                return K_sparse, V_sparse
            
    return MPSCoordinateGatherScatter.apply(k, v, indices)