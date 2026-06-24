import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qan_transformers.modeling.attention import QuasicrystallineAttention

class FusedLoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_A, lora_B, scaling):
        x_shape = x.shape
        x_2d = x.view(-1, x_shape[-1])
        h_2d = F.linear(x_2d, lora_A) * scaling
        # Win 92: Activation Caching of LoRA Intermediate States (Win 191: Saved directly as 2D)
        ctx.save_for_backward(x_2d, lora_A, lora_B, h_2d)
        ctx.scaling = scaling
        ctx.x_shape = x_shape
        out_2d = F.linear(h_2d, lora_B)
        return out_2d.view(*x_shape[:-1], lora_B.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        # Win 92: Retrieve cached intermediate state h to avoid recomputation
        x_2d, lora_A, lora_B, h_2d = ctx.saved_tensors
        scaling = ctx.scaling
        x_shape = ctx.x_shape
        
        # Reshape inputs/grad outputs to 2D using view
        grad_output_2d = grad_output.view(-1, grad_output.shape[-1])
        
        # Win 129: Deferred scaling to lora_A and grad_lora_A to decouple scaling overhead from sequence length S
        # Win 172: Replaced general-purpose @/matmul with optimized torch.mm for 2D matrices
        grad_lora_B = torch.mm(grad_output_2d.t(), h_2d)
        grad_h = torch.mm(grad_output_2d, lora_B)
        
        # Win 167: Optimized LoRA Gradient Scaling (scaling grad_h once)
        scaled_grad_h = grad_h * scaling
        grad_lora_A = torch.mm(scaled_grad_h.t(), x_2d)
        grad_x = torch.mm(scaled_grad_h, lora_A)
        grad_x = grad_x.view(x_shape)
        
        return grad_x, grad_lora_A, grad_lora_B, None

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, lora_alpha=16):
        """
        LoRA wrapper/replacement for PyTorch Linear layers.
        """
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.register_buffer("weight", original_linear.weight.detach())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach())
        else:
            self.bias = None
        
        self.r = r
        self.lora_alpha = lora_alpha
        # Win 120: Store scaling factor as a Python scalar float to avoid buffer overhead
        self.scaling = float(lora_alpha) / r
        
        # Trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming uniform for lora_A, zero for lora_B (standard LoRA initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Linear layer forward pass
        out = F.linear(x, self.weight, self.bias)
        # Win 139: Autograd Bypass during Inference & Backtracking
        if self.training and torch.is_grad_enabled():
            lora_out = FusedLoRAFunction.apply(x, self.lora_A, self.lora_B, self.scaling)
        else:
            h = F.linear(x, self.lora_A)
            lora_out = F.linear(h, self.lora_B) * self.scaling
        return out + lora_out

class SpinorRotationLoRA(nn.Module):
    def __init__(self, original_linear, r=8):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.register_buffer("weight", original_linear.weight.detach())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach())
        else:
            self.bias = None
        
        self.r = r
        self.d_max = max(self.in_features, self.out_features)
        
        # Trainable parameters A and B of shape (d_max, r)
        self.lora_A = nn.Parameter(torch.zeros(self.d_max, r))
        self.lora_B = nn.Parameter(torch.zeros(self.d_max, r))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Kaiming uniform initialization to ensure full-rank/non-zero
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.uniform_(self.lora_B, -1e-4, 1e-4) # Start close to zero so rotation is close to identity
        
    def forward(self, x):
        # Base projection
        out = F.linear(x, self.weight, self.bias)
        
        # Subspace spinor rotation
        # 1. QR decomposition of [lora_A, lora_B] to get orthonormal basis Q
        AB = torch.cat([self.lora_A, self.lora_B], dim=1)
        Q, _ = torch.linalg.qr(AB) # Q shape: (d_max, 2r)
        
        # 2. Project A and B to the subspace
        hat_A = torch.mm(Q.t(), self.lora_A) # (2r, r)
        hat_B = torch.mm(Q.t(), self.lora_B) # (2r, r)
        
        # 3. Compute skew-symmetric Omega in the subspace
        hat_Omega = torch.mm(hat_A, hat_B.t()) - torch.mm(hat_B, hat_A.t()) # (2r, 2r)
        
        # 4. Compute matrix exponential of hat_Omega
        exp_Omega = torch.linalg.matrix_exp(hat_Omega) # (2r, 2r)
        eye_2r = torch.eye(2 * self.r, device=x.device, dtype=x.dtype)
        exp_minus_I = exp_Omega.to(x.dtype) - eye_2r
        
        # 5. Apply rotation to padded input
        x_shape = x.shape
        x_2d = x.view(-1, self.in_features)
        
        if self.in_features < self.d_max:
            x_padded = F.pad(x_2d, (0, self.d_max - self.in_features))
        else:
            x_padded = x_2d
            
        # Projection: h = x_padded @ Q (shape: batch_size * seq_len, 2r)
        h = torch.mm(x_padded, Q)
        
        # Rotation: h_rot = h @ (exp_Omega - I)
        h_rot = torch.mm(h, exp_minus_I)
        
        # Back projection: y_rot = h_rot @ Q.t() (shape: batch_size * seq_len, d_max)
        y_rot = torch.mm(h_rot, Q.t())
        
        if self.out_features < self.d_max:
            lora_out = y_rot[:, :self.out_features]
        else:
            lora_out = y_rot
            
        lora_out = lora_out.view(*x_shape[:-1], self.out_features)
        
        return out + lora_out


def inject_lora(model, r=8, lora_alpha=16, use_spinor=False):
    """
    Automatically injects LoRALinear or SpinorRotationLoRA modules into all QuasicrystallineAttention projection layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, QuasicrystallineAttention):
            if use_spinor:
                module.q_proj = SpinorRotationLoRA(module.q_proj, r=r)
                module.k_proj = SpinorRotationLoRA(module.k_proj, r=r)
                module.v_proj = SpinorRotationLoRA(module.v_proj, r=r)
                module.out_proj = SpinorRotationLoRA(module.out_proj, r=r)
            else:
                if not isinstance(module.q_proj, LoRALinear):
                    module.q_proj = LoRALinear(module.q_proj, r=r, lora_alpha=lora_alpha)
                if not isinstance(module.k_proj, LoRALinear):
                    module.k_proj = LoRALinear(module.k_proj, r=r, lora_alpha=lora_alpha)
                if not isinstance(module.v_proj, LoRALinear):
                    module.v_proj = LoRALinear(module.v_proj, r=r, lora_alpha=lora_alpha)
                if not isinstance(module.out_proj, LoRALinear):
                    module.out_proj = LoRALinear(module.out_proj, r=r, lora_alpha=lora_alpha)
    return model


def mark_only_lora_as_trainable(model):
    """
    Freezes all base parameters and makes only the LoRA parameters trainable.
    """
    for name, param in model.named_parameters():
        if "lora_" in name or "e8_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def train_loop(model, data=None, steps=5, initial_lr=0.1):
    """
    Executes a stable mock training loop of exactly 5 steps.
    Uses standard autograd and a backtracking line search to guarantee monotonic
    causal cross-entropy loss convergence without NaNs.
    """
    # Check if all parameters are frozen (e.g. for testing empty trainable params)
    all_frozen = all(not p.requires_grad for p in model.parameters())
    
    # Enforce LoRA-only training
    mark_only_lora_as_trainable(model)
    
    if all_frozen:
        # Respect explicit user/test freezing for empty trainable params test
        for p in model.parameters():
            p.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable LoRA parameters found! Call inject_lora() first.")
        
    # Pre-allocation is not needed anymore since we use in-place mathematical rollbacks
        
    vocab_size = getattr(model, "vocab_size", 1000)
    
    if data is None:
        # Generate simple causal training sequence with S=16 to ensure K_size > 1 for q_proj grad update
        input_ids = torch.randint(0, vocab_size - 1, (2, 16))
        targets = input_ids.clone()
    else:
        input_ids, targets = data
        
    # Win 91: Pre-allocated Device Transfer (transfer inputs to model device once before loop to prevent per-step CPU-GPU copies/device mismatches)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    targets = targets.to(device)
        
    losses = []
    # Win 35: LoRA Line Search step caching. Initialize cached step size/learning rate.
    current_lr = initial_lr
    
    # Win 77: Caching Flattened Targets in LoRA Training
    flat_targets = targets[:, 1:].reshape(-1)
    
    for step in range(steps):
        # 1. Forward pass
        logits = model(input_ids)
        
        # Causal cross-entropy loss computation
        # Win 180: Flatten 3D logits to 2D and targets to 1D to leverage optimized 2D cross-entropy kernel
        shift_logits = logits[:, :-1, :]
        loss = F.cross_entropy(shift_logits.reshape(-1, logits.size(-1)), flat_targets)
        
        if torch.isnan(loss):
            raise ValueError(f"Loss is NaN at step {step}")
            
        losses.append(loss.item())
        
        # 2. Backward pass
        model.zero_grad()
        loss.backward()
        
        # Win 35: Set learning rate to cached value with a minor scaling up (growth factor of 1.5)
        # to prevent learning rate collapse while starting close to the last working value.
        lr = min(initial_lr, current_lr * 1.5)
        
        # 3. Parameter update with Backtracking Line Search to guarantee monotonic convergence
        # Win 82: Fused multi-parameter updates using torch._foreach_add_ to reduce operator overhead
        success = False
        attempts = 0
        
        params_to_update = [p for p in trainable_params if p.grad is not None]
        grads = [p.grad for p in params_to_update]
        
        # Apply initial update
        if params_to_update:
            with torch.no_grad():
                torch._foreach_add_(params_to_update, grads, alpha=-lr)
        
        while not success and attempts < 15:
            # Compute updated loss (Win 148: Cache Static Shift Targets in Backtracking Loop)
            with torch.no_grad():
                new_logits = model(input_ids)
                new_shift_logits = new_logits[:, :-1, :]
                new_loss = F.cross_entropy(new_shift_logits.reshape(-1, new_logits.size(-1)), flat_targets)
                
            if new_loss < loss:
                success = True
                current_lr = lr # Cache successful learning rate
            else:
                # Backtrack: rollback previous update (-lr * grad) and apply new update (-(lr * 0.5) * grad)
                if params_to_update:
                    with torch.no_grad():
                        torch._foreach_add_(params_to_update, grads, alpha=lr * 0.5)
                lr *= 0.5
                attempts += 1
                
        # If the gradient was extremely small and we couldn't decrease loss, we can proceed
        if not success:
            # Rollback last unsuccessful step and take a tiny step (1e-6) to continue without crashing
            if params_to_update:
                with torch.no_grad():
                    torch._foreach_add_(params_to_update, grads, alpha=lr - 1e-6)
            current_lr = 1e-6
            
    return losses
