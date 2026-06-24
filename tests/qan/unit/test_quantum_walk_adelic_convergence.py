import pytest
import torch
import torch.nn as nn
import numpy as np
from qan_transformers.optim.adelic import QuantumWalkAdelicOptimizer

class DummyLoRAModel(nn.Module):
    def __init__(self, in_features=16, out_features=16, r=4):
        super().__init__()
        # Base parameters (non-submanifold)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA parameters (submanifold)
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.scaling = 2.0
        
    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

@pytest.mark.parametrize("coin_type", ["hadamard", "grover"])
def test_dummy_lora_convergence_and_submanifold_handling(coin_type):
    """
    Empirically verify that QuantumWalkAdelicOptimizer successfully converges
    on a dummy LoRA model and handles non-submanifold parameters correctly.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    in_features = 8
    out_features = 8
    model = DummyLoRAModel(in_features, out_features, r=2)
    named_params = list(model.named_parameters())
    
    # Generate some dummy data
    X = torch.randn(20, in_features)
    Y_target = torch.randn(20, out_features)
    
    optimizer = QuantumWalkAdelicOptimizer(
        model.parameters(),
        named_parameters=named_params,
        coin_type=coin_type,
        lr=0.01,
        damping_rate=0.1,
        T_0=0.001
    )
    
    # Track initial losses and losses over time
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = nn.functional.mse_loss(Y_pred, Y_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    # 1. Convergence verification: Final loss must be less than 50% of the initial loss
    assert losses[-1] < losses[0] * 0.5, f"Optimizer {coin_type} failed to converge: {losses[0]} -> {losses[-1]}"
    
    # 2. Submanifold vs non-submanifold handling verification
    for name, p in named_params:
        state = optimizer.state[p]
        is_sub = any(pat in name for pat in ["lora_A", "lora_B", "lora_a", "lora_b", "e8_proj"])
        
        assert len(state) > 0, f"Parameter {name} has no state after stepping!"
        
        if is_sub:
            # Submanifold parameters must get quantum walk specific state keys
            assert "rho_c" in state, f"Submanifold parameter {name} should have 'rho_c' in state"
            assert "x_p" in state, f"Submanifold parameter {name} should have 'x_p' in state"
        else:
            # Non-submanifold parameters must NOT get quantum walk specific state keys
            assert "rho_c" not in state, f"Non-submanifold parameter {name} should NOT have 'rho_c' in state"
            assert "x_p" not in state, f"Non-submanifold parameter {name} should NOT have 'x_p' in state"
