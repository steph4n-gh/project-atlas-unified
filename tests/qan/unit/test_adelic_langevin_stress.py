import pytest
import torch
import torch.nn as nn
import numpy as np
from qan_transformers.optim.adelic import AdelicLangevinOptimizer

def test_nan_gradients_adelic():
    """
    Test that NaN gradients on AdelicLangevinOptimizer are cleaned and do not cause NaNs.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    opt = AdelicLangevinOptimizer([p], lr=0.1)
    
    p.grad = torch.tensor([[float('nan'), -0.5], [1.0, float('nan')]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    assert torch.isnan(p.grad).sum() == 0
    assert torch.allclose(p.grad, torch.tensor([[0.0, -0.5], [1.0, 0.0]]))
    assert torch.isnan(p_after).sum() == 0
    assert torch.isinf(p_after).sum() == 0
    assert not torch.equal(p_before, p_after)

def test_infinite_gradients_adelic():
    """
    Test that infinite gradients on AdelicLangevinOptimizer are cleaned and do not cause NaNs/Infs.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    opt = AdelicLangevinOptimizer([p], lr=0.1)
    
    p.grad = torch.tensor([[float('inf'), -0.5], [1.0, float('-inf')]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    assert torch.isinf(p.grad).sum() == 0
    assert torch.allclose(p.grad, torch.tensor([[0.0, -0.5], [1.0, 0.0]]))
    assert torch.isnan(p_after).sum() == 0
    assert torch.isinf(p_after).sum() == 0
    assert not torch.equal(p_before, p_after)

def test_zero_learning_rate_adelic():
    """
    Test that a learning rate of 0.0 prevents updates on AdelicLangevinOptimizer.
    """
    p = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True))
    opt = AdelicLangevinOptimizer([p], lr=0.0)
    
    p.grad = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    
    p_before = p.clone().detach()
    opt.step()
    p_after = p.clone().detach()
    
    assert torch.equal(p_before, p_after)
    assert len(opt.state[p]) == 0
