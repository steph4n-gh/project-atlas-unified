import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ultrametric_ce.diffusion import UltrametricDiffusion
from ultrametric_ce.distillation import topological_distance_loss, build_toy_arithmetic_tree
from ultrametric_ce.tree import FiniteTree

def test_schwarzschild_warp_contraction():
    p = 3
    depth = 2
    dim = 4
    
    diff_warp = UltrametricDiffusion(
        p=p, depth=depth, dim=dim, num_layers=1, alpha=0.5,
        schwarzschild_warp=True, r_s=1.0, wormhole_gate=False
    )
    
    diff_no_warp = UltrametricDiffusion(
        p=p, depth=depth, dim=dim, num_layers=1, alpha=0.5,
        schwarzschild_warp=False, wormhole_gate=False
    )
    
    # Sync weights to make them identical
    for w_mix, nw_mix in zip(diff_warp.mix_linears, diff_no_warp.mix_linears):
        nw_mix.weight = w_mix.weight
        nw_mix.bias = w_mix.bias
        
    states = {
        (0, 0): mx.array([0.0, 0.0, 0.0, 0.0]),
        (1, 0): mx.array([1.0, 1.0, 1.0, 1.0]),  # Parent at depth 1
        (2, 0): mx.array([1.2, 1.2, 1.2, 1.2]),  # Child 0 (close to parent)
        (2, 1): mx.array([2.5, 2.5, 2.5, 2.5]),  # Child 1 (far from parent)
    }
    
    out_warp = diff_warp(states)
    out_no_warp = diff_no_warp(states)
    
    # Contraction output should differ from baseline
    assert not mx.allclose(out_warp[(2, 0)], out_no_warp[(2, 0)])
    assert not mx.allclose(out_warp[(2, 1)], out_no_warp[(2, 1)])

def test_einstein_rosen_wormholes():
    p = 3
    depth = 2
    dim = 4
    
    diff_wormhole = UltrametricDiffusion(
        p=p, depth=depth, dim=dim, num_layers=1, alpha=0.5,
        schwarzschild_warp=False, wormhole_gate=True, epsilon=0.5
    )
    
    # States with high correlation (cos_sim > 0.85) at disjoint branches
    states = {
        (2, 0): mx.array([1.0, 0.0, 0.0, 0.0]),
        (2, 2): mx.array([0.9, 0.1, 0.0, 0.0]),
    }
    
    out = diff_wormhole(states)
    
    # Disjoint branches should mix feature info via the wormhole shortcut
    assert not mx.allclose(out[(2, 0)], states[(2, 0)])
    assert not mx.allclose(out[(2, 2)], states[(2, 2)])

def test_topological_loss_warped():
    tree, _, _, _ = build_toy_arithmetic_tree()
    
    dim = 4
    diffused = {
        (tree.depth, 0): mx.array([1.0, 0.0, 0.0, 0.0]),
        (tree.depth, 1): mx.array([0.0, 1.0, 0.0, 0.0]),
    }
    
    loss_val = topological_distance_loss(diffused, [0, 1], tree, alpha=0.1, r_s=0.5)
    assert loss_val.ndim == 0
    assert not mx.isnan(loss_val)
    
    # Optimization step stability check
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = mx.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        def __call__(self):
            return {
                (tree.depth, 0): self.param[0],
                (tree.depth, 1): self.param[1],
            }
            
    m = ToyModel()
    def loss_fn(model):
        diff = model()
        return topological_distance_loss(diff, [0, 1], tree, alpha=0.1, r_s=0.5)
        
    loss_init = loss_fn(m)
    loss_val, grads = nn.value_and_grad(m, loss_fn)(m)
    m.param = m.param - 0.1 * grads["param"]
    loss_after = loss_fn(m)
    
    assert loss_after < loss_init
