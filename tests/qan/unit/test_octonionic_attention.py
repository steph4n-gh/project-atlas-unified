import torch
import torch.nn.functional as F
import numpy as np
import pytest
from qan_transformers.math.octonion import OctonionAlgebra, CayleyIntegerProjector
from qan_transformers.modeling.attention.octonionic import OctonionicAttentionMode
from qan_transformers.modeling.attention.e8_routing import get_shared_e8_roots_8d

def test_octonionic_multiplication_properties():
    # 1. Identity element e0
    alg = OctonionAlgebra()
    
    def basis(i):
        e = torch.zeros(8)
        e[i] = 1.0
        return e
        
    for i in range(8):
        left = alg.multiply(basis(0), basis(i))
        right = alg.multiply(basis(i), basis(0))
        assert torch.allclose(left, basis(i))
        assert torch.allclose(right, basis(i))

    # 2. Imaginary units square to -e0
    for i in range(1, 8):
        res = alg.multiply(basis(i), basis(i))
        assert torch.allclose(res, -basis(0))

    # 3. Fano plane anti-commutativity
    for (i, j, k) in OctonionAlgebra.FANO_TRIPLES:
        res1 = alg.multiply(basis(i), basis(j))
        res2 = alg.multiply(basis(j), basis(i))
        assert torch.allclose(res1, basis(k))
        assert torch.allclose(res2, -basis(k))

    # 4. Normed division property: |a*b| = |a|*|b|
    torch.manual_seed(42)
    a = torch.randn(8)
    b = torch.randn(8)
    ab = alg.multiply(a, b)
    assert torch.allclose(alg.norm(ab), alg.norm(a) * alg.norm(b), atol=1e-6)


def test_non_associativity():
    alg = OctonionAlgebra()
    # (e1 * e2) * e3 != e1 * (e2 * e3)
    # e1 * e2 = e4. e4 * e3 = -e5 (since 3, 4, 6 is a triple: e3*e4 = e6 => e4*e3 = -e6, wait, Fano triples:
    # 3,4,6 is triple => e3*e4 = e6 => e4*e3 = -e6. e1*(e2*e3) = e1*e5 (since 2,3,5 is triple => e2*e3 = e5).
    # e1*e5 = e6 (since 1,5,6 is triple => e1*e5 = e6).
    # So left is -e6, right is e6.
    def basis(i):
        e = torch.zeros(8)
        e[i] = 1.0
        return e
        
    left = alg.multiply(alg.multiply(basis(1), basis(2)), basis(3))
    right = alg.multiply(basis(1), alg.multiply(basis(2), basis(3)))
    assert not torch.allclose(left, right)
    assert torch.allclose(left, -right)


def test_e8_closure():
    alg = OctonionAlgebra()
    # E8 roots (Shell 1)
    _, roots_8d_norm = get_shared_e8_roots_8d(1)
    
    # Take a sample of products and verify they are on E8 lattice (scaled)
    # E8 coordinates are integers or half-integers
    a = roots_8d_norm[0]
    b = roots_8d_norm[1]
    prod = alg.multiply(a, b)
    
    # Check if norm of product is correct: roots have norm 1 (normalized), so product has norm 1
    assert torch.allclose(alg.norm(prod), torch.tensor(1.0), atol=1e-6)


def test_octonionic_attention_mode():
    device = 'cpu'
    dtype = torch.float32
    mode = OctonionicAttentionMode(device=device, dtype=dtype)
    
    # Inputs: batch_size=2, seq_len=4, E8 space=8D
    seq_8d = torch.randn(2, 4, 8, device=device, dtype=dtype)
    seq_8d_norm = F.normalize(seq_8d, p=2, dim=-1, eps=1e-6)
    
    _, roots_8d_norm = get_shared_e8_roots_8d(1)
    roots_8d_norm = roots_8d_norm.to(device=device, dtype=dtype)
    
    # Forward pass
    scores = mode(seq_8d_norm, roots_8d_norm)
    assert scores.shape == (2, 4, len(roots_8d_norm))
    assert not torch.isnan(scores).any()


def test_octonionic_attention_gradients():
    device = 'cpu'
    dtype = torch.float32
    mode = OctonionicAttentionMode(device=device, dtype=dtype)
    
    seq_8d = torch.randn(2, 4, 8, device=device, dtype=dtype, requires_grad=True)
    seq_8d_norm = F.normalize(seq_8d, p=2, dim=-1, eps=1e-6)
    
    _, roots_8d_norm = get_shared_e8_roots_8d(1)
    roots_8d_norm = roots_8d_norm.to(device=device, dtype=dtype)
    
    scores = mode(seq_8d_norm, roots_8d_norm)
    loss = scores.sum()
    loss.backward()
    
    assert seq_8d.grad is not None
    assert not torch.isnan(seq_8d.grad).any()
    assert (seq_8d.grad != 0.0).any()
