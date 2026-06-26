import torch
import pytest
from qan_transformers.modeling.persistent_homology_loss import (
    SoftRipsComplex,
    PersistenceLandscapeVectorizer,
    PersistentHomologyLoss,
)

def test_soft_rips_complex_differentiable():
    # Test that SoftRipsComplex computes diagrams and allows gradient flow
    rips = SoftRipsComplex(max_filtration=2.0)
    
    # 2 batches, 6 vertices
    skeleton = torch.randn(2, 6, 6, requires_grad=True)
    h0_diags, h1_diags = rips.forward(skeleton)
    
    assert len(h0_diags) == 2
    assert len(h1_diags) == 2
    
    # Verify elements in diagrams
    for b in range(2):
        h0_b, h0_d = h0_diags[b]
        h1_b, h1_d = h1_diags[b]
        
        # Check shapes
        # H0 has K components (vertices), one is infinite (retains max_filtration)
        # So it should have 6 entries
        assert h0_b.shape == (6,)
        assert h0_d.shape == (6,)
        
        # H1 loops depend on edges that did not cause merge
        # Total edges = 6 * 5 / 2 = 15. K-1 merges = 5.
        # So remaining 10 edges create loops.
        assert h1_b.shape == (10,)
        assert h1_d.shape == (10,)
        
        # Verify gradient flow
        loss = h0_d.sum() + h1_b.sum()
        loss.backward(retain_graph=True)
        assert skeleton.grad is not None
        assert not torch.isnan(skeleton.grad).any()
        assert (skeleton.grad != 0.0).any()
        skeleton.grad.zero_()


def test_persistence_landscape_vectorizer():
    vectorizer = PersistenceLandscapeVectorizer(num_landscapes=5, resolution=50, max_filtration=2.0)
    
    # Sample birth/death times
    births = torch.tensor([0.1, 0.2, 0.5], requires_grad=True)
    deaths = torch.tensor([0.8, 1.2, 0.9], requires_grad=True)
    
    landscapes = vectorizer(births, deaths)
    assert landscapes.shape == (5, 50)
    assert not torch.isnan(landscapes).any()
    
    # Backward pass
    loss = landscapes.sum()
    loss.backward()
    
    assert births.grad is not None
    assert deaths.grad is not None
    assert (births.grad != 0.0).any()
    assert (deaths.grad != 0.0).any()


def test_persistent_homology_loss_gradient_flow():
    ph_loss = PersistentHomologyLoss(h0_weight=1.0, h1_weight=0.5, ref_ema_decay=0.9)
    
    # Pre-pass with detached skeleton to initialize reference
    skeleton_init = torch.randn(2, 6, 6)
    _ = ph_loss(skeleton_init)
    
    # Now verify gradient flow on a new skeleton
    skeleton = torch.randn(2, 6, 6, requires_grad=True)
    
    # Forward pass
    loss = ph_loss(skeleton)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    
    # Backward pass
    loss.backward()
    
    assert skeleton.grad is not None
    assert not torch.isnan(skeleton.grad).any()
    assert (skeleton.grad != 0.0).any()


def test_ema_updates():
    ph_loss = PersistentHomologyLoss(h0_weight=1.0, h1_weight=1.0, ref_ema_decay=0.9)
    
    # Verify initial state
    assert not ph_loss.has_ref.item()
    
    # Batch 1
    skeleton_1 = torch.randn(2, 5, 5)
    loss_1 = ph_loss(skeleton_1)
    
    # After first batch, reference landscapes should be initialized
    assert ph_loss.has_ref.item()
    ref_h0_init = ph_loss.ref_landscape_h0.clone()
    ref_h1_init = ph_loss.ref_landscape_h1.clone()
    assert not torch.allclose(ref_h0_init, torch.zeros_like(ref_h0_init))
    
    # Batch 2 in training mode
    ph_loss.train()
    skeleton_2 = torch.randn(2, 5, 5)
    loss_2 = ph_loss(skeleton_2)
    
    # Reference landscapes should update via EMA
    ref_h0_updated = ph_loss.ref_landscape_h0.clone()
    ref_h1_updated = ph_loss.ref_landscape_h1.clone()
    assert not torch.allclose(ref_h0_updated, ref_h0_init)
    assert not torch.allclose(ref_h1_updated, ref_h1_init)
    
    # Batch 3 in eval mode
    ph_loss.eval()
    skeleton_3 = torch.randn(2, 5, 5)
    loss_3 = ph_loss(skeleton_3)
    
    # Reference landscapes should not change in eval mode
    ref_h0_eval = ph_loss.ref_landscape_h0.clone()
    assert torch.allclose(ref_h0_eval, ref_h0_updated)
