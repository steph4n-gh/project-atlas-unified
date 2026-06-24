import pytest
import mlx.core as mx
import numpy as np
import torch
from qan_transformers.mlx.attention import (
    DiracSpinorAttention,
    FibonacciQuasicrystallineEncoding,
    AperiodicPenroseMasking,
    LegendrePolynomialSoftmax,
    ThermodynamicAttention
)
from qan_transformers.mlx.e8_swap import (
    CellularAutomataCompactor,
    QuantumWalkMetropolisSearch,
    DNASelfAssemblingLookup,
    FloquetResonanceDoping,
    SlimeMoldPaging
)
from qan_transformers.mlx.modeling import (
    GaloisRingGR4dWeights,
    NonCommutativeBraidBisectionRouter,
    SheafRestrictiveGating,
    ReactionDiffusionSemanticGating,
    SymplecticFourierMLP
)
from qan_transformers.optim.adelic import (
    LieGroupSymplecticOptimizer,
    AdelicFeynmanPathOptimizer
)
from qan_transformers.firewall.cohomology import CohomologicalLayerGating
from qan_transformers.mlx.wave_solver import NeuralDendriticBranchSpeculation

# --- Attention Tests ---

def test_dirac_spinor_attention():
    # embed_dim=32, num_heads=2 -> head_dim=16 (divisible by 4)
    attn = DiracSpinorAttention(embed_dim=32, num_heads=2)
    x = mx.random.normal((1, 10, 32))
    out = attn(x)
    assert out.shape == (1, 10, 32)
    assert not mx.any(mx.isnan(out)).item()

def test_fibonacci_quasicrystalline_encoding():
    encoding = FibonacciQuasicrystallineEncoding(dims=16)
    x = mx.random.normal((1, 20, 16))
    out = encoding(x)
    assert out.shape == (1, 20, 16)
    assert not mx.any(mx.isnan(out)).item()

def test_aperiodic_penrose_masking():
    masking = AperiodicPenroseMasking(size=128)
    mask = masking.get_mask(q_len=10, k_len=10)
    assert mask.shape == (10, 10)
    # Check that mask values are either 0.0 or -1e9
    assert mx.all((mask == 0.0) | (mask == -1e9)).item()

def test_legendre_polynomial_softmax():
    softmax = LegendrePolynomialSoftmax()
    logits = mx.random.normal((2, 5))
    probs = softmax(logits)
    assert probs.shape == (2, 5)
    assert mx.allclose(mx.sum(probs, axis=-1), mx.ones((2,)), atol=1e-4).item()

def test_thermodynamic_attention():
    attn = ThermodynamicAttention(embed_dim=32, num_heads=2, decay_rate=0.5, cold_threshold=0.5)
    x = mx.random.normal((1, 5, 32))
    
    class MockCache:
        def __init__(self):
            self.keys = None
            self.values = None
            
    cache = MockCache()
    out1 = attn(x, cache=cache)
    # Feed input again to trigger thermodynamic cooling and compaction in cache
    x2 = mx.random.normal((1, 3, 32))
    out2 = attn(x2, cache=cache)
    
    assert out1.shape == (1, 5, 32)
    assert out2.shape == (1, 3, 32)
    assert cache.keys is not None
    assert cache.keys.shape[2] > 0

# --- Swap DB Tests ---

def test_cellular_automata_compactor():
    compactor = CellularAutomataCompactor(size=8)
    states = mx.array([True, False, True, False, True, False, True, False])
    next_states = compactor.compact(states)
    assert next_states.shape == (8,)
    assert next_states.dtype == mx.bool_

def test_quantum_walk_metropolis_search():
    search = QuantumWalkMetropolisSearch(size=240)
    psi = mx.zeros((240,))
    psi[10] = 1.0 # start at state 10
    psi_walked = search.walk(psi, steps=2)
    assert psi_walked.shape == (240,)
    assert mx.allclose(mx.sum(psi_walked), mx.array(1.0), atol=1e-4).item()

def test_dna_self_assembling_lookup():
    lookup = DNASelfAssemblingLookup(size=4)
    # DNA complementary: A-T (00 ^ 11 = 3), C-G (01 ^ 10 = 3)
    q_dna = mx.array([[[[0, 1, 2, 3]]]], dtype=mx.uint8) # [1, 1, 1, 4]
    db_dna = mx.array([[3, 2, 1, 0]], dtype=mx.uint8)   # [1, 4] (complementary)
    scores = lookup.hybridize(q_dna, db_dna)
    assert scores.shape == (1, 1, 1, 1)
    assert scores.item() == 4 # perfect match

def test_floquet_resonance_doping():
    doping = FloquetResonanceDoping(d_model=16, omega=10.0)
    coords = mx.random.normal((10, 8))
    modulated = doping.modulate(coords, time_step=5)
    assert modulated.shape == (10, 8)
    assert not mx.allclose(modulated, coords).item()

def test_slime_mold_paging():
    paging = SlimeMoldPaging(capacity=50, decay=0.1)
    active = mx.array([2, 5, 12, 43], dtype=mx.int32)
    paging.update_flux(active, capacity_len=50)
    assert paging.flux[2].item() > 0.0
    assert paging.flux[3].item() == 0.0
    mask = paging.prune_inactive(threshold=0.05)
    assert mask.shape == (50,)
    assert mask[2].item() is True
    assert mask[3].item() is False

# --- Modeling & Routing Tests ---

def test_galois_ring_gr4d_weights():
    gr = GaloisRingGR4dWeights(in_features=16, out_features=16)
    W = gr.unpack()
    assert W.shape == (16, 16)
    assert not mx.any(mx.isnan(W)).item()

def test_non_commutative_braid_bisection_router():
    router = NonCommutativeBraidBisectionRouter(embed_dim=32, num_experts=4, depth=4)
    x = mx.random.normal((2, 5, 32))
    probs = router(x)
    assert probs.shape == (2, 5, 4)
    assert mx.allclose(mx.sum(probs, axis=-1), mx.ones((2, 5)), atol=1e-4).item()

def test_sheaf_restrictive_gating():
    gating = SheafRestrictiveGating(embed_dim=32, num_experts=3)
    x = mx.random.normal((2, 5, 32))
    probs = gating(x)
    assert probs.shape == (2, 5, 3)
    assert mx.allclose(mx.sum(probs, axis=-1), mx.ones((2, 5)), atol=1e-4).item()

def test_reaction_diffusion_gating():
    gating = ReactionDiffusionSemanticGating(embed_dim=16)
    x = mx.random.normal((2, 5, 16))
    gate_mask, u = gating(x)
    assert gate_mask.shape == (2, 5, 1)
    assert u.shape == (2, 5, 1)
    assert gate_mask.dtype == mx.bool_

def test_symplectic_fourier_mlp():
    mlp = SymplecticFourierMLP(embed_dim=32, hidden_dim=64)
    x = mx.random.normal((2, 10, 32))
    out = mlp(x)
    assert out.shape == (2, 10, 32)
    assert not mx.any(mx.isnan(out)).item()

# --- Optimizer Tests ---

def test_lie_group_symplectic_optimizer():
    linear = torch.nn.Linear(16, 8)
    optimizer = LieGroupSymplecticOptimizer(linear.parameters(), lr=1e-2)
    
    # Run one step
    x = torch.randn(2, 16)
    loss = linear(x).sum()
    loss.backward()
    
    optimizer.step()
    assert linear.weight.grad is not None

def test_adelic_feynman_path_optimizer():
    linear = torch.nn.Linear(8, 4)
    optimizer = AdelicFeynmanPathOptimizer(linear.parameters(), lr=1e-2, primes=[2, 3])
    
    x = torch.randn(2, 8)
    loss = linear(x).sum()
    loss.backward()
    
    optimizer.step()
    assert linear.weight.grad is not None

# --- Firewall & Speculation Tests ---

def test_cohomological_layer_gating():
    gating = CohomologicalLayerGating(threshold=0.5)
    x1 = mx.random.normal((1, 5, 16))
    x2 = x1 + mx.random.normal((1, 5, 16)) * 0.1 # close representation
    x3 = x1 + mx.random.normal((1, 5, 16)) * 2.0 # distinct representation
    
    assert gating.should_exit_early(x2, x1) is True
    assert gating.should_exit_early(x3, x1) is False

def test_neural_dendritic_branch_speculation():
    spec = NeuralDendriticBranchSpeculation(branching_factor=4, depth=4)
    path_logits = mx.random.normal((1, 4, 6, 32)) # B=1, paths=4, len=6, vocab=32
    potentials = spec.integrate_potentials(path_logits)
    assert potentials.shape == (1, 4)
    assert not mx.any(mx.isnan(potentials)).item()
