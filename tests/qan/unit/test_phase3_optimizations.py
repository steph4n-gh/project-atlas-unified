import pytest
import mlx.core as mx
import numpy as np
import torch
from qan_transformers.mlx.attention import (
    SolitonicAttentionWaveguide,
    DiracKählerSpectralCompression,
    ZeroCopyOctonionicAttention,
    AutomorphicModularFormsPooling
)
from qan_transformers.mlx.e8_swap import (
    DNAReassociationCotAnalysis,
    AntColonyPheromoneTrails,
    AdelicAdjointGroupLayout,
    CytoskeletalMicrotubuleRouting
)
from qan_transformers.mlx.modeling import (
    KnotEquivalenceCacheTopology,
    AnyonicBraidingGating,
    AllostericCooperativityGating,
    MyelinShearSparseInsulation,
    TropicalSemiringSpeculation,
    GrothendieckTopologyGating,
    GaloisFieldGF28SIMD
)
from qan_transformers.optim.adelic import (
    OrthogonalProcrustesProjection,
    RelativisticTimeDilationCache
)
from qan_transformers.mlx.wave_solver import (
    AsynchronousParallelSpeculation,
    SymplecticIntegrationSymDRAM,
    StochasticResonanceDithering
)

# --- Attention Optimizations Tests ---

def test_solitonic_attention_waveguide():
    waveguide = SolitonicAttentionWaveguide(dt=0.01, dx=1.0)
    x = mx.random.normal((1, 2, 8, 16))
    out = waveguide(x)
    assert out.shape == (1, 2, 8, 16)
    assert not mx.any(mx.isnan(out)).item()

def test_dirac_kahler_spectral_compression():
    compression = DiracKählerSpectralCompression(keep_ratio=0.5)
    x = mx.random.normal((1, 2, 8, 16))
    out = compression.compress(x)
    assert out.shape == (1, 2, 8, 16)
    assert not mx.any(mx.isnan(out)).item()

def test_zero_copy_octonionic_attention():
    oct_attn = ZeroCopyOctonionicAttention(embed_dim=16)
    q = mx.random.normal((1, 2, 5, 16))
    k = mx.random.normal((1, 2, 5, 16))
    out = oct_attn.multiply(q, k)
    assert out.shape == (1, 2, 5, 16)
    assert not mx.any(mx.isnan(out)).item()

def test_automorphic_modular_forms_pooling():
    pooling = AutomorphicModularFormsPooling(k_weight=2)
    x = mx.random.normal((1, 2, 6, 16))
    out = pooling.pool(x)
    assert out.shape == (1, 2, 6, 16)
    assert not mx.any(mx.isnan(out)).item()

# --- Memory Swapping DB Tests ---

def test_dna_reassociation_cot_analysis():
    cot = DNAReassociationCotAnalysis(k_rate=0.1, c0=1.0)
    t = mx.array([0.0, 1.0, 10.0, 100.0])
    frac = cot.get_reassociation_fraction(t)
    assert frac[0].item() == 1.0
    assert frac[1].item() < 1.0
    assert frac[3].item() > 0.0

def test_ant_colony_pheromone_trails():
    colony = AntColonyPheromoneTrails(num_pages=10, decay=0.1, alpha=1.0)
    visited = mx.array([2, 5, 5])
    colony.update_trail(visited)
    assert colony.pheromones[2].item() > 1.0
    assert colony.pheromones[5].item() > 1.0
    assert colony.pheromones[0].item() < 1.0
    top_3 = colony.get_priority_pages(3)
    assert top_3.shape == (3,)

def test_adelic_adjoint_group_layout():
    layout = AdelicAdjointGroupLayout(prime=2, max_depth=4)
    coords = mx.array([[1, 0, 1, 0], [0, 1, 1, 1]])
    offsets = layout.get_physical_offset(coords)
    assert offsets[0].item() == 5
    assert offsets[1].item() == 14

def test_cytoskeletal_microtubule_routing():
    routing = CytoskeletalMicrotubuleRouting(num_slots=20, velocity=2)
    active = mx.array([2, 10, 19])
    next_active = routing.step(active)
    assert next_active[0].item() == 4
    assert next_active[1].item() == 12
    assert next_active[2].item() == 1

# --- Modeling & Gating Tests ---

def test_knot_equivalence_cache_topology():
    topology = KnotEquivalenceCacheTopology()
    x = mx.random.normal((2, 6, 12))
    crossing = topology.compute_crossing_number(x)
    assert crossing.shape == (2,)
    assert not mx.any(mx.isnan(crossing)).item()

def test_anyonic_braiding_gating():
    gating = AnyonicBraidingGating(num_experts=4)
    x = mx.random.normal((2, 5, 8))
    routes = gating.route(x)
    assert routes.shape == (2, 5)
    assert mx.min(routes).item() >= 0
    assert mx.max(routes).item() < 4

def test_allosteric_cooperativity_gating():
    gating = AllostericCooperativityGating(hill_n=4.0, k_half=0.5)
    scores = mx.array([0.1, 0.4, 0.6, 0.9])
    exits = gating.should_exit(scores)
    assert exits[0].item() is False
    assert exits[1].item() is False
    assert exits[2].item() is True
    assert exits[3].item() is True

def test_myelin_shear_sparse_insulation():
    insulation = MyelinShearSparseInsulation(keep_ratio=0.5)
    w = mx.array([1.0, 2.0, 3.0, 4.0])
    insulated = insulation.insulate(w)
    assert mx.sum(insulated == 0.0).item() == 2
    assert insulated[2].item() == 3.0
    assert insulated[3].item() == 4.0

def test_tropical_semiring_speculation():
    spec = TropicalSemiringSpeculation()
    costs = mx.array([[[0.0, 5.0], [2.0, 0.0]]])
    min_costs = spec.evaluate_path_costs(costs)
    assert min_costs.shape == (1, 2, 2)
    assert min_costs[0, 0, 1].item() == 5.0

def test_grothendieck_topology_gating():
    gating = GrothendieckTopologyGating(threshold=0.6)
    logits = mx.array([[[10.0, 0.0], [10.0, 0.0]], [[5.0, 5.0], [5.0, 5.0]]])
    exits = gating.verify_coverage(logits)
    assert exits[0].item() is True
    assert exits[1].item() is False

def test_galois_field_gf28_simd():
    gf = GaloisFieldGF28SIMD()
    a = mx.array([3, 7, 0, 12])
    b = mx.array([5, 9, 8, 0])
    prod = gf.multiply(a, b)
    assert prod.shape == (4,)
    assert prod[2].item() == 0
    assert prod[3].item() == 0
    assert prod[0].item() == 15

# --- Optimizer Tests ---

def test_orthogonal_procrustes_projection():
    proj = OrthogonalProcrustesProjection()
    A = torch.randn(4, 4)
    B = torch.randn(4, 4)
    R = proj.project(A, B)
    assert R.shape == (4, 4)
    RTR = torch.matmul(R.t(), R)
    assert torch.allclose(RTR, torch.eye(4), atol=1e-4)

def test_relativistic_time_dilation_cache():
    cache = RelativisticTimeDilationCache(num_slots=5, c_limit=10.0)
    active = torch.tensor([1, 3])
    grad_norms = torch.tensor([2.0, 8.0])
    weight_norms = torch.tensor([5.0, 10.0])
    
    cache.step(active, grad_norms, weight_norms)
    assert cache.ages[0].item() == 1.0
    assert cache.ages[1].item() < 1.0
    assert cache.ages[3].item() < 1.0
    assert abs(cache.velocities[1].item() - 0.4) < 1e-4
    assert abs(cache.velocities[3].item() - 0.8) < 1e-4

# --- Speculation & Systems Tests ---

def test_asynchronous_parallel_speculation():
    spec = AsynchronousParallelSpeculation()
    def draft_fn(x):
        return x * 2
    def target_fn(x):
        return x + 5
    x = mx.array([1.0, 2.0])
    d_res, t_res = spec.run_speculation(draft_fn, target_fn, x)
    assert mx.allclose(d_res, mx.array([2.0, 4.0])).item()
    assert mx.allclose(t_res, mx.array([6.0, 7.0])).item()

def test_symplectic_integration_symdram():
    symdram = SymplecticIntegrationSymDRAM(dt=0.1)
    q = mx.array([1.0, 2.0])
    p = mx.array([0.0, 0.0])
    q_next, p_next = symdram.step_leapfrog(q, p)
    assert q_next.shape == (2,)
    assert p_next.shape == (2,)
    assert not mx.any(mx.isnan(q_next)).item()
    assert not mx.any(mx.isnan(p_next)).item()

def test_stochastic_resonance_dithering():
    dithering = StochasticResonanceDithering(noise_scale=0.01)
    x = mx.random.normal((3, 4))
    dithered = dithering.dither(x)
    assert dithered.shape == (3, 4)
    assert not mx.allclose(dithered, x).item()
