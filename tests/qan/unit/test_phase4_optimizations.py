import pytest
import mlx.core as mx
import numpy as np
import torch
from qan_transformers.mlx.attention import (
    SchwarzschildMetricPositionalWarp,
    FeynmanDiagramPerturbationAttention,
    GrassmannianSubspaceProjection,
    ThermodynamicPartitionFunctionSoftmax
)
from qan_transformers.mlx.e8_swap import (
    STDPPlasticitySwappingPriority,
    SIREpidemicSemanticHubbing,
    ZeroCopyCacheSliceMapping,
    PrecompiledMetalJITKernels
)
from qan_transformers.mlx.modeling import (
    JonesPolynomialKnotInvariants,
    IsingSpinGlassExpertRouting,
    TranscriptionFactorGeneGating,
    ProteinChaperoneQuantization,
    QuantumDecoherencePathCollapse,
    HamiltonJacobiTrajectoryRouting,
    LaplacianSpectralGraphPooling
)
from qan_transformers.optim.adelic import BiquaternionSpinorLoRA
from qan_transformers.firewall.cohomology import SheafCohomologicalAuditFirewall
from qan_transformers.mlx.wave_solver import (
    AsynchronousPipelineRingBuffers,
    OctalSIMDParallelMatrixMath,
    DynamicClockScalingScheduler
)

# --- Attention Tests ---

def test_schwarzschild_metric_positional_warp():
    warp = SchwarzschildMetricPositionalWarp(rs_factor=0.1)
    positions = mx.array([1.0, 2.0, 3.0])
    importance = mx.array([0.1, 0.2, 0.1]) # rs = 0.04
    warped = warp.warp(positions, importance)
    assert warped.shape == (3,)
    assert warped[0].item() > 1.0 # Schwarzschild gravitational dilation warps it larger

def test_feynman_diagram_perturbation_attention():
    attn = FeynmanDiagramPerturbationAttention(epsilon=0.1)
    q = mx.random.normal((1, 2, 5, 8))
    k = mx.random.normal((1, 2, 5, 8))
    v = mx.random.normal((1, 2, 5, 8))
    out = attn(q, k, v)
    assert out.shape == (1, 2, 5, 8)
    assert not mx.any(mx.isnan(out)).item()

def test_grassmannian_subspace_projection():
    proj = GrassmannianSubspaceProjection(subspace_dim=4)
    x = mx.random.normal((1, 2, 10, 8))
    out = proj.project(x)
    assert out.shape == (1, 2, 10, 8)
    assert not mx.any(mx.isnan(out)).item()

def test_thermodynamic_partition_function_softmax():
    softmax = ThermodynamicPartitionFunctionSoftmax(beta=1.5)
    logits = mx.random.normal((2, 5))
    probs = softmax(logits)
    assert probs.shape == (2, 5)
    assert mx.allclose(mx.sum(probs, axis=-1), mx.ones((2,)), atol=1e-4).item()

# --- Memory Swapping DB Tests ---

def test_stdp_plasticity_swapping_priority():
    stdp = STDPPlasticitySwappingPriority(num_pages=5, tau_stdp=5.0)
    stdp.register_spike(page_idx=2, current_time=1.0)
    stdp.register_spike(page_idx=2, current_time=2.0)
    candidate = stdp.get_eviction_candidate(active_pages=mx.array([2]))
    assert candidate != 2
    assert 0 <= candidate < 5

def test_sir_epidemic_semantic_hubbing():
    sir = SIREpidemicSemanticHubbing(num_nodes=5, beta=0.5)
    adj = mx.array([
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ])
    active = mx.array([0])
    sir.step_epidemic(active, adj)
    # Node 1 should be infected by Node 0
    assert sir.states[0].item() == 1.0
    assert sir.states[1].item() == 1.0
    hubs = sir.get_semantic_hubs(k=2)
    assert hubs.shape == (2,)

def test_zero_copy_cache_slice_mapping():
    mapper = ZeroCopyCacheSliceMapping()
    arr = mx.array([1, 2, 3, 4, 5])
    sliced = mapper.map_slice(arr, 1, 4)
    assert mx.allclose(sliced, mx.array([2, 3, 4])).item()

def test_precompiled_metal_jit_kernels():
    jit = PrecompiledMetalJITKernels()
    k1 = jit.get_kernel("fused_e8_decode")
    k2 = jit.get_kernel("fused_e8_decode")
    assert "fused_e8_decode" in k1
    assert k1 == k2

# --- Modeling & Gating Tests ---

def test_jones_polynomial_knot_invariants():
    knot = JonesPolynomialKnotInvariants()
    crossings = mx.array([1, -1, 1, 1])
    bracket = knot.compute_kauffman_bracket(crossings)
    assert bracket > 0.0

def test_ising_spin_glass_expert_routing():
    routing = IsingSpinGlassExpertRouting(num_experts=3)
    logits = mx.array([[1.0, -1.0, 0.5]])
    coupling = mx.array([
        [0.0, -0.5, 0.2],
        [-0.5, 0.0, -0.1],
        [0.2, -0.1, 0.0]
    ])
    expert = routing.route(logits, coupling)
    assert expert.shape == (1,)
    assert 0 <= expert.item() < 3

def test_transcription_factor_gene_gating():
    gating = TranscriptionFactorGeneGating(num_factors=4)
    concentrations = mx.array([[0.8, 0.9, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
    skip = gating.gate_layers(concentrations)
    assert skip[0].item() is False # TF_0 AND TF_1 > 0.5 -> active, do not skip
    assert skip[1].item() is True  # inactive TF -> skip layer

def test_protein_chaperone_quantization():
    chaperone = ProteinChaperoneQuantization(num_bits=4)
    w = mx.array([0.18, 0.45, -0.92])
    scale = mx.array(0.2)
    chaperoned = chaperone.chaperone_regularization(w, scale)
    assert chaperoned.shape == (3,)
    # rounded centers: [0.2, 0.4, -1.0]
    # chaperoned pulls closer to centers
    assert abs(chaperoned[0].item() - 0.2) < abs(w[0].item() - 0.2)

def test_quantum_decoherence_path_collapse():
    collapse = QuantumDecoherencePathCollapse(coherence_threshold=0.6)
    # density matrix representing a pure coherent state
    rho1 = mx.array([[[1.0, 0.0], [0.0, 0.0]]])
    # density matrix representing a maximally mixed decohered state
    rho2 = mx.array([[[0.5, 0.0], [0.0, 0.5]]])
    
    keep1 = collapse.collapse_paths(rho1)
    keep2 = collapse.collapse_paths(rho2)
    assert keep1[0].item() is True  # Tr(rho^2) = 1.0 >= 0.6
    assert keep2[0].item() is False # Tr(rho^2) = 0.5 < 0.6


def test_hamilton_jacobi_trajectory_routing():
    routing = HamiltonJacobiTrajectoryRouting(time_steps=5)
    values = mx.array([[1.0, 2.0, 3.0, 2.5, 2.0]])
    exit_idx = routing.get_optimal_exit(values)
    assert exit_idx.shape == (1,)
    assert exit_idx.item() == 2 # optimal exit is at step index 2 (derivative becomes negative)

def test_laplacian_spectral_graph_pooling():
    pooling = LaplacianSpectralGraphPooling()
    sim = mx.array([[[0.0, 0.8], [0.8, 0.0]]])
    x = mx.random.normal((1, 2, 8))
    pooled = pooling.pool_graph(sim, x)
    assert pooled.shape == (1, 2, 8)
    assert not mx.any(mx.isnan(pooled)).item()

# --- Optimizer Tests ---

def test_biquaternion_spinor_lora():
    lora = BiquaternionSpinorLoRA(in_features=8, out_features=12)
    x = torch.randn(2, 8)
    out = lora.forward(x)
    assert out.shape == (2, 12)
    assert not torch.any(torch.isnan(out))

# --- Firewalls & Speculation Tests ---

def test_sheaf_cohomological_audit_firewall():
    firewall = SheafCohomologicalAuditFirewall(threshold=0.1)
    # consistent section data: overlap region perfectly overlaps
    consistent_h = mx.ones((1, 20, 8))
    # compromised/fractured section data: overlap has high variance / discrepancy
    compromised_h = mx.array(consistent_h)
    compromised_h[:, 11:, :] = 5.0 # create discrepancy
    
    assert firewall.audit([consistent_h, consistent_h]) is False
    assert firewall.audit([compromised_h, compromised_h]) is True

def test_asynchronous_pipeline_ring_buffers():
    buf = AsynchronousPipelineRingBuffers(capacity=8)
    tokens = mx.array([1, 2, 3, 4], dtype=mx.uint32)
    buf.push(tokens)
    popped = buf.pop(2)
    assert mx.allclose(popped, mx.array([1, 2])).item()
    
    tokens2 = mx.array([5, 6, 7, 8, 9, 10], dtype=mx.uint32)
    buf.push(tokens2) # circular wrap around
    popped2 = buf.pop(4)
    assert mx.allclose(popped2, mx.array([3, 4, 5, 6])).item()

def test_octal_simd_parallel_matrix_math():
    math = OctalSIMDParallelMatrixMath()
    W = mx.random.normal((8, 4, 4))
    x = mx.random.normal((8, 4, 1))
    out = math.parallel_matmul_8x(W, x)
    assert out.shape == (8, 4, 1)

def test_dynamic_clock_scaling_scheduler():
    scheduler = DynamicClockScalingScheduler(baseline_clock=1000.0)
    f1 = scheduler.update_clock_frequency(cfi_metric=1.5)
    assert f1 == 1000.0
    f2 = scheduler.update_clock_frequency(cfi_metric=0.5)
    assert f2 == 600.0
