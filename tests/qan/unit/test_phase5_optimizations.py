import pytest
import mlx.core as mx
import numpy as np
import math

from qan_transformers.mlx.optimizations_phase5 import (
    # Group 1
    CosmicInflationScaleExpansion, HawkingRadiationKVLeakage, DarkMatterLatentCoupling,
    GravitationalLensingAttention, KerrErgosphereEnergyExtraction, ChandrasekharMassLimit,
    CosmicMicrowaveBackgroundAnisotropy, WormholeShortcutRouting, BigBangNucleosynthesisAssembly,
    FLRWMetricSpaceScaling, TachyonSuperluminalPropagation, SchwarzschildRadCachePruning,
    NebulaGasAccretionPooling, SupernovaCollapseQuantization, KeplerianOrbitCacheScheduling,
    # Group 2
    GaugeInvarianceProjection, YangMillsColorGating, CasimirForceAttraction,
    HiggsInertialMassWeighting, GoldstoneBosonPhaseTransition, SUSYSupermultipletMapping,
    NeutrinoFlavorOscillation, QuarkGluonPlasmaPhaseShift, BaryonAcousticOscillationMask,
    QCDColorConfinementFirewall, DiracEquationSpinorCoupling, KleinGordonWavePropagation,
    CherenkovRadiationDamping, CooperPairElectronCoupling, FermiDiracDistributionGating,
    # Group 3
    SynapticPruningScheduler, DendriticSpineGrowthPathway, DNAMethylationFreezing,
    RNAInterferencePathway, GlycolyticOscillationOptimizer, NeurotransmitterModulation,
    HippocampalConsolidationDB, ApoptosisExpertDropout, MitochondrialATPRespiration,
    GeneticCrossoverRecombination, MyelinSheathInsulationLayer, DNAHybridizationCotPrefetch,
    AntPheromoneTrailReinforcement, CellularPottsTokenAdhesion, ActionPotentialSpikeGating,
    # Group 4
    BoltzmannPartitionGating, GibbsFreeEnergyRouting, ClausiusEntropyDegradation,
    MaxwellDemonFilter, JouleThomsonExpansionExpansion, OnsagerReciprocalSymmetry,
    BravaisLatticeProjection, StefanBoltzmannRadiationLoss, CarnotEfficiencyLimitOptimizer,
    IsingExpertSpinFlip, BraggDiffractionPeakFiltering, PhononVibrationalWaveguide,
    HelimagnetismChiralRouting, VanDerWaalsAdhesionPooling, SebeckThermoelectricFeedback,
    # Group 5
    deRhamCohomologyFirewall, SymplecticPhaseSpaceAttention, LieAlgebraso3Rotations,
    GaloisFieldGF256Mapping, padicCantorSetIndexing, RiemannZetaZeroAlignment,
    CategoryTheoryMonadPipeline, HomotopyEquivalenceRouting, CoxeterGroupReflection,
    CliffordAlgebraSpinorRepresentations, MobiusStripAttentionLoop, HilbertSpaceProjection,
    PoincareDualityFirewall, EulerCharacteristicClassifier, TeichmullerSpaceDeformation,
    # Group 6
    LorenzAttractorNoiseDithering, MandelbrotPruningBoundary, LyapunovExponentMonitor,
    SelfOrganizedCriticalityAvalanche, CellularAutomataPooling, StrangeAttractorRouting,
    PhaseSpaceReconstruction, FeigenbaumBifurcationScheduler, RenormalizationGroupKVRGFlow,
    VolterraLotkaPredatorPreyRouting, FitzHughNagumoOscillator, DuffingOscillatorResonance,
    KuramotoSynchronizer, ChuaCircuitChaoticGating, NavierStokesFluidAttention,
    # Group 7
    PrecompiledMetalJITCache, ZeroCopySharedMemorySwap, AsynchronousPrefetchQueue,
    MultiThreadedVectorMath, MetalGPUWatchdogBypass, InstructionCacheAlignment,
    SIMD16VectorGather, DoubleBufferPipeSwap, ZeroAllocTensorReuse, MetalFusedFlashAttention
)

# ==========================================
# GROUP 1 TESTS: Cosmology & Astrophysics
# ==========================================

def test_cosmic_inflation():
    inflation = CosmicInflationScaleExpansion(h_factor=0.05)
    lr = inflation.expand_lr(base_lr=0.01, iteration=10)
    assert lr > 0.01
    assert abs(lr - 0.01 * math.exp(0.5)) < 1e-5

def test_hawking_radiation():
    leakage = HawkingRadiationKVLeakage(temperature=1.0)
    k = mx.array([[0.1, 0.2], [5.0, 5.0]])
    v = mx.array([[1.0, 1.0], [2.0, 2.0]])
    k_l, v_l = leakage.leak(k, v)
    assert k_l.shape == (2, 2)
    assert v_l.shape == (2, 2)

def test_dark_matter_coupling():
    coupling = DarkMatterLatentCoupling(decay=0.9)
    w = mx.array([1.0, 2.0])
    l1 = coupling.update(w)
    assert mx.allclose(l1, mx.array([0.1, 0.2])).item()

def test_gravitational_lensing():
    lensing = GravitationalLensingAttention(mass_coeff=0.5)
    scores = mx.zeros((1, 2, 2, 2))
    saliency = mx.array([[1.0, 2.0]])
    lensed = lensing.lens_attention(scores, saliency)
    assert mx.allclose(lensed, mx.array([[[[0.5, 1.0], [0.5, 1.0]], [[0.5, 1.0], [0.5, 1.0]]]])).item()

def test_kerr_ergosphere():
    kerr = KerrErgosphereEnergyExtraction(spin=0.5)
    grads = mx.array([1.0, 2.0])
    boosted = kerr.extract_gradient(grads)
    assert boosted[0].item() > 1.0

def test_chandrasekhar_limit():
    chandra = ChandrasekharMassLimit(limit=5.0)
    w = mx.array([10.0, -10.0])
    capped = chandra.cap_weights(w)
    assert mx.allclose(capped, mx.array([5.0, -5.0])).item()

def test_cmb_anisotropy():
    cmb = CosmicMicrowaveBackgroundAnisotropy(amplitude=1e-3)
    attn = mx.ones((2, 2))
    injected = cmb.inject_cmb(attn)
    assert injected.shape == (2, 2)

def test_wormhole_shortcut():
    wormhole = WormholeShortcutRouting(threshold=0.5)
    entropy = mx.array([0.2, 0.8])
    shortcuts = wormhole.get_shortcut(entropy)
    assert shortcuts[0].item() is True
    assert shortcuts[1].item() is False

def test_big_bang_nucleosynthesis():
    bbn = BigBangNucleosynthesisAssembly()
    w1 = mx.array([1.0, 1.0])
    w2 = mx.array([2.0, 2.0])
    assembled = bbn.assemble(w1, w2, ratio=0.4)
    assert mx.allclose(assembled, mx.array([1.6, 1.6])).item()

def test_flrw_metric():
    flrw = FLRWMetricSpaceScaling(scale_factor=1.5)
    x = mx.array([1.0, 2.0])
    scaled = flrw.scale_embeddings(x)
    assert mx.allclose(scaled, mx.array([1.5, 3.0])).item()

def test_tachyon_propagation():
    tach = TachyonSuperluminalPropagation(speed=3.0)
    x = mx.array([1.0, 2.0])
    prop = tach.propagate(x)
    assert mx.allclose(prop, mx.array([3.0, 6.0])).item()

def test_schwarzschild_rad_pruning():
    pruner = SchwarzschildRadCachePruning(rs=2.0)
    keys = mx.array([[1.0, 1.0], [3.0, 3.0]])
    pruned = pruner.prune(keys)
    assert pruned[0, 0].item() == 0.0
    assert pruned[1, 0].item() == 3.0

def test_nebula_gas_accretion():
    nebula = NebulaGasAccretionPooling(density=0.1)
    x = mx.array([[1.0, 2.0], [3.0, 4.0]])
    pooled = nebula.pool(x)
    assert pooled.shape == (2, 1)

def test_supernova_collapse():
    supernova = SupernovaCollapseQuantization(boundary=0.0)
    w = mx.array([0.5, -0.5])
    collapsed = supernova.collapse(w)
    assert mx.allclose(collapsed, mx.array([1.0, -1.0])).item()

def test_keplerian_orbit():
    orbit = KeplerianOrbitCacheScheduling(period=5)
    idx = orbit.get_orbit_index(step=12, num_pages=4)
    assert idx == 2

# ==========================================
# GROUP 2 TESTS: Quantum Field Theory
# ==========================================

def test_gauge_invariance():
    gauge = GaugeInvarianceProjection()
    x = mx.array([3.0, 4.0])
    proj = gauge.project(x)
    assert mx.allclose(proj, mx.array([0.6, 0.8])).item()

def test_yang_mills():
    ym = YangMillsColorGating(num_charges=3)
    color = mx.array([0.1, 0.8, 0.2])
    gate = ym.get_charge_gate(color)
    assert gate.item() == 1

def test_casimir_force():
    casimir = CasimirForceAttraction(distance_coeff=0.1)
    x1 = mx.array([1.0, 0.0])
    x2 = mx.array([2.0, 0.0])
    x1_n, x2_n = casimir.attract(x1, x2)
    assert x1_n[0].item() > 1.0
    assert x2_n[0].item() < 2.0

def test_higgs_inertial_mass():
    higgs = HiggsInertialMassWeighting(mass=2.0)
    updates = mx.array([1.0, 1.0])
    history = mx.array([0.5, 0.0])
    scaled = higgs.scale_updates(updates, history)
    assert mx.allclose(scaled, mx.array([0.5, 1.0])).item()

def test_goldstone_boson():
    gb = GoldstoneBosonPhaseTransition(target_val=1.0)
    x = mx.array([0.5, 4.0])
    trans = gb.transition(x)
    assert mx.allclose(trans, mx.array([0.25, 2.0])).item()

def test_susy_mapping():
    susy = SUSYSupermultipletMapping()
    w = mx.array([[1.0, 2.0], [3.0, 4.0]])
    boson, fermion = susy.map_susy(w)
    assert mx.allclose(boson, mx.array([[1.0, 2.5], [2.5, 4.0]])).item()

def test_neutrino_oscillation():
    neutrino = NeutrinoFlavorOscillation(theta=math.pi / 2.0) # c=0, s=1
    x = mx.array([[[1.0, 2.0, 3.0]]])
    osc = neutrino.oscillate(x)
    assert abs(osc[0, 0, 0].item() - (-2.0)) < 1e-5
    assert abs(osc[0, 0, 1].item() - 1.0) < 1e-5
    assert osc[0, 0, 2].item() == 3.0

def test_quark_gluon_plasma():
    qgp = QuarkGluonPlasmaPhaseShift(critical_temp=2.0)
    x = mx.array([1.0, 2.0, 3.0])
    out1 = qgp.shift(x, temp=1.0)
    out2 = qgp.shift(x, temp=3.0)
    assert mx.allclose(out1, x).item()
    assert abs(mx.std(out2).item() - 1.0) < 1e-4

def test_baryon_acoustic_mask():
    bao = BaryonAcousticOscillationMask(frequency=0.0)
    attn = mx.ones((1, 2, 3))
    masked = bao.apply_mask(attn)
    assert mx.allclose(masked, attn).item()

def test_qcd_confinement():
    qcd = QCDColorConfinementFirewall(bound=5.0)
    x1 = mx.array([3.0, 4.0])
    x2 = mx.array([5.0, 5.0])
    assert qcd.is_confined(x1).item() is True
    assert qcd.is_confined(x2).item() is False

def test_dirac_equation():
    dirac = DiracEquationSpinorCoupling()
    left = mx.array([1.0, 2.0])
    right = mx.array([2.0, 4.0])
    coupled = dirac.couple(left, right)
    assert mx.allclose(coupled, mx.array([1.2, 2.4])).item()

def test_klein_gordon():
    kg = KleinGordonWavePropagation(mass=1.0)
    x = mx.array([2.0, 2.0])
    x_prev = mx.array([1.0, 1.0])
    x_next = kg.step(x, x_prev)
    assert mx.allclose(x_next, mx.array([1.0, 1.0])).item()

def test_cherenkov_damping():
    cher = CherenkovRadiationDamping(max_speed=2.0)
    vels = mx.array([1.0, 4.0])
    damped = cher.damp(vels)
    assert damped[0].item() == 1.0
    assert abs(damped[1].item() - 2.0) < 1e-5

def test_cooper_pair():
    cooper = CooperPairElectronCoupling(coupling_strength=0.5)
    k = mx.array([[1.0, 2.0], [3.0, 4.0]])
    coupled = cooper.couple_keys(k)
    assert mx.allclose(coupled, mx.array([[2.0, 2.5], [5.0, 5.5]])).item()

def test_fermi_dirac():
    fd = FermiDiracDistributionGating(mu=2.0, kbT=1.0)
    energy = mx.array([2.0])
    gate_val = fd.gate(energy)
    assert abs(gate_val[0].item() - 0.5) < 1e-5

# ==========================================
# GROUP 3 TESTS: Biological Systems
# ==========================================

def test_synaptic_pruning():
    pruner = SynapticPruningScheduler(initial_threshold=0.5)
    w = mx.array([0.1, 0.8, -0.6])
    pruned = pruner.prune(w)
    assert pruned[0].item() == 0.0
    assert abs(pruned[1].item() - 0.8) < 1e-5

def test_dendritic_spine():
    growth = DendriticSpineGrowthPathway(threshold=0.5)
    acts = mx.array([0.2, 0.7])
    routed = growth.route_growth(acts)
    assert routed[0].item() is False
    assert routed[1].item() is True

def test_dna_methylation():
    meth = DNAMethylationFreezing(rate=0.5)
    w = mx.array([1.0, 2.0])
    freq = mx.array([0.2, 0.8])
    frozen = meth.freeze(w, freq)
    assert abs(frozen[0].item() - 0.9) < 1e-5
    assert frozen[1].item() == 2.0

def test_rna_interference():
    target = mx.array([1.0, 0.0])
    rna = RNAInterferencePathway(target_pattern=target)
    x1 = mx.array([1.0, 0.0])
    x2 = mx.array([0.0, 1.0])
    out1 = rna.inhibit(x1)
    out2 = rna.inhibit(x2)
    assert abs(out1[0].item() - 0.1) < 1e-5
    assert out2[1].item() == 1.0

def test_glycolytic_oscillation():
    opt = GlycolyticOscillationOptimizer(freq=0.0)
    lr = opt.step_lr(0.01, 100)
    assert lr == 0.01

def test_neurotransmitter_modulation():
    mod = NeurotransmitterModulation(dop=0.5, ser=0.1)
    attn = mx.array([1.0, 2.0])
    out = mod.modulate(attn)
    assert mx.allclose(out, mx.array([1.4, 2.9])).item()

def test_hippocampal_consolidation():
    hippo = HippocampalConsolidationDB()
    db = mx.array([10, 20, 30])
    hits = mx.array([1.0, 5.0, 2.0])
    cons = hippo.consolidate(db, hits)
    assert mx.allclose(cons, mx.array([20, 30, 10])).item()

def test_apoptosis_dropout():
    dropout = ApoptosisExpertDropout(death_rate=0.2)
    perf = mx.array([0.1, 0.8])
    dead = dropout.apoptosis(perf)
    assert dead[0].item() is True
    assert dead[1].item() is False

def test_mitochondrial_respiration():
    atp = MitochondrialATPRespiration(atp_pool=5.0)
    pool = atp.scale_compute(complexity=10.0)
    assert pool == 4.0

def test_genetic_crossover():
    cross = GeneticCrossoverRecombination(crossover_rate=1.0)
    p1 = mx.array([1.0, 2.0])
    p2 = mx.array([3.0, 4.0])
    res = cross.recombine(p1, p2)
    assert mx.allclose(res, p1).item()

def test_myelin_insulation():
    sheath = MyelinSheathInsulationLayer(insulation=0.2)
    x = mx.array([0.1, 0.5])
    out = sheath.insulate(x)
    assert out[0].item() == 0.0
    assert out[1].item() == 0.5

def test_dna_hybridization():
    cot = DNAHybridizationCotPrefetch(c0t_threshold=0.5)
    overlap = mx.array([0.2, 0.8])
    prio = cot.priority(overlap)
    assert prio[0].item() is False
    assert prio[1].item() is True

def test_ant_pheromone():
    ant = AntPheromoneTrailReinforcement(evaporation=0.9)
    pheromones = mx.array([1.0, 2.0])
    hits = mx.array([0.5, 0.0])
    new_pher = ant.reinforce(pheromones, hits)
    assert mx.allclose(new_pher, mx.array([1.4, 1.8])).item()

def test_cellular_potts():
    energy = mx.array([0.5, 1.2, -0.4])
    potts = CellularPottsTokenAdhesion(energy_matrix=energy)
    tokens = mx.array([2, 0])
    adh = potts.get_adhesion(tokens)
    assert mx.allclose(adh, mx.array([-0.4, 0.5])).item()

def test_action_potential():
    spike = ActionPotentialSpikeGating(thresh=2.0)
    pot = mx.array([1.5, 2.5])
    should = spike.should_spike(pot)
    assert should[0].item() is False
    assert should[1].item() is True

# ==========================================
# GROUP 4 TESTS: Thermodynamics
# ==========================================

def test_boltzmann_partition():
    partition = BoltzmannPartitionGating(temp=1.0)
    logits = mx.array([0.0, 0.0])
    gated = partition.gate(logits)
    assert mx.allclose(gated, mx.array([0.5, 0.5])).item()

def test_gibbs_free_energy():
    gibbs = GibbsFreeEnergyRouting(enthalpy=2.0, entropy=1.0)
    logits = mx.array([1.0, 2.0])
    g = gibbs.get_free_energy(logits, temp=0.5)
    assert g.shape == (2,)

def test_clausius_entropy():
    clausius = ClausiusEntropyDegradation()
    logits = mx.array([1.0, 1.0])
    ent = clausius.check_degradation(logits)
    assert abs(ent.item() - math.log(2.0)) < 1e-4

def test_maxwell_demon():
    demon = MaxwellDemonFilter(threshold=0.5)
    x = mx.array([0.2, 0.8])
    filtered = demon.filter_signals(x)
    assert filtered[0].item() == 0.0
    assert abs(filtered[1].item() - 0.8) < 1e-5

def test_joule_thomson():
    jt = JouleThomsonExpansionExpansion(coeff=0.1)
    logits = mx.array([1.0, 2.0])
    cooled = jt.cool_logits(logits)
    assert mx.allclose(cooled, mx.array([0.9, 1.8])).item()

def test_onsager_reciprocal():
    onsager = OnsagerReciprocalSymmetry()
    mat = mx.array([[1.0, 2.0], [4.0, 5.0]])
    sym = onsager.enforce_reciprocity(mat)
    assert mx.allclose(sym, mx.array([[1.0, 3.0], [3.0, 5.0]])).item()

def test_bravais_lattice():
    bravais = BravaisLatticeProjection(grid_size=0.5)
    x = mx.array([0.1, 0.7, 1.1])
    proj = bravais.project_to_lattice(x)
    assert mx.allclose(proj, mx.array([0.0, 0.5, 1.0])).item()

def test_stefan_boltzmann():
    sb = StefanBoltzmannRadiationLoss(sigma=0.01)
    w = mx.array([10.0])
    diss = sb.dissipate(w, temp=2.0)
    assert abs(diss[0].item() - 8.4) < 1e-5

def test_carnot_limit():
    carnot = CarnotEfficiencyLimitOptimizer(t_cold=300, t_hot=500)
    lr = carnot.scale_lr(0.1)
    assert abs(lr - 0.04) < 1e-5

def test_ising_expert():
    ising = IsingExpertSpinFlip(field_strength=0.1)
    spins = mx.array([1.0, -1.0])
    coupling = mx.array([[0.5, -0.5], [0.5, -0.5]])
    new_spins = ising.route_expert(spins, coupling)
    assert new_spins.shape == (2,)

def test_bragg_diffraction():
    bragg = BraggDiffractionPeakFiltering(d_spacing=1.0)
    waves = mx.array([0.5, 2.5])
    filtered = bragg.filter_peaks(waves)
    assert filtered[0].item() == 0.5
    assert filtered[1].item() == 0.0

def test_phonon_guide():
    phonon = PhononVibrationalWaveguide(frequency=0.0)
    x = mx.array([1.0, 2.0])
    guided = phonon.guide_signals(x)
    assert mx.allclose(guided, x).item()

def test_helimagnetism_routing():
    heli = HelimagnetismChiralRouting(chirality=0.5)
    x = mx.array([1.0, 2.0])
    routed = heli.route_chiral(x)
    assert mx.allclose(routed, mx.array([2.0, 2.5])).item()

def test_vanderwaals_pooling():
    vdw = VanDerWaalsAdhesionPooling(distance_coeff=1.0)
    keys = mx.array([[1.0, 0.0], [1.1, 0.0]])
    pooled = vdw.pool_vectors(keys)
    assert pooled.shape == (2, 1)

def test_sebeck_feedback():
    sebeck = SebeckThermoelectricFeedback(factor=0.1)
    grads = mx.array([1.0, 2.0])
    t_grad = mx.array([2.0, 0.0])
    feed = sebeck.feedback(grads, t_grad)
    assert mx.allclose(feed, mx.array([1.2, 2.0])).item()

# ==========================================
# GROUP 5 TESTS: Abstract Math & Topology
# ==========================================

def test_derham_firewall():
    firewall = deRhamCohomologyFirewall(threshold=1.5)
    x = mx.array([[1.0, 1.0], [1.0, 3.0], [1.0, 4.0]])
    valid = firewall.audit_differential(x)
    assert valid[0].item() is True
    assert valid[1].item() is False

def test_symplectic_attention():
    sym = SymplecticPhaseSpaceAttention()
    q = mx.array([[1.0, 2.0]])
    k = mx.array([[3.0, 4.0]])
    q_r, k_r = sym.preserve_volume(q, k)
    assert mx.allclose(q_r, mx.array([[-2.0, 1.0]])).item()

def test_lie_so3():
    so3 = LieAlgebraso3Rotations(theta=math.pi / 2.0)
    x = mx.array([1.0, 2.0, 3.0])
    rot = so3.rotate_so3(x)
    assert abs(rot[0].item() - (-2.0)) < 1e-5
    assert abs(rot[1].item() - 1.0) < 1e-5
    assert rot[2].item() == 3.0

def test_galois_field_gf256():
    author = GaloisFieldGF256Mapping()
    assert author.add(3, 5) == 6
    assert author.mul(2, 4) == 8

def test_padic_cantor():
    cantor = padicCantorSetIndexing(p=3)
    val = cantor.get_cantor_value(5)
    assert val > 0.0

def test_riemann_zeta_zero():
    zeta = RiemannZetaZeroAlignment()
    freqs = mx.array([1.0, 1.0])
    warped = zeta.warp_frequency(freqs)
    assert abs(warped[0].item() - 14.1347) < 1e-4

def test_category_monad():
    monad = CategoryTheoryMonadPipeline()
    x = mx.array([1.0])
    u = monad.unit(x)
    res = monad.bind(u, lambda val: val * 2.0)
    assert res[0].item() == 2.0

def test_homotopy_equivalence():
    homo = HomotopyEquivalenceRouting(tolerance=0.1)
    p1 = mx.array([1.0, 2.0])
    p2 = mx.array([1.05, 2.0])
    p3 = mx.array([1.5, 2.0])
    assert homo.is_equivalent(p1, p2).item() is True
    assert homo.is_equivalent(p1, p3).item() is False

def test_coxeter_reflection():
    ref = CoxeterGroupReflection(reflection_vector=mx.array([1.0, 0.0]))
    x = mx.array([2.0, 3.0])
    out = ref.reflect(x)
    assert mx.allclose(out, mx.array([-2.0, 3.0])).item()

def test_clifford_spinor():
    cliff = CliffordAlgebraSpinorRepresentations()
    s1 = mx.array([1.0, 2.0])
    s2 = mx.array([3.0, 4.0])
    out = cliff.multiply_spinors(s1, s2)
    assert mx.allclose(out, mx.array([11.0, 10.0])).item()

def test_mobius_strip():
    mobius = MobiusStripAttentionLoop()
    pos = mx.array([0.5])
    coords = mobius.warp_indices(pos)
    assert abs(coords[0, 0].item() - 0.0) < 1e-5
    assert abs(coords[0, 1].item() - 1.0) < 1e-5

def test_hilbert_space():
    hilb = HilbertSpaceProjection(dimensions=5)
    x = mx.array([1.0, 2.0])
    proj = (hilb.project_hilbert(x))
    assert proj.shape == (5,)
    assert proj[0].item() == 1.0
    assert proj[4].item() == 0.0

def test_poincare_duality():
    poincare = PoincareDualityFirewall()
    x1 = mx.array([1.0, 2.0, 2.0, 1.0])
    x2 = mx.array([1.0, 2.0, 3.0, 4.0])
    assert poincare.dual_check(x1).item() is True
    assert poincare.dual_check(x2).item() is False

def test_euler_characteristic():
    euler = EulerCharacteristicClassifier(threshold=0.5)
    adj = mx.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])
    assert euler.compute_euler(adj) == 1

def test_teichmuller_deformation():
    teich = TeichmullerSpaceDeformation(scaling=1.1)
    coords = mx.array([1.0, 2.0])
    deformed = teich.deform(coords)
    assert mx.allclose(deformed, mx.array([1.1, 2.2])).item()

# ==========================================
# GROUP 6 TESTS: Complexity & Chaos
# ==========================================

def test_lorenz_dithering():
    lorenz = LorenzAttractorNoiseDithering()
    s = lorenz.step_attractor()
    assert s.shape == (3,)

def test_mandelbrot_pruning():
    mandel = MandelbrotPruningBoundary(max_iter=5)
    assert mandel.in_set(0.0, 0.0) is True
    assert mandel.in_set(2.0, 2.0) is False

def test_lyapunov_monitor():
    lyap = LyapunovExponentMonitor()
    traj = mx.array([1.0, 2.0, 1.5, 3.0])
    val = lyap.compute_lyapunov(traj)
    assert isinstance(val, float)

def test_soc_criticality():
    soc = SelfOrganizedCriticalityAvalanche(threshold=4)
    grid = mx.array([1.0, 5.0, 2.0])
    new_grid, avalanche = soc.trigger_avalanche(grid)
    assert avalanche is True
    assert new_grid[1].item() == 1.0

def test_ca_pooling():
    ca = CellularAutomataPooling()
    grid = mx.zeros((3, 3), dtype=mx.int32)
    grid[0, 1] = 1
    grid[1, 1] = 1
    grid[2, 1] = 1
    new_grid = ca.step_conway(grid)
    assert new_grid.shape == (3, 3)

def test_strange_attractor():
    routing = StrangeAttractorRouting()
    logits = mx.array([0.5, 2.0, -1.0])
    expert = routing.route_chaotic(logits)
    assert expert.ndim == 0

def test_phase_space_reconstruct():
    recon = PhaseSpaceReconstruction(delay=1)
    ts = mx.array([1.0, 2.0, 3.0, 4.0])
    rec = recon.reconstruct(ts)
    assert rec.shape == (3, 2)

def test_feigenbaum_scheduler():
    scheduler = FeigenbaumBifurcationScheduler(r_start=4.0)
    x = scheduler.step_logistic(0.5)
    assert x == 1.0

def test_renormalization_group():
    rg = RenormalizationGroupKVRGFlow(step_size=0.1)
    keys = mx.array([1.0, 2.0])
    flowed = rg.flow_step(keys)
    assert mx.allclose(flowed, mx.array([0.9, 1.8])).item()

def test_predator_prey():
    vp = VolterraLotkaPredatorPreyRouting(alpha=0.1, beta=0.02)
    prey, pred = vp.step_populations(10.0, 5.0)
    assert prey > 0
    assert pred > 0

def test_fitzhugh_nagumo():
    neuron = FitzHughNagumoOscillator()
    v = neuron.step_neuron(i_ext=1.0)
    assert isinstance(v, float)

def test_duffing_resonance():
    duffing = DuffingOscillatorResonance()
    x = duffing.step_duffing(f_force=0.5)
    assert isinstance(x, float)

def test_kuramoto_sync():
    sync = KuramotoSynchronizer(coupling_k=0.5)
    phases = mx.array([0.1, 0.2, 0.3])
    synced = sync.synchronize_phases(phases)
    assert synced.shape == (3,)

def test_chua_circuit():
    chua = ChuaCircuitChaoticGating()
    x = chua.step_chua()
    assert isinstance(x, float)

def test_navier_stokes():
    fluid = NavierStokesFluidAttention(viscosity=0.1)
    vel = mx.array([1.0, 2.0, 3.0])
    diff = fluid.diffuse(vel)
    assert diff.shape == (3,)

# ==========================================
# GROUP 7 TESTS: Hardware & Systems
# ==========================================

def test_precompiled_metal_jit():
    cache = PrecompiledMetalJITCache()
    shader = cache.retrieve_shader("kernel_1")
    assert "kernel_1" in shader

def test_zerocopy_shared():
    swap = ZeroCopySharedMemorySwap()
    arr = np.array([1.0, 2.0])
    mx_arr = swap.map_shared(arr)
    assert mx.allclose(mx_arr, mx.array([1.0, 2.0])).item()

def test_prefetch_queue():
    queue = AsynchronousPrefetchQueue(capacity=2)
    idx1 = mx.array([1, 2])
    idx2 = mx.array([3, 4])
    queue.prefetch(idx1)
    queue.prefetch(idx2)
    assert mx.allclose(queue.pop_prefetch(), idx1).item()
    assert mx.allclose(queue.pop_prefetch(), idx2).item()
    assert queue.pop_prefetch() is None

def test_multithreaded_math():
    math_op = MultiThreadedVectorMath()
    a = mx.array([[1.0, 2.0]])
    b = mx.array([[3.0], [4.0]])
    res = math_op.compute_matmul(a, b)
    assert res[0, 0].item() == 11.0

def test_gpu_watchdog():
    bypass = MetalGPUWatchdogBypass(max_elements=2)
    a = mx.ones((5, 3))
    b = mx.ones((3, 2))
    res = bypass.slice_operation(a, b)
    assert res.shape == (5, 2)

def test_icache_alignment():
    align = InstructionCacheAlignment()
    addr = align.align_offset(100)
    assert addr == 128

def test_simd16_gather():
    simd = SIMD16VectorGather()
    arr = mx.array([10, 20, 30, 40])
    idx = mx.array([1, 3])
    gathered = simd.gather(arr, idx)
    assert mx.allclose(gathered, mx.array([20, 40])).item()

def test_double_buffer():
    db = DoubleBufferPipeSwap()
    x1 = mx.array([1.0])
    x2 = mx.array([2.0])
    db.write(x1)
    assert db.get_read_buffer()[0].item() == 1.0
    db.write(x2)
    assert db.get_read_buffer()[0].item() == 2.0

def test_zero_alloc_reuse():
    reuse = ZeroAllocTensorReuse(shape=(2,), dtype=mx.float32)
    x = mx.array([1.0, 2.0])
    res = reuse.reuse(x)
    assert mx.allclose(res, x).item()

def test_metal_fused_flash():
    flash = MetalFusedFlashAttention()
    q = mx.random.normal((1, 2, 4, 8))
    k = mx.random.normal((1, 2, 4, 8))
    v = mx.random.normal((1, 2, 4, 8))
    out = flash.flash_attn(q, k, v)
    assert out.shape == (1, 2, 4, 8)
