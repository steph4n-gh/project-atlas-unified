import torch
import pytest
from qan_transformers.firewall.motivic import MotivicCohomologyFirewall
from qan_transformers.firewall.cohomology import CohomologyFirewall

def test_motivic_cohomology_firewall_betti_numbers():
    # K=6 nodes
    firewall = MotivicCohomologyFirewall(
        weak_threshold=0.15,
        strong_threshold=0.45,
        halt_threshold_h11=1.0,
        rollback_threshold_h10=1.0,
        sharpen_threshold_h01=3.0
    )
    
    # 1. Disconnected/Identity skeleton (Fragmented)
    # 6 nodes, 0 off-diagonal edges.
    skeleton_disconnected = torch.eye(6).unsqueeze(0) # [1, 6, 6]
    res_disc = firewall(skeleton_disconnected)
    
    # H00: weak connected components. Since weak_threshold = 0.15, diagonal (1.0 > 0.15) is active,
    # but no off-diagonal connections, so 6 connected components.
    assert res_disc["h00"].item() == 6.0
    # H10: weak loops. No edges, so 0 loops.
    assert res_disc["h10"].item() == 0.0
    # H01: strong connected components. 6 connected components.
    assert res_disc["h01"].item() == 6.0
    # H11: strong loops. 0 loops.
    assert res_disc["h11"].item() == 0.0
    
    assert res_disc["action"] == ["sharpen"] # since h01 (6) >= sharpen_threshold_h01 (3)

    # 2. Cycle graph (Weak loop)
    # We form a single cycle: 0-1-2-3-4-5-0 with weak weight (e.g. 0.3)
    skeleton_cycle = torch.zeros(6, 6)
    for i in range(6):
        skeleton_cycle[i, (i + 1) % 6] = 0.3
        skeleton_cycle[(i + 1) % 6, i] = 0.3
    skeleton_cycle = skeleton_cycle.unsqueeze(0)
    
    res_cycle = firewall(skeleton_cycle)
    # At weak_threshold = 0.15, all cycle edges are active.
    # 1 connected component, 1 loop.
    assert res_cycle["h00"].item() == 1.0
    assert res_cycle["h10"].item() == 1.0
    
    # At strong_threshold = 0.45, cycle edges are inactive.
    # Vertices themselves are 0 on diagonal, so 0 active vertices?
    # Wait, the eigenvalues of zero Laplacian are all 0, so h0 = 6 components. Edges = 0.
    # So h01 = 6, h11 = 0.
    assert res_cycle["h01"].item() == 6.0
    assert res_cycle["h11"].item() == 0.0
    
    assert res_cycle["action"] == ["rollback"] # since h10 (1) >= rollback_threshold_h10 (1.0)


def test_motivic_cohomology_firewall_strong_loop():
    # 3. Strong loop (Adversarial)
    # A single cycle with strong weight (e.g. 0.8)
    firewall = MotivicCohomologyFirewall(
        weak_threshold=0.15,
        strong_threshold=0.45,
        halt_threshold_h11=1.0,
        rollback_threshold_h10=1.0,
        sharpen_threshold_h01=3.0
    )
    
    skeleton_strong_cycle = torch.zeros(6, 6)
    for i in range(6):
        skeleton_strong_cycle[i, (i + 1) % 6] = 0.8
        skeleton_strong_cycle[(i + 1) % 6, i] = 0.8
    skeleton_strong_cycle = skeleton_strong_cycle.unsqueeze(0)
    
    res_strong = firewall(skeleton_strong_cycle)
    # At strong_threshold = 0.45, cycle is active.
    # h01 = 1, h11 = 1.
    assert res_strong["h11"].item() == 1.0
    assert res_strong["action"] == ["halt"] # since h11 (1) >= halt_threshold_h11 (1.0)


def test_motivic_cohomology_differentiability():
    firewall = MotivicCohomologyFirewall()
    
    # Random skeleton with grad
    skeleton = torch.randn(2, 6, 6, requires_grad=True)
    res = firewall(skeleton)
    
    # Verify soft Betti numbers have grad
    loss = res["soft_h00"].sum() + res["soft_h10"].sum() + res["soft_h01"].sum() + res["soft_h11"].sum()
    loss.backward()
    
    assert skeleton.grad is not None
    assert not torch.isnan(skeleton.grad).any()
    assert (skeleton.grad != 0.0).any()


def test_two_tier_escalation():
    # Create CohomologyFirewall with a reasonable threshold
    cf_firewall = CohomologyFirewall(threshold=1.5)
    
    # Case A: Normal attention matrix (low CFI, no escalation)
    # 8x8 matrix representing clean focused attention
    attn_normal = torch.zeros(8, 8)
    attn_normal[range(8), range(8)] = 1.0
    
    is_fractured, cfi, alt_idx = cf_firewall.check_obstruction(attn_normal)
    assert not is_fractured
    assert cf_firewall.last_motivic_diagnostics is None
    
    # Case B: Pathological/Fractured attention matrix (CFI > 0.8 * threshold)
    # Let's create a highly fragmented/alternating pattern that triggers Čech warning
    attn_fractured = torch.zeros(8, 8)
    # Create 2 disjoint cliques of size 4
    for i in range(4):
        for j in range(4):
            attn_fractured[i, j] = 0.9 if i != j else 0.1
    for i in range(4, 8):
        for j in range(4, 8):
            attn_fractured[i, j] = 0.9 if i != j else 0.1
            
    is_fractured, cfi, alt_idx = cf_firewall.check_obstruction(attn_fractured)
    # Verification: Motivic firewall must be escalated
    assert cf_firewall.last_motivic_diagnostics is not None
    assert "h00" in cf_firewall.last_motivic_diagnostics
    assert "action" in cf_firewall.last_motivic_diagnostics
