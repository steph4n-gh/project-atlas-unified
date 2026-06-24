import pytest
import mlx.core as mx
import mlx.nn as nn
from qan_transformers.mlx.tricore import TriCorePipeline
from qan_transformers.firewall.cohomology import CohomologyFirewall

class DummyModel(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        
    def __call__(self, x, **kwargs):
        return self.linear(x)

def test_initialization():
    model = DummyModel()
    pipeline = TriCorePipeline(model)
    
    assert pipeline.model == model
    assert isinstance(pipeline.firewall, CohomologyFirewall)
    assert isinstance(pipeline.gpu_stream, mx.Stream)
    assert isinstance(pipeline.cpu_stream, mx.Stream)
    assert isinstance(pipeline.ane_stream, mx.Stream)
    
    # Verify device types of streams
    assert pipeline.gpu_stream.device.type == mx.gpu
    assert pipeline.cpu_stream.device.type == mx.cpu
    assert pipeline.ane_stream.device.type == mx.cpu

def test_routing_and_output_equivalence():
    model = DummyModel()
    pipeline = TriCorePipeline(model)
    
    # Make sure weights are stable
    x_pref = mx.random.normal((1, 5, 64))
    x_dec = mx.random.normal((1, 1, 64))
    
    # Standard direct evaluation
    mx.set_default_device(mx.Device(mx.gpu))
    ref_pref = model(x_pref)
    ref_dec = model(x_dec)
    
    # Tri-Core Pipeline evaluation
    # 1. Prefill (S = 5 > 1) -> GPU routing
    out_pref = pipeline(x_pref)
    # 2. Decode (S = 1) -> CPU routing
    out_dec = pipeline(x_dec)
    
    # Verify outputs are identical
    assert mx.allclose(ref_pref, out_pref, atol=1e-5).item()
    assert mx.allclose(ref_dec, out_dec, atol=1e-5).item()

def test_pipelined_ane_firewall():
    model = DummyModel()
    firewall = CohomologyFirewall(threshold=0.1) # low threshold to trigger easily
    pipeline = TriCorePipeline(model, firewall=firewall)
    
    # First forward pass with an explicit NaN attention matrix to guarantee topological fracture detection
    x1 = mx.random.normal((1, 10, 64))
    attn_matrix = mx.full((1, 10, 10), float("nan"))
    out1 = pipeline(x1, attn_matrix=attn_matrix)
    
    # At first step, prev_attn_matrix is set but not yet audited because no prev step existed
    assert pipeline.prev_attn_matrix is not None
    assert not pipeline.anomaly_triggered
    
    # Second forward pass: triggers the background ANE audit of x1's attention matrix
    x2 = mx.random.normal((1, 1, 64))
    out2 = pipeline(x2)
    
    # Third forward pass: checks the result of the second step's audit
    x3 = mx.random.normal((1, 1, 64))
    out3 = pipeline(x3)
    
    # Since threshold=0.1, the anomaly should have triggered and set the flag
    assert pipeline.anomaly_triggered
