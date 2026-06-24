import mlx.core as mx
import mlx.nn as nn
from qan_transformers.firewall.cohomology import CohomologyFirewall

class TriCorePipeline:
    def __init__(self, model: nn.Module, firewall: CohomologyFirewall = None):
        self.model = model
        self.firewall = firewall if firewall is not None else CohomologyFirewall()
        
        # Initialize explicit device type objects
        self.gpu_device = mx.Device(mx.gpu)
        self.cpu_device = mx.Device(mx.cpu)
        
        # Initialize independent streams for concurrent execution
        self.gpu_stream = mx.new_stream(self.gpu_device)
        self.cpu_stream = mx.new_stream(self.cpu_device)
        self.ane_stream = mx.new_stream(self.cpu_device)
        
        # Pipelining state variables
        self.prev_attn_matrix = None
        self.prev_firewall_future = None
        self.anomaly_triggered = False
        
    def shutdown(self):
        # No threads to shut down in the stream-based model
        pass
        
    def __call__(self, x, **kwargs):
        B, S, C = x.shape
        
        # 1. Check the background safety audit from the PREVIOUS step
        if self.prev_firewall_future is not None:
            if isinstance(self.prev_firewall_future, bool):
                is_fractured = self.prev_firewall_future
            elif isinstance(self.prev_firewall_future, list):
                is_fractured = any(self.prev_firewall_future)
            else:
                is_fractured = bool(self.prev_firewall_future.item())
                
            if is_fractured:
                self.anomaly_triggered = True
                print("[ANE Firewall Alert] Topological fracture detected in previous step!")
            self.prev_firewall_future = None
            
        # 2. Pipelined ANE Firewall Audit:
        # Launch cohomology check on previous step's attention matrix on the ANE stream
        if self.prev_attn_matrix is not None:
            # We explicitly execute the check on the ANE stream (which runs concurrently on CPU)
            with mx.stream(self.ane_stream):
                is_fractured, cfi_val, alt_idx = self.firewall.check_obstruction(self.prev_attn_matrix)
                # Materialize the boolean result on the stream asynchronously
                mx.eval(is_fractured)
                self.prev_firewall_future = is_fractured
                
        # 3. Task Routing:
        # Pre-fill (S > 1) -> GPU stream
        # Autoregressive Decode (S == 1) -> CPU stream (AMX)
        if S > 1:
            with mx.stream(self.gpu_stream):
                mx.set_default_device(self.gpu_device)
                res = self.model(x, **kwargs)
                mx.eval(res)
        else:
            with mx.stream(self.cpu_stream):
                mx.set_default_device(self.cpu_device)
                res = self.model(x, **kwargs)
                mx.eval(res)
                
        # Cache attention matrix for next step (or simulate one if not returned)
        attn_matrix = kwargs.get("attn_matrix", None)
        if attn_matrix is not None:
            self.prev_attn_matrix = attn_matrix
        else:
            # Simulate a 3D attention-like matrix using outer product of inputs
            # Run on CPU stream to keep it lightweight
            with mx.stream(self.ane_stream):
                self.prev_attn_matrix = mx.matmul(x, mx.transpose(x, (0, 2, 1)))[..., -1, :]
                mx.eval(self.prev_attn_matrix)
                
        # Always restore default device to GPU for consistency
        mx.set_default_device(self.gpu_device)
        
        return res
