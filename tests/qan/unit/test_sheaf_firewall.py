import pytest
import mlx.core as mx
from qan_transformers.mlx.sheaf import SheafFirewall

def test_sheaf_consistency_normal():
    # Model layers embed_dim = 32
    firewall = SheafFirewall(embed_dim=32, threshold=0.1)
    
    # Under a linear mapping, W_sheaf will learn to align x_l and x_next
    x_l = mx.random.normal((2, 10, 32))
    W_actual = mx.random.normal((32, 32)) * 0.1
    x_next = x_l @ W_actual
    
    # Warm up / align connection
    for _ in range(20):
        firewall.update_connection(x_l, x_next, lr=0.1)
        
    is_fractured, E_l = firewall.check_consistency(x_l, x_next)
    
    # E_l should be low because W_sheaf has aligned to W_actual
    assert E_l.item() < 0.1
    assert not is_fractured.item()

def test_sheaf_consistency_fractured():
    firewall = SheafFirewall(embed_dim=32, threshold=0.1)
    
    x_l = mx.random.normal((2, 10, 32))
    # Align connection to identity
    x_next = x_l
    for _ in range(10):
        firewall.update_connection(x_l, x_next, lr=0.1)
        
    # Introduce anomalous out-of-distribution fracture (adversarial injection)
    x_fractured = x_next + mx.random.normal(x_next.shape) * 5.0
    
    is_fractured, E_l = firewall.check_consistency(x_l, x_fractured)
    
    # Cohomology obstruction should spike
    assert E_l.item() > 0.1
    assert is_fractured.item()

def test_online_adaptation():
    firewall = SheafFirewall(embed_dim=16, threshold=0.2)
    
    x_l = mx.random.normal((1, 5, 16))
    x_next = x_l * 2.0
    
    # Before training/adaptation, identity W_sheaf gives large obstruction
    is_frac_before, E_l_before = firewall.check_consistency(x_l, x_next)
    
    # Train online
    for _ in range(50):
        firewall.update_connection(x_l, x_next, lr=0.05)
        
    # After training, obstruction should be reduced
    is_frac_after, E_l_after = firewall.check_consistency(x_l, x_next)
    
    assert E_l_after.item() < E_l_before.item()
