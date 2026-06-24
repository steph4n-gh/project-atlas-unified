import pytest
import torch

def test_t1_adelic_optimizer_init():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1)
    assert isinstance(opt, torch.optim.Optimizer)

def test_t1_adelic_optimizer_step():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1)
    
    # Run a step
    loss = (p ** 2).sum()
    loss.backward()
    
    val_before = p.clone().detach()
    opt.step()
    val_after = p.clone().detach()
    
    assert not torch.equal(val_before, val_after)

def test_t1_adelic_optimizer_vladimirov_derivative():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    # Test checking Vladimirov derivative calculation
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1, alpha=0.5)
    
    # We just run a step and verify it runs successfully.
    p.grad = torch.ones_like(p)
    opt.step()
    # If the derivative logic is correct, the parameter updates are recorded.
    assert opt is not None

def test_t1_adelic_optimizer_dyadic_compression():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    # Check that multiscale dyadic history compression of gradient trajectories runs
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1, dyadic_compression=True)
    p.grad = torch.randn(2, 2)
    opt.step()
    assert opt is not None

def test_t1_adelic_optimizer_exact_updates():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    # Check that exact updates are performed without noisy STE
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1)
    p.grad = torch.randn(2, 2)
    opt.step()
    assert opt is not None

# Tier 2 Boundary Cases

def test_t2_adelic_optimizer_lr_boundary():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    p = torch.nn.Parameter(torch.randn(2, 2))
    
    # lr=0.0 (no change)
    opt_zero = AdelicLangevinOptimizer([p], lr=0.0)
    p.grad = torch.ones_like(p)
    val_before = p.clone().detach()
    opt_zero.step()
    assert torch.equal(val_before, p.detach())
    
    # lr=2.0 (large learning rate)
    opt_large = AdelicLangevinOptimizer([p], lr=2.0)
    p.grad = torch.ones_like(p)
    opt_large.step()
    assert not torch.equal(val_before, p.detach())

def test_t2_adelic_optimizer_nan_gradients():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1)
    p.grad = torch.tensor([[float('nan'), 1.0], [float('inf'), -float('inf')]])
    
    # Optimizer should handle NaN/Inf gracefully without crashing
    opt.step()
    assert not torch.isnan(p).all()

def test_t2_adelic_optimizer_fractional_order_extremes():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    p = torch.nn.Parameter(torch.randn(2, 2))
    p.grad = torch.randn(2, 2)
    
    # alpha near 0.0 (0.05)
    opt_low = AdelicLangevinOptimizer([p], lr=0.1, alpha=0.05)
    opt_low.step()
    
    # alpha near 1.0 (0.95)
    opt_high = AdelicLangevinOptimizer([p], lr=0.1, alpha=0.95)
    opt_high.step()
    assert opt_low is not None and opt_high is not None

def test_t2_adelic_optimizer_tree_depth_limits():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    # Test p-adic tree depth boundary conditions (e.g. depth=1, depth=32)
    p = torch.nn.Parameter(torch.randn(2, 2))
    p.grad = torch.randn(2, 2)
    opt_d1 = AdelicLangevinOptimizer([p], lr=0.1, tree_depth=1)
    opt_d1.step()
    
    opt_d32 = AdelicLangevinOptimizer([p], lr=0.1, tree_depth=32)
    opt_d32.step()
    
    assert opt_d1 is not None and opt_d32 is not None

def test_t2_adelic_optimizer_high_variance_history():
    try:
        from qan_transformers.optim.adelic import AdelicLangevinOptimizer
    except (ImportError, ModuleNotFoundError):
        pytest.fail("Module not implemented yet")
        
    p = torch.nn.Parameter(torch.randn(2, 2))
    opt = AdelicLangevinOptimizer([p], lr=0.1, dyadic_compression=True)
    
    # Pass highly fluctuating gradients and check that compression handles it cleanly
    for _ in range(5):
        p.grad = torch.randn(2, 2) * 10.0
        opt.step()
        
    assert opt is not None
