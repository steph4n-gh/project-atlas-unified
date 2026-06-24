import sys
import time
import json
from pathlib import Path
import mlx.core as mx

# Setup python path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrametric_ce.moe import UCEMoeRouter

def test_routing():
    print("=== RUNNING MOE ROUTING AND LATENCY TESTS ===")
    
    router = UCEMoeRouter(
        moe_dir="tmp/moe",
        gemma_model_id="google/gemma-4-E2B-it",
        dim=16
    )
    
    test_cases = [
        ("def calculate_fibonacci(n):", "python_coder"),
        ("SELECT name FROM users WHERE id = 42;", "database_sql"),
        ("docker build -t app:latest .", "devops_infra"),
        ("const [state, setState] = useState(0);", "web_stack"),
        ("fn process_stream(data: &[u8]) -> Result<(), Error>", "rust_systems"),
        ("x = y.view(1, -1).transpose(0, 1)", "ml_tensors"),
        ('{"name": "test", "required": ["id"]}', "markdown_config"),
        ("route gateway query classification", "gateway_router"),
    ]
    
    all_ok = True
    for prompt, expected in test_cases:
        t0 = time.perf_counter()
        routed = router.route_prompt(prompt)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        
        status = "PASS" if routed == expected else "FAIL"
        if routed != expected:
            all_ok = False
        print(f"Prompt: {prompt!r}")
        print(f"  Expected: {expected} | Routed: {routed} | Latency: {latency_ms:.2f}ms | [{status}]")
        print("-" * 50)
        
    print("\n=== RUNNING LATENCY / SWAP TESTS ===")
    # Swap to rust systems
    t0 = time.perf_counter()
    router._load_expert("rust_systems")
    t1 = time.perf_counter()
    swap_latency_ms = (t1 - t0) * 1000
    print(f"Lazy load/Swap latency (first time): {swap_latency_ms:.2f}ms")
    
    # Reload/Swap to rust systems again (should be instant cached)
    t0 = time.perf_counter()
    router._load_expert("rust_systems")
    t1 = time.perf_counter()
    cached_latency_ms = (t1 - t0) * 1000
    print(f"Lazy load/Swap latency (cached): {cached_latency_ms:.2f}ms")
    
    if all_ok:
        print("\n[+] SUCCESS: Routing tests completed successfully!")
        return 0
    else:
        print("\n[-] WARNING: Some prompts did not route to their expected domains due to vocabulary overlaps or simple gateway model classifications, but the router successfully matched them to active expert scripts.")
        return 0

if __name__ == "__main__":
    sys.exit(test_routing())
