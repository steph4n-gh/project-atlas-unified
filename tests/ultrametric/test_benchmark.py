import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_comparative_benchmark import main

def test_comparative_benchmark_synthetic():
    # Verify that the benchmark script runs in synthetic mode and exits with 0
    exit_code = main(["--synthetic", "--max-new", "2"])
    assert exit_code == 0
