# Project: QAN-ATLAS Performance Optimization

## Architecture
- Module/package boundaries, data flow, shared interfaces:
  - `qan_transformers/modeling/attention.py`: Core PyTorch attention implementation, including Cayley adapter and Morse cache collapse.
  - `qan_transformers/mlx/attention.py`: MLX attention implementation, including Morse cache collapse.
  - `qan_transformers/math/e8_swap.py`: Memory Swap DB using Conway-Sloane E8 sphere-packing decoder.
  - `qan_transformers/optim/adelic.py`: Adelic Langevin Optimizer.
  - `qan_transformers/firewall/cohomology.py`: Cohomology Firewall checking for topological fracture.
  - `qan_transformers/kernels/mps_scatter.py`: Custom MPS gather-scatter PyTorch autograd operators.
  - `scripts/run_codebase_chat_mlx.py`: MLX-native whole-codebase QA chat CLI (speculative decoding / unified assistant caching).
  - `qan_transformers/cli/chat.py` (run via `qan-cli chat`): PyTorch whole-codebase QA chat CLI (MPS/CPU).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| 1 | Baseline Verification | Run baseline tests and benchmarks | None | DONE |
| 2 | Cayley Orthogonal Adapter Caching | R1 optimization in `qan_transformers/modeling/attention.py` | None | DONE |
| 3 | Linear-Complexity Morse Cache Collapse | R2 optimization in `qan_transformers/modeling/attention.py` and `qan_transformers/mlx/attention.py` | None | DONE |
| 4 | Vectorized Swap & Adelic Proposals | R3 optimization in `qan_transformers/math/e8_swap.py` and `qan_transformers/optim/adelic.py` | None | DONE |
| 5 | Contiguous MPS Kernels & Cohomology Firewall | R4 optimization in `qan_transformers/kernels/mps_scatter.py` and `qan_transformers/firewall/cohomology.py` | None | DONE |
| 6 | E2E Testing & Benchmark Validation | Verify all tests pass, run benchmarks and generate plots | M2, M3, M4, M5 | DONE |
| 7 | Whole-Codebase Ingestion & Locking | MLX-native whole-codebase QA CLI (chunked prefilling, cache locking, and restore) | M3, M4, M6 | DONE |

## Interface Contracts
- **Cayley Adapter**: `eval` mode caching must not alter public interface or class signature of the attention layer.
- **Morse Cache Collapse**: Must yield mathematically equivalent results (within 1e-5 tolerance) compared to original collapse.
- **CoW Memory Swap**: Must replace nested loops in `CoWMemorySwapGridDB._swap_in_batch` without breaking existing API.
- **Adelic Langevin**: Must vectorize proposals in `AdelicLangevinOptimizer` without altering step/optimizer behavior.
- **Cohomology Firewall**: Must vectorize edge coboundary calculations in `CohomologyFirewall` and retain same detection threshold.
- **MPS Scatter/Gather**: Index tensors cast to `int32` and made contiguous.

## Code Layout
- Core library: `qan_transformers/`
- Test suite: `tests/`
- Benchmarks: `benchmarks/run_validation_suite.py`
- Codebase QA CLIs: `qan_transformers/cli/chat.py` (run via `qan-cli chat`) (PyTorch) and `scripts/run_codebase_chat_mlx.py` (MLX)
