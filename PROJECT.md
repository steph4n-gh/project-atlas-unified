# Project: QAN-ATLAS Documentation Audit, Verification, and Enrichment

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
| 1 | Baseline & Exploration | Explore documentation, analyze code implementation of key concepts, draft Mermaid structures and flavor image prompts. | None | DONE |
| 2 | Math & Technical Alignment | Audit and rewrite equations to use valid LaTeX blocks. Ensure mathematical accuracy of Čech Cohomology, Morse KV caching, ELQ, and E8 lattices. Explain ELQ sliding cache, Metal fused matmul execution, and gate_proj interaction with ELQLinear. | M1 | DONE |
| 3 | Visual Enrichment | Generate and insert Mermaid diagrams (flows, architectures) and creative flavor/concept images (using image generation tools) for major subsystems. | M2 | DONE |
| 4 | Reference & Link Validation | Statically and programmatically verify and fix all relative links, file paths, and codebase symbol references. Ensure zero broken references. | M3 | DONE |
| 5 | Validation & Final Verification | Run `pytest` test suite to verify no errors. Run reviewer, challenger, and forensic auditor loops to ensure final clean verdict. | M4 | DONE |

## Interface Contracts
- **Documentation style**: Tone must be accessible to readers of all levels, jargon-free, with a dry, witty humor.
- **LaTeX Math Equations**: Must use valid LaTeX formatting (e.g. `$$` or `$`).
- **Mermaid Diagrams**: Must render correctly with valid Mermaid syntax.
- **Links**: Must use relative paths or file:/// schema, and must exist.
- **Code Snippets**: Code blocks in docs must represent valid python/bash syntax and match codebase API names.

## Code Layout
- Core library: `qan_transformers/`
- Documentation: `docs/` and root `README.md`
- Test suite: `tests/`
- Benchmarks: `benchmarks/run_validation_suite.py`
- Codebase QA CLIs: `qan_transformers/cli/chat.py` (run via `qan-cli chat`) (PyTorch) and `scripts/run_codebase_chat_mlx.py` (MLX)
