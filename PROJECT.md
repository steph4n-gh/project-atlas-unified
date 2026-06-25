# Project: QAN-ATLAS Documentation Audit, Verification, and Enrichment

## Architecture
- Documentation files under `docs/` and root `README.md`
- Core subsystems to document and align:
  - **ELQ Quantization**: splits weights into E8 coordinate lattices and sparse outlier matrices ($W = \text{Dequant}(W_{\text{quant}}) + \Delta W_{\text{outliers}}$). Interacts with `gate_proj` and `ELQLinear` projections.
  - **Morse KV Caching Mechanics**: Linear-complexity cache collapse.
  - **Čech Cohomology & Ultrametric Spaces**: Topological fracture firewall.
  - **E8 Root Projections / Sphere Packing**: Memory swap.
- Codebase implementations to verify against:
  - `qan_transformers/math/e8_swap.py`
  - `qan_transformers/modeling/attention.py`
  - `qan_transformers/mlx/attention.py`
  - `qan_transformers/firewall/cohomology.py`
  - `qan_transformers/optim/adelic.py`

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
