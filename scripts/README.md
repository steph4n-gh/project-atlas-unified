# scripts/ — UCE MVP Pipeline Scripts

These scripts implement the full reproducible pipeline for the Ultrametric Cognitive Engine MVP (Tasks 4–8 + polish). All are **synthetic-first** (no Gemma weights or `mlx-lm` required for smoke/CI/dev). Real-Gemma paths are opt-in when you have weights.

**Always run from repo root with the established pattern**:

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/<name>.py [args...]
```

After `pip install -e .[dev]`, the console entry points (from `[project.scripts]` in `pyproject.toml`) are registered in the venv. Invoke them with `PYTHONPATH=.` so the `scripts.*` namespace is importable (the wrappers are located in the venv bin, so implicit cwd is not the project root):

- `PYTHONPATH=. uce-build-tree ...`
- `PYTHONPATH=. uce-distill-phase0 ...`
- etc.

See root `README.md` for details. The explicit `PYTHONPATH=src python scripts/*.py` form (shown in all examples below) is the primary/reliable documented way and is what the tests use via subprocess.

Each script has a rich module docstring with exact examples (synthetic + real). Run `python scripts/foo.py --help` for CLI.

## Scripts Overview + Examples

### build_tree_from_gemma.py (Task 4)

Build `FiniteTree` by hierarchical clustering on embeddings (synthetic structured or real Gemma via `gemma_interface`).

**Synthetic (TDD/smoke, no weights):**

```bash
PYTHONPATH=src python scripts/build_tree_from_gemma.py \
    --synthetic --p 2 --depth 2 --num-tokens 4 --out /tmp/synth_tree.json
```

**Real (user supplies model):**

```bash
# gemma-2 / mlx 4bit (mlx backend)
PYTHONPATH=src python scripts/build_tree_from_gemma.py \
    --gemma-model mlx-community/gemma-2-2b-4bit \
    --p 3 --depth 6 --out /tmp/gemma_induced_tree.json

# gemma-4 (storage cache + transformers backend for E2B/12B; or direct snapshot for embed)
export HF_HOME=/Volumes/Storage/huggingface_cache
PYTHONPATH=src python scripts/build_tree_from_gemma.py \
    --gemma-model google/gemma-4-E2B-it \
    --p 8 --depth 2 --max-tokens 64 --out /tmp/real_gemma4_tree.json
# or pass local mlx snapshot dir for embed-only tree from cache 4bit without full teacher load
```

Output: JSON with `p`, `depth`, `address_map`, `source`, `num_tokens`. Reconstruct with `FiniteTree(p, depth, address_map=...)` (see script header for exact snippet).

See: `src/ultrametric_ce/tree.py` (cluster_and_assign_addresses), `gemma_interface.py` (lazy), `tests/test_tree.py` (roundtrip smoke via subprocess).

### distill_phase0_heads.py (Task 5)

Warm-start `DigitHeads` via factorization/distillation from teacher sub-distributions over tree balls (few gradient steps on heads only).

**Synthetic:**

```bash
PYTHONPATH=src python scripts/distill_phase0_heads.py \
    --synthetic --out /tmp/phase0_heads.safetensors --steps 25
```

**With prebuilt tree or real Gemma (ids must align):**

```bash
PYTHONPATH=src python scripts/distill_phase0_heads.py \
    --tree-config /tmp/synth_tree.json --out /tmp/phase0_heads.safetensors

# real
... --gemma-model mlx-community/gemma-2-2b-4bit ...
```

Produces `.safetensors` (+ optional meta). Verification in script: warmed heads alone already beat random on structural prefixes (for toy).

See: `src/ultrametric_ce/distillation.py` (warm_start_phase0_heads, predict_with_warmed_heads, save/load), `routing.py`.

### run_distillation.py (Task 6)

Full Phase 1 (diffusion focus, heads frozen) + Phase 2 (light joint) training via multi-part distillation (KL on leaves, hidden alignment, hierarchical prefix CE, ultrametric reg) against teacher.

**Synthetic smoke (fast, produces ready-to-use ckpt):**

```bash
PYTHONPATH=src python scripts/run_distillation.py \
    --synthetic --phase 1 --steps 30 --out /tmp/uce_phase1.safetensors

# with Phase 0 heads + smoke mode
PYTHONPATH=src python scripts/run_distillation.py \
    --synthetic --phase 1 --heads-ckpt /tmp/phase0_heads.safetensors --smoke --out /tmp/uce_smoke.safetensors
```

**Real:**

```bash
PYTHONPATH=src python scripts/run_distillation.py \
    --tree-config /tmp/gemma_tree.json --gemma-model mlx-community/gemma-2-2b-4bit \
    --phase 1 --steps 200 --out /tmp/uce_gemma_distilled.safetensors
```

Saves UCE weights + sibling `.meta.json` (p/depth/dim/alpha/address_map + provenance). Roundtrip load verified inside. For synthetic also prints post-train prefix/validity.

See: `src/ultrametric_ce/distillation.py` (run_distillation_phase + the 4 losses + batcher + ToyStructuralTeacher), `model.py`, `scripts/run_distillation.py` (save_full_checkpoint / load_full_checkpoint helpers, also mirrored in public `inference.load_model_and_tree`).

### generate_with_mvp.py (Task 7)

Load full distilled checkpoint (via public `inference.load_model_and_tree`), run autoregressive generation with **sparse active-path** digit routing. Prints per-step "active balls touched: X" demonstrating efficiency.

**Example (after a ckpt from run_distillation):**

```bash
PYTHONPATH=src python scripts/generate_with_mvp.py \
    --checkpoint /tmp/uce_distilled.safetensors --prompt "((1+2)*" --max-new 12
```

- Uses toy sym maps for prompt decode (synthetic ckpts align; real would need Gemma tokenizer mapping).
- Always emits only registered leaves (enforced by tree live navigation).
- `--no-verbose` suppresses the per-step active prints (summary note remains).

See: `src/ultrametric_ce/inference.py` (`load_model_and_tree`, `generate` — the core sparse impl using `embed_and_diffuse(active_balls=...)` + per-digit heads + categorical over live children).

### eval_structural.py (Task 8)

Full evaluation harness on a UCE checkpoint. Runs validity checker (`is_structurally_valid_toy_expr` / `check_structural_validity`), prefix_accuracy, ultrametric_spearman_correlation, structural_validity_rate, active-ball measurement (by capturing verbose generate), plus random baselines.

**Example:**

```bash
PYTHONPATH=src python scripts/run_distillation.py --synthetic --steps 30 --out /tmp/uce_mvp.safetensors
PYTHONPATH=src python scripts/eval_structural.py \
    --checkpoint /tmp/uce_mvp.safetensors --num-samples 10 --seed 42
```

Reports numbers proving structural coherence (high good/bad separation on checker, prefix gains vs ~1/p random, positive Spearman for ultrametric geometry, active frac << 1.0 for O(p log V) routing). See script for exact output format and interpretation.

Uses only public `ultrametric_ce.{inference,distillation,evaluation,...}`.

See: `src/ultrametric_ce/evaluation.py` (all the metrics + toy checker/parser), `test_distillation_synthetic.py` (asserts improvement).

## Prerequisites & Notes

- **Synthetic path (default)**: fully self-contained. Uses hard-coded `build_toy_arithmetic_tree` + `ToyStructuralTeacher` (structural bias injected into mock targets). Perfect for TDD, CI, and demos.
- **Real path**: use storage HF cache gemma-4 (12B 4bit flat etc via short name or resolved path). Contract (real_gemma_contract) enforces high original tids + prompt overlap. Use `load_gemma_tokenizer` (only, no model weights) for encode/decode in gen. Canonical evidence: `python scripts/run_gemma4_verif.py` (tok-only, calls assert_tree_talkable, one log).
- Small trees / `--smoke` / few `--steps` / `--num-samples` for speed in tests.
- Checkpoints: always `.safetensors` (MLX) + `.meta.json` sidecar for tree reconstruction. Use `load_model_and_tree` (public).
- All scripts set up their own `sys.path` hack for `PYTHONPATH=src` runs (consistent).
- Help + examples: each script's top docstring is authoritative and copy-paste runnable.

## Reproducibility

The end-to-end synthetic smoke test (`pytest ...::test_mvp_end_to_end_synthetic_smoke`) runs the scripts in subprocess exactly as a user/CI would (build → phase0 → distillation(ckpt+meta) → generate → eval) and asserts structural metrics + sparsity. This + the README quickstarts make the MVP claims reproducible without external weights.

See root `README.md` for high-level pipeline, expected numbers, and links to the design spec.

Run `python -m pytest tests/test_tree.py -q -k build_tree` to exercise the build script smoke too.
