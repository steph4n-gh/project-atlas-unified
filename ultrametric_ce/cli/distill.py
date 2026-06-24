#!/usr/bin/env python
"""Task 6 main script: Full diffusion training with distillation (Phase 1 + Phase 2).

Trains the UCE diffusion layers (+ optionally light heads) via multi-part distillation
(kl on leaves + hidden alignment + hierarchical prefix CE + ultrametric reg) against
a teacher (GemmaInterface or mock).

Synthetic path (default, no weights): hard-coded toy grammar tree + ToyStructuralTeacher.
Produces runnable checkpoint: UCEModel weights (.safetensors) + sidecar tree meta .json .

The checkpoint can be loaded for inference later (reconstruct tree, UCEModel(tree, ...), load_weights).

Usage examples:
  # Synthetic smoke (fast, self-contained, recommended for test):
  source .venv/bin/activate
  PYTHONPATH=src python scripts/run_distillation.py --synthetic --phase 1 --steps 30 --out /tmp/uce_phase1.safetensors

  # With pre-warmed heads from Phase 0:
  PYTHONPATH=src python scripts/run_distillation.py --synthetic --phase 1 --heads-ckpt phase0_heads.safetensors --steps 50

  # Real (user provides Gemma + tree from build_tree...):
  PYTHONPATH=src python scripts/run_distillation.py \
      --tree-config /tmp/gemma_tree.json --gemma-model mlx-community/gemma-2-2b-4bit \
      --phase 1 --steps 200 --out /tmp/uce_gemma_distilled.safetensors

See src/ultrametric_ce/distillation.py for run_distillation_phase, losses, batcher.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import mlx.core as mx

from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.model import UCEModel
from ultrametric_ce.distillation import (
    build_toy_arithmetic_tree,
    VALID_TOY_EXPRS,
    expr_to_address_sequence,
    ToyStructuralTeacher,
    run_distillation_phase,
    load_warmed_heads,
)
from ultrametric_ce.routing import DigitHeads

# lazy real gemma
_load_gemma = None
try:
    from ultrametric_ce.gemma_interface import load_gemma as _load_gemma  # type: ignore
except Exception:
    _load_gemma = None


def load_tree_from_config(cfg_path: str | Path) -> Tuple[FiniteTree, Dict]:
    """Reconstruct FiniteTree + raw cfg from json (as produced by build_tree_from_gemma.py)."""
    cfg = json.loads(Path(cfg_path).read_text())
    p = int(cfg["p"])
    depth = int(cfg["depth"])
    am = {int(a): int(t) for a, t in cfg["address_map"].items()}
    tree = FiniteTree(p, depth, address_map=am)
    print(f"[tree] loaded from {cfg_path}: p={p} depth={depth} leaves={len(tree)} source={cfg.get('source')}")
    return tree, cfg


def save_full_checkpoint(
    model: UCEModel,
    tree: FiniteTree,
    out_path: str | Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save UCEModel weights (safetensors) + tree meta sidecar (.meta.json) for reload."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out_p))
    full_meta: Dict[str, Any] = {
        "p": tree.p,
        "depth": tree.depth,
        "dim": model.dim,
        "num_diff_layers": getattr(model.diffusion, "num_layers", 1),
        "alpha": getattr(model.diffusion, "alpha", 0.5),
        "num_leaves": len(tree),
        "address_map": {str(a): int(tree.address_to_token(a)) for a in tree.leaf_addresses()},
        "note": "Reload: tree=FiniteTree(p,depth,address_map=am); model=UCEModel(tree,dim=...,num_diff_layers=...,alpha=...); model.load_weights(ckpt)",
    }
    if meta:
        full_meta.update(meta)
    meta_p = out_p.with_suffix(".meta.json")
    meta_p.write_text(json.dumps(full_meta, indent=2, sort_keys=True))
    return out_p


def load_full_checkpoint(ckpt_path: str | Path, meta_path: Optional[str | Path] = None) -> Tuple[FiniteTree, UCEModel]:
    """Reconstruct tree + UCEModel and load weights from the script's checkpoint format."""
    ckpt_p = Path(ckpt_path)
    if meta_path is None:
        meta_p = ckpt_p.with_suffix(".meta.json")
    else:
        meta_p = Path(meta_path)
    meta = json.loads(meta_p.read_text())
    p = int(meta["p"])
    depth = int(meta["depth"])
    dim = int(meta.get("dim", 16))
    ndl = int(meta.get("num_diff_layers", 1))
    alpha = float(meta.get("alpha", 0.5))
    am = {int(a): int(t) for a, t in meta["address_map"].items()}
    tree = FiniteTree(p, depth, address_map=am)
    model = UCEModel(tree, dim=dim, num_diff_layers=ndl, alpha=alpha)
    model.load_weights(str(ckpt_p))
    return tree, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 6: train UCE diffusion with full distillation (Phase1/2).")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true", help="Use built-in toy tree + structural mock teacher (no Gemma; default for smoke).")
    parser.add_argument("--tree-config", type=str, default=None, help="Path to tree_config.json for custom/prebuilt tree.")
    parser.add_argument("--gemma-model", type=str, default=None, help="Gemma path/HF id for real teacher logits (requires matching tree token ids).")
    parser.add_argument("--cached-dataset", type=str, default=None, help="Path to pre-computed dataset cache (.pkl) to bypass teacher.")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2], help="1=freeze heads/diffusion focus; 2=light joint heads+diffusion.")
    parser.add_argument("--dim", type=int, default=16, help="State dim (must match heads if loading).")
    parser.add_argument("--num-diff-layers", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=100, help="Training steps.")
    parser.add_argument("--lr", type=float, default=0.005, help="Adam lr.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (num examples per step).")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="uce_distilled.safetensors", help="Output checkpoint path for full UCE weights.")
    parser.add_argument("--heads-ckpt", type=str, default=None, help="Optional Phase0 warmed heads .safetensors to init model.heads before training.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Force tiny steps for test smoke (overrides --steps to small).")

    args = parser.parse_args(argv)

    if args.smoke:
        args.steps = min(args.steps, 12)
        args.log_every = min(args.log_every, 4)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Tree
    use_toy = False
    tree = None
    sym_to_token = token_to_sym = symbols = None
    tree_meta: Dict = {}
    if args.synthetic or (args.tree_config is None and args.gemma_model is None):
        tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
        use_toy = True
        print(f"[synthetic] toy grammar tree p={tree.p} depth={tree.depth} leaves={len(tree)}")
    elif args.tree_config:
        tree, tree_meta = load_tree_from_config(args.tree_config)
        # treat as "toy-like" for verify only if it contains all 21 toy tokens
        has_all_toy_tokens = False
        try:
            has_all_toy_tokens = all(tree.token_to_address(i) is not None for i in range(21))
        except KeyError:
            has_all_toy_tokens = False
        if has_all_toy_tokens:
            use_toy = True
            _, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
            print("[note] small toy tree; reusing toy sym maps for structural teacher/verify.")
    else:
        print("ERROR: provide --synthetic or --tree-config", file=sys.stderr)
        return 2

    # 2. Teacher
    teacher: Any = None
    precomputed = None
    if args.cached_dataset:
        from ultrametric_ce.distillation import load_dataset_cache
        precomputed = load_dataset_cache(args.cached_dataset)
        print(f"[cache] Loaded {len(precomputed)} batches from {args.cached_dataset}")
    elif args.gemma_model:
        if _load_gemma is None:
            print("ERROR: --gemma-model requires mlx-lm (load_gemma).", file=sys.stderr)
            return 2
        gm = args.gemma_model
        try:
            from ultrametric_ce.gemma_interface import find_local_gemma_on_storage as _fl  # type: ignore
            resolved = _fl(gm)
            if resolved:
                gm = resolved
                print(f"[real] Resolved short id to local storage copy: {gm}")
        except Exception:
            pass
        print(f"[real] loading Gemma: {gm} (auto; transformers for gemma-4)")
        import os
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = "/Volumes/Storage/huggingface_cache"
        try:
            iface = _load_gemma(gm, backend="auto", strict=False)
            _ = iface.get_logits([0])
            teacher = iface
            print(f"[real] teacher ready V~{iface.vocab_size}")
        except Exception as exc:
            print(f"[real] failed: {exc}", file=sys.stderr)
            print("Hint: use google/gemma-4-E2B-it or storage HF cache for gemma4 weights.")
            return 1
        # log real text batch count (per strategy)
        try:
            from ultrametric_ce.distillation import _get_tokenizer, iter_text_batches
            tok = _get_tokenizer(teacher)
            if tok is not None:
                rb = iter_text_batches(tree, teacher, tok, batch_size=args.batch, max_pairs=24, dim=args.dim, seed=args.seed)
                print(f"[real] real text batches for distillation: {len(rb)} batches (sequential from tokenizer)")
        except Exception:
            pass
    else:
        if use_toy and sym_to_token is not None:
            teacher = ToyStructuralTeacher(tree, sym_to_token, token_to_sym, symbols)
            print("[synthetic] ToyStructuralTeacher (structural bias for mock distillation)")
        else:
            class _Neutral:
                def __init__(self, V: int): self.V = V
                def get_logits(self, inp: Any): return np.zeros((self.V,), dtype=np.float32)
            max_tid = max([tree.address_to_token(a) for a in tree.leaf_addresses()] or [31]) + 1
            teacher = _Neutral(max_tid)
            print("[note] neutral mock teacher")

    # 3. Model init (optionally with warmed heads)
    model = UCEModel(tree, dim=args.dim, num_diff_layers=args.num_diff_layers, alpha=args.alpha)
    if args.heads_ckpt:
        try:
            heads = load_warmed_heads(args.heads_ckpt, tree.p, tree.depth, args.dim)
            model.heads = heads
            print(f"[heads] loaded Phase0 warm from {args.heads_ckpt}")
        except Exception as exc:
            print(f"[heads] load failed {exc}; continuing with random heads", file=sys.stderr)

    print(f"Starting Phase {args.phase}: steps={args.steps} lr={args.lr} batch={args.batch} freeze_heads={args.phase==1}")

    # 4. Run
    trained, logs = run_distillation_phase(
        model,
        teacher,
        tree,
        sym_to_token=sym_to_token,
        phase=args.phase,
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        log_every=args.log_every,
        seed=args.seed,
        dim=args.dim,
        precomputed_batches=precomputed,
    )

    # 5. Save full checkpoint + meta
    final_meta = {
        "phase": args.phase,
        "steps": args.steps,
        "source": "synthetic-toy" if use_toy else f"tree-config:{args.tree_config}",
        "heads_warmed": bool(args.heads_ckpt),
        "final_metrics": logs[-1] if logs else {},
    }
    out_p = save_full_checkpoint(trained, tree, args.out, meta=final_meta)
    print(f"Saved full UCE checkpoint to {out_p} (+ .meta.json)")

    # 6. Quick smoke verify load roundtrip
    tree2, model2 = load_full_checkpoint(out_p)
    _ = model2([])
    print(f"Checkpoint roundtrip load+forward OK (leaves={len(tree2)})")

    # 7. If synthetic, run a small structural verify (similar to phase0 script)
    if use_toy:
        from ultrametric_ce.evaluation import prefix_accuracy, structural_validity_rate
        held = []
        for expr in VALID_TOY_EXPRS[:3]:
            try:
                addrs = expr_to_address_sequence(expr, sym_to_token, tree)
                for i in range(1, min(len(addrs), 3)):
                    held.append((addrs[:i], addrs[i]))
            except Exception:
                pass
        pa = prefix_accuracy(model2, tree2, held)
        vr = structural_validity_rate(model2, tree2, token_to_sym, num_samples=4, max_len=5, seed=args.seed)
        print(f"[verify] post-train prefix_acc={pa:.3f} valid_rate={vr:.3f} on toy structural")

    print(f"\nPhase {args.phase} complete. Use checkpoint for generation/eval (Tasks 7/8).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
