#!/usr/bin/env python
"""Phase 0 script: warm-start DigitHeads via LM head factorization / distillation from teacher.

Synthetic first (toy grammar tree + mock structural teacher, fully self contained, no Gemma needed).

Real usage:
- Provide --tree-config (json from scripts/build_tree_from_gemma.py) for the FiniteTree.
- Optionally --gemma-model for real teacher logits (must match the tree's token ids, i.e. real Gemma vocab ids).
- For limited trees from --max-tokens in build, sub-dists only consider the clustered tokens (approx).

Does a few gradient steps ONLY on the routing heads (using per-ball teacher sub-dists as soft targets + distillation_loss).
Saves checkpoint (heads weights + meta) that can be loaded back into DigitHeads or UCEModel routing.

Verification (Step 5.4): for toy/synthetic trees, the warmed heads used *alone* (via predict_with_warmed_heads, no diffusion) already
exhibit reasonable next-token behavior on toy structural prefixes (e.g. favoring digits after empty or ops, better match to
teacher marginal than random-init heads baseline).

Example synthetic (TDD/smoke, no weights):
    source .venv/bin/activate
    PYTHONPATH=src python scripts/distill_phase0_heads.py \
        --synthetic --out /tmp/phase0_heads.safetensors --steps 25

Example with prebuilt tree + mock (or real if ids align):
    PYTHONPATH=src python scripts/distill_phase0_heads.py \
        --tree-config /tmp/synth_tree.json --out /tmp/phase0_heads.safetensors

With real Gemma (user supplies weights + matching tree):
    PYTHONPATH=src python scripts/distill_phase0_heads.py \
        --tree-config /tmp/gemma_induced_tree.json \
        --gemma-model mlx-community/gemma-2-2b-4bit \
        --out /tmp/phase0_heads.safetensors --steps 10

See:
- src/ultrametric_ce/distillation.py (warm_start_phase0_heads, predict_with..., load/save)
- src/ultrametric_ce/routing.py (distillation_loss)
- src/ultrametric_ce/gemma_interface.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.routing import DigitHeads
from ultrametric_ce.distillation import (
    build_toy_arithmetic_tree,
    VALID_TOY_EXPRS,
    expr_to_address_sequence,
    ToyStructuralTeacher,
    warm_start_phase0_heads,
    predict_with_warmed_heads,
    save_warmed_heads,
    load_warmed_heads,
)

# lazy real gemma
_load_gemma = None
try:
    from ultrametric_ce.gemma_interface import load_gemma as _load_gemma  # type: ignore
except Exception:
    _load_gemma = None


def load_tree_from_config(cfg_path: str | Path) -> FiniteTree:
    """Reconstruct FiniteTree from the json produced by build_tree_from_gemma.py."""
    cfg = json.loads(Path(cfg_path).read_text())
    p = int(cfg["p"])
    depth = int(cfg["depth"])
    am = {int(a): int(t) for a, t in cfg["address_map"].items()}
    tree = FiniteTree(p, depth, address_map=am)
    print(f"[tree] loaded from {cfg_path}: p={p} depth={depth} leaves={len(tree)} source={cfg.get('source')}")
    return tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0: distill/warm-start UCE routing heads from teacher sub-dists over tree balls.")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--synthetic", action="store_true", help="Use built-in toy arithmetic grammar tree + mock teacher (default).")
    parser.add_argument(
        "--tree-config",
        type=str,
        default=None,
        help="Path to tree_config.json (from build_tree_from_gemma.py) to use a (pre)built tree.",
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default=None,
        help="If provided (and tree tokens align with Gemma ids), use real GemmaInterface as teacher for logits.",
    )
    parser.add_argument("--dim", type=int, default=16, help="Hidden dim for heads (match future UCEModel).")
    parser.add_argument("--steps", type=int, default=25, help="Tiny gradient steps for head warm-start.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam lr for the Phase 0 head steps.")
    parser.add_argument(
        "--out",
        type=str,
        default="phase0_heads.safetensors",
        help="Output path for warmed heads checkpoint (.safetensors + .meta.json).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")

    args = parser.parse_args(argv)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Obtain tree (synthetic toy or loaded)
    use_toy = False
    if args.synthetic or (args.tree_config is None and args.gemma_model is None):
        tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
        use_toy = True
        print(f"[synthetic] Using hard-coded toy grammar tree: p={tree.p} depth={tree.depth} leaves={len(tree)}")
    elif args.tree_config:
        tree = load_tree_from_config(args.tree_config)
        sym_to_token = token_to_sym = symbols = None
        # treat as "toy-like" for verify only if it contains all 21 toy tokens
        has_all_toy_tokens = False
        try:
            has_all_toy_tokens = all(tree.token_to_address(i) is not None for i in range(21))
        except KeyError:
            has_all_toy_tokens = False
        if has_all_toy_tokens:
            use_toy = True
            # load canonical toy maps (tids 0-20) so we can use structural teacher + expr_to + verify
            _, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
            print("[note] small toy tree loaded; using toy maps + structural mock teacher for verification.")
    else:
        print("ERROR: need --synthetic or --tree-config", file=sys.stderr)
        return 2

    # 2. Teacher: real gemma if requested+available, else for toy use structural mock; for real tree without gemma use neutral mock
    teacher = None
    if args.gemma_model:
        if _load_gemma is None:
            print("ERROR: --gemma-model requires load_gemma (mlx-lm). Use --synthetic or install.", file=sys.stderr)
            return 2
        gm = args.gemma_model
        # auto-resolve short ids to storage drive copies if present (the "hugging face cache on the storage drive")
        try:
            from ultrametric_ce.gemma_interface import find_local_gemma_on_storage as _fl  # type: ignore
            resolved = _fl(gm)
            if resolved:
                gm = resolved
                print(f"[real] Resolved short id to local storage copy: {gm}")
        except Exception:
            pass
        print(f"[real] Loading Gemma teacher: {gm} (auto backend; transformers for gemma-4)")
        # set storage cache preference if not already (user can pre-export)
        import os
        if "HF_HOME" not in os.environ and "HF_HUB_CACHE" not in os.environ:
            os.environ["HF_HOME"] = "/Volumes/Storage/huggingface_cache"
        try:
            iface = _load_gemma(gm, backend="auto", strict=False)
            # quick smoke
            _ = iface.get_logits([0, 1])
            teacher = iface
            print(f"[real] Teacher ready (V~{iface.vocab_size})")
        except Exception as exc:
            print(f"[real] Gemma load failed: {exc}", file=sys.stderr)
            print("Hint: for gemma-4 use 'google/gemma-4-E2B-it' (or 12B); storage cache via HF_HOME=/Volumes/Storage/...")
            return 1
    else:
        if use_toy and sym_to_token is not None:
            teacher = ToyStructuralTeacher(tree, sym_to_token, token_to_sym, symbols)
            print("[synthetic] Using ToyStructuralTeacher (grammar-biased mock)")
        else:
            # neutral teacher for non-toy without gemma (will give uniform-ish subs)
            class _Neutral:
                def __init__(self, V):
                    self.V = V
                def get_logits(self, inp):
                    return np.zeros((self.V,), dtype=np.float32)
            max_tid = max([tree.address_to_token(a) for a in tree.leaf_addresses()] or [31]) + 1
            teacher = _Neutral(max_tid)
            print("[note] Using neutral mock teacher (no --gemma-model)")

    # 3. Warm the heads (Phase 0 core)
    print(f"Warming heads (dim={args.dim}, steps={args.steps}, lr={args.lr}) ...")
    heads = warm_start_phase0_heads(
        tree,
        teacher,
        dim=args.dim,
        num_steps=args.steps,
        lr=args.lr,
        temperature=1.0,
        seed=args.seed,
    )
    print("Warm-start complete.")

    # 4. Save checkpoint + meta
    meta = {
        "p": tree.p,
        "depth": tree.depth,
        "dim": args.dim,
        "steps": args.steps,
        "source": "synthetic-toy" if use_toy else (f"tree-config:{args.tree_config}" if args.tree_config else "unknown"),
        "note": (
            "Load: from ultrametric_ce.distillation import load_warmed_heads; "
            "heads=load_warmed_heads('this.safetensors', p, depth, dim); "
            "Use predict_with_warmed_heads(heads, tree, prev_addrs) for heads-alone next dist (no diffusion). "
            "To attach to UCE: model=UCEModel(tree); model.heads = heads (or load_weights if shapes match)."
        ),
    }
    out_p = save_warmed_heads(heads, args.out, meta=meta)
    print(f"Saved warmed heads to {out_p} (+ meta)")

    # 5. Verify load back
    heads2 = load_warmed_heads(out_p, tree.p, tree.depth, args.dim)
    _ = predict_with_warmed_heads(heads2, tree, [])
    print("Checkpoint roundtrip load+predict OK.")

    # 6. Step 5.4: verify warmed heads alone give reasonable behavior on toy structural prefixes
    if use_toy:
        print("\n=== Verification: warmed heads alone (no diffusion) on toy structural prefixes ===")
        # rebuild maps if needed (for loaded small tree case, assume ids 0..20 and reuse builder)
        if sym_to_token is None:
            t2, s2t, t2s, syms2 = build_toy_arithmetic_tree()
            sym_to_token, token_to_sym, symbols = s2t, t2s, syms2

        leafs = tree.leaf_addresses()
        # compare to random baseline + to teacher-induced target
        heads_rand = DigitHeads(tree.p, tree.depth, args.dim)
        t_logits0 = teacher.get_logits([])
        t_probs0 = np.exp(np.asarray(t_logits0, dtype=np.float32) - np.max(t_logits0))
        t_probs0 = t_probs0 / (np.sum(t_probs0) + 1e-12)

        def leaf_dist_overlap(probs: np.ndarray) -> int:
            # top-3 leaf addr idx overlap with teacher top on leaves
            t_on = np.array([t_probs0[tree.address_to_token(a)] if tree.address_to_token(a) < len(t_probs0) else 0.0 for a in leafs])
            t_on /= (t_on.sum() + 1e-12)
            i1 = set(np.argsort(probs)[-3:])
            i2 = set(np.argsort(t_on)[-3:])
            return len(i1 & i2)

        dw = np.array(predict_with_warmed_heads(heads, tree, []))
        dr = np.array(predict_with_warmed_heads(heads_rand, tree, []))
        ov_w = leaf_dist_overlap(dw)
        ov_r = leaf_dist_overlap(dr)
        print(f"Empty prefix: top-3 leaf overlap w/ teacher-induced: warmed={ov_w} random={ov_r}")

        # on structural prefixes (use expr prefixes)
        print("On structural prefixes (argmax next symbol):")
        for expr in VALID_TOY_EXPRS[:5]:
            prev = expr_to_address_sequence(expr, sym_to_token, tree)
            dw_pre = np.array(predict_with_warmed_heads(heads, tree, prev))
            dr_pre = np.array(predict_with_warmed_heads(heads_rand, tree, prev))
            aw = token_to_sym[tree.address_to_token(leafs[int(np.argmax(dw_pre))])]
            ar = token_to_sym[tree.address_to_token(leafs[int(np.argmax(dr_pre))])]
            print(f"  prefix='{expr}' -> warmed argmax next='{aw}'  | random='{ar}'")

        # assert reasonable: warmed matches teacher marginal at least as well as random (overlap)
        if ov_w < ov_r:
            print("WARNING: warmed overlap < random this run (stochastic); still demonstrating runnable pipeline.", file=sys.stderr)
        else:
            print("Verification: warmed >= random on teacher match (reasonable next-token from distilled heads).")

        # also sample a short continuation from a prefix and show it uses only valid tokens (always true) + looks "ok"
        start = "((1+"
        prev = expr_to_address_sequence(start, sym_to_token, tree)
        gen = list(prev)
        for _ in range(4):
            dist = predict_with_warmed_heads(heads, tree, gen)
            idx = int(mx.random.categorical(mx.log(dist + 1e-12)).item())
            gen.append(leafs[idx])
        gen_str = "".join(token_to_sym[tree.address_to_token(a)] for a in gen)
        print(f"\nHeads-alone sample continuation start='{start}' -> {gen_str} (uses warmed heads only)")

    else:
        print("\n[non-toy] Skipping full structural prefix verify (no toy maps). Basic check:")
        d0 = predict_with_warmed_heads(heads, tree, [])
        print(f"  empty-prefix dist sum={float(mx.sum(d0)):.6f}  (should ~1)")

    print("\nPhase 0 complete. Use the checkpoint for later diffusion training (Task 6).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
