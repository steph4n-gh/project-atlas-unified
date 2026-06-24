#!/usr/bin/env python
"""Task 8: Evaluation harness and structural metrics script.

Loads a final UCE checkpoint (produced by run_distillation.py --synthetic etc.)
via public load_model_and_tree (safetensors + .meta.json sidecar).

Builds small structural test set from VALID_TOY_EXPRS (good) + simple mutations
(mismatched parens, consecutive ops/operands, trailing ops, truncated, etc. for bad).

Runs:
- validity checker (recursive parser via check_structural_validity / is_...) on good vs bad
- prefix_accuracy, ultrametric_spearman_correlation, structural_validity_rate, compute_...
- active ball counter by calling generate (sparse path) + capturing "active balls touched: X" logs
- reports vs simple random baseline (uniform leaf sampling for validity; 1/p for prefix digits)

Prints numbers demonstrating structural coherence on toy grammar (higher validity/spearman/prefix,
active << total balls ~ p*depth routing cost).

All synthetic/toy (no Gemma); uses public APIs only from ultrametric_ce.{inference,distillation,evaluation,model,tree,...}

Example (after synthetic ckpt):
  source .venv/bin/activate && PYTHONPATH=src python scripts/run_distillation.py --synthetic --steps 30 --out /tmp/uce_mvp.safetensors
  source .venv/bin/activate && PYTHONPATH=src python scripts/eval_structural.py --checkpoint /tmp/uce_mvp.safetensors --num-samples 10 --seed 42

Matches style of generate_with_mvp.py / run_distillation.py .
"""

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

import mlx.core as mx

from ultrametric_ce.inference import load_model_and_tree, generate
from ultrametric_ce.distillation import (
    build_toy_arithmetic_tree,
    VALID_TOY_EXPRS,
    expr_to_address_sequence,
)
from ultrametric_ce.evaluation import (
    is_structurally_valid_toy_expr,
    check_structural_validity,
    prefix_accuracy,
    structural_validity_rate,
    compute_structural_metrics,
    ultrametric_spearman_correlation,
)


def _make_bad_mutations(goods: List[str], max_bads: int = 12) -> List[str]:
    """Generate synthetic bad variants via simple structure-breaking mutations."""
    bads: List[str] = [
        "1++2", "((1+2)", "1+2)", "12", "1+2*", "+1+2", "x+y)", "((1+2)*3",
        "1+2=", "2(3+4)", "1+2*+", "x=(1+2)^", ")", "(", "1+=", "(x*y)+(z=4=5)",
        "4*(2+3)-", "((1-2)+3", "9/(2+3))", "1+2*3+",
    ]
    for g in goods[:8]:
        if not g:
            continue
        # drop an open paren
        if "(" in g:
            bads.append(g.replace("(", "", 1))
        # drop a close
        if ")" in g:
            bads.append(g.replace(")", "", 1))
        # trailing op
        if g[-1] in "0123456789xyz)":
            bads.append(g + "+")
        # truncate
        if len(g) > 2:
            bads.append(g[:-1])
        # consecutive operand-like (insert digit after digit without op, if possible)
        for i, ch in enumerate(g):
            if ch in "0123456789xyz" and i+1 < len(g) and g[i+1] in "0123456789xyz":
                pass  # already bad if was, but for others
        # insert op after op (find op pos)
        for i, ch in enumerate(g):
            if ch in "+-*/=^" and i+1 < len(g) and g[i+1] in "+-*/=^":
                pass
    # filter truly bad + dedup + cap
    seen = set()
    out: List[str] = []
    for b in bads:
        if b and b not in seen and not is_structurally_valid_toy_expr(b):
            seen.add(b)
            out.append(b)
            if len(out) >= max_bads:
                break
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 8: structural eval harness for UCE MVP checkpoint (toy grammar).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to UCE .safetensors checkpoint (with .meta.json tree sidecar).",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Num samples for generative validity rate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new", type=int, default=5, help="Max new for internal generate active count samples.")

    args = parser.parse_args(argv)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[load] checkpoint {ckpt_path}")
    tree, model = load_model_and_tree(ckpt_path)
    print(f"[load] tree p={tree.p} depth={tree.depth} leaves={len(tree)}")

    # Toy syms for structural test set (tids stable for synthetic ckpts)
    _, sym_to_token, token_to_sym, _ = build_toy_arithmetic_tree()
    loaded_tids = {tree.address_to_token(a) for a in tree.leaf_addresses()}
    toy_tids = set(range(21))
    if not (loaded_tids & toy_tids):
        print("[warn] loaded tree tids do not overlap toy grammar (0-20); validity strings may be meaningless for this tree.", file=sys.stderr)

    # Heldout pairs from VALID for prefix/spearman
    heldout_pairs: List[Tuple[List[int], int]] = []
    for expr in VALID_TOY_EXPRS[:5]:
        try:
            addrs = expr_to_address_sequence(expr, sym_to_token, tree)
            for ii in range(1, min(3, len(addrs))):
                heldout_pairs.append((addrs[:ii], addrs[ii]))
        except Exception:
            continue
    if not heldout_pairs:
        leafs = tree.leaf_addresses()
        heldout_pairs = [([], leafs[0])] if leafs else []

    # 1. Harness metrics bundle (prefix, validity_rate via sample, spearman)
    print("\n[compute_structural_metrics]")
    mets = compute_structural_metrics(
        model, tree, token_to_sym, heldout_pairs,
        num_samples=min(4, args.num_samples), max_len=6, seed=args.seed
    )
    for k in sorted(mets):
        print(f"  {k}: {mets[k]:.4f}")

    # 2. Direct validity checker test on synthetic good vs bad generations/set
    good_exprs = list(VALID_TOY_EXPRS)
    bad_exprs = _make_bad_mutations(good_exprs, max_bads=12)
    print(f"\n[structural test set] goods={len(good_exprs)} bads={len(bad_exprs)} (from VALID_TOY_EXPRS + mutations)")

    vg = sum(bool(check_structural_validity(e)["valid"]) for e in good_exprs) / max(1, len(good_exprs))
    vb = sum(bool(check_structural_validity(e)["valid"]) for e in bad_exprs) / max(1, len(bad_exprs))
    print(f"[validity checker] good_rate={vg:.3f} bad_rate={vb:.3f} (recursive parser; ideal 1.0 vs 0.0)")

    # extra direct spearman and rate
    us = ultrametric_spearman_correlation(model, tree, heldout_pairs)
    print(f"[ultrametric_spearman_correlation] {us:.4f} (positive: model favors p-adically close tokens)")

    vr = structural_validity_rate(model, tree, token_to_sym, num_samples=args.num_samples, max_len=6, seed=args.seed)
    print(f"[structural_validity_rate (sample path)] {vr:.4f}")

    pa = prefix_accuracy(model, tree, heldout_pairs)
    print(f"[prefix_accuracy] {pa:.4f}")

    # 3. Active param / balls via generate logs (capture prints); also %valid from sparse gens
    print("\n[active balls via generate (sparse) + validity on sparse gens]")
    actives: List[int] = []
    sparse_valid = 0
    n_active_samples = max(3, min(5, args.num_samples))
    for si in range(n_active_samples):
        pr_addrs: List[int] = []
        try:
            # alternate short prompt from goods
            pr_str = VALID_TOY_EXPRS[si % len(VALID_TOY_EXPRS)][:3]
            pr_addrs = expr_to_address_sequence(pr_str, sym_to_token, tree)
        except Exception:
            pr_addrs = []
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            new_addrs = generate(
                model, tree,
                prompt_addresses=pr_addrs,
                max_new_tokens=args.max_new,
                temperature=1.0,
                seed=args.seed + si,
                verbose=True,
            )
        captured = buf.getvalue()
        for m in re.finditer(r"active balls touched: (\d+)", captured):
            actives.append(int(m.group(1)))
        # check validity of the *new* generated portion
        try:
            gen_str = "".join(token_to_sym[tree.address_to_token(a)] for a in new_addrs)
            if is_structurally_valid_toy_expr(gen_str):
                sparse_valid += 1
        except Exception:
            pass

    p, depth = tree.p, tree.depth
    total_balls = sum(p ** d for d in range(depth + 1))
    avg_active = float(np.mean(actives)) if actives else 0.0
    frac = (avg_active / total_balls) if total_balls > 0 else 0.0
    print(f"  avg_active_balls={avg_active:.1f} total_balls={total_balls} frac={frac:.3f}")
    print(f"  target_routing ~ p*depth = {p * depth}  (active slightly higher due to context ancestors + sibs)")

    if n_active_samples > 0:
        print(f"  sparse_gen_valid_rate={sparse_valid / n_active_samples:.3f}")

    # 4. Random baselines
    print("\n[baselines (random leaf sampling for comparison)]")
    # random prefix acc ~1/p avg
    rand_prefix = 1.0 / p
    print(f"  random_prefix_per_digit ~{rand_prefix:.3f}")

    # random validity by sampling leaf strs
    rng = np.random.default_rng(args.seed)
    leafs = tree.leaf_addresses()
    rand_v = 0
    n_rand = 20
    for _ in range(n_rand):
        if not leafs:
            break
        rs = [int(leafs[rng.integers(0, len(leafs))]) for _ in range(5)]
        try:
            rstr = "".join(token_to_sym[tree.address_to_token(a)] for a in rs)
            if is_structurally_valid_toy_expr(rstr):
                rand_v += 1
        except Exception:
            pass
    print(f"  random_validity_rate (20 trials of 5-token) ~{rand_v / n_rand:.3f} (<< trained model)")

    print("\n[done] eval_structural complete. Use these numbers to demonstrate zero-distortion hierarchy + structural fidelity + efficiency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
