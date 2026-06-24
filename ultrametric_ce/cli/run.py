#!/usr/bin/env python
"""Task 7 MVP generation script: load UCE checkpoint (safetensors + .meta.json), run sparse active-path generate.

Demonstrates:
- Loading full distilled (or toy) checkpoint via inference.load_model_and_tree
- Prompt -> addresses (toy symbols or real text via tokenizer from gemma id, filtered to registered leaves)
- Autoregressive generate() with digit routing + active ball tracking (pure UCE, no attention)
- Text decode back (real tokenizer or toy) + "active balls touched: X" sparsity logs

Real path (Gemma-4 tree from storage 12B 4bit etc): pass --gemma-model (resolved from storage cache) + real ckpt
  (tokenizer supplies encode/decode for high-tid Gemma tokens; generation still 100% UCE tree+diffusion+heads).

Example toy:
  ... --checkpoint /tmp/uce_distilled.safetensors --prompt "((1+2)*" --max-new 12

Example real (storage gemma4):
  ... --checkpoint /tmp/uce_gemma4.safetensors --gemma-model google/gemma-4-12B-it-4bit --prompt "The quick brown" --max-new 12

Uses only public APIs from ultrametric_ce.* .
"""

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from ultrametric_ce.inference import load_model_and_tree, generate
from ultrametric_ce.distillation import (
    build_toy_arithmetic_tree,
    expr_to_address_sequence,
    text_to_address_sequence,
    addresses_to_text,
)
from ultrametric_ce import real_gemma_contract as contract
# lazy for real tokenizer (only, no model)
_load_tok = None
try:
    from ultrametric_ce.gemma_interface import load_gemma_tokenizer as _load_tok  # type: ignore
except Exception:
    _load_tok = None
_find_local = None
try:
    from ultrametric_ce.gemma_interface import find_local_gemma_on_storage as _find_local  # type: ignore
except Exception:
    _find_local = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Task 7: sparse UCE generation demo (MVP).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to UCE .safetensors checkpoint (must have sibling .meta.json with tree).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="((1+2)*",
        help="Starting prompt string. For real Gemma-4 tree+tokenizer: natural text (will filter to registered tokens). For toy: symbols from grammar.",
    )
    parser.add_argument("--max-new", type=int, default=20, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-verbose", action="store_true", help="Suppress per-step active ball prints.")
    parser.add_argument(
        "--gemma-model",
        type=str,
        default=None,
        help="Optional: gemma id or local path (storage resolved) to load tokenizer for real tree prompt encode/decode. If omitted for real tree, falls back to warning and may fail decode.",
    )

    args = parser.parse_args(argv)

    mx.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[load] checkpoint {ckpt_path}")
    tree, model = load_model_and_tree(ckpt_path)
    print(f"[load] tree p={tree.p} depth={tree.depth} leaves={len(tree)}")

    # Load tokenizer (real) or toy syms for prompt<->text. Real path uses GemmaInterface tokenizer for high-tid Gemma models.
    tokenizer = None
    sym_to_token = token_to_sym = symbols = None
    use_real_tokenizer = False
    if args.gemma_model and _load_tok is not None:
        gm = args.gemma_model
        if _find_local is not None:
            try:
                resolved = _find_local(gm)
                if resolved:
                    gm = resolved
            except Exception:
                pass
        try:
            tokenizer = _load_tok(gm)
            use_real_tokenizer = tokenizer is not None
            print(f"[tokenizer] loaded real tokenizer (only, no model weights) from {gm}")
        except Exception as exc:
            print(f"[warn] could not load gemma tokenizer for encode/decode: {exc}; falling back (may only work for toy trees)")
    if not use_real_tokenizer:
        # toy path (unchanged behavior)
        _, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
        loaded_tids = {tree.address_to_token(a) for a in tree.leaf_addresses()}
        toy_tids = set(range(21))
        if not (loaded_tids & toy_tids):
            print("[warn] loaded tree tids do not look like toy grammar (0-20); prompt decode may fail or be meaningless for real trees without --gemma-model.", file=sys.stderr)

    # Convert prompt str to addresses: use real tokenizer helper (filters to registered) or toy expr helper
    try:
        if use_real_tokenizer and tokenizer is not None:
            _ = contract.assert_tree_talkable(tree, tokenizer, args.prompt)
            prompt_addrs = text_to_address_sequence(args.prompt, tokenizer, tree)
        else:
            prompt_addrs = expr_to_address_sequence(args.prompt, sym_to_token or {}, tree)
    except Exception as exc:
        print(f"[error] failed to map prompt to addresses: {exc}", file=sys.stderr)
        return 1
    print(f"[prompt] '{args.prompt}' -> {len(prompt_addrs)} addrs")

    # Generate with active path sparse execution
    print(f"[generate] max_new={args.max_new} temp={args.temperature} seed={args.seed}")
    new_addrs = generate(
        model,
        tree,
        prompt_addresses=prompt_addrs,
        max_new_tokens=args.max_new,
        temperature=args.temperature,
        seed=args.seed,
        verbose=not args.no_verbose,
    )

    # Reconstruct human string using real tokenizer or toy sym (supports high-tid real Gemma trees)
    full_addrs = prompt_addrs + new_addrs
    try:
        if use_real_tokenizer and tokenizer is not None:
            full_str = addresses_to_text(full_addrs, tokenizer, tree)
            prompt_str = addresses_to_text(prompt_addrs, tokenizer, tree)
            new_str = addresses_to_text(new_addrs, tokenizer, tree)
        else:
            full_str = "".join(token_to_sym[tree.address_to_token(a)] for a in full_addrs)
            prompt_str = "".join(token_to_sym[tree.address_to_token(a)] for a in prompt_addrs)
            new_str = "".join(token_to_sym[tree.address_to_token(a)] for a in new_addrs)
    except Exception as exc:
        print(f"[error] decode failed: {exc}", file=sys.stderr)
        return 1

    print(f"\n[prompt] {prompt_str}")
    print(f"[generated +{len(new_addrs)}] {new_str}")
    print(f"[full] {full_str}")

    # Small note on sparsity observed (logs per step from generate)
    print("\n[note] 'active balls touched: X' above are O(p * depth) per step vs full ball count (sparse UCE routing).")
    print("[note] All emitted addresses were registered leaves (enforced via tree.children live navigation + public API).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
