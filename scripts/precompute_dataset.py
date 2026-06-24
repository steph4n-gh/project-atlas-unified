#!/usr/bin/env python
"""Precompute dataset caches for UCE distillation.

Loads the FiniteTree and the designated teacher model, runs forward passes
over a text corpus, extracts logits and target hidden states, and serializes
the resulting batches to a binary file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

import mlx.core as mx

# Insert src directory
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.distillation import (
    build_toy_arithmetic_tree,
    ToyStructuralTeacher,
    iter_synthetic_batches,
    iter_text_batches,
    serialize_dataset_cache,
)

# Lazy loading of real gemma
_load_gemma = None
try:
    from ultrametric_ce.gemma_interface import load_gemma as _load_gemma, load_gemma_tokenizer
except Exception:
    _load_gemma = None


def load_tree_from_config(cfg_path: str | Path) -> FiniteTree:
    cfg = json.loads(Path(cfg_path).read_text())
    p = int(cfg["p"])
    depth = int(cfg["depth"])
    am = {int(a): int(t) for a, t in cfg["address_map"].items()}
    tree = FiniteTree(p, depth, address_map=am)
    print(f"[tree] Loaded p={p} depth={depth} leaves={len(tree)}")
    return tree


def gather_local_code_corpus(root_path: Path) -> list[str]:
    """Gathers all Python files in the repository to form a high-quality structural text corpus."""
    texts = []
    # Walk src and tests
    for subdir in ["src", "tests", "tests_e2e"]:
        path = root_path / subdir
        if not path.exists():
            continue
        for pfile in path.glob("**/*.py"):
            try:
                content = pfile.read_text(encoding="utf-8")
                # Split content into smaller chunks (e.g., classes/functions or paragraphs)
                # to feed into tokenizer as individual corpus inputs
                chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 10]
                texts.extend(chunks)
            except Exception:
                pass
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-compute UCE distillation dataset caches.")
    parser.add_argument("--tree-config", type=str, default=None, help="Path to tree_config.json")
    parser.add_argument("--gemma-model", type=str, default=None, help="Gemma model id/path for real teacher")
    parser.add_argument("--synthetic", action="store_true", help="Run on toy synthetic grammar tree and mock teacher")
    parser.add_argument("--num-tokens", type=int, default=100000, help="Maximum number of target pairs/tokens to cache")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for UCE training batches")
    parser.add_argument("--dim", type=int, default=16, help="Model/projection dimension")
    parser.add_argument("--out", type=str, required=True, help="Output path for the serialized cache (e.g. e2b_cache.pkl)")
    parser.add_argument("--corpus-file", type=str, default=None, help="Optional text file containing training sentences")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load the Tree
    if args.synthetic or (args.tree_config is None and args.gemma_model is None):
        tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
        use_toy = True
    elif args.tree_config:
        tree = load_tree_from_config(args.tree_config)
        sym_to_token = token_to_sym = symbols = None
        use_toy = False
    else:
        print("ERROR: Must provide --tree-config or --synthetic", file=sys.stderr)
        return 2

    # 2. Build or Load Batches
    if args.synthetic:
        print("[synthetic] Generating toy synthetic grammar batches...")
        teacher = ToyStructuralTeacher(tree, sym_to_token, token_to_sym, symbols)
        # Use synthetic helper
        batches = iter_synthetic_batches(
            tree, teacher, sym_to_token=sym_to_token, batch_size=args.batch_size,
            max_pairs=args.num_tokens, dim=args.dim, seed=args.seed
        )
    else:
        # Real Gemma path
        if not args.gemma_model:
            print("ERROR: --gemma-model is required for real dataset pre-computation.", file=sys.stderr)
            return 2
        if _load_gemma is None:
            print("ERROR: Real Gemma model requires mlx-lm / transformers package.", file=sys.stderr)
            return 2

        gm = args.gemma_model
        # Resolve short id if on storage drive
        try:
            from ultrametric_ce.gemma_interface import find_local_gemma_on_storage as fl
            res = fl(gm)
            if res:
                gm = res
        except Exception:
            pass

        print(f"[real] Loading teacher model: {gm}")
        os.environ.setdefault("HF_HOME", "/Volumes/Storage/huggingface_cache")
        teacher = _load_gemma(gm, backend="auto", strict=False)
        tokenizer = load_gemma_tokenizer(gm)

        # Build corpus
        canned_texts = None
        if args.corpus_file:
            print(f"[real] Reading corpus from file: {args.corpus_file}")
            path = Path(args.corpus_file)
            if path.suffix == ".jsonl":
                canned_texts = []
                max_lines = int(args.num_tokens * 1.5)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                text = f"{data.get('dialect', '')} {data.get('instruction', '')} {data.get('pattern', '')}"
                                canned_texts.append(text)
                                if len(canned_texts) >= max_lines:
                                    break
                            except Exception:
                                pass
                print(f"[real] Parsed {len(canned_texts)} JSONL records from {args.corpus_file} (early break optimized)")
            else:
                canned_texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            print("[real] No corpus-file provided. Gathering local codebase files (.py) as a structural corpus...")
            canned_texts = gather_local_code_corpus(ROOT)
            print(f"[real] Gathered {len(canned_texts)} code/text chunks from repository.")

        print(f"[real] Pre-computing targets with teacher (limit={args.num_tokens} pairs)...")
        batches = iter_text_batches(
            tree, teacher, tokenizer, batch_size=args.batch_size,
            max_pairs=args.num_tokens, dim=args.dim, seed=args.seed,
            canned_texts=canned_texts
        )

    num_pairs = sum(len(b) for b in batches)
    print(f"Pre-computation complete. Generated {len(batches)} batches containing {num_pairs} total pairs.")

    # 3. Serialize
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialize_dataset_cache(batches, str(out_path))
    print(f"Successfully saved pre-computed dataset cache to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
