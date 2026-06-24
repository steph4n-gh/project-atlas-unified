#!/usr/bin/env python
"""Build a Finite p-adic tree by clustering token embeddings from Gemma (or synthetic).

This is the Task 4 script stub. It can be run immediately on synthetic data
(fully self-contained, no Gemma weights or mlx-lm required for --synthetic).

For real induction:
- User must have mlx-lm installed and supply Gemma weights (via --gemma-model
  path or mlx-community HF repo id such as 'mlx-community/gemma-2-2b-4bit').
- Gemma weights / checkpoints are NOT part of this repository.
- Loading quantized models is supported by choosing an appropriate repo id
  or local 4-bit/8-bit converted checkpoint (mlx_lm.load handles it).
- Full-vocab extraction + clustering on 100k+ tokens is memory/CPU intensive;
  start with --max-tokens for experiments or use activations sampling in future.

Example synthetic (TDD / smoke / no weights):
    cd /path/to/worktree
    source .venv/bin/activate
    PYTHONPATH=src python scripts/build_tree_from_gemma.py \
        --synthetic --p 2 --depth 2 --num-tokens 4 --out /tmp/synth_tree.json

Example real (user provides model):
    PYTHONPATH=src python scripts/build_tree_from_gemma.py \
        --gemma-model mlx-community/gemma-2-2b-4bit \
        --p 3 --depth 6 --out /tmp/gemma_induced_tree.json

The saved JSON contains:
  p, depth, source, num_tokens, address_map (string(addr) -> token_id),
  and a note for reconstruction.

Reconstruction (no pickle needed):
    import json
    from ultrametric_ce.tree import FiniteTree
    cfg = json.loads(open("...json").read())
    address_map = {int(a): int(t) for a, t in cfg["address_map"].items()}
    tree = FiniteTree(cfg["p"], cfg["depth"], address_map=address_map)
    # now tree.token_to_address etc. ready; pass to UCEModel etc.

See also:
- src/ultrametric_ce/gemma_interface.py (load + get_embeddings)
- src/ultrametric_ce/tree.py (cluster_and_assign_addresses)
- tests/test_tree.py (roundtrip + script smoke tests)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from ultrametric_ce.tree import FiniteTree
from ultrametric_ce import real_gemma_contract as contract

# Gemma import is lazy / guarded inside the gemma_interface (and here).
# We only import the symbol if --gemma-model is actually used.
_load_gemma = None
try:
    from ultrametric_ce.gemma_interface import load_gemma as _load_gemma  # type: ignore
except Exception:
    _load_gemma = None

_extract_snapshot = None
_find_local = None
_load_tok = None
try:
    from ultrametric_ce.gemma_interface import (
        extract_embeddings_from_mlx_snapshot as _extract_snapshot,
        find_local_gemma_on_storage as _find_local,
        load_gemma_tokenizer as _load_tok,
    )  # type: ignore
except Exception:
    _extract_snapshot = None
    _find_local = None
    _load_tok = None


def make_structured_synthetic_embeddings(
    n: int, dim: int = 8, p: int = 2, depth: int = 2, seed: int = 42
) -> np.ndarray:
    """Create tiny embeddings with explicit multi-scale clusters for p-ary splits.

    Offsets ensure that the deterministic max-var + chunk partition groups
    "similar" items under shared low digits (for roundtrip demo + test).
    """
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, dim)).astype(np.float32) * 0.03

    if n < 2:
        return embs

    # Top-level separation into p coarse clusters (affects lowest digit)
    chunk = max(1, n // p)
    for i in range(p):
        start = i * chunk
        end = min(start + chunk, n)
        if start >= end:
            continue
        offset = np.zeros(dim, dtype=np.float32)
        offset[0] = float(i) * 4.0  # dominant separation
        if dim > 1:
            offset[1] = float((i % 3) - 1) * 1.5
        embs[start:end] += offset

        # Finer sub-structure inside the coarse (affects higher digits)
        sub_n = end - start
        if sub_n >= 2 and depth > 1:
            sub_chunk = max(1, sub_n // p)
            for j in range(p):
                ss = start + j * sub_chunk
                se = min(start + (j + 1) * sub_chunk, end)
                if ss >= se:
                    continue
                sub_off = np.zeros(dim, dtype=np.float32)
                if dim > 1:
                    sub_off[1] = float(j) * 0.6
                else:
                    sub_off[0] += float(j) * 0.3
                embs[ss:se] += sub_off

    return embs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Induce FiniteTree via clustering of Gemma (or synthetic) embeddings."
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic embeddings (default behavior when --gemma-model omitted).",
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default=None,
        metavar="PATH_OR_REPO",
        help="Gemma via mlx_lm (e.g. 'mlx-community/gemma-2-2b-4bit' or local dir). "
             "User must supply the model/weights; nothing is bundled.",
    )
    parser.add_argument("--p", type=int, default=2, help="p (branching, >=2).")
    parser.add_argument("--depth", type=int, default=3, help="Tree depth K (p**K >= #tokens).")
    parser.add_argument(
        "--num-tokens", type=int, default=8, help="Size of synthetic token set to cluster."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tree_config.json",
        help="Where to write the JSON config (address_map etc.).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG for synthetic only.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="When using --gemma-model, number of high-tid leaves to induce (uses contract for high ids + prompt overlap).",
    )
    parser.add_argument(
        "--seed-prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog",
        help="Natural text prompt whose tokens are guaranteed included in the induced tree (for talkable real path).",
    )

    args = parser.parse_args(argv)

    p = args.p
    depth = args.depth
    if p < 2 or depth < 1:
        print("ERROR: p>=2 and depth>=1 required", file=sys.stderr)
        return 2

    print("[debug] PYTHONPATH:", os.environ.get("PYTHONPATH"))
    print("[debug] sys.path:", sys.path)
    use_real = bool(args.gemma_model)

    if use_real:
        # Prefer direct snapshot extract for local mlx-community gemma-4 4bit dirs on storage
        # (avoids mlx_lm.load "parameters not in model" for gemma4 k_norm etc).
        gm = args.gemma_model
        if _find_local is not None:
            resolved = _find_local(gm)
            if resolved:
                gm = resolved
                print(f"[real] Resolved to local on storage drive: {gm}")
        used_direct = False
        full_embs = None
        try:
            model_path = Path(gm)
            if model_path.exists() and _extract_snapshot is not None:
                # loose check: extract func itself will search for safetensors (supports HF snapshot, flat scratch dirs on storage, or .safetensors file)
                print(f"[real] Direct embed extract from mlx snapshot (storage cache friendly): {gm}")
                full_embs = _extract_snapshot(gm)
                used_direct = True
        except Exception as e:
            print(f"[real] direct snapshot extract failed, will try load: {e}")
        if not used_direct:
            if _load_gemma is None:
                print(
                    "ERROR: Could not import load_gemma (mlx_lm probably missing or "
                    "gemma_interface import failed). Install mlx-lm and retry, or use --synthetic.",
                    file=sys.stderr,
                )
                return 2
            print(f"[real] Loading Gemma: {gm} (quantized if you chose a 4-bit repo; transformers for gemma-4)")
            try:
                # backend=auto will pick transformers for google/gemma-4* ; mlx otherwise
                iface = _load_gemma(gm, backend="auto", strict=False)
                full_embs = iface.get_embeddings()
            except Exception as exc:
                print(f"[real] Gemma load/extract failed: {exc}", file=sys.stderr)
                print(
                    "Hints: ensure the model id/path is correct, you have internet/HF token if needed, "
                    "and enough unified memory (quantized 2B ~1-2 GB). For gemma-4 try google/gemma-4-E2B-it "
                    "(transformers path) or pass local mlx snapshot dir for embed-only. "
                    "Set HF_HOME=/Volumes/Storage/huggingface_cache to target storage drive cache. "
                    "For testing the pipeline use --synthetic.",
                    file=sys.stderr,
                )
                return 1
        print(f"[real] Embeddings extracted: {full_embs.shape} (D={full_embs.shape[1]})")
        if args.max_tokens is not None and args.max_tokens < full_embs.shape[0]:
            V = full_embs.shape[0]
            n = args.max_tokens
            # contract: high ids + prompt overlap (no inline hardcoded sentence, no demo_text)
            prompt_tok = []
            try:
                tok = None
                if _load_tok is not None:
                    tok = _load_tok(gm)
                elif 'iface' in locals() and hasattr(iface, 'tokenizer'):
                    tok = iface.tokenizer
                if tok is not None:
                    ids = tok.encode(args.seed_prompt)
                    if isinstance(ids, dict) and 'input_ids' in ids:
                        ids = ids['input_ids']
                    if hasattr(ids, 'tolist'):
                        ids = ids.tolist()
                    if ids and isinstance(ids[0], (list, tuple)):
                        ids = ids[0]
                    prompt_tok = [int(x) for x in ids if 0 <= x < V]
            except Exception as e:
                print(f"[debug] prompt token extraction failed: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
            token_ids = contract.select_induction_token_ids(V, n, prompt_tok)
            # fetch via contract (avoids unbound iface)
            embs_arr = contract.fetch_embeddings_for_ids(
                gm, token_ids,
                used_direct=used_direct,
                iface=iface if 'iface' in locals() else None,
                extract_func=_extract_snapshot if used_direct else None,
            )
            embs = embs_arr
        else:
            embs = full_embs
            token_ids = list(range(embs.shape[0]))
        source = f"gemma:{gm}"
    else:
        n = int(args.num_tokens)
        embs = make_structured_synthetic_embeddings(n, dim=8, p=p, depth=depth, seed=args.seed)
        token_ids = list(range(n))
        source = "synthetic"
        print(f"[synthetic] Generated structured embeddings: {embs.shape}")

    if len(token_ids) > p ** depth:
        print(f"ERROR: {len(token_ids)} tokens > capacity p**depth = {p**depth}", file=sys.stderr)
        return 2

    print(f"Clustering (p={p}, depth={depth}) on {len(token_ids)} tokens ...")
    tree = FiniteTree.cluster_and_assign_addresses(
        embs, p=p, depth=depth, token_ids=token_ids
    )
    print(f"Induced tree: {len(tree)} leaves registered.")

    # address_map for persistence: {str(addr): token_id} so JSON serializable
    # (reconstruct via {int(a):int(t) for a,t ...} into FiniteTree(..., address_map=that)
    address_map: dict[str, int] = {}
    for tid in token_ids:
        addr = tree.token_to_address(tid)
        address_map[str(addr)] = int(tid)

    cfg = {
        "p": p,
        "depth": depth,
        "source": source,
        "num_tokens": len(token_ids),
        "address_map": address_map,
        "note": (
            "Rebuild: from ultrametric_ce.tree import FiniteTree; "
            "import json; cfg=json.load(open('this.json')); "
            "am={int(a):int(t) for a,t in cfg['address_map'].items()}; "
            "tree=FiniteTree(cfg['p'], cfg['depth'], address_map=am). "
            "Gemma weights not included in repo -- user supplied for real runs. "
            "See docs and gemma_interface.py."
        ),
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(cfg, indent=2, sort_keys=True))
    print(f"Wrote {out_p}")

    # Self-test roundtrip using the saved map (and a bit of hierarchy for synthetic)
    try:
        am = {int(a): int(t) for a, t in address_map.items()}
        tree_rt = FiniteTree(p, depth, address_map=am)
        assert len(tree_rt) == len(tree)
        print("Roundtrip (json -> FiniteTree) OK.")
        if source == "synthetic" and len(token_ids) >= 2:
            a0 = tree.token_to_address(0)
            a1 = tree.token_to_address(min(1, len(token_ids)-1))
            lca = tree.lca_depth(a0, a1)
            print(f"  Synthetic example LCA (tokens 0 and 1): {lca} (higher is tighter cluster)")
    except Exception as ve:
        print(f"Roundtrip self-check warning: {ve}", file=sys.stderr)

    print("Success. Use the produced JSON with later distillation / model scripts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
