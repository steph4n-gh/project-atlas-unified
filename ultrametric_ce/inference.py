"""Inference and generation for UCE: autoregressive sampling with active-path sparse execution.

Uses the tree for registered leaves + live child navigation (ensures only valid addresses),
MLX categorical sampling on per-digit heads (O(p) per depth level), and active ball tracking
for diffusion + heads (path ancestors + p siblings at decision depths).

Supports loading full checkpoints (UCE .safetensors + .meta.json tree sidecar) via load_model_and_tree.

Public API only; no peeking at privates in tree/model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from ultrametric_ce.model import UCEModel, WeightManager
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.padic import address_to_digits

__all__ = ["generate", "load_model_and_tree"]


def load_model_and_tree(
    ckpt_path: str | Path,
    meta_path: Optional[str | Path] = None,
    weight_manager: Optional[WeightManager] = None,
) -> Tuple[FiniteTree, UCEModel]:
    """Reconstruct FiniteTree from sidecar meta + UCEModel, then load weights.

    Mirrors the logic in scripts/run_distillation.py but provided here as public helper
    for inference users and the generate script. Uses only public model/tree APIs.
    """
    ckpt_p = Path(ckpt_path)
    if meta_path is None:
        meta_p = ckpt_p.with_suffix(".meta.json")
    else:
        meta_p = Path(meta_path)
    import json

    meta = json.loads(meta_p.read_text())
    p = int(meta["p"])
    depth = int(meta["depth"])
    dim = int(meta.get("dim", 16))
    ndl = int(meta.get("num_diff_layers", 1))
    alpha = float(meta.get("alpha", 0.5))
    am = {int(a): int(t) for a, t in meta["address_map"].items()}
    tree = FiniteTree(p, depth, address_map=am)

    # Dynamically check if checkpoint contains wormhole_gate parameters
    try:
        weights_data = mx.load(str(ckpt_p))
        has_gate = any("diffusion.floquet" in k or "learned_gate" in k for k in weights_data.keys())
    except Exception:
        has_gate = False

    model = UCEModel(
        tree,
        dim=dim,
        num_diff_layers=ndl,
        alpha=alpha,
        weight_manager=weight_manager,
        wormhole_gate=has_gate
    )
    model.load_weights(str(ckpt_p))
    return tree, model


def generate(
    model: UCEModel,
    tree: FiniteTree,
    prompt_addresses: Optional[List[int]] = None,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[int]:
    """Autoregressive generation using digit-by-digit routing + sparse active path.

    - Accepts pre-loaded model + tree (use load_model_and_tree for checkpoints).
    - Uses tree.children and leaf checks (public) to only ever emit *registered* leaf addresses.
    - At each depth decision, gathers active = context path ancestors + sibling group (p children of parent)
      at the decision ball's depth; calls embed_and_diffuse(..., active_balls=...) to restrict compute.
    - Samples digit via mx.random.categorical over *live* children only (renormalized), using heads on path ball state.
    - Prints "active balls touched: X" (X small, ~O(p * depth) per step or unioned) if verbose.
    - Returns list of the *new* generated addresses (continuation after prompt).

    This demonstrates O(p log V) routing and sparsity (only active branch + siblings "touched" for diffusion/heads).
    """
    if seed is not None:
        mx.random.seed(seed)

    if prompt_addresses is None:
        prompt_addresses = []
    # work on copy
    context: List[int] = list(prompt_addresses)

    generated: List[int] = []

    p = tree.p
    depth = tree.depth

    for step in range(max_new_tokens):
        # Determine active balls for this step: ancestors of recent context + siblings at potential decisions
        active_balls: List[Tuple[int, int]] = []
        # Context paths (recent prev for injection relevance)
        recent_ctx = context[-5:] if context else []
        for addr in recent_ctx:
            try:
                ancs = tree.get_ancestors(addr)  # [0, ball_d1, ..., full]
                for d in range(len(ancs)):
                    pref = ancs[d]
                    active_balls.append((d, pref))
            except Exception:
                # ignore bad addr
                pass

        # Also ensure root always
        active_balls.append((0, 0))

        # For the upcoming next-token's decisions, we'll collect siblings dynamically during digit build
        # (see inside loop below; we union more actives per decision point)

        # We will build the next addr digit-by-digit, restricting diffusion per decision depth for demo
        # (small depth; each decision uses its local sibling group + context)
        next_addr = 0
        prefix = 0
        touched_this_step: set[Tuple[int, int]] = set(active_balls)

        for d in range(depth):
            # current ball at this decision depth
            curr_key = (d, prefix)
            touched_this_step.add(curr_key)

            # Siblings group at this depth d for the current prefix (p or fewer "branches" at level).
            # For d==0: only root (0,0); the p first-digit siblings are the children of root.
            # For d>0: siblings = direct children list from parent ball (populated via public tree.children).
            if d == 0:
                sibs_at_d: List[int] = [0]
            else:
                parent_pref = prefix % (p ** (d - 1)) if d > 1 else 0
                sibs_at_d = tree.children(d - 1, parent_pref)
            for sib_pref in sibs_at_d:
                touched_this_step.add((d, sib_pref))
            touched_this_step.add((d, prefix))

            # Additionally, to include "p siblings at decision points", union the child group (next level)
            # for the balls we are deciding (this helps diffusion mixing at decision).
            child_sibs = tree.children(d, prefix)
            for cs in child_sibs:
                touched_this_step.add((d + 1, cs))

            # Now, for this decision, prepare active list: context + this level's sibs group + curr path so far
            local_active = list(touched_this_step)
            # dedup preserve order simple
            seen = set()
            dedup_active: List[Tuple[int,int]] = []
            for k in local_active:
                if k not in seen:
                    seen.add(k)
                    dedup_active.append(k)

            # sparse-restricted call (only active path + siblings for this decision)
            diffused = model.embed_and_diffuse(context, active_balls=dedup_active)

            # Get state at curr ball (may fallback if not returned)
            state_vec = diffused.get(curr_key, mx.zeros((model.dim,), dtype=mx.float32))
            if state_vec.ndim > 1:
                state_vec = mx.reshape(state_vec, (model.dim,))

            # head logits (p,)
            digit_logits = model.heads(state_vec, d)

            # Live children at this ball: use public children(d, prefix) -- these are child *prefixes* at d+1
            live_children = tree.children(d, prefix)
            if not live_children:
                # no way forward? fallback pick 0 or break; but tree should allow from root
                # for safety pick a live from root or something
                live_children = [prefix]  # degenerate

            # Map live child prefixes to their digit at this level
            power = p ** d
            live_digits: List[int] = []
            for cpref in live_children:
                # the digit chosen to go from prefix to cpref at this step
                dig = (cpref - prefix) // power   # since low order
                if 0 <= dig < p:
                    live_digits.append(dig)
            if not live_digits:
                live_digits = [0]

            # Now, get probs for all p, then subselect live, renormalize
            probs = mx.softmax(digit_logits)
            live_probs_list = []
            live_digits_sorted = sorted(set(live_digits))  # unique
            for dig in live_digits_sorted:
                live_probs_list.append( float(probs[dig].item()) if dig < probs.shape[0] else 0.0 )
            s = sum(live_probs_list) or 1e-12
            live_probs = [pp / s for pp in live_probs_list]

            # sample from live using categorical on (sub) logits scaled by temp
            if temperature != 1.0:
                # approx: use log of adjusted
                live_logits = [mx.log(mx.array(max(pp, 1e-12))) / temperature for pp in live_probs]
                live_logits_arr = mx.stack(live_logits)
            else:
                live_logits_arr = mx.log( mx.array(live_probs) + 1e-12 )

            # categorical over the sub vocab size = len(live_digits_sorted)
            sub_idx = int( mx.random.categorical( live_logits_arr ).item() )
            chosen_dig = live_digits_sorted[ sub_idx ]

            # advance
            next_addr = next_addr + chosen_dig * power
            prefix = prefix + chosen_dig * power

            # also track the *next level ball* we'll use
            next_key = (d+1, prefix)
            touched_this_step.add(next_key)

        # After full digits, next_addr should be a full-depth address; verify registered via public API
        try:
            _ = tree.address_to_token(next_addr)
        except KeyError:
            # Should not happen if live children respected; fallback to a leaf
            if tree.leaf_addrs:
                next_addr = tree.leaf_addrs[0]

        generated.append(next_addr)
        context.append(next_addr)

        active_count = len(touched_this_step)
        if verbose:
            print(f"active balls touched: {active_count}")

    return generated
