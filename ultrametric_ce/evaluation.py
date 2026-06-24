"""Evaluation helpers for structural metrics (Task 6/8).

Implements:
- prefix_accuracy: fraction of correct next-digit predictions (coarse-to-fine) on (prefix, target) pairs,
  using model's embed+diffuse states fed to (frozen or not) heads.
- structural_validity_rate: fraction of autoregressively sampled sequences that satisfy
  basic toy grammar structural constraints (balanced parens, operator/operand alternation, etc.).
- is_structurally_valid_toy_expr and check_structural_validity (simple recursive descent parser for toy exprs).
- ultrametric_spearman_correlation: Spearman rho showing model probs rank ultrametrically-close leaves higher.
- compute_structural_metrics convenience bundling all.
- (active ball counts via inference.generate logs in calling scripts)

All use public model/tree APIs only (e.g. model.embed_and_diffuse, tree.address_to_token, padic.address_to_digits, tree.lca_depth, padic.distance).
Intended for both synthetic toy and (later) real structural evals. No private peeking.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx

from ultrametric_ce.padic import address_to_digits, distance
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.model import UCEModel


__all__ = [
    "prefix_accuracy",
    "structural_validity_rate",
    "compute_structural_metrics",
    "is_structurally_valid_toy_expr",
    "check_structural_validity",
    "ultrametric_spearman_correlation",
]


def prefix_accuracy(
    model: UCEModel,
    tree: FiniteTree,
    pairs: List[Tuple[List[int], int]],
    use_diffusion: bool = True,
) -> float:
    """Compute average per-digit prefix accuracy for next-token address prediction on given pairs.

    For each (prefix_addrs, target_addr):
      - compute diffused states = model.embed_and_diffuse(prefix_addrs)  (always uses diffusion stack)
      - for d in 0..depth-1: ball_state at current prefix ball, digit_logits = model.heads(state, d)
        pred_dig = argmax( softmax(digit_logits) )
        target_dig = digits_of(target)[d]
        match += (pred == target)
    Returns mean match rate over (all digits across all pairs). 1.0 = perfect on these.

    Note: even if 'heads warmed from phase0', this measures how well the *current* (diffused) states
    make the heads pick the correct digits for the supervised targets. Diffusion training moves the states.
    """
    if not pairs:
        return 0.0
    p = tree.p
    depth = tree.depth
    total_matches = 0
    total_digits = 0
    for pref, tgt_addr in pairs:
        # always use diffusion for full UCE (even Phase0 heads test can call with model)
        diffused = model.embed_and_diffuse(pref)
        digits = address_to_digits(tgt_addr, p, depth)
        curr_prefix = 0
        for d in range(depth):
            bkey = (d, curr_prefix)
            stv = diffused.get(bkey, mx.zeros((model.dim,), dtype=mx.float32))
            if stv.ndim > 1:
                stv = mx.reshape(stv, (model.dim,))
            dlogits = model.heads(stv, d)  # (p,)
            dprobs = mx.softmax(dlogits)
            pred_dig = int(mx.argmax(dprobs).item())
            total_matches += int(pred_dig == digits[d])
            total_digits += 1
            curr_prefix = curr_prefix + digits[d] * (p ** d)
    if total_digits == 0:
        return 0.0
    return total_matches / total_digits


def is_structurally_valid_toy_expr(expr: str) -> bool:
    """Simple grammar checker for the toy arithmetic expressions using recursive descent parser.

    Distinguishes structurally good (valid nesting, op/operand alternation per toy grammar)
    from bad (mismatched parens, consecutive ops/operands, start/end errors, invalid = placement, etc.).
    """
    if not expr:
        return False
    alphabet = set("0123456789+-*/()xyz=^")
    if any(ch not in alphabet for ch in expr):
        return False

    def parse_expr(pos: int) -> int:
        pos = parse_addsub(pos)
        if pos < 0:
            return -1
        if pos < len(expr) and expr[pos] == "=":
            pos += 1
            pos = parse_addsub(pos)
            if pos < 0:
                return -1
        return pos

    def parse_addsub(pos: int) -> int:
        pos = parse_muldiv(pos)
        if pos < 0:
            return -1
        while pos < len(expr) and expr[pos] in "+-":
            pos += 1
            pos = parse_muldiv(pos)
            if pos < 0:
                return -1
        return pos

    def parse_muldiv(pos: int) -> int:
        pos = parse_pow(pos)
        if pos < 0:
            return -1
        while pos < len(expr) and expr[pos] in "*/":
            pos += 1
            pos = parse_pow(pos)
            if pos < 0:
                return -1
        return pos

    def parse_pow(pos: int) -> int:
        pos = parse_atom(pos)
        if pos < 0:
            return -1
        while pos < len(expr) and expr[pos] == "^":
            pos += 1
            pos = parse_atom(pos)
            if pos < 0:
                return -1
        return pos

    def parse_atom(pos: int) -> int:
        if pos >= len(expr):
            return -1
        ch = expr[pos]
        if ch in "0123456789xyz":
            return pos + 1
        if ch == "(":
            pos2 = parse_expr(pos + 1)
            if pos2 < 0 or pos2 >= len(expr) or expr[pos2] != ")":
                return -1
            return pos2 + 1
        return -1

    final = parse_expr(0)
    return final == len(expr) and final > 0


def check_structural_validity(expr_str: str) -> Dict[str, Any]:
    """Full validity checker returning details.

    Uses the recursive parser based is_structurally_valid_toy_expr.
    Returns dict with 'valid': bool and 'reason': optional str.
    """
    valid = is_structurally_valid_toy_expr(expr_str)
    reason = None if valid else "invalid_toy_structure"
    return {"valid": valid, "reason": reason}


def structural_validity_rate(
    model: UCEModel,
    tree: FiniteTree,
    token_to_sym: Dict[int, str],
    num_samples: int = 8,
    max_len: int = 8,
    seed: int = 42,
) -> float:
    """Fraction of model.sample generations (from empty or short start) that form structurally valid toy exprs.

    Samples autoregressively using the full UCEModel (embed+diffuse+heads path).
    Converts addr seq -> char str using token_to_sym, checks with is_structurally_valid_toy_expr.
    """
    mx.random.seed(seed)
    if num_samples <= 0:
        return 0.0
    valid = 0
    leafs = tree.leaf_addresses()
    for s in range(num_samples):
        # start empty or very short structural start (use [] to let model show its bias)
        generated: List[int] = []
        for _ in range(max_len):
            next_a = model.sample(generated)
            generated.append(next_a)
        # to str
        try:
            gen_str = "".join(token_to_sym[tree.address_to_token(a)] for a in generated)
        except KeyError:
            continue
        if is_structurally_valid_toy_expr(gen_str):
            valid += 1
    return valid / num_samples


def _spearman_rho(x: List[float] | np.ndarray, y: List[float] | np.ndarray) -> float:
    """Pure-numpy Spearman rank correlation (no scipy). Handles short/constant inputs -> nan."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = len(x)
    if n < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    # ranks with order tie-break (sufficient for MVP)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def ultrametric_spearman_correlation(
    model: UCEModel,
    tree: FiniteTree,
    pairs: List[Tuple[List[int], int]],
) -> float:
    """Average Spearman rho between (p-adic dist from target leaf to all leaves) and (model prob rank).

    For each (prefix, target_addr): run model to get leaf probs (aligned to tree/model.leaf_addrs),
    compute p_dists to target, ranks (0= highest prob), rho(p_dists, ranks).
    Positive rho means model assigns higher prob to ultrametrically closer leaves: geometry respected.
    Uses only public APIs (model(), tree.leaf_addresses, padic.distance).
    """
    if not pairs:
        return float("nan")
    p = tree.p
    leaf_addrs = model.leaf_addrs  # public (as used in distillation)
    if not leaf_addrs:
        return float("nan")
    rhos: List[float] = []
    for pref, tgt_addr in pairs:
        try:
            probs = np.asarray(model(pref))
        except Exception:
            continue
        if probs.shape[0] != len(leaf_addrs):
            continue
        p_dists = [distance(tgt_addr, la, p) for la in leaf_addrs]
        # rank: smaller rank number for larger prob
        order = np.argsort(-probs)
        ranks = np.argsort(order).astype(float)
        rho = _spearman_rho(p_dists, ranks)
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan")
    return float(np.mean(rhos))


def compute_structural_metrics(
    model: UCEModel,
    tree: FiniteTree,
    token_to_sym: Dict[int, str],
    heldout_pairs: List[Tuple[List[int], int]],
    num_samples: int = 4,
    max_len: int = 6,
    seed: int = 123,
) -> Dict[str, float]:
    """Bundle structural metrics for a model/tree on heldout prefix pairs + generative samples.

    Returns: prefix_accuracy (per-digit), structural_validity_rate (on model.sample gens),
    ultrametric_spearman (model prob mass prefers p-adically close leaves to targets).
    Active ball counts measured via inference.generate (logs) in eval scripts.
    """
    pa = prefix_accuracy(model, tree, heldout_pairs)
    vr = structural_validity_rate(model, tree, token_to_sym, num_samples=num_samples, max_len=max_len, seed=seed)
    try:
        us = ultrametric_spearman_correlation(model, tree, heldout_pairs)
    except Exception:
        us = float("nan")
    return {
        "prefix_accuracy": float(pa),
        "structural_validity_rate": float(vr),
        "ultrametric_spearman": float(us) if not np.isnan(us) else 0.0,
    }
