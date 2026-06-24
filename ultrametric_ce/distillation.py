"""Phase 0: Head factorization / distillation bootstrap.

Warm-starts the DigitHeads (clopen routing) by distilling teacher (Gemma or mock)
sub-distributions over tree balls. Uses the distillation_loss helper.

- Synthetic/toy first (hard-coded arithmetic grammar tree + structural mock teacher).
- Real path: load prebuilt tree (from scripts/build_tree_from_gemma.py json) + real GemmaInterface.
- Tiny supervised steps on heads (no diffusion in this phase).
- Provides save/load for heads checkpoint + "heads alone" (no diffusion) predictor helper
  for verification that warmed heads already give reasonable next-token behavior on
  toy structural prefixes (better than random init baseline).

Persistence: heads saved via mlx nn save_weights (.safetensors). Load back with
load_warmed_heads(ckpt, p, depth, dim) which constructs DigitHeads and loads.

The predict_with_warmed_heads enables using heads for digit-by-digit next-token
without UCEModel or its diffusion stack (uses simple position-zero + leaf/ancestor
injection for limited history dependence during Phase 0 verification).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from ultrametric_ce.padic import address_to_digits
from ultrametric_ce.routing import DigitHeads, distillation_loss
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.model import UCEModel
from ultrametric_ce.evaluation import (
    prefix_accuracy,
    structural_validity_rate,
    compute_structural_metrics,
)

# GemmaInterface used for type/duck; import inside real path only (mockable)
# We accept any duck with .get_logits(list[int] | array) -> np.ndarray (V,)

__all__ = [
    "distillation_loss",  # re-export for convenience
    "build_toy_arithmetic_tree",
    "VALID_TOY_EXPRS",
    "ToyStructuralTeacher",
    "compute_teacher_sub_dist",
    "warm_start_phase0_heads",
    "predict_with_warmed_heads",
    "load_warmed_heads",
    "save_warmed_heads",
    # Task 6 additions
    "distillation_kl",
    "hidden_alignment",
    "hierarchical_prefix_loss",
    "ultrametric_reg",
    "get_synthetic_ball_target_states",
    "prepare_synthetic_training_item",
    "iter_synthetic_batches",
    "iter_text_batches",
    "text_to_address_sequence",
    "addresses_to_text",
    "run_distillation_phase",
    "serialize_dataset_cache",
    "load_dataset_cache",
]


# --- Toy grammar data (duplicated from tests/test_model_toy.py for script self-containment
# and to obey "only modify listed files"; small, stable, used for synthetic Phase 0 path + verify)

def build_toy_arithmetic_tree() -> Tuple[FiniteTree, Dict[str, int], Dict[int, str], List[str]]:
    """Hard-code the tiny grammar tree (p=3, K=4) for 21 synthetic expression tokens.

    See tests/test_model_toy.py for origin. Groups digits/ops/closers for structural demo.
    """
    from ultrametric_ce.padic import padic_address

    p = 3
    depth = 4
    symbols = [str(d) for d in range(10)] + ["+", "-", "*", "/", "(", ")", "x", "y", "z", "=", "^"]
    token_ids = list(range(len(symbols)))
    addresses: List[int] = []
    # digits 0-9: low digit 0
    for i in range(10):
        d1 = i % 3
        d2 = (i // 3) % 3
        d3 = (i // 9) % 3
        addr = padic_address([0, d1, d2, d3], p)
        addresses.append(addr)
    # ops + - * / ( : low digit 1
    for i in range(5):
        d1 = i % 3
        d2 = (i // 3) % 3
        addr = padic_address([1, d1, d2, 0], p)
        addresses.append(addr)
    # ) x y z = ^ : low digit 2
    for i in range(6):
        d1 = i % 3
        d2 = (i // 3) % 3
        d3 = (i // 9) % 3
        addr = padic_address([2, d1, d2, d3], p)
        addresses.append(addr)
    assert len(set(addresses)) == len(addresses), "address collision in toy tree"
    assert all(0 <= a < (p ** depth) for a in addresses)
    tree = FiniteTree.build_from_addresses(addresses, p=p, depth=depth, token_ids=token_ids)
    sym_to_token = {sym: tid for tid, sym in enumerate(symbols)}
    token_to_sym = {tid: sym for tid, sym in enumerate(symbols)}
    return tree, sym_to_token, token_to_sym, symbols


VALID_TOY_EXPRS: List[str] = [
    "1+2",
    "3*4",
    "5-6",
    "7/8",
    "(1+2)",
    "((1+2)*3)",
    "(4*(5+6))",
    "1+2*3",
    "(x+y)*z",
    "((1-2)+3)",
    "4*(2+3)-1",
    "x=(1+2)^3",
    "9/(2+3)",
    "(x*y)+(z=4)",
]


def expr_to_address_sequence(
    expr: str, sym_to_token: Dict[str, int], tree: FiniteTree
) -> List[int]:
    """Map toy expr str to list of leaf addresses (for contexts / generation checks)."""
    addrs: List[int] = []
    for ch in expr:
        if ch in sym_to_token:
            tid = sym_to_token[ch]
            addrs.append(tree.token_to_address(tid))
    return addrs


class ToyStructuralTeacher:
    """Mock GemmaInterface-like teacher for toy grammar.

    get_logits returns (V,) favoring structurally plausible next symbols given
    recent context (paren balance, after-op vs after-digit rules). The biases
    induce non-uniform sub-dists inside balls when restricted/grouped.
    """

    def __init__(
        self,
        tree: FiniteTree,
        sym_to_token: Dict[str, int],
        token_to_sym: Dict[int, str],
        symbols: List[str],
    ) -> None:
        self.tree = tree
        self.sym_to_token = sym_to_token
        self.token_to_sym = token_to_sym
        self.symbols = symbols
        tids = [tree.address_to_token(a) for a in tree.leaf_addresses()]
        self.V = max(tids) + 1 if tids else 32

    def get_logits(
        self, input_ids: List[int] | mx.array | np.ndarray
    ) -> np.ndarray:
        """Return biased logits over the toy V (indexed by token_id)."""
        logits = np.full((self.V,), -8.0, dtype=np.float32)
        # base: all registered get neutral
        for addr in self.tree.leaf_addresses():
            tid = self.tree.address_to_token(addr)
            if 0 <= tid < self.V:
                logits[tid] = 0.0

        if not input_ids:
            # empty: favor starters (digits, vars, open)
            for s in list("0123456789xyz("):
                if s in self.sym_to_token:
                    tid = self.sym_to_token[s]
                    if 0 <= tid < self.V:
                        logits[tid] += 2.5
            return logits

        # convert to syms (last few)
        recent_syms: List[str] = []
        for tid in list(input_ids)[-4:]:
            if isinstance(tid, (mx.array, np.integer)):
                tid = int(tid)
            recent_syms.append(self.token_to_sym.get(int(tid), "?"))

        last = recent_syms[-1] if recent_syms else ""
        paren_depth = sum(1 for s in recent_syms if s == "(") - sum(1 for s in recent_syms if s == ")")

        boosts: List[str] = []
        if last in "0123456789xyz)":
            # after operand/close: prefer ops or close (if open) or =
            boosts = ["+", "-", "*", "/", ")", "="]
            if paren_depth > 0:
                boosts.append(")")
            else:
                # no dangling close
                if ")" in boosts:
                    boosts.remove(")")
        elif last in "+-*/=^(":
            # after op or open: prefer digit, var, open
            boosts = list("0123456789xyz(")
        else:
            boosts = list("0123456789xyz(")

        for s in boosts:
            if s in self.sym_to_token:
                tid = self.sym_to_token[s]
                if 0 <= tid < self.V:
                    logits[tid] += 4.0

        # small global structural prior (helps balls)
        for s in "0123456789xyz":
            if s in self.sym_to_token:
                tid = self.sym_to_token[s]
                if 0 <= tid < self.V:
                    logits[tid] += 0.5

        return logits


def _collect_leaves_in_subtree(tree: FiniteTree, d: int, pref: int) -> List[int]:
    """Recursively collect registered token_ids under ball (d, pref) using public API only."""
    if d == tree.depth:
        return list(tree.tokens_in_ball(d, pref))
    leaves: List[int] = []
    for child_pref in tree.children(d, pref):
        leaves.extend(_collect_leaves_in_subtree(tree, d + 1, child_pref))
    return leaves


def compute_teacher_sub_dist(
    teacher_logits: np.ndarray, tree: FiniteTree, ball_depth: int, ball_prefix: int
) -> np.ndarray:
    """Given full teacher logits (np, shape (V,)), return normalized (p,) sub-dist over children digits for this ball.

    Groups leaf mass under each of 0..p-1 child at ball_depth+1.
    """
    p = tree.p
    power = p ** ball_depth
    child_masses: List[float] = []
    tprobs = None
    for dig in range(p):
        cpref = ball_prefix + dig * power
        leaves = _collect_leaves_in_subtree(tree, ball_depth + 1, cpref)
        if tprobs is None:
            tprobs = np.exp(teacher_logits - np.max(teacher_logits))
            tprobs = tprobs / (np.sum(tprobs) + 1e-12)
        mass = 0.0
        for tid in leaves:
            if 0 <= tid < len(tprobs):
                mass += float(tprobs[tid])
        child_masses.append(mass)
    s = sum(child_masses)
    if s > 1e-12:
        sub = np.array([m / s for m in child_masses], dtype=np.float32)
    else:
        sub = np.full((p,), 1.0 / p, dtype=np.float32)
    return sub


def _compute_states_with_inject_and_propagate(
    tree: FiniteTree, dim: int, previous_addresses: List[int], delta_scale: float = 1.0
) -> Dict[Tuple[int, int], mx.array]:
    """States for Phase 0: base zeros + additive 'was here' delta at prev leaves + decayed to ancestors.

    This gives limited history dependence to ball states (for heads alone and Phase 0 fit)
    without calling the diffusion module.
    """
    states: Dict[Tuple[int, int], mx.array] = {}
    # all possible balls (to have entries); use same enumeration as model
    p = tree.p
    for d in range(tree.depth + 1):
        max_pref = p ** d
        for pref in range(max_pref):
            if d == 0 and pref != 0:
                continue
            i_arr = mx.arange(dim, dtype=mx.float32)
            base = 0.12 * mx.sin(float(d) * 1.1 + float(pref) * 0.35 + i_arr * 0.18)
            states[(d, pref)] = base

    if not previous_addresses:
        return states

    # fixed delta vector (differentiable signal); vary slightly by leaf addr for category signal
    for addr in previous_addresses[-6:]:  # recent
        if addr not in set(tree.leaf_addresses()):
            continue
        # delta depends on addr (so different leaves give different-ish vectors)
        tid = tree.address_to_token(addr)
        phase = float(tid % 7) * 0.9
        angles = phase + mx.arange(dim, dtype=mx.float32) * 0.7
        delta = 0.8 * mx.sin(angles)
        delta = delta * delta_scale
        # inject at leaf
        leaf_key = (tree.depth, addr)
        if leaf_key in states:
            states[leaf_key] = states[leaf_key] + delta
        # propagate to ancestors (decayed)
        try:
            ancs = tree.get_ancestors(addr)  # [d0=0, ..., d=depth]
            for level, apref in enumerate(ancs):
                decay = 0.7 ** max(0, (len(ancs) - 1 - level))
                akey = (level, apref)  # level == depth for the list index
                if akey in states:
                    states[akey] = states[akey] + decay * delta
        except Exception:
            # robust
            pass
    return states


def warm_start_phase0_heads(
    tree: FiniteTree,
    teacher: Any,
    dim: int = 16,
    num_steps: int = 25,
    lr: float = 0.08,
    temperature: float = 1.0,
    seed: int = 123,
) -> DigitHeads:
    """Phase 0 bootstrap: collect per-ball teacher sub-dists (from provided contexts),
    run tiny gradient steps on DigitHeads using distillation_loss + context-modulated states.

    Returns warmed heads (ready for save or direct use in predict_with... or load into UCEModel.heads).
    """
    mx.random.seed(seed)
    p = tree.p
    depth = tree.depth
    heads = DigitHeads(p, depth, dim)

    # discover splittable balls via public children
    balls_to_supervise: List[Tuple[int, int]] = []

    def _visit(d: int, pref: int) -> None:
        ch = tree.children(d, pref)
        if ch:
            balls_to_supervise.append((d, pref))
            for c in ch:
                _visit(d + 1, c)

    _visit(0, 0)

    # build varying contexts ( [] + recent leaves ) to get history modulation
    # overweight [] to ensure marginal/empty factorization (head warmstart from LM head) dominates
    leaf_addrs = tree.leaf_addresses()
    contexts: List[List[int]] = [[]] * 5
    if leaf_addrs:
        for i in range(min(4, len(leaf_addrs))):
            contexts.append([leaf_addrs[i]])
        if len(leaf_addrs) >= 3:
            contexts.append(list(leaf_addrs[:3]))

    # precompute training examples (state, target, d) -- teacher calls outside grad
    training_examples: List[Tuple[mx.array, mx.array, int]] = []
    for ctx in contexts:
        try:
            t_np = np.asarray(teacher.get_logits(ctx), dtype=np.float32)
        except Exception:
            # fallback neutral for bad teacher
            t_np = np.zeros((max(leaf_addrs or [0]) + 2,), dtype=np.float32)
        t_mx = mx.array(t_np)
        t_probs = mx.softmax(t_mx)
        states = _compute_states_with_inject_and_propagate(tree, dim, ctx)
        for bd, bpref in balls_to_supervise:
            power = p ** bd
            child_masses: List[float] = []
            for dig in range(p):
                cpref = bpref + dig * power
                leaves = _collect_leaves_in_subtree(tree, bd + 1, cpref)
                mass = 0.0
                for tid in leaves:
                    if 0 <= tid < t_probs.shape[0]:
                        mass += float(t_probs[tid].item())
                child_masses.append(mass)
            s = sum(child_masses)
            if s > 1e-12:
                t_sub = mx.array([m / s for m in child_masses])
            else:
                t_sub = mx.full((p,), 1.0 / p)
            stv = states.get((bd, bpref), mx.zeros((dim,), dtype=mx.float32))
            if stv.ndim > 1:
                stv = mx.reshape(stv, (dim,))
            training_examples.append((stv, t_sub, bd))

    if not training_examples:
        # still return heads (edge)
        return heads

    opt = optim.Adam(learning_rate=lr)

    for step in range(num_steps):
        def loss_fn(hds: DigitHeads) -> mx.array:
            tot = mx.array(0.0)
            n = float(len(training_examples) or 1)
            for stv, tsub, dd in training_examples:
                lg = hds(stv, dd)
                tot = tot + distillation_loss(lg, tsub, temperature)
            return tot / n

        loss_val, grads = nn.value_and_grad(heads, loss_fn)(heads)
        opt.update(heads, grads)
        mx.eval(heads.parameters(), opt.state)
        mx.clear_cache()
        if (step + 1) % 10 == 0 or step == num_steps - 1:
            print(f"[phase0] step {step+1}/{num_steps} loss={float(loss_val):.4f}")

    # ready
    return heads


def predict_with_warmed_heads(
    heads: DigitHeads,
    tree: FiniteTree,
    previous_addresses: Optional[List[int]] = None,
    ball_states: Optional[Dict[Tuple[int, int], mx.array]] = None,
) -> mx.array:
    """Compute next-token distribution over tree leaves using *only* the (warmed) heads + states.

    No UltrametricDiffusion / UCEModel required. Uses zero-base + ancestor-propagated
    injection for (limited) context sensitivity from previous_addresses.
    """
    if previous_addresses is None:
        previous_addresses = []
    dim = heads.dim
    if ball_states is not None:
        states = ball_states
    else:
        states = _compute_states_with_inject_and_propagate(tree, dim, previous_addresses)

    leaf_addrs = tree.leaf_addresses()
    p = tree.p
    depth = tree.depth
    logps: List[mx.array] = []
    for addr in leaf_addrs:
        digits = address_to_digits(addr, p, depth)
        logp = mx.array(0.0)
        prefix = 0
        for d in range(depth):
            bkey = (d, prefix)
            stv = states.get(bkey, mx.zeros((dim,), dtype=mx.float32))
            if stv.ndim > 1:
                stv = mx.reshape(stv, (dim,))
            dlogits = heads(stv, d)
            dprob = mx.softmax(dlogits)[digits[d]]
            logp = logp + mx.log(dprob + 1e-12)
            prefix = prefix + digits[d] * (p ** d)
        logps.append(logp)

    logps_arr = mx.stack(logps) if logps else mx.array([0.0])
    probs = mx.softmax(logps_arr)
    return probs


def save_warmed_heads(
    heads: DigitHeads, out_path: str | Path, *, meta: Optional[Dict[str, Any]] = None
) -> Path:
    """Save heads weights (safetensors via MLX) + optional json meta sidecar.

    The meta should contain p, depth, dim for easy reload.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    heads.save_weights(str(out_p))
    if meta is not None:
        import json

        meta_p = out_p.with_suffix(".meta.json")
        meta_p.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return out_p


def load_warmed_heads(ckpt_path: str | Path, p: int, depth: int, dim: int) -> DigitHeads:
    """Reconstruct DigitHeads and load the warmed weights from Phase 0 checkpoint."""
    heads = DigitHeads(p, depth, dim)
    heads.load_weights(str(ckpt_path))
    return heads


# =============================================================================
# Task 6: Full diffusion training with distillation (Phase 1 + 2)
# =============================================================================

def distillation_kl(
    uce_leaf_probs: mx.array,  # (L,) softmaxed dist over registered leaves (from UCE recon)
    teacher_soft_probs_on_leaves: mx.array,  # (L,) normalized soft target probs aligned to same leaf order
    temperature: float = 1.0,
) -> mx.array:
    """KL( teacher || uce ) distillation loss on the final leaf distribution.

    Uses exact MLX ops (softmax not needed here as caller provides probs; logsumexp style inside if logits passed upstream).
    Lower is better; encourages UCE's reconstructed leaf dist to match teacher's soft over the tree leaves.
    """
    eps = 1e-12
    if temperature != 1.0:
        # temp scaling would normally be on logits before softmax; here assume probs pre-adjusted or ignore for MVP
        pass
    log_uce = mx.log(uce_leaf_probs + eps)
    # KL(t || s) = sum t*(log t - log s) ; ignore entropy(t) for loss value
    kl = mx.sum(teacher_soft_probs_on_leaves * (mx.log(teacher_soft_probs_on_leaves + eps) - log_uce))
    return kl


def hidden_alignment(
    uce_diffused_states: Dict[Tuple[int, int], mx.array],
    teacher_hidden_projections: Dict[Tuple[int, int], mx.array],
) -> mx.array:
    """MSE alignment between UCE post-diffusion ball states (on active/relevant balls) and teacher-provided projected hiddens.

    For synthetic: teacher_hidden_projections typically the 'target' states from get_synthetic_ball_target_states (pre or ideal).
    For real Gemma: would be projected last-hidden slices or per-position mapped to balls along path.
    Only aligns on keys present in both (common balls for the example).
    """
    if not uce_diffused_states or not teacher_hidden_projections:
        return mx.array(0.0)
    loss = mx.array(0.0)
    count = 0
    for key, uvec in uce_diffused_states.items():
        if key in teacher_hidden_projections:
            tvec = teacher_hidden_projections[key]
            # ensure vectors
            if uvec.ndim > 1:
                uvec = mx.reshape(uvec, (-1,))
            if tvec.ndim > 1:
                tvec = mx.reshape(tvec, (-1,))
            # pad or truncate if dim mismatch (robust for mixed)
            d = uvec.shape[0]
            if tvec.shape[0] != d:
                tvec = mx.concatenate([tvec[:d], mx.zeros((max(0, d - tvec.shape[0]),))])[:d]
            loss = loss + mx.mean((uvec - tvec) ** 2)
            count += 1
    if count == 0:
        return mx.array(0.0)
    return loss / count





def hierarchical_prefix_loss(
    diffused_states: Dict[Tuple[int, int], mx.array],
    heads: DigitHeads,
    target_address: int,
    tree: FiniteTree,
    coarse_to_fine_weight: float = 3.0,
) -> mx.array:
    """Weighted CE on the digits of the target address (teacher-forced path), higher weight on coarse (root) digits.

    Uses the diffused state at each ball along the target's address path.
    Directly supervises the hierarchical structure (coarse decisions first).
    """
    p = tree.p
    depth = tree.depth
    digits = address_to_digits(target_address, p, depth)
    loss = mx.array(0.0)
    w_sum = mx.array(0.0)
    prefix = 0
    for d in range(depth):
        bkey = (d, prefix)
        stv = diffused_states.get(bkey, mx.zeros((heads.dim,), dtype=mx.float32))
        if stv.ndim > 1:
            stv = mx.reshape(stv, (heads.dim,))
        dlogits = heads(stv, d)  # (p,)
        # stable log_softmax equiv
        logprobs = dlogits - mx.logsumexp(dlogits)
        ce = -logprobs[digits[d]]
        # weight: root (d=0) gets highest, e.g. w = coarse** (depth-1-d)
        w = coarse_to_fine_weight ** (depth - 1 - d)
        loss = loss + w * ce
        w_sum = w_sum + w
        prefix = prefix + digits[d] * (p ** d)
    if float(w_sum) < 1e-12:
        return loss
    return loss / w_sum


_ULTRAMETRIC_D_CACHE = {}


def ultrametric_reg(
    leaf_probs: mx.array,  # (L,)
    leaf_addrs: List[int],
    tree: FiniteTree,
    weight: float = 1.0,
) -> mx.array:
    """Light ultrametric regularization: encourage probability mass to concentrate on leaves that are close in p-adic metric.

    Uses weighted sum of p_i * p_j * lca_depth(i,j) (higher better => we return negative for loss term).
    Vectorized via pre-computed/cached LCA depth matrix to avoid O(L^2) intermediate allocations.
    """
    L = len(leaf_addrs)
    if L > 512:
        return mx.array(0.0)
    if L < 2 or leaf_probs.size < 2:
        return mx.array(0.0)

    cache_key = tuple(leaf_addrs)
    if cache_key not in _ULTRAMETRIC_D_CACHE:
        D_np = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            ai = leaf_addrs[i]
            for j in range(i, L):
                aj = leaf_addrs[j]
                d = float(tree.lca_depth(ai, aj))
                D_np[i, j] = d
                D_np[j, i] = d
        _ULTRAMETRIC_D_CACHE[cache_key] = mx.array(D_np)

    D = _ULTRAMETRIC_D_CACHE[cache_key]
    ps = leaf_probs[:L]
    reg = mx.sum(ps[:, None] * ps[None, :] * D)
    return -weight * reg


def get_synthetic_ball_target_states(
    tree: FiniteTree, dim: int, previous_addresses: List[int], delta_scale: float = 1.0
) -> Dict[Tuple[int, int], np.ndarray]:
    """Public helper for synthetic teacher hidden projections (re-uses Phase0 style base+inject logic).

    Provides 'target' ball states that hidden_alignment can pull the diffused states toward.
    In synthetic path this acts as a structural prior (no real Gemma hiddens).
    """
    states: Dict[Tuple[int, int], np.ndarray] = {}
    p = tree.p
    i_arr_np = np.arange(dim, dtype=np.float32)
    for d in range(tree.depth + 1):
        max_pref = p ** d
        for pref in range(max_pref):
            if d == 0 and pref != 0:
                continue
            base_np = 0.12 * np.sin(float(d) * 1.1 + float(pref) * 0.35 + i_arr_np * 0.18)
            states[(d, pref)] = base_np

    if not previous_addresses:
        return states

    for addr in previous_addresses[-6:]:
        if addr not in set(tree.leaf_addresses()):
            continue
        tid = tree.address_to_token(addr)
        phase = float(tid % 7) * 0.9
        angles_np = phase + np.arange(dim, dtype=np.float32) * 0.7
        delta_np = 0.8 * np.sin(angles_np)
        delta_np = delta_np * delta_scale
        leaf_key = (tree.depth, addr)
        if leaf_key in states:
            states[leaf_key] = states[leaf_key] + delta_np
        try:
            ancs = tree.get_ancestors(addr)
            for level, apref in enumerate(ancs):
                decay = 0.7 ** max(0, (len(ancs) - 1 - level))
                akey = (level, apref)
                if akey in states:
                    states[akey] = states[akey] + decay * delta_np
        except Exception:
            pass
    return states


def topological_distance_loss(
    diffused: Dict[Tuple[int, int], mx.array],
    prefix_addrs: List[int],
    tree: FiniteTree,
    alpha: float = 0.1,
    r_s: float = 0.0
) -> mx.array:
    """Penalizes deviations between hidden-state Euclidean distance and tree p-adic distance."""
    valid_addrs = [a for a in prefix_addrs if (tree.depth, a) in diffused]
    S = len(valid_addrs)
    if S < 2:
        return mx.array(0.0)
    
    states_list = [diffused[(tree.depth, a)] for a in valid_addrs]
    states = mx.stack(states_list)  # shape: (S, dim)
    
    D_np = np.zeros((S, S), dtype=np.float32)
    for i in range(S):
        for j in range(i + 1, S):
            d = float(tree.lca_depth(valid_addrs[i], valid_addrs[j]))
            D_np[i, j] = D_np[j, i] = d
    D = mx.array(D_np)
    
    if r_s > 0.0:
        D_warped = D * (1.0 - (r_s / (r_s + D + 1e-6)))
    else:
        D_warped = D
        
    diffs = states[:, None, :] - states[None, :, :]
    dist_matrix = mx.sum(diffs ** 2, axis=-1)
    
    return mx.mean((dist_matrix - alpha * D_warped) ** 2)


def prepare_synthetic_training_item(
    prefix_addrs: List[int],
    target_addr: int,
    tree: FiniteTree,
    teacher: Any,
    dim: int,
) -> Tuple[List[int], int, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """For one (prefix, target) produce the 4-tuple for batcher/losses: (prefix, target, teacher_leaf_probs_np, hidden_target_dict_np)."""
    # Map prev addrs -> tids (works for both toy tids and real Gemma token ids in tree)
    prev_tids: List[int] = []
    for a in prefix_addrs:
        try:
            prev_tids.append(tree.address_to_token(a))
        except KeyError:
            pass
    try:
        tlog_np = np.asarray(teacher.get_logits(prev_tids), dtype=np.float32)
    except Exception:
        max_tid = max([tree.address_to_token(a) for a in tree.leaf_addresses()] or [0]) + 1
        tlog_np = np.zeros((max_tid,), dtype=np.float32)
    
    # Numpy softmax
    tlog_np_shifted = tlog_np - np.max(tlog_np)
    tprobs_full = np.exp(tlog_np_shifted) / np.sum(np.exp(tlog_np_shifted))

    # align to leaves order
    leaf_addrs = tree.leaf_addresses()
    leaf_t_list: List[float] = []
    for a in leaf_addrs:
        tid = tree.address_to_token(a)
        if 0 <= tid < tprobs_full.shape[0]:
            leaf_t_list.append(float(tprobs_full[tid]))
        else:
            leaf_t_list.append(0.0)
    leaf_tprobs = np.array(leaf_t_list, dtype=np.float32) if leaf_t_list else np.array([1.0], dtype=np.float32)
    leaf_tprobs = leaf_tprobs / (np.sum(leaf_tprobs) + 1e-12)

    h_target = get_synthetic_ball_target_states(tree, dim, prefix_addrs)
    return (prefix_addrs, target_addr, leaf_tprobs, h_target)


def iter_synthetic_batches(
    tree: FiniteTree,
    teacher: Any,
    sym_to_token: Optional[Dict[str, int]] = None,
    batch_size: int = 4,
    max_pairs: int = 32,
    dim: int = 16,
    seed: int = 42,
) -> List[List[Tuple[List[int], int, mx.array, Dict[Tuple[int, int], mx.array]]]]:
    """Build list of batches from VALID_TOY_EXPRS (or provided) for synthetic training.

    Returns list-of-batches; each batch is list of 4-tuples.
    Variable len prefixes handled naturally (no padding; per-item forward in step).
    """
    rng = np.random.default_rng(seed)
    pairs: List[Tuple[List[int], int]] = []
    exprs = VALID_TOY_EXPRS
    for expr in exprs:
        try:
            addrs = expr_to_address_sequence(expr, sym_to_token or {}, tree)
        except Exception:
            continue
        for i in range(1, len(addrs)):  # at least one prev? or allow i=0
            pairs.append((addrs[:i], addrs[i]))
    if not pairs:
        # fallback simple
        leafs = tree.leaf_addresses()
        for a in leafs[:min(4, len(leafs))]:
            pairs.append(([], a))
    # shuffle + cap
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    batches: List = []
    for bstart in range(0, len(pairs), batch_size):
        bitems = []
        for pref, tgt in pairs[bstart : bstart + batch_size]:
            item = prepare_synthetic_training_item(pref, tgt, tree, teacher, dim)
            bitems.append(item)
        if bitems:
            batches.append(bitems)
    return batches


def _zero_grads_for_module_prefix(grads: Dict[str, Any], prefix: str = "heads.") -> Dict[str, Any]:
    """Helper for Phase 1 freeze: zero gradients for heads (and sub) so Adam does not update them."""
    out: Dict[str, Any] = {}
    for k, g in grads.items():
        if k.startswith(prefix):
            out[k] = mx.zeros_like(g) if (g is not None and hasattr(g, "shape")) else g
        else:
            out[k] = g
    return out


def _get_tokenizer(teacher: Any) -> Any | None:
    """Duck-type: return .tokenizer if present on gemma teacher iface (for real text roundtrips and data)."""
    if teacher is None:
        return None
    tok = getattr(teacher, "tokenizer", None)
    if tok is not None:
        return tok
    # fallback if teacher itself quacks like tokenizer
    if hasattr(teacher, "encode") and hasattr(teacher, "decode"):
        return teacher
    return None


def _ids_from_text(tokenizer: Any, text: str) -> List[int]:
    if tokenizer is None or not text:
        return []
    try:
        ids = tokenizer.encode(text)
        if isinstance(ids, dict) and "input_ids" in ids:
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(x) for x in (ids or [])]
    except Exception:
        return []


def _text_from_ids(tokenizer: Any, ids: List[int]) -> str:
    if tokenizer is None or not ids:
        return ""
    try:
        txt = tokenizer.decode(ids)
        if isinstance(txt, (list, tuple)):
            txt = txt[0] if txt else ""
        return str(txt)
    except Exception:
        return ""


def iter_text_batches(
    tree: FiniteTree,
    teacher: Any,
    tokenizer: Any,
    batch_size: int = 4,
    max_pairs: int = 32,
    dim: int = 16,
    seed: int = 42,
    canned_texts: Optional[List[str]] = None,
) -> List[List[Tuple[List[int], int, mx.array, Dict[Tuple[int, int], mx.array]]]]:
    """Build batches of (prefix_addrs, target_addr, teacher_probs, h_target) from tokenized real text snippets.

    Only tokens registered in the (sub)tree are kept. Produces sequential real-data pairs for KL+alignment
    when a real Gemma-4 teacher + tokenizer is available. Falls back to empty if no tokenizer.
    Synthetic path (toy exprs) remains completely unchanged.
    """
    if tokenizer is None:
        return []
    rng = np.random.default_rng(seed)
    if canned_texts is None:
        canned_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Gemma models are developed by Google.",
            "Ultrametric routing uses p adic balls and diffusion instead of attention.",
            "This lets us talk to the base model through the UCE tree at inference time.",
            "High dimensional token embeddings from the 12B are clustered into the finite tree.",
            "Phase one distillation aligns the diffusion states using the real teacher logits.",
            "Generation decodes only registered leaves back to text via the tokenizer.",
            "Active balls touched remain small compared to the full vocabulary size.",
        ]
    registered_tids = {tree.address_to_token(a) for a in tree.leaf_addresses()}
    pairs: List[Tuple[List[int], int]] = []
    for txt in canned_texts:
        ids = _ids_from_text(tokenizer, txt)
        if len(ids) < 2:
            continue
        for i in range(1, len(ids)):
            tgt_tid = ids[i]
            if tgt_tid not in registered_tids:
                continue
            pref_ids = ids[:i]
            pref_addrs: List[int] = []
            for tid in pref_ids:
                if tid in registered_tids:
                    try:
                        pref_addrs.append(tree.token_to_address(tid))
                    except KeyError:
                        pass
            if not pref_addrs:
                continue
            try:
                tgt_addr = tree.token_to_address(tgt_tid)
            except KeyError:
                continue
            pairs.append((pref_addrs, tgt_addr))
    if not pairs:
        return []
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    batches: List = []
    total_batches = (len(pairs) + batch_size - 1) // batch_size
    for idx, bstart in enumerate(range(0, len(pairs), batch_size), 1):
        bitems = []
        for pref, tgt in pairs[bstart : bstart + batch_size]:
            item = prepare_synthetic_training_item(pref, tgt, tree, teacher, dim)
            bitems.append(item)
        if bitems:
            batches.append(bitems)
        if idx % 10 == 0 or idx == total_batches:
            print(f"Processed batch {idx}/{total_batches} (pairs: {len(batches) * batch_size})")
    return batches


def text_to_address_sequence(
    text: str, tokenizer: Any, tree: FiniteTree
) -> List[int]:
    """Public thin helper for real (or toy) text->registered-addresses roundtrip.

    Uses tokenizer.encode (real GemmaInterface.tokenizer) or sym_to_token dict (toy).
    Only returns addresses for tokens that are leaves registered in the tree (filters prompt).
    """
    if tokenizer is None or not text:
        return []
    # real tokenizer path (has encode)
    if hasattr(tokenizer, "encode"):
        ids = _ids_from_text(tokenizer, text)
        registered = {tree.address_to_token(a) for a in tree.leaf_addresses()}
        addrs: List[int] = []
        for tid in ids:
            if tid in registered:
                try:
                    addrs.append(tree.token_to_address(tid))
                except KeyError:
                    pass
        return addrs
    # toy fallback (sym_to_token dict)
    if isinstance(tokenizer, dict):
        sym_to_token = tokenizer
        return expr_to_address_sequence(text, sym_to_token, tree)
    return []


def addresses_to_text(
    addrs: List[int], tokenizer: Any, tree: FiniteTree
) -> str:
    """Public thin helper for addresses->text decode using real tokenizer or toy token_to_sym."""
    if not addrs:
        return ""
    if tokenizer is None:
        return ""
    tids: List[int] = []
    for a in addrs:
        try:
            tids.append(tree.address_to_token(a))
        except KeyError:
            pass
    if hasattr(tokenizer, "decode"):
        return _text_from_ids(tokenizer, tids)
    if isinstance(tokenizer, dict):
        token_to_sym = tokenizer
        return "".join(token_to_sym.get(tid, "?") for tid in tids)
    return ""


def run_distillation_phase(
    model: UCEModel,
    teacher: Any,
    tree: FiniteTree,
    sym_to_token: Optional[Dict[str, int]] = None,
    phase: int = 1,
    steps: int = 20,
    batch_size: int = 4,
    lr: float = 0.01,
    log_every: int = 5,
    loss_weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
    dim: Optional[int] = None,
    precomputed_batches: Optional[List] = None,
) -> Tuple[UCEModel, List[Dict[str, float]]]:
    """Core training loop for Phase 1 (diffusion focus, heads frozen) + Phase 2 (light joint).

    - Uses synthetic batches from iter_synthetic_batches (toy teacher).
    - Full multi-part loss: kl + hidden + hierarchical (weighted) + light ultrametric.
    - Hybrid: Adam on all, but for phase==1 zero grads on 'heads.*' (freeze completely).
    - Teacher forcing on prefixes; logs structural metrics (from evaluation) on heldout every log_every.
    - Scheduled sampling note: for MVP we stay with full prefix forcing (no random replace yet).
    - Returns (model after steps, list of log dicts per step or logged step).

    For real Gemma path caller would pass real teacher iface + same tree, batches would call on-the-fly or cache logits/hiddens.
    The model is updated in place (its params).
    """
    mx.random.seed(seed)
    np.random.seed(seed)
    if dim is None:
        dim = model.dim

    if loss_weights is None:
        loss_weights = {"kl": 1.0, "hidden": 0.2, "hier": 1.5, "ultra": 0.05}

    tokenizer = _get_tokenizer(teacher)

    # Prepare heldout for logging metrics (small fixed)
    heldout_pairs: List[Tuple[List[int], int]] = []
    if tokenizer is not None:
        # for real, derive heldout pairs from text tokenization for meaningful prefix/spearman on sequential
        try:
            txt_b = iter_text_batches(tree, teacher, tokenizer, batch_size=4, max_pairs=8, dim=dim, seed=seed)
            for b in txt_b:
                for pref, tgt, _, _ in b:
                    if pref:
                        heldout_pairs.append((pref, tgt))
            if len(heldout_pairs) > 6:
                heldout_pairs = heldout_pairs[:6]
        except Exception:
            pass
    if not heldout_pairs:
        for expr in VALID_TOY_EXPRS[:4]:
            try:
                addrs = expr_to_address_sequence(expr, sym_to_token or {}, tree)
                for ii in range(1, min(3, len(addrs))):
                    heldout_pairs.append((addrs[:ii], addrs[ii]))
            except Exception:
                continue
    if not heldout_pairs:
        leafs = tree.leaf_addresses()
        heldout_pairs = [([], leafs[0])]

    # batches (recompute each epoch for variety, but tiny so pregen ok)
    if precomputed_batches is not None:
        batches = precomputed_batches
    elif tokenizer is not None:
        batches = iter_text_batches(
            tree, teacher, tokenizer, batch_size=batch_size, max_pairs=24, dim=dim, seed=seed
        )
    else:
        batches = iter_synthetic_batches(
            tree, teacher, sym_to_token=sym_to_token, batch_size=batch_size, max_pairs=24, dim=dim, seed=seed
        )
    if not batches:
        batches = [[prepare_synthetic_training_item([], tree.leaf_addresses()[0], tree, teacher, dim)]]

    opt = optim.Adam(learning_rate=lr)
    log: List[Dict[str, float]] = []
    freeze_heads = phase == 1

    for step in range(steps):
        # pick a batch (cycle)
        bidx = step % len(batches)
        batch = batches[bidx]

        def loss_fn(m: UCEModel) -> mx.array:
            tot = mx.array(0.0)
            n = float(len(batch) or 1)
            leafs = m.leaf_addrs  # public
            for pref, tgt, tleaf_probs, htarget in batch:
                # Convert from numpy arrays to mx.array dynamically inside loss_fn
                tleaf_probs_mx = mx.array(tleaf_probs) if not isinstance(tleaf_probs, mx.array) else tleaf_probs
                htarget_mx = {k: (mx.array(v) if not isinstance(v, mx.array) else v) for k, v in htarget.items()}
                # 1. get uce leaf dist (full forward uses diffusion+heads)
                uce_probs = m(pref)
                # 2. kl on leaves
                kl_l = distillation_kl(uce_probs, tleaf_probs_mx)
                tot = tot + loss_weights.get("kl", 1.0) * kl_l
                # 3. diffuse states for this pref (expose via new API)
                diffused = m.embed_and_diffuse(pref)
                # 4. hidden align
                ha = hidden_alignment(diffused, htarget_mx)
                tot = tot + loss_weights.get("hidden", 0.2) * ha
                # 5. hierarchical prefix CE (on target digits, using diffused)
                hp = hierarchical_prefix_loss(diffused, m.heads, tgt, tree)
                tot = tot + loss_weights.get("hier", 1.5) * hp
                # 6. ultra reg on the uce_probs
                ur = ultrametric_reg(uce_probs, leafs, tree)
                tot = tot + loss_weights.get("ultra", 0.05) * ur
                # 7. topological distance loss
                r_s = getattr(model.diffusion, "r_s", 0.5)
                loss_topo = topological_distance_loss(diffused, pref, tree, 0.1, r_s=r_s)
                tot = tot + loss_weights.get("topo", 0.1) * loss_topo
            return tot / n

        loss_val, grads = nn.value_and_grad(model, loss_fn)(model)
        if freeze_heads:
            grads = _zero_grads_for_module_prefix(grads, "heads.")
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        mx.clear_cache()

        step_log: Dict[str, float] = {"step": step, "total_loss": float(loss_val)}
        if (step + 1) % log_every == 0 or step == steps - 1:
            # compute structural on heldout (use current model)
            try:
                mets = compute_structural_metrics(model, tree, sym_to_token or {}, heldout_pairs, num_samples=3, seed=seed)
                step_log.update(mets)
            except Exception as e:
                step_log["metric_err"] = str(e)[:50]
            print(f"[phase{phase}] step {step+1}/{steps} loss={float(loss_val):.4f} prefix_acc={step_log.get('prefix_accuracy', -1):.3f}")
        log.append(step_log)

    return model, log


def serialize_dataset_cache(batches: List, output_path: str) -> None:
    """Serializes dataset batches to a file using pickle."""
    import pickle
    portable_batches = []
    for batch in batches:
        portable_batch = []
        for pref, tgt, tprobs, htarget in batch:
            tprobs_np = np.array(tprobs)
            htarget_np = {k: np.array(v) for k, v in htarget.items()}
            portable_batch.append((pref, tgt, tprobs_np, htarget_np))
        portable_batches.append(portable_batch)

    with open(output_path, "wb") as f:
        pickle.dump(portable_batches, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset_cache(cache_path: str) -> List:
    """Loads dataset batches from a serialized cache file keeping them as numpy arrays."""
    import pickle
    with open(cache_path, "rb") as f:
        return pickle.load(f)
