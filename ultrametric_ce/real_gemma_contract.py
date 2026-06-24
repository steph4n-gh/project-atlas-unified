"""Pure contract for real Gemma-4 path to enforce high-tid trees, prompt overlap, tok-only surface.

No I/O except via injected deps (gm path or tokenizer passed in).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from ultrametric_ce.tree import FiniteTree


def select_induction_token_ids(
    vocab_size: int, max_tokens: int, prompt_token_ids: List[int], *, seed: int = 42
) -> List[int]:
    """Select N high original token ids (strided from mid) union prompt ids.

    Deterministic, no hardcoded sentence in selection (demo prompt ids are input).
    Returns sorted unique list of length <= max_tokens.
    """
    if max_tokens <= 0 or vocab_size <= 0:
        return []
    n = min(max_tokens, vocab_size)
    # strided high ids, start mid to get "original high" not low special/unused
    start = max(1000, vocab_size // 4)
    step = max(1, (vocab_size - start) // (n + 10))
    base = [start + i * step for i in range(n)]
    base = [min(b, vocab_size - 1) for b in base]
    # union prompt ids (passed in, e.g. from seed prompt)
    all_ids = list(dict.fromkeys(prompt_token_ids + base))  # dedup preserve order
    selected = all_ids[:n]
    return sorted(selected)  # return sorted for determinism in tree


def assert_tree_talkable(
    tree: FiniteTree, tokenizer: Any, prompt: str
) -> List[int]:
    """Assert the tree is usable for 'talk' with this prompt: high tids, prompt maps to >0 addrs.

    Returns the mapped addrs for the prompt (after filter to registered).
    Raises AssertionError on violation (for tests and verif script).
    """
    tids = [tree.address_to_token(a) for a in tree.leaf_addresses()]
    if not tids:
        raise AssertionError("empty tree")
    if min(tids) <= 100:
        raise AssertionError(f"tree must use high original tids (min={min(tids)})")
    # map prompt
    try:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(prompt)
            if isinstance(ids, dict) and "input_ids" in ids:
                ids = ids["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            ids = [int(x) for x in (ids or [])]
        else:
            ids = []
    except Exception:
        ids = []
    registered = set(tids)
    mapped_tids = [tid for tid in ids if tid in registered]
    if not mapped_tids:
        raise AssertionError(f"prompt {prompt!r} maps to 0 registered addrs (tids={mapped_tids})")
    addrs = [tree.token_to_address(tid) for tid in mapped_tids]
    return addrs


def fetch_embeddings_for_ids(
    gm: str, token_ids: List[int], *, used_direct: bool = False, iface: Any = None, extract_func: Any = None
) -> np.ndarray:
    """Fetch the embed rows for the given original token_ids.

    Single branch to avoid unbound iface.
    If used_direct (mlx snapshot), use extract_func(gm, token_ids=...).
    Else use iface.get_embeddings(token_ids=...).
    """
    if not token_ids:
        return np.empty((0, 1), dtype=np.float32)
    if used_direct and extract_func is not None:
        embs = extract_func(gm, token_ids=token_ids)
        return np.asarray(embs, dtype=np.float32)
    if iface is not None:
        embs = iface.get_embeddings(token_ids=token_ids)
        return np.asarray(embs, dtype=np.float32)
    raise ValueError("need either extract_func (direct) or iface for embeddings")
