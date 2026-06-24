"""Finite p-adic tree construction and ball management.

Pure Python (no MLX except for optional Gemma path in other modules). Addresses use
the low-order digits for coarse-to-fine branching so that p-adic closeness (high
valuation of difference) corresponds to sharing a deep common ancestor (i.e. being
in the same small ball/subtree).

Balls are identified by their prefix integer (address % p**depth_for_ball).

Clustering helper (cluster_and_assign_addresses) added in Task 4: requires numpy
(project dep) only for that code path; other tree functionality stays pure Python.
Supports inducing addresses from (Gemma) embeddings via simple deterministic
divisive partitioning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .padic import valuation

__all__ = ["FiniteTree"]


class FiniteTree:
    """Finite-depth p-adic tree mapping token_ids <-> p-adic addresses (leaves).

    Supports construction via add_leaf or convenience builders (build_from_addresses,
    from_tokens, cluster_and_assign_addresses), plus optional address_map in __init__
    (dict addr->token for bulk). Provides ball prefix queries (get_ball), ancestors,
    lca_depth, token<->addr lookups, and ball management via children(depth, prefix) /
    tokens_in_ball(depth, prefix).

    Lookups (token_to_address, address_to_token) raise informative KeyError
    (not raw) for unregistered items. All addresses validated to tree range where applicable.
    Internal _ball_* dicts populated for ball/subtree management (now queryable).

    cluster_and_assign_addresses (Task 4) performs hierarchical clustering on a
    provided embedding matrix (e.g. from GemmaInterface) to assign addresses.
    """

    def __init__(
        self,
        p: int,
        depth: int,
        *,
        address_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """Create empty tree (or pre-populated via address_map).

        address_map: optional {address: token_id, ...} to bulk-register via add_leaf
        during construction. (Provided for convenience; main APIs are add_leaf and
        the build_from_* classmethods. Not part of the core plan "Support" list but
        implemented and now documented.)
        """
        if not isinstance(p, int) or p < 2:
            raise ValueError("p must be an integer >= 2")
        if not isinstance(depth, int) or depth < 1:
            raise ValueError("depth must be an integer >= 1")
        self.p = p
        self.depth = depth

        # token <-> address maps (tokens are leaves at full depth only)
        self._token_to_addr: Dict[int, int] = {}
        self._addr_to_token: Dict[int, int] = {}

        # Ball management structures:
        # (depth, ball_prefix) -> list of direct child prefixes (at depth+1)
        self._ball_children: Dict[Tuple[int, int], List[int]] = {}
        # (depth, ball_prefix) -> list of token_ids directly at this leaf ball (usually 0 or 1)
        self._ball_tokens: Dict[Tuple[int, int], List[int]] = {}

        if address_map is not None:
            for addr, tok in address_map.items():
                self.add_leaf(tok, addr)

    def _validate_address(self, address: int) -> None:
        """Shared helper to dedupe prefix math / range validation for addresses.

        Raises ValueError with consistent message for non-int or out-of-range.
        """
        max_addr = self.p ** self.depth
        if not isinstance(address, int) or not (0 <= address < max_addr):
            raise ValueError(f"address {address} out of range for this tree")

    def add_leaf(self, token_id: int, address: int) -> None:
        """Register a token at a full-depth leaf address.

        Idempotent if the (token, address) pair already exists.
        Raises on collisions (different token at same addr, or same token at different addr).
        Populates internal children lists for ball management.
        """
        self._validate_address(address)
        if not isinstance(token_id, int):
            raise ValueError("token_id must be int")

        existing_addr = self._token_to_addr.get(token_id)
        if existing_addr is not None:
            if existing_addr != address:
                raise ValueError(
                    f"token_id {token_id} already mapped to address {existing_addr}, "
                    f"cannot remap to {address}"
                )
            return  # already present, nothing to do

        existing_tok = self._addr_to_token.get(address)
        if existing_tok is not None:
            if existing_tok != token_id:
                raise ValueError(
                    f"address {address} already occupied by token {existing_tok}, "
                    f"cannot add token {token_id}"
                )
            return

        # Record mapping
        self._token_to_addr[token_id] = address
        self._addr_to_token[address] = token_id

        # Insert into ball hierarchy (list of children per ball)
        self._insert_path(address)

    def _insert_path(self, address: int) -> None:
        """Walk digits low-to-high and populate _ball_children lists and leaf tokens."""
        p = self.p
        current_prefix = 0
        # depth 0: implicit root ball 0, no need to store in children at -1
        for d in range(1, self.depth + 1):
            # digit for this level (d-1 in 0-based digits list)
            digit = (address // (p ** (d - 1))) % p
            next_prefix = current_prefix + digit * (p ** (d - 1))

            parent_key: Tuple[int, int] = (d - 1, current_prefix)
            children = self._ball_children.setdefault(parent_key, [])
            if next_prefix not in children:
                children.append(next_prefix)

            current_prefix = next_prefix

        # Record token at the exact leaf ball (depth, address)
        leaf_key: Tuple[int, int] = (self.depth, address)
        toks = self._ball_tokens.setdefault(leaf_key, [])
        if self._addr_to_token[address] not in toks:
            toks.append(self._addr_to_token[address])

    def get_ball(self, address: int, depth: int) -> int:
        """Return the ball identifier (prefix) for the given address at 'depth'.

        depth=0 -> 0 (root ball)
        depth=d -> address % (p**d)   (agrees on first d low-order digits)
        Raises ValueError for address out of range, same as get_ancestors/lca_depth.
        """
        self._validate_address(address)
        if not isinstance(depth, int) or depth < 0 or depth > self.depth:
            raise ValueError(f"depth {depth} out of range [0, {self.depth}]")
        if depth == 0:
            return 0
        return address % (self.p ** depth)

    def get_ancestors(self, address: int) -> List[int]:
        """Return ball prefixes from root (depth 0) through to the full leaf address.

        Length == self.depth + 1
        """
        # Validate address is in range (even if not added as leaf yet)
        self._validate_address(address)
        ancs: List[int] = [0]
        for d in range(1, self.depth + 1):
            ancs.append(self.get_ball(address, d))
        return ancs

    def lca_depth(self, addr1: int, addr2: int) -> int:
        """Return depth of the lowest common ancestor ball of two addresses.

        Addresses must be in-range for the tree but need not be registered leaves
        (any valid p-adic addresses within depth work; doc updated for accuracy).

        This is exactly the p-adic valuation of (addr1 - addr2) (capped at tree depth).
        Used for distance validation: distance(addr1, addr2, p) == 0 if lca==depth else p**(-lca)
        """
        # Accept any in-range addresses (do not require them to be registered leaves)
        self._validate_address(addr1)
        self._validate_address(addr2)
        if addr1 == addr2:
            return self.depth
        v = valuation(addr1 - addr2, self.p)
        return min(v, self.depth)

    def token_to_address(self, token_id: int) -> int:
        """Look up the p-adic address for a registered token_id.

        Raises:
            KeyError: with message "token_id {token_id} not registered" if not present.
        """
        if token_id not in self._token_to_addr:
            raise KeyError(f"token_id {token_id} not registered")
        return self._token_to_addr[token_id]

    def address_to_token(self, address: int) -> int:
        """Look up the original token_id for a registered full-depth address.

        Raises:
            KeyError: with message "address {address} not registered" if not present.
        """
        if address not in self._addr_to_token:
            raise KeyError(f"address {address} not registered")
        return self._addr_to_token[address]

    def __len__(self) -> int:
        return len(self._token_to_addr)

    # --- Ball management queries (expose the populated _ball_* structures) ---

    def children(self, depth: int, prefix: int) -> List[int]:
        """Return direct child ball prefixes (at depth+1) for ball identified by (depth, prefix).

        These are the prefixes stored in the internal _ball_children for ball management.
        Returns [] for unknown balls or balls with no children. Order follows insertion.
        """
        key = (depth, prefix)
        return list(self._ball_children.get(key, []))

    def tokens_in_ball(self, depth: int, prefix: int) -> List[int]:
        """Return token_ids directly recorded at exact (depth, prefix) ball.

        Populated in _ball_tokens only for full-depth leaf balls (usually 0 or 1 token).
        Non-leaf depths return [] with current population strategy.
        Returns [] for unknown balls.
        """
        key = (depth, prefix)
        return list(self._ball_tokens.get(key, []))

    def leaf_addresses(self) -> List[int]:
        """Return sorted list of registered full-depth leaf addresses.

        Provides public API for consumers (e.g. UCEModel) needing the ordered
        registered leaves, avoiding private _addr_to_token access.
        """
        return sorted(self._addr_to_token.keys())

    # --- Builders for tests and early use (clustering stub here; real clustering in Task 4) ---

    @classmethod
    def build_from_addresses(
        cls,
        addresses: List[int],
        p: int,
        depth: int,
        token_ids: Optional[List[int]] = None,
    ) -> FiniteTree:
        """Build a tree directly from a list of (full-depth) addresses.

        If token_ids is None, uses range(len(addresses)) as synthetic token ids.
        """
        if token_ids is None:
            token_ids = list(range(len(addresses)))
        if len(token_ids) != len(addresses):
            raise ValueError("token_ids and addresses must have same length")
        tree = cls(p, depth)
        for tid, addr in zip(token_ids, addresses):
            tree.add_leaf(tid, addr)
        return tree

    @classmethod
    def cluster_and_assign_addresses(
        cls,
        embeddings: np.ndarray,
        p: int,
        depth: int,
        token_ids: Optional[List[int]] = None,
    ) -> FiniteTree:
        """Hierarchical clustering on embeddings to induce p-adic addresses + build tree.

        This is the Task 4 implementation (replaces the from_tokens placeholder logic
        when embeddings are available). Similar tokens by Euclidean distance in the
        supplied embedding space are placed so they share longer common low-order
        digit prefixes (higher LCA depth, smaller p-adic distance).

        Uses a lightweight, deterministic, pure-numpy divisive procedure (no scipy
        or sklearn required, though both are optional for future stronger linkage):
        at each level, for the current group of items, project onto the coordinate
        of max variance, sort, and chunk into (up to) p ordered sub-groups. The
        digit for that level is the chunk index. This induces a p-ary hierarchy
        reflecting the dominant axes of variation at multiple scales.

        embeddings: shape (N, D) float array. N must be <= p**depth.
        token_ids: if None, range(N) is used.
        Returns a populated FiniteTree(p, depth) with exactly the given tokens.

        The clustering is a *proxy* for semantic hierarchy when the embeddings
        come from a real teacher (Gemma input embeddings or activations). For
        perfectly separable synthetic data the induced balls exactly recover
        the generating clusters (up to label permutation of digits at each split).

        Raises on capacity exceeded (same message style as from_tokens).
        If after partitioning any address collision occurs (possible only for
        pathological size distributions exceeding subtree capacity), add_leaf
        will raise.
        """
        if token_ids is None:
            token_ids = list(range(len(embeddings)))
        n = len(token_ids)
        if embeddings.ndim != 2 or embeddings.shape[0] != n:
            raise ValueError("embeddings must be 2D with shape (len(token_ids), dim)")
        max_leaves = p ** depth
        if n > max_leaves:
            raise ValueError(
                f"token_ids length {n} exceeds tree capacity "
                f"p**depth={max_leaves} (p={p}, depth={depth}); "
                "cluster_and_assign_addresses does not truncate"
            )
        if n == 0:
            raise ValueError("Zero/Empty Vocab: cannot cluster 0 tokens")

        # Work with float64 for stability in means/vars (but accept any)
        embs = np.asarray(embeddings, dtype=np.float64)

        # Compute address for each local index 0..n-1 by successive partitioning
        addrs = np.zeros(n, dtype=int)
        # groups: lists of local indices sharing the address prefix so far
        current_groups: List[List[int]] = [list(range(n))]

        for lev in range(depth):
            power = p ** lev
            next_groups: List[List[int]] = []
            for grp in current_groups:
                m = len(grp)
                if m <= 1:
                    next_groups.append(grp)
                    continue
                grp_embs = embs[grp]  # (m, D)
                subgroups = cls._partition_group_into_p(grp_embs, p)
                for digit, sub_local in enumerate(subgroups):
                    if not sub_local:
                        continue
                    sub_global = [grp[i] for i in sub_local]
                    for gi in sub_global:
                        addrs[gi] += digit * power
                    next_groups.append(sub_global)
            current_groups = next_groups

        # Build + populate (add_leaf will validate uniqueness / range)
        tree = cls(p, depth)
        for tid, addr in zip(token_ids, addrs.tolist()):
            tree.add_leaf(tid, addr)
        return tree

    @staticmethod
    def _partition_group_into_p(embs: np.ndarray, p: int) -> List[List[int]]:
        """Deterministic divisive split of a group's embeddings into <=p subgroups.

        Project on the single coordinate of maximum variance (no RNG), argsort,
        then divide the ordered items into p (as-equal-as-possible) contiguous
        chunks. This is simple, reproducible, and sufficient for MVP tree
        induction on both synthetic structured data and real embeddings.
        """
        m = embs.shape[0]
        if m == 0 or p <= 1:
            return [list(range(m))] + [[] for _ in range(p - 1)]
        # max var axis (0 if all equal)
        axis = int(np.argmax(embs.var(axis=0))) if embs.shape[1] > 0 else 0
        projs = embs[:, axis]
        order = np.argsort(projs, kind="stable")
        # chunk
        groups: List[List[int]] = [[] for _ in range(p)]
        chunk = m / p
        for pos, local_idx in enumerate(order):
            g = min(int(pos // chunk), p - 1)
            groups[g].append(int(local_idx))
        return groups

    @classmethod
    def from_tokens(cls, token_ids: List[int], p: int, depth: int) -> FiniteTree:
        """Build tree assigning token_ids to sequential addresses 0,1,... .

        Errors (rather than silent truncate) if len(token_ids) > p**depth (max leaves).

        This remains available for grammar / hard-coded trees (e.g. toy arithmetic
        in Task 3) where no embeddings are present. For embedding-driven induction
        from real (or synthetic) data use cluster_and_assign_addresses (added Task 4).
        """
        max_leaves = p ** depth
        if len(token_ids) > max_leaves:
            raise ValueError(
                f"token_ids length {len(token_ids)} exceeds tree capacity "
                f"p**depth={max_leaves} (p={p}, depth={depth}); "
                "from_tokens does not truncate"
            )
        tree = cls(p, depth)
        for i, tid in enumerate(token_ids):
            tree.add_leaf(tid, i)
        return tree
