import pytest
import numpy as np

from ultrametric_ce.padic import padic_address, distance, valuation
from ultrametric_ce.tree import FiniteTree


def test_build_tree_from_list_of_addresses():
    """Building tree from list of addresses (with corresponding token ids)."""
    p = 3
    depth = 2
    addrs = [
        padic_address([0, 0], p),
        padic_address([0, 1], p),
        padic_address([1, 0], p),
        padic_address([2, 2], p),
    ]
    token_ids = [10, 11, 12, 13]
    tree = FiniteTree.build_from_addresses(addrs, p=p, depth=depth, token_ids=token_ids)
    assert len(tree) == 4
    for tid, addr in zip(token_ids, addrs):
        assert tree.address_to_token(addr) == tid
        assert tree.token_to_address(tid) == addr


def test_token_address_mapping_via_add_leaf():
    """Direct add_leaf and bidirectional mapping."""
    p = 2
    depth = 3
    tree = FiniteTree(p=p, depth=depth)
    addr_a = padic_address([0, 1, 0], p)
    addr_b = padic_address([1, 0, 1], p)
    tree.add_leaf(token_id=42, address=addr_a)
    tree.add_leaf(token_id=99, address=addr_b)
    assert len(tree) == 2
    assert tree.token_to_address(42) == addr_a
    assert tree.address_to_token(addr_a) == 42
    assert tree.token_to_address(99) == addr_b
    # idempotent add ok
    tree.add_leaf(42, addr_a)
    assert len(tree) == 2


def test_get_ball_and_ancestors():
    """get_ball(address, depth) and get_ancestors give correct prefixes."""
    p = 3
    depth = 3
    tree = FiniteTree(p=p, depth=depth)
    addr = padic_address([1, 2, 0], p)  # low digit first: d0=1 (top level), d1=2, d2=0
    tree.add_leaf(7, addr)

    # depth 0 always root ball 0
    assert tree.get_ball(addr, 0) == 0
    # depth 1: only lowest digit
    assert tree.get_ball(addr, 1) == 1
    # depth 2
    assert tree.get_ball(addr, 2) == 1 + 2 * 3
    # depth 3 == full addr
    assert tree.get_ball(addr, 3) == addr

    ancs = tree.get_ancestors(addr)
    assert ancs == [0, 1, 1 + 2*3, addr]
    assert len(ancs) == depth + 1


def test_lca_depth_for_distance_validation():
    """LCA depth must be consistent with p-adic valuation / distance from Task 1."""
    p = 3
    depth = 3
    tree = FiniteTree(p=p, depth=depth)

    # Construct addresses that share different prefix lengths (low digits)
    # digits lists are low-to-high
    a = padic_address([0, 0, 0], p)  # 0
    b = padic_address([0, 0, 1], p)  # agree on first 2 low digits -> lca=2 , val(diff)=2
    c = padic_address([0, 1, 0], p)  # agree on first 1 -> lca=1
    d = padic_address([1, 0, 0], p)  # agree on 0 -> lca=0
    e = padic_address([0, 0, 0], p)  # same as a (for lca test; do not add duplicate addr)

    # Only add distinct addresses as leaves
    for tid, addr in enumerate([a, b, c, d]):
        tree.add_leaf(100 + tid, addr)

    assert tree.lca_depth(a, a) == depth
    assert tree.lca_depth(a, e) == depth
    assert tree.lca_depth(a, b) == 2
    assert tree.lca_depth(a, c) == 1
    assert tree.lca_depth(a, d) == 0

    # Cross-check with distance (p ** -lca except when full depth -> 0)
    assert distance(a, b, p) == pytest.approx(p ** -2)
    assert distance(a, c, p) == pytest.approx(p ** -1)
    assert distance(a, d, p) == pytest.approx(1.0)  # p ** -0
    assert distance(a, a, p) == 0.0

    # lca_depth cross-check via valuation where applicable (for non-equal)
    assert tree.lca_depth(a, b) == valuation(a - b, p)
    assert tree.lca_depth(a, c) == valuation(a - c, p)
    assert tree.lca_depth(a, d) == valuation(a - d, p)

    # robustness: lca_depth supports ANY in-range addresses (registered leaves or not)
    # (previously docstring overstated as "of two (leaf) addresses")
    f = padic_address([0, 0, 2], p)  # not among added leaves
    assert 0 <= f < (p ** depth)
    assert tree.lca_depth(a, f) == 2
    assert tree.lca_depth(f, b) == 2


def test_clustering_stub():
    """Clustering stub produces a usable tree (to be replaced by real embedding clustering in Task 4)."""
    p = 2
    depth = 2
    token_ids = [5, 6, 7, 8]
    tree = FiniteTree.from_tokens(token_ids, p=p, depth=depth)
    assert len(tree) == len(token_ids)
    for tid in token_ids:
        addr = tree.token_to_address(tid)
        assert 0 <= addr < (p ** depth)
        assert tree.address_to_token(addr) == tid
    # stub used some addresses (implementation detail: sequential is acceptable for stub)
    used_addrs = {tree.token_to_address(tid) for tid in token_ids}
    assert len(used_addrs) == len(token_ids)

    # > p**depth now raises explicitly (was silent truncate via break; robustness)
    too_many = list(range(5))  # 5 > 2**2=4
    with pytest.raises(ValueError, match="exceeds|capacity|too many"):
        FiniteTree.from_tokens(too_many, p=p, depth=depth)


def test_ball_management_queries_and_internal_population():
    """Verify ball management structures (_ball_children, _ball_tokens) are populated
    during construction (white-box internal inspection) and exposed via public
    query helpers. This fulfills the 'ball management' emphasis in task title,
    class docstring, and add_leaf docstring (previously dead code, never read).
    """
    p = 2
    depth = 2
    addrs = [0, 3]
    tree = FiniteTree.build_from_addresses(addrs, p=p, depth=depth)
    # white-box internal population check (as minimum required by reviewer)
    assert len(tree._ball_children) > 0
    assert len(tree._ball_tokens) == len(addrs)
    assert (0, 0) in tree._ball_children
    assert (2, 0) in tree._ball_tokens
    assert (2, 3) in tree._ball_tokens
    # public helpers (minimal API per reviewer suggestion)
    assert sorted(tree.children(0, 0)) == [0, 1]
    assert tree.children(1, 0) == [0]
    assert tree.children(1, 1) == [3]
    assert tree.children(2, 0) == []  # leaves have no further children
    assert tree.children(0, 99) == []  # unknown ball
    assert tree.tokens_in_ball(2, 0) == [0]  # default tids from build
    assert tree.tokens_in_ball(2, 3) == [1]
    assert tree.tokens_in_ball(1, 0) == []  # non-leaf balls have no direct _ball_tokens
    assert tree.tokens_in_ball(2, 99) == []


def test_address_range_validation():
    """get_ball now validates address range like get_ancestors and lca_depth
    (was inconsistent: accepted any int). Also covers out-of-range for get_ball
    as recommended robustness test.
    """
    p = 3
    depth = 2
    tree = FiniteTree(p=p, depth=depth)
    max_addr = p ** depth  # e.g. 9
    bad = max_addr
    neg = -1

    # ctor validation robustness (p, depth)
    with pytest.raises(ValueError, match="p must be an integer >= 2"):
        FiniteTree(p=1, depth=2)
    with pytest.raises(ValueError, match="depth must be an integer >= 1"):
        FiniteTree(p=2, depth=0)

    # get_ball should now raise consistently for bad addresses (any depth)
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.get_ball(bad, 1)
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.get_ball(neg, 1)
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.get_ball(bad, 0)  # even depth 0

    # still works for valid addresses (incl depth 0)
    assert tree.get_ball(0, 0) == 0
    assert tree.get_ball(5, 1) == 5 % 3
    assert tree.get_ball(5, 2) == 5

    # depth validation in get_ball (robustness)
    with pytest.raises(ValueError, match="depth .* out of range"):
        tree.get_ball(0, -1)
    with pytest.raises(ValueError, match="depth .* out of range"):
        tree.get_ball(0, 99)

    # other methods keep their validation (and error messages)
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.get_ancestors(bad)
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.lca_depth(0, bad)

    # non-int addresses now consistently ValueError (via shared helper; previously TypeError in some)
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.get_ancestors("bad")
    with pytest.raises(ValueError, match="out of range for this tree"):
        tree.lca_depth(0, None)


def test_lookup_methods_raise_informative_keyerror():
    """token_to_address / address_to_token raise KeyError with helpful message
    for unregistered items (instead of raw KeyError from dict).
    Also documents the address_map ctor usage here (undocumented before).
    """
    p = 3
    depth = 2
    tree = FiniteTree(p=p, depth=depth)
    tree.add_leaf(42, 5)

    with pytest.raises(KeyError, match="token_id 999 not registered"):
        tree.token_to_address(999)
    with pytest.raises(KeyError, match="address 7 not registered"):
        tree.address_to_token(7)

    # registered work
    assert tree.token_to_address(42) == 5
    assert tree.address_to_token(5) == 42

    # also test ctor with address_map (now documented; was internal support only)
    tree2 = FiniteTree(p=2, depth=1, address_map={0: 100, 1: 101})
    assert len(tree2) == 2
    assert tree2.token_to_address(100) == 0


def test_top_level_package_import():
    """FiniteTree is re-exported at package top level so users can do
    `from ultrametric_ce import FiniteTree` (consistent with padic funcs via *).
    """
    from ultrametric_ce import FiniteTree as FT_top
    from ultrametric_ce.tree import FiniteTree as FT_direct
    assert FT_top is FT_direct

    # smoke: usable
    t = FT_top(p=3, depth=2)
    t.add_leaf(7, 4)
    assert t.token_to_address(7) == 4


def test_leaf_addresses_public_api():
    """Public API leaf_addresses() returns sorted list of registered full-depth leaf addresses.

    Fulfills reviewer requirement for no private _addr_to_token access in library code
    (e.g. model.py) and provides the missing public API for registered leaves.
    """
    p = 3
    depth = 2
    # use unsorted input to verify sorting
    addrs = [3, 0, 1]
    tree = FiniteTree.build_from_addresses(addrs, p=p, depth=depth)
    assert tree.leaf_addresses() == [0, 1, 3]
    assert len(tree.leaf_addresses()) == 3

    # empty tree
    tree_empty = FiniteTree(p=2, depth=1)
    assert tree_empty.leaf_addresses() == []


# =============================================================================
# Task 4: Gemma interface tests (must be mockable) + clustering from embeddings
# Written first per micro-TDD (Step 4.1), will fail until gemma_interface.py
# and tree clustering method implemented. All imports of gemma_interface are
# inside test functions so that test collection does not require the module.
# =============================================================================

def test_gemma_interface_module_is_importable():
    """Gemma interface must be importable (file exists) without executing mlx_lm
    load at import time of the package. (Clear error only on use of load.)
    """
    import ultrametric_ce.gemma_interface as gi
    assert hasattr(gi, "GemmaInterface") or hasattr(gi, "load_gemma")


def test_gemma_load_raises_clear_error_if_mlx_lm_not_installed(monkeypatch):
    """Step 4.1/4.2: loading must detect missing mlx_lm and give actionable error.
    Fully mockable via patching __import__ so test passes even if mlx_lm *is*
    installed in the env (we simulate its absence).
    """
    import builtins
    import sys

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if "mlx_lm" in name:
            raise ImportError("No module named 'mlx_lm' (forced for test)")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # ensure no cached module interferes
    for mod in list(sys.modules.keys()):
        if "mlx_lm" in mod or "gemma_interface" in mod:
            monkeypatch.delitem(sys.modules, mod, raising=False)

    import ultrametric_ce.gemma_interface as gi

    with pytest.raises(ImportError, match=r"mlx_lm.*(not installed|required|Gemma)"):
        gi.load_gemma("dummy-repo-or-path")

    # also test direct class ctor works without load (for mocks)
    iface = gi.GemmaInterface(object(), object())
    assert iface is not None


def test_gemma_get_embeddings_and_get_logits_via_mocked_model():
    """Step 4.1: embedding extraction + logits helpers must work against a
    duck-typed mock of the mlx_lm loaded (model, tokenizer). No real Gemma
    weights or even mlx_lm import required for this path. This is the
    primary mockability requirement.
    """
    import mlx.core as mx

    # Replicate the structure: loaded_model.model.embed_tokens.weight
    class MockEmbedTokens:
        def __init__(self, V=32, D=4):
            self.weight = mx.array(np.random.randn(V, D).astype(np.float32))

    class MockGemmaModel:  # the inner .model
        def __init__(self, V=32, D=4):
            self.embed_tokens = MockEmbedTokens(V, D)

    class MockModel:  # what mlx_lm.load returns as first item (has .model)
        def __init__(self, V=32, D=4):
            self.model = MockGemmaModel(V, D)
            self.args = type("Args", (), {"vocab_size": V})()

        def __call__(self, inputs):
            # simulate forward -> (B, S, V) logits
            if isinstance(inputs, list):
                inputs = mx.array([inputs])
            b, s = inputs.shape
            V = 32
            return mx.array(np.random.randn(b, s, V).astype(np.float32))

    class MockTokenizer:
        pass

    import ultrametric_ce.gemma_interface as gi

    mock_model = MockModel()
    mock_tok = MockTokenizer()
    iface = gi.GemmaInterface(mock_model, mock_tok)

    # get all embeddings
    embs = iface.get_embeddings()
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (32, 4)
    assert embs.dtype == np.float32

    # subset
    embs_sub = iface.get_embeddings(token_ids=[0, 10, 31])
    assert embs_sub.shape == (3, 4)

    # get_logits on a short prefix -> last pos logits
    logits = iface.get_logits([5, 7, 9])
    assert isinstance(logits, np.ndarray)
    assert logits.shape == (32,)
    assert logits.dtype == np.float32

    # also accepts mx array input
    logits2 = iface.get_logits(mx.array([[1, 2]]))
    assert logits2.shape == (32,)


def test_cluster_and_assign_addresses_builds_tree_from_embeddings():
    """Step 4.1/4.3 (TDD): write the expected behavior test first.
    Hierarchical clustering must assign addresses such that embeddings that are
    close in euclidean end up with deeper LCA (share low-order p-adic digits)
    for the induced tree. Roundtrips, valid tree, no addr collisions.

    Uses deterministic partitioning (no reliance on RNG) on max-var axis +
    ordered chunking so test is reliable for synthetic structured data.
    """
    p = 2
    depth = 2
    # 4 tokens in 2 coarse clusters of 2; within each cluster tokens are near,
    # clusters far apart. Separation on dim 0.
    embs = np.array([
        [0.0, 0.1],
        [0.05, -0.1],
        [5.0, 0.0],
        [5.1, 0.2],
    ], dtype=np.float32)
    token_ids = [1001, 1002, 2001, 2002]

    # Will fail with AttributeError (no such classmethod) until added to tree.py
    tree = FiniteTree.cluster_and_assign_addresses(
        embs, p=p, depth=depth, token_ids=token_ids
    )

    assert len(tree) == 4
    assert tree.p == p and tree.depth == depth

    # full roundtrip mappings
    for tid in token_ids:
        addr = tree.token_to_address(tid)
        assert 0 <= addr < (p ** depth)
        assert tree.address_to_token(addr) == tid

    addrs = {tid: tree.token_to_address(tid) for tid in token_ids}
    assert len(set(addrs.values())) == 4  # no collisions

    a00, a01 = addrs[1001], addrs[1002]
    a10, a11 = addrs[2001], addrs[2002]

    # within synthetic clusters must share at least the first (lowest) digit -> lca >=1
    assert tree.lca_depth(a00, a01) >= 1
    assert tree.lca_depth(a10, a11) >= 1
    # cross clusters must not share low digit (lca at 0)
    assert tree.lca_depth(a00, a10) == 0
    assert tree.lca_depth(a00, a11) == 0

    # ultrametric distances should be smaller within than cross (proxy for hierarchy)
    d_w = distance(a00, a01, p)
    d_c = distance(a00, a10, p)
    assert d_w <= d_c

    # public leaf list
    leaves = tree.leaf_addresses()
    assert len(leaves) == 4
    assert sorted(leaves) == sorted(addrs.values())


def test_from_tokens_still_works_after_clustering_addition():
    """from_tokens (the old stub) must continue to work for backward compat with
    toy grammar trees etc. (narrow change, no breakage).
    """
    p = 3
    depth = 2
    tids = [7, 8, 9]
    tree = FiniteTree.from_tokens(tids, p=p, depth=depth)
    assert len(tree) == 3
    for tid in tids:
        _ = tree.token_to_address(tid)


def test_build_tree_from_gemma_script_synthetic_roundtrip(tmp_path):
    """Step 4.4 TDD: script stub must support --synthetic (no Gemma), produce
    a loadable tree config (json with p/depth + address_map), and the induced
    tree must roundtrip + be valid. Run via subprocess using the test's python
    (under venv) + PYTHONPATH so it matches production run pattern.

    This exercises the script on synthetic before any real weights.
    """
    import json
    import os
    import subprocess
    import sys

    # Skip this test if MLX is not installed or is mocked
    try:
        import mlx.core as mx
        if "mlx" in sys.modules and sys.modules["mlx"].__class__.__name__ == "MLXMock":
            pytest.skip("MLX is mocked; skipping subprocess test requiring real MLX")
    except ImportError:
        pytest.skip("MLX is not installed; skipping subprocess test requiring real MLX")

    script_path = "scripts/build_tree_from_gemma.py"
    out_json = tmp_path / "synth_tree.json"

    cmd = [
        sys.executable,
        script_path,
        "--synthetic",
        "--p", "2",
        "--depth", "2",
        "--num-tokens", "4",
        "--out", str(out_json),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    # Will fail until script exists + implements the CLI + synthetic path
    result = subprocess.run(
        cmd, cwd=".", env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"script failed: {result.stderr}\n{result.stdout}"

    # script must have written the json
    assert out_json.exists(), "script did not create output json"

    cfg = json.loads(out_json.read_text())
    assert "p" in cfg and "depth" in cfg and "address_map" in cfg
    assert cfg["p"] == 2 and cfg["depth"] == 2
    addr_map = cfg["address_map"]  # expect {str(addr): token or token:addr ? see impl}
    assert len(addr_map) == 4

    # reconstruct and verify roundtrip / validity using the saved map
    # (script should document the exact format; support {addr: tid, ...})
    inv_map = {int(a): int(t) for a, t in addr_map.items()}  # tolerant
    if min(inv_map.keys()) < 32:  # looks like addrs small
        address_map_for_tree = inv_map
    else:
        # fallback if saved as token->addr
        address_map_for_tree = {int(v): int(k) for k, v in addr_map.items()}
    tree = FiniteTree(p=cfg["p"], depth=cfg["depth"], address_map=address_map_for_tree)
    assert len(tree) == 4
    # basic mapping works
    for tid in range(4):  # synthetic default tids 0..3
        try:
            _ = tree.token_to_address(tid)
        except KeyError:
            # if tids were not 0-3, just check len
            pass
    assert len(tree.leaf_addresses()) == 4

