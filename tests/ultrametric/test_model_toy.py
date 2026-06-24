"""Toy synthetic data + basic UCE model forward tests (tree + routing + diffusion skeleton).

TDD: this test is written first and must fail before implementation.
"""

import mlx.core as mx

from ultrametric_ce.padic import padic_address, address_to_digits
from ultrametric_ce.tree import FiniteTree
from ultrametric_ce.model import UCEModel


def build_toy_arithmetic_tree():
    """Hard-code a tiny grammar tree (p=3, K=4) for 21 synthetic expression tokens.

    Tokens grouped roughly by category in p-adic prefixes for 'grammar' flavor:
    - digits under low-digit 0 sector
    - operators and open-paren under ~1
    - close-paren, vars, =, ^ under ~2
    Addresses manually assigned (not pure sequential) to demonstrate hard-coded layout.
    """
    p = 3
    depth = 4
    # 21 tokens (target ~20-30)
    symbols = [str(d) for d in range(10)] + ["+", "-", "*", "/", "(", ")", "x", "y", "z", "=", "^"]
    token_ids = list(range(len(symbols)))
    # Hard-coded addresses: use different low digits for rough categories, ensure unique and <81
    addresses = []
    # digits 0-9: force low digit 0, vary higher to spread but stay in 'digit-ish' subtrees
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
    # ) x y z = ^ : low digit 2 (close paren, vars, equals, power; 6 tokens)
    for i in range(6):
        d1 = i % 3
        d2 = (i // 3) % 3
        d3 = (i // 9) % 3
        addr = padic_address([2, d1, d2, d3], p)
        addresses.append(addr)
    # ensure unique
    assert len(set(addresses)) == len(addresses), "address collision in toy tree hard-code"
    assert all(0 <= a < (p ** depth) for a in addresses)
    tree = FiniteTree.build_from_addresses(addresses, p=p, depth=depth, token_ids=token_ids)
    sym_to_token = {sym: tid for tid, sym in enumerate(symbols)}
    token_to_sym = {tid: sym for tid, sym in enumerate(symbols)}
    return tree, sym_to_token, token_to_sym, symbols


def expr_to_address_sequence(expr: str, sym_to_token: dict, tree: FiniteTree) -> list[int]:
    """Map a hard-coded valid expression string to its sequence of p-adic addresses."""
    addrs = []
    for ch in expr:
        if ch not in sym_to_token:
            # allow only controlled exprs; skip unexpected for robustness in toy
            continue
        tid = sym_to_token[ch]
        addr = tree.token_to_address(tid)
        addrs.append(addr)
    return addrs


# Hard-coded valid (structurally plausible for later validators) nested arithmetic exprs
VALID_TOY_EXPRS = [
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


def test_toy_model_forward_pass_distribution_sums_to_one_and_can_sample():
    """End-to-end toy forward: model(prev_addrs) -> dist over leaves summing to 1, and sample works.

    This is the primary Step 3.1 failing test (written first, per TDD).
    """
    tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
    assert len(tree) == 21
    assert tree.p == 3
    assert tree.depth == 4

    model = UCEModel(tree)
    # Verify leaf list uses public API on tree (sorted registered addrs); white-box for integration
    assert model.leaf_addrs == tree.leaf_addresses()
    assert model.num_leaves == len(tree)
    # Use a valid prefix from hard-coded expr
    expr = "(1+2)"
    prev_addrs = expr_to_address_sequence(expr, sym_to_token, tree)
    assert len(prev_addrs) > 0

    dist = model(prev_addrs)
    assert isinstance(dist, mx.array)
    assert dist.shape == (len(tree),)
    s = float(mx.sum(dist))
    assert abs(s - 1.0) < 1e-5, f"distribution must sum to 1, got {s}"

    # sampling must produce a registered address (leaf in tree)
    sampled_addr = model.sample(prev_addrs)
    assert isinstance(sampled_addr, int)
    # must be registered
    tok = tree.address_to_token(sampled_addr)
    assert tok in token_to_sym

    # empty prefix case must also yield valid dist
    dist0 = model([])
    assert dist0.shape == (len(tree),)
    s0 = float(mx.sum(dist0))
    assert abs(s0 - 1.0) < 1e-5


def test_toy_data_generator_produces_valid_address_sequences():
    """Smoke that hard-coded exprs map to proper address seqs of correct length."""
    tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
    for expr in VALID_TOY_EXPRS:
        addrs = expr_to_address_sequence(expr, sym_to_token, tree)
        # length should match non-ignored chars
        expected_len = sum(1 for ch in expr if ch in sym_to_token)
        assert len(addrs) == expected_len
        for a in addrs:
            assert 0 <= a < (3 ** 4)
            # roundtrips
            sym = token_to_sym[tree.address_to_token(a)]
            assert sym in symbols


def test_toy_generation_loop_runs_and_produces_valid_token_sequence():
    """Generation loop test (Step 3.5): autoregressive sampling works for multiple steps.

    Uses random weights (skeleton), so output is not yet 'expression-like' semantically,
    but must only emit valid leaves from the hard-coded tree and keep valid dists.
    """
    tree, sym_to_token, token_to_sym, symbols = build_toy_arithmetic_tree()
    model = UCEModel(tree, dim=8, num_diff_layers=1)

    # start from a prefix taken from hard-coded valid expr data
    start_expr = "((1+2)*"
    prev_addrs = expr_to_address_sequence(start_expr, sym_to_token, tree)
    assert len(prev_addrs) >= 3

    generated = list(prev_addrs)
    for step in range(6):
        next_addr = model.sample(generated)
        generated.append(next_addr)
        # after each step, the 'current' dist from model must still sum to 1
        dist = model(generated)
        assert dist.shape == (len(tree),)
        assert abs(float(mx.sum(dist)) - 1.0) < 1e-5

    assert len(generated) == len(prev_addrs) + 6

    # all generated addresses must correspond to registered tokens in the toy tree
    for addr in generated:
        tok_id = tree.address_to_token(addr)
        sym = token_to_sym[tok_id]
        assert sym in symbols

    # reconstruct for visibility (will be gibberish at random init, but structurally 'tokens')
    gen_str = "".join(token_to_sym[tree.address_to_token(a)] for a in generated)
    # just ensure non-empty and uses only our alphabet
    assert len(gen_str) > 0
    assert all(ch in symbols for ch in gen_str)
    # demo output (random init so not meaningful yet, but proves loop + sampling + reconstruction)
    print(f"\\n[toy generation demo start='{start_expr}'] -> {gen_str}")
