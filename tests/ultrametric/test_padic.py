import pytest
from ultrametric_ce.padic import padic_address, valuation, norm, distance, address_to_digits

def test_padic_address_roundtrip():
    digits = [1, 0, 2]
    p = 3
    addr = padic_address(digits, p)
    assert addr == 1 + 0*3 + 2*9
    recovered = address_to_digits(addr, p, len(digits))
    assert recovered == digits

def test_valuation_and_norm():
    assert valuation(9, 3) == 2  # 3^2 divides 9
    assert norm(9, 3) == pytest.approx(3 ** -2)
    assert norm(0, 3) == 0.0

def test_ultrametric_distance():
    """Explicit cases for p-adic distance (satisfies ultrametric inequality)."""
    p = 3
    # identity
    assert distance(0, 0, p) == 0
    assert distance(1, 1, p) == 0
    # examples from comments/specs
    assert distance(1, 4, p) == 1 / 3  # 1 and 4=1+3 differ at p^1 place, val=1
    assert distance(0, 1, p) == 1.0    # differ at p^0, val(diff)=0 -> norm= p^0 =1
    assert distance(0, 3, p) == 1 / 3  # 3 divisible by p once
    # ultrametric demo: points in different top level
    assert distance(0, 2, p) == 1
    assert distance(1, 2, p) == 1
    # e.g. d(0,2) <= max(d(0,1),d(1,2)) holds as 1 <= max(1,1)


def test_ultrametric_inequality():
    """Verify the strong triangle inequality: d(x, z) <= max(d(x, y), d(y, z)) for all triples."""
    p = 3
    # Test on a range of small addresses (covers various digit prefixes)
    test_vals = list(range(27)) + [81, 243]  # up to p^3 and some higher powers
    for x in test_vals:
        for y in test_vals:
            for z in test_vals:
                dxy = distance(x, y, p)
                dyz = distance(y, z, p)
                dxz = distance(x, z, p)
                assert dxz <= max(dxy, dyz), (
                    f"Ultrametric inequality violated for p={p}, x={x}, y={y}, z={z}: "
                    f"d(x,z)={dxz} > max(d(x,y)={dxy}, d(y,z)={dyz})"
                )
    # Specific sanity: identity and the example from original test
    assert distance(1, 1, p) == 0
    # 1 and 4 differ at first digit: dist should be p^{-0}=1? wait compute: actually 1/3 as val(3)=1
    assert distance(1, 4, p) == 1 / 3
