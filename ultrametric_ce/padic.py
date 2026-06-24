from typing import List

__all__ = ["padic_address", "address_to_digits", "valuation", "norm", "distance"]


def padic_address(digits: List[int], p: int) -> int:
    """Convert list of base-p digits (low to high) to integer address."""
    addr = 0
    power = 1
    for d in digits:
        addr += d * power
        power *= p
    return addr

def address_to_digits(addr: int, p: int, k: int) -> List[int]:
    digits = []
    for _ in range(k):
        digits.append(addr % p)
        addr //= p
    return digits

def valuation(x: int, p: int) -> int | float:
    if x == 0:
        return float('inf')
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v

def norm(x: int, p: int) -> float:
    if x == 0:
        return 0.0
    return p ** -valuation(x, p)

def distance(x: int, y: int, p: int) -> float:
    return norm(x - y, p)
