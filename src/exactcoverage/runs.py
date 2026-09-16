"""Closed-form distribution of run structure in i.i.d. Bernoulli hit sequences.

A binary sequence of length ``n`` is summarised by its *run signature*
``(x, r, first, last)``: the number of ones, the number of maximal runs of
ones, and the first and last bits. The signature determines every transition
count used by Christoffersen's tests, and the number of sequences sharing a
signature has a closed form, so exact null distributions need no simulation.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Dict, Iterator, NamedTuple, Sequence, Tuple, Union

Signature = Tuple[int, int, int, int]


class Transitions(NamedTuple):
    n00: int
    n01: int
    n10: int
    n11: int


def _check_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"sequence length n must be a positive integer, got {n!r}")


def run_count(n: int, x: int, r: int, first: int, last: int) -> int:
    """Number of binary sequences of length n with x ones in r runs and the given end bits.

    The ones form r runs, i.e. a composition of x into r positive parts:
    C(x-1, r-1) ways. The zeros form s = r + 1 - first - last runs (one more
    than r when both ends are zero, one fewer when both ends are one), a
    composition of n-x into s parts: C(n-x-1, s-1) ways. Runs alternate, so the
    two choices are independent.
    """
    _check_n(n)
    if first not in (0, 1) or last not in (0, 1):
        raise ValueError("first and last must be 0 or 1")
    if not 0 <= x <= n or r < 0:
        return 0
    z = n - x
    s = r + 1 - first - last
    if x == 0:
        return 1 if (r == 0 and first == 0 and last == 0) else 0
    if z == 0:
        return 1 if (r == 1 and first == 1 and last == 1) else 0
    if r < 1 or s < 1:
        return 0
    return math.comb(x - 1, r - 1) * math.comb(z - 1, s - 1)


def transitions(n: int, x: int, r: int, first: int, last: int) -> Transitions:
    """Transition counts (n00, n01, n10, n11) implied by a run signature.

    Every run of ones except one starting on day 1 is entered from a zero
    (n01 = r - first), every run except one ending on day n is left to a zero
    (n10 = r - last), and the remaining adjacent pairs lie inside runs.
    """
    s = r + 1 - first - last
    return Transitions(n - x - s, r - first, r - last, x - r)


def signatures(n: int) -> Iterator[Tuple[int, int, int, int, int]]:
    """Yield ``(x, r, first, last, count)`` for every signature with count > 0."""
    _check_n(n)
    for x in range(n + 1):
        z = n - x
        if x == 0:
            yield 0, 0, 0, 0, 1
            continue
        if z == 0:
            yield n, 1, 1, 1, 1
            continue
        for r in range(1, min(x, z + 1) + 1):
            for first in (0, 1):
                for last in (0, 1):
                    c = run_count(n, x, r, first, last)
                    if c:
                        yield x, r, first, last, c


def signature_of(hits: Sequence[int]) -> Signature:
    """Run signature ``(x, r, first, last)`` of an explicit 0/1 sequence."""
    if len(hits) == 0:
        raise ValueError("hit sequence is empty")
    x = 0
    r = 0
    prev = 0
    for h in hits:
        if h not in (0, 1):
            raise ValueError(f"hits must be 0 or 1, got {h!r}")
        h = int(h)
        x += h
        if h == 1 and prev == 0:
            r += 1
        prev = h
    return x, r, int(hits[0]), int(hits[-1])


def count_transitions(hits: Sequence[int]) -> Transitions:
    """Transition counts computed directly from adjacent pairs (reference implementation)."""
    counts = [0, 0, 0, 0]
    for a, b in zip(hits, hits[1:]):
        counts[2 * int(a) + int(b)] += 1
    return Transitions(*counts)


def exact_distribution(n: int, p: Union[Fraction, int]) -> Dict[Signature, Fraction]:
    """Null distribution over run signatures in exact rational arithmetic."""
    p = Fraction(p)
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1]")
    q = 1 - p
    return {(x, r, first, last): c * p**x * q ** (n - x) for x, r, first, last, c in signatures(n)}


def log_distribution(n: int, p: float) -> Iterator[Tuple[int, int, int, int, float]]:
    """Yield ``(x, r, first, last, probability)`` computed in log space.

    Binomial coefficients enter as log-factorials so n in the thousands neither
    overflows nor loses the tails to cancellation.
    """
    _check_n(n)
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie strictly between 0 and 1")
    lf = log_factorials(n)
    logp, logq = math.log(p), math.log1p(-p)
    yield 0, 0, 0, 0, math.exp(n * logq)
    for x in range(1, n):
        z = n - x
        base = x * logp + z * logq
        for r in range(1, min(x, z + 1) + 1):
            ones = lf[x - 1] - lf[r - 1] - lf[x - r]
            for first in (0, 1):
                for last in (0, 1):
                    s = r + 1 - first - last
                    if s < 1 or s > z:
                        continue
                    zeros = lf[z - 1] - lf[s - 1] - lf[z - s]
                    yield x, r, first, last, math.exp(base + ones + zeros)
    yield n, 1, 1, 1, math.exp(n * logp)


def log_factorials(n: int) -> list:
    return [math.lgamma(k + 1) for k in range(n + 1)]


def brute_force_counts(n: int) -> Dict[Signature, int]:
    """Count run signatures by enumerating all 2^n sequences."""
    _check_n(n)
    counts: Dict[Signature, int] = {}
    for seq in itertools.product((0, 1), repeat=n):
        key = signature_of(seq)
        counts[key] = counts.get(key, 0) + 1
    return counts


def brute_force_distribution(n: int, p: Union[Fraction, int]) -> Dict[Signature, Fraction]:
    """Null distribution over run signatures by summing the probability of all 2^n sequences."""
    _check_n(n)
    p = Fraction(p)
    q = 1 - p
    # A sequence's probability depends only on its number of ones.
    weight = [p**k * q ** (n - k) for k in range(n + 1)]
    dist: Dict[Signature, Fraction] = {}
    for seq in itertools.product((0, 1), repeat=n):
        key = signature_of(seq)
        dist[key] = dist.get(key, Fraction(0)) + weight[key[0]]
    return dist
