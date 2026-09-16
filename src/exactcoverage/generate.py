"""Synthetic hit sequences: i.i.d. Bernoulli nulls and clustered Markov alternatives."""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional, Tuple


def _run_length(rng: random.Random, leave: float, remaining: int) -> int:
    """Length of a run that ends each day with probability ``leave`` (geometric, capped)."""
    if leave >= 1.0:
        return 1
    if leave <= 0.0:
        return remaining
    # Inverse-CDF sampling draws a whole run with one uniform instead of one per day,
    # which is what makes 200k-sequence Monte Carlo checks affordable in pure Python.
    u = 1.0 - rng.random()
    return min(remaining, int(math.log(u) / math.log1p(-leave)) + 1)


def markov_hits(
    n: int,
    p01: float,
    p11: float,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """First-order Markov hit sequence with P(1 | 0) = p01 and P(1 | 1) = p11.

    The first day is drawn from the stationary distribution, so the sequence is
    stationary with exception rate p01 / (1 - p11 + p01).
    """
    out: List[int] = []
    for state, length in _runs(n, p01, p11, rng if rng is not None else random.Random(seed)):
        out.extend([state] * length)
    return out


def markov_signature(
    n: int, p01: float, p11: float, rng: random.Random
) -> Tuple[int, int, int, int]:
    """Run signature (x, r, first, last) of a Markov hit sequence, without materialising it."""
    x = r = 0
    first = last = -1
    for state, length in _runs(n, p01, p11, rng):
        if first < 0:
            first = state
        if state:
            x += length
            r += 1
        last = state
    return x, r, first, last


def _runs(n: int, p01: float, p11: float, rng: random.Random) -> Iterator[Tuple[int, int]]:
    if n < 1:
        raise ValueError("n must be a positive integer")
    for name, value in (("p01", p01), ("p11", p11)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    denom = 1.0 - p11 + p01
    stationary = p01 / denom if denom > 0 else 1.0
    state = 1 if rng.random() < stationary else 0
    done = 0
    while done < n:
        length = _run_length(rng, (1.0 - p11) if state else p01, n - done)
        yield state, length
        done += length
        state = 1 - state


def bernoulli_hits(
    n: int, p: float, seed: Optional[int] = None, rng: Optional[random.Random] = None
) -> List[int]:
    """I.i.d. Bernoulli(p) hit sequence: the null hypothesis of every test here."""
    return markov_hits(n, p, p, seed=seed, rng=rng)


def clustered_p01(p: float, p11: float) -> float:
    """P(1 | 0) that keeps the unconditional exception rate at p when P(1 | 1) = p11."""
    if not 0.0 < p < 1.0 or not 0.0 <= p11 < 1.0:
        raise ValueError("need 0 < p < 1 and 0 <= p11 < 1")
    p01 = p * (1.0 - p11) / (1.0 - p)
    if p01 > 1.0:
        raise ValueError("no Markov chain has this exception rate and P(1|1)")
    return p01
