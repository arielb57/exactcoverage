"""Likelihood-ratio statistics and their exact null distributions."""

from __future__ import annotations

import math
from bisect import bisect_left
from functools import lru_cache
from typing import Iterable, List, Tuple

from .runs import log_distribution, log_factorials, transitions

# Statistics computed from different integer counts can agree mathematically yet
# differ by rounding noise; outcomes this close are treated as ties and counted
# as "at least as extreme", which keeps the exact p-value conservative.
TIE_TOLERANCE = 1e-9


def _xlogy(k: int, prob: float) -> float:
    # 0 * log(0) = 0 is the limit and the value of the maximised likelihood,
    # which is what keeps Christoffersen's statistic finite when n11 = 0.
    return 0.0 if k == 0 else k * math.log(prob)


def lr_pof(n: int, x: int, p: float) -> float:
    """Kupiec's proportion-of-failures likelihood ratio."""
    phat = x / n
    ll_null = _xlogy(n - x, 1.0 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1.0 - phat) + _xlogy(x, phat)
    return max(0.0, 2.0 * (ll_alt - ll_null))


def lr_ind(n00: int, n01: int, n10: int, n11: int) -> float:
    """Christoffersen's independence likelihood ratio (first-order Markov vs i.i.d.)."""
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0
    ones = n01 + n11
    pi = ones / total
    ll_null = _xlogy(total - ones, 1.0 - pi) + _xlogy(ones, pi)
    ll_alt = 0.0
    from_zero = n00 + n01
    if from_zero:
        pi01 = n01 / from_zero
        ll_alt += _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
    from_one = n10 + n11
    if from_one:
        pi11 = n11 / from_one
        ll_alt += _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    return max(0.0, 2.0 * (ll_alt - ll_null))


def chi2_sf(stat: float, df: int) -> float:
    """Survival function of the chi-square distribution for df = 1 or 2."""
    if stat <= 0.0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(stat / 2.0))
    if df == 2:
        return math.exp(-stat / 2.0)
    raise ValueError("only df = 1 and df = 2 are needed by these tests")


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class Tail:
    """Upper-tail probabilities of a discrete distribution over statistic values."""

    def __init__(self, outcomes: Iterable[Tuple[float, float]]):
        ordered = sorted(outcomes)
        stats: List[float] = []
        probs: List[float] = []
        for stat, prob in ordered:
            if stats and stat - stats[-1] <= TIE_TOLERANCE * max(1.0, abs(stat)):
                probs[-1] += prob
            else:
                stats.append(stat)
                probs.append(prob)
        suffix = [0.0] * (len(stats) + 1)
        running = 0.0
        for i in range(len(stats) - 1, -1, -1):
            running += probs[i]
            suffix[i] = running
        # The whole distribution is at least its smallest value; pin that to 1
        # so rounding in the total never makes the least extreme outcome look rare.
        suffix[0] = 1.0
        self.stats = stats
        self.suffix = suffix

    def sf(self, stat: float) -> float:
        """P(statistic >= stat), counting ties within TIE_TOLERANCE."""
        i = bisect_left(self.stats, stat - TIE_TOLERANCE * max(1.0, abs(stat)))
        return min(1.0, self.suffix[i])

    def strictly_at_least(self, threshold: float) -> float:
        """P(statistic >= threshold) with no tie tolerance (for fixed critical values)."""
        return min(1.0, self.suffix[bisect_left(self.stats, threshold)])

    def exact_test_size(self, alpha: float) -> float:
        """Rejection probability of the test that rejects when sf(observed) <= alpha."""
        for value in self.suffix:
            if value <= alpha:
                return value
        return 0.0


def check_p(p: float) -> float:
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"coverage rate p must lie strictly between 0 and 1, got {p!r}")
    return p


@lru_cache(maxsize=64)
def pof_tail(n: int, p: float) -> Tail:
    lf = log_factorials(n)
    logp, logq = math.log(p), math.log1p(-p)
    return Tail(
        (lr_pof(n, x, p), math.exp(lf[n] - lf[x] - lf[n - x] + x * logp + (n - x) * logq))
        for x in range(n + 1)
    )


@lru_cache(maxsize=4096)
def ind_conditional_tail(n: int, x: int) -> Tail:
    """Null distribution of LR_ind given x exceptions.

    Given x, every arrangement of the exceptions is equally likely under any
    i.i.d. hit rate, so this distribution is free of the unknown rate.
    """
    if x == 0 or x == n:
        return Tail([(0.0, 1.0)])
    lf = _log_factorials(n)
    z = n - x
    log_total = lf[n] - lf[x] - lf[z]
    outcomes = []
    for r in range(1, min(x, z + 1) + 1):
        ones = lf[x - 1] - lf[r - 1] - lf[x - r]
        for first in (0, 1):
            for last in (0, 1):
                s = r + 1 - first - last
                if 1 <= s <= z:
                    zeros = lf[z - 1] - lf[s - 1] - lf[z - s]
                    outcomes.append(
                        (
                            lr_ind(*transitions(n, x, r, first, last)),
                            math.exp(ones + zeros - log_total),
                        )
                    )
    return Tail(outcomes)


@lru_cache(maxsize=16)
def _log_factorials(n: int) -> Tuple[float, ...]:
    return tuple(log_factorials(n))


@lru_cache(maxsize=16)
def _joint_tails(n: int, p: float) -> Tuple[Tail, Tail]:
    pof_by_x = [lr_pof(n, x, p) for x in range(n + 1)]
    cc = []
    ind = []
    for x, r, first, last, prob in log_distribution(n, p):
        li = lr_ind(*transitions(n, x, r, first, last))
        ind.append((li, prob))
        cc.append((pof_by_x[x] + li, prob))
    return Tail(ind), Tail(cc)


def ind_unconditional_tail(n: int, p: float) -> Tail:
    return _joint_tails(n, p)[0]


def cc_tail(n: int, p: float) -> Tail:
    return _joint_tails(n, p)[1]


def binomial_pmf(n: int, p: float) -> List[float]:
    """[P(X = k) for k = 0..n] for X ~ Binomial(n, p), evaluated in log space."""
    lf = log_factorials(n)
    logp, logq = math.log(p), math.log1p(-p)
    return [math.exp(lf[n] - lf[k] - lf[n - k] + k * logp + (n - k) * logq) for k in range(n + 1)]


def binomial_cdf_table(n: int, p: float) -> List[float]:
    """[P(X <= k) for k = 0..n] for X ~ Binomial(n, p)."""
    pmf = binomial_pmf(n, p)
    return [min(1.0, math.fsum(pmf[: k + 1])) for k in range(n + 1)]


def binomial_upper_tail(n: int, p: float, x: int) -> float:
    """P(X >= x), summed directly so small tails keep their precision."""
    return min(1.0, math.fsum(binomial_pmf(n, p)[x:]))
