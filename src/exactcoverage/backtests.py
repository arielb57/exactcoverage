"""Public VaR backtests.

Kupiec POF, Christoffersen independence and conditional coverage, and the Basel
traffic light, each with an exact and an asymptotic p-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .runs import signature_of, transitions
from .stats import (
    binomial_cdf_table,
    binomial_pmf,
    binomial_upper_tail,
    cc_tail,
    check_p,
    chi2_sf,
    ind_conditional_tail,
    ind_unconditional_tail,
    lr_ind,
    lr_pof,
    normal_sf,
    pof_tail,
)

# Plus factors from the 1996 Basel supervisory framework; they are defined only
# for 250 observations of a 99% VaR.
BASEL_PLUS_FACTORS = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.40,
    6: 0.50,
    7: 0.65,
    8: 0.75,
    9: 0.85,
}
GREEN_LIMIT = 0.95
RED_LIMIT = 0.9999


@dataclass(frozen=True)
class BacktestResult:
    test: str
    n: int
    exceptions: int
    statistic: float
    p_value_exact: float
    p_value_asymptotic: float
    df: int

    def reject(self, alpha: float = 0.05, exact: bool = True) -> bool:
        return (self.p_value_exact if exact else self.p_value_asymptotic) <= alpha


@dataclass(frozen=True)
class TrafficLightResult:
    test: str
    n: int
    exceptions: int
    statistic: float
    p_value_exact: float
    p_value_asymptotic: float
    cumulative_probability: float
    zone: str
    yellow_from: int
    red_from: int
    plus_factor: Optional[float]


def _as_hits(hits: Iterable) -> List[int]:
    out = []
    for h in hits:
        if isinstance(h, bool):
            out.append(int(h))
        elif isinstance(h, int) and h in (0, 1):
            out.append(h)
        elif isinstance(h, float) and h in (0.0, 1.0):
            out.append(int(h))
        else:
            raise ValueError(f"hits must be 0/1 values, got {h!r}")
    if not out:
        raise ValueError("hit sequence is empty")
    return out


def pof(hits: Iterable, p: float) -> BacktestResult:
    """Kupiec's proportion-of-failures test that the exception rate equals p (two-sided)."""
    p = check_p(p)
    seq = _as_hits(hits)
    n, x = len(seq), sum(seq)
    stat = lr_pof(n, x, p)
    return BacktestResult("pof", n, x, stat, pof_tail(n, p).sf(stat), chi2_sf(stat, 1), 1)


def independence(hits: Iterable) -> BacktestResult:
    """Christoffersen's independence test.

    The exact p-value is conditional on the observed number of exceptions,
    which makes it exact whatever the true (i.i.d.) exception rate is.
    """
    seq = _as_hits(hits)
    n = len(seq)
    x, r, first, last = signature_of(seq)
    stat = lr_ind(*transitions(n, x, r, first, last))
    return BacktestResult(
        "independence", n, x, stat, ind_conditional_tail(n, x).sf(stat), chi2_sf(stat, 1), 1
    )


def conditional_coverage(hits: Iterable, p: float) -> BacktestResult:
    """Christoffersen's conditional coverage test: LR_cc = LR_pof + LR_ind."""
    p = check_p(p)
    seq = _as_hits(hits)
    n = len(seq)
    x, r, first, last = signature_of(seq)
    stat = lr_pof(n, x, p) + lr_ind(*transitions(n, x, r, first, last))
    return BacktestResult(
        "conditional_coverage", n, x, stat, cc_tail(n, p).sf(stat), chi2_sf(stat, 2), 2
    )


def traffic_light_zones(n: int, p: float) -> Tuple[int, int, List[float]]:
    """(first yellow count, first red count, cumulative probabilities P(X <= k))."""
    p = check_p(p)
    cdf = binomial_cdf_table(n, p)
    yellow = next((k for k, c in enumerate(cdf) if c >= GREEN_LIMIT), n + 1)
    red = next((k for k, c in enumerate(cdf) if c >= RED_LIMIT), n + 1)
    return yellow, red, cdf


def traffic_light(hits: Iterable, p: float = 0.01) -> TrafficLightResult:
    """Basel traffic-light classification from exact binomial cumulative probabilities.

    The exact p-value is P(X >= exceptions); the asymptotic one is the normal
    approximation to the same one-sided tail.
    """
    p = check_p(p)
    seq = _as_hits(hits)
    n, x = len(seq), sum(seq)
    yellow, red, cdf = traffic_light_zones(n, p)
    zone = "green" if x < yellow else ("yellow" if x < red else "red")
    plus = None
    if n == 250 and abs(p - 0.01) < 1e-15:
        plus = BASEL_PLUS_FACTORS.get(x, 1.0)
    z = (x - n * p) / math.sqrt(n * p * (1.0 - p))
    return TrafficLightResult(
        test="traffic_light",
        n=n,
        exceptions=x,
        statistic=float(x),
        p_value_exact=binomial_upper_tail(n, p, x),
        p_value_asymptotic=normal_sf(z),
        cumulative_probability=cdf[x],
        zone=zone,
        yellow_from=yellow,
        red_from=red,
        plus_factor=plus,
    )


@dataclass(frozen=True)
class SizeRow:
    n: int
    p: float
    alpha: float
    test: str
    asymptotic_size: float
    exact_size: float


def _chi2_critical(alpha: float, df: int) -> float:
    if df == 2:
        return -2.0 * math.log(alpha)
    lo, hi = 0.0, 1.0
    while chi2_sf(hi, 1) > alpha:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, 1) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def size_report(n: int, p: float, alpha: float = 0.05) -> List[SizeRow]:
    """True rejection rates under the null, computed exactly.

    ``asymptotic_size`` is the probability that the chi-square p-value is at
    most alpha; ``exact_size`` is the same for the exact p-value, which can only
    be at most alpha (discreteness keeps it below).
    """
    p = check_p(p)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    crit1 = _chi2_critical(alpha, 1)
    crit2 = _chi2_critical(alpha, 2)

    pt = pof_tail(n, p)
    it = ind_unconditional_tail(n, p)
    ct = cc_tail(n, p)

    pmf = binomial_pmf(n, p)
    ind_exact = math.fsum(
        pmf[x] * ind_conditional_tail(n, x).exact_test_size(alpha) for x in range(n + 1)
    )
    return [
        SizeRow(n, p, alpha, "pof", pt.strictly_at_least(crit1), pt.exact_test_size(alpha)),
        SizeRow(n, p, alpha, "independence", it.strictly_at_least(crit1), ind_exact),
        SizeRow(
            n,
            p,
            alpha,
            "conditional_coverage",
            ct.strictly_at_least(crit2),
            ct.exact_test_size(alpha),
        ),
    ]
