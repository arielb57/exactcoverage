"""Seeded Monte Carlo checks: exact tests are valid, size computations are right, power grows."""

import math
import random
import statistics

import pytest

from exactcoverage import (
    bernoulli_hits,
    clustered_p01,
    conditional_coverage,
    markov_hits,
    size_report,
)
from exactcoverage.generate import markov_signature
from exactcoverage.runs import count_transitions, signature_of, transitions
from exactcoverage.stats import (
    cc_tail,
    chi2_sf,
    ind_conditional_tail,
    lr_ind,
    lr_pof,
    pof_tail,
)

ALPHA = 0.05
Z = 3.29  # two-sided 99.9% normal quantile


def _decisions(n, p, signature):
    x, r, first, last = signature
    lp = lr_pof(n, x, p)
    li = lr_ind(*transitions(n, x, r, first, last))
    return (
        pof_tail(n, p).sf(lp) <= ALPHA,
        ind_conditional_tail(n, x).sf(li) <= ALPHA,
        cc_tail(n, p).sf(lp + li) <= ALPHA,
        chi2_sf(lp, 1) <= ALPHA,
        chi2_sf(li, 1) <= ALPHA,
        chi2_sf(lp + li, 2) <= ALPHA,
    )


@pytest.mark.parametrize("n,p,draws,seed", [(250, 0.01, 200_000, 20260916), (250, 0.05, 50_000, 7)])
def test_monte_carlo_under_null_confirms_validity_and_exact_size(n, p, draws, seed):
    rng = random.Random(seed)
    cache = {}
    rejections = [0] * 6
    for _ in range(draws):
        sig = markov_signature(n, p, p, rng)
        decision = cache.get(sig)
        if decision is None:
            decision = cache[sig] = _decisions(n, p, sig)
        for i, rejected in enumerate(decision):
            rejections[i] += rejected

    rows = {row.test: row for row in size_report(n, p, ALPHA)}
    predicted = [
        rows["pof"].exact_size,
        rows["independence"].exact_size,
        rows["conditional_coverage"].exact_size,
        rows["pof"].asymptotic_size,
        rows["independence"].asymptotic_size,
        rows["conditional_coverage"].asymptotic_size,
    ]
    for i, (count, expected) in enumerate(zip(rejections, predicted)):
        rate = count / draws
        margin = Z * math.sqrt(expected * (1 - expected) / draws) + 1.0 / draws
        assert abs(rate - expected) <= margin, (i, rate, expected)
        if i < 3:
            # Validity: the exact tests reject at most alpha, up to binomial noise.
            assert rate <= ALPHA + Z * math.sqrt(ALPHA * (1 - ALPHA) / draws), (i, rate)


def test_asymptotic_pof_over_rejects_at_one_year_of_99_percent_var():
    rows = {row.test: row for row in size_report(250, 0.01, ALPHA)}
    assert rows["pof"].asymptotic_size > 0.09
    assert rows["pof"].exact_size <= ALPHA


def test_signature_sampler_matches_materialised_sequences():
    for seed in range(50):
        hits = markov_hits(300, 0.02, 0.4, seed=seed)
        assert markov_signature(300, 0.02, 0.4, random.Random(seed)) == signature_of(hits)


def test_power_of_conditional_coverage_grows_with_sample_size():
    p, p11 = 0.01, 0.3
    p01 = clustered_p01(p, p11)
    means = []
    reject_rates = []
    for n in (250, 500, 1000):
        rng = random.Random(1000 + n)
        values = [
            conditional_coverage(markov_hits(n, p01, p11, rng=rng), p).p_value_exact
            for _ in range(400)
        ]
        means.append(statistics.fmean(values))
        reject_rates.append(sum(v <= ALPHA for v in values) / len(values))
    assert means[0] > means[1] > means[2]
    assert reject_rates[0] < reject_rates[1] < reject_rates[2]


def test_bernoulli_generator_has_the_requested_rate_and_no_clustering():
    n, p = 400_000, 0.05
    hits = bernoulli_hits(n, p, seed=3)
    assert len(hits) == n
    rate = sum(hits) / n
    assert abs(rate - p) < 4 * math.sqrt(p * (1 - p) / n)
    t = count_transitions(hits)
    p11_hat = t.n11 / (t.n10 + t.n11)
    assert abs(p11_hat - p) < 4 * math.sqrt(p * (1 - p) / (t.n10 + t.n11))


def test_markov_generator_has_requested_transition_probabilities():
    n, p, p11 = 400_000, 0.02, 0.3
    hits = markov_hits(n, clustered_p01(p, p11), p11, seed=11)
    t = count_transitions(hits)
    ones = t.n10 + t.n11
    assert abs(t.n11 / ones - p11) < 4 * math.sqrt(p11 * (1 - p11) / ones)
    assert abs(sum(hits) / n - p) < 0.003


def test_generators_are_deterministic_per_seed_and_validate_arguments():
    assert bernoulli_hits(500, 0.1, seed=5) == bernoulli_hits(500, 0.1, seed=5)
    assert bernoulli_hits(500, 0.1, seed=5) != bernoulli_hits(500, 0.1, seed=6)
    assert markov_hits(10, 0.0, 0.0, seed=1) == [0] * 10
    assert markov_hits(10, 1.0, 1.0, seed=1) == [1] * 10
    with pytest.raises(ValueError):
        markov_hits(0, 0.1, 0.1)
    with pytest.raises(ValueError):
        markov_hits(10, 1.2, 0.1)
    with pytest.raises(ValueError):
        clustered_p01(0.6, 0.0)
