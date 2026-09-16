import itertools
import math
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from exactcoverage.runs import (
    Transitions,
    brute_force_counts,
    brute_force_distribution,
    count_transitions,
    exact_distribution,
    log_distribution,
    run_count,
    signature_of,
    signatures,
    transitions,
)

PROBABILITIES = [Fraction(1, 100), Fraction(1, 20), Fraction(2, 7), Fraction(1, 2)]


@pytest.mark.parametrize("n", range(1, 17))
def test_closed_form_counts_match_enumeration(n):
    closed = {row[:4]: row[4] for row in signatures(n)}
    assert closed == brute_force_counts(n)
    assert sum(closed.values()) == 2**n


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 11, 14, 16])
@pytest.mark.parametrize("p", PROBABILITIES)
def test_closed_form_distribution_matches_enumeration_in_fractions(n, p):
    closed = exact_distribution(n, p)
    brute = brute_force_distribution(n, p)
    assert closed == brute
    assert sum(closed.values()) == 1


@pytest.mark.parametrize("n", range(1, 17))
def test_every_n_up_to_16_matches_enumeration_at_one_percent(n):
    assert exact_distribution(n, Fraction(1, 100)) == brute_force_distribution(n, Fraction(1, 100))


@pytest.mark.parametrize("n", [1, 2, 7, 12])
@pytest.mark.parametrize("p", [0.01, 0.3])
def test_log_space_distribution_matches_exact_fractions(n, p):
    exact = exact_distribution(n, Fraction(p))
    approx = {row[:4]: row[4] for row in log_distribution(n, p)}
    assert set(approx) == set(exact)
    for key, prob in approx.items():
        assert prob == pytest.approx(float(exact[key]), rel=1e-11, abs=1e-300)


@pytest.mark.parametrize("n", [250, 500, 1000])
@pytest.mark.parametrize("p", [0.01, 0.05, 0.5])
def test_distribution_sums_to_one_for_large_n(n, p):
    total = math.fsum(prob for *_, prob in log_distribution(n, p))
    assert abs(total - 1.0) < 1e-12


@pytest.mark.parametrize("n", range(1, 13))
def test_transitions_from_signature_match_every_sequence(n):
    for seq in itertools.product((0, 1), repeat=n):
        assert transitions(n, *signature_of(seq)) == count_transitions(seq)


@settings(max_examples=300, deadline=2000)
@given(st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=400))
def test_transitions_property_on_long_sequences(seq):
    n = len(seq)
    t = transitions(n, *signature_of(seq))
    assert t == count_transitions(seq)
    assert sum(t) == n - 1
    assert min(t) >= 0


def test_signature_examples():
    assert signature_of([0, 1, 1, 0, 1]) == (3, 2, 0, 1)
    assert transitions(5, 3, 2, 0, 1) == Transitions(n00=0, n01=2, n10=1, n11=1)
    assert signature_of([0, 0, 0]) == (0, 0, 0, 0)
    assert signature_of([1]) == (1, 1, 1, 1)
    assert transitions(1, 1, 1, 1, 1) == Transitions(0, 0, 0, 0)


def test_run_count_rejects_impossible_signatures():
    assert run_count(5, 0, 1, 0, 0) == 0
    assert run_count(5, 5, 2, 1, 1) == 0
    assert run_count(5, 2, 3, 0, 0) == 0
    assert run_count(3, 2, 2, 0, 0) == 0  # two separated ones need three zero runs
    assert run_count(5, 6, 1, 1, 1) == 0


def test_run_count_small_cases_by_hand():
    assert run_count(4, 2, 2, 1, 1) == 1  # 1001
    assert run_count(5, 2, 2, 0, 0) == 1  # 01010
    assert run_count(6, 2, 2, 0, 0) == 3  # 010100, 010010, 001010
    assert run_count(5, 2, 1, 0, 1) == 1  # 00011
    assert run_count(6, 3, 2, 0, 1) == 4  # ones split 1+2 or 2+1, zeros split 1+2 or 2+1
    assert run_count(5, -1, 0, 0, 0) == 0


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        run_count(0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        run_count(3, 1, 1, 2, 0)
    with pytest.raises(ValueError):
        signature_of([])
    with pytest.raises(ValueError):
        signature_of([0, 2, 1])
    with pytest.raises(ValueError):
        list(log_distribution(10, 0.0))
    with pytest.raises(ValueError):
        exact_distribution(4, Fraction(3, 2))
