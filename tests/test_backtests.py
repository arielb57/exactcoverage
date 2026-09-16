import itertools
import math

import pytest

from exactcoverage import (
    conditional_coverage,
    independence,
    pof,
    traffic_light,
    traffic_light_zones,
)
from exactcoverage.runs import count_transitions
from exactcoverage.stats import TIE_TOLERANCE, chi2_sf, lr_ind, lr_pof


def _at_least(values_probs, observed):
    tol = TIE_TOLERANCE * max(1.0, observed)
    return math.fsum(prob for value, prob in values_probs if value >= observed - tol)


def _enumerate(n, p):
    rows = []
    for seq in itertools.product((0, 1), repeat=n):
        x = sum(seq)
        prob = p**x * (1 - p) ** (n - x)
        rows.append((seq, x, prob, lr_pof(n, x, p), lr_ind(*count_transitions(seq))))
    return rows


@pytest.mark.parametrize("n,p", [(9, 0.1), (12, 0.25), (13, 0.02)])
def test_exact_p_values_match_brute_force_enumeration(n, p):
    rows = _enumerate(n, p)
    pof_pairs = [(lp, prob) for _, _, prob, lp, _ in rows]
    cc_pairs = [(lp + li, prob) for _, _, prob, lp, li in rows]
    checked = set()
    for seq, x, _, lp, li in rows:
        key = (x, count_transitions(seq))
        if key in checked:
            continue
        checked.add(key)
        assert pof(seq, p).p_value_exact == pytest.approx(_at_least(pof_pairs, lp), rel=1e-9)
        assert conditional_coverage(seq, p).p_value_exact == pytest.approx(
            _at_least(cc_pairs, lp + li), rel=1e-9
        )
        same_x = [(r[4], 1.0) for r in rows if r[1] == x]
        expected_ind = _at_least(same_x, li) / len(same_x)
        assert independence(seq).p_value_exact == pytest.approx(expected_ind, rel=1e-9)


def test_asymptotic_p_values_use_chi_square():
    hits = [0] * 240 + [1, 1, 0, 1, 0, 0, 1, 0, 0, 1]
    r = pof(hits, 0.01)
    assert r.p_value_asymptotic == pytest.approx(chi2_sf(r.statistic, 1))
    c = conditional_coverage(hits, 0.01)
    assert c.p_value_asymptotic == pytest.approx(math.exp(-c.statistic / 2))
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-12)
    assert chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-12)


def test_pof_statistic_matches_textbook_formula():
    n, x, p = 250, 7, 0.01
    phat = x / n
    expected = -2 * (
        (n - x) * math.log(1 - p)
        + x * math.log(p)
        - (n - x) * math.log(1 - phat)
        - x * math.log(phat)
    )
    hits = [1] * x + [0] * (n - x)
    assert pof(hits, p).statistic == pytest.approx(expected, rel=1e-12)
    assert pof(hits, p).reject(0.05)


def test_independence_statistic_matches_textbook_formula_when_all_cells_positive():
    hits = [0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    n00, n01, n10, n11 = count_transitions(hits)
    pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    ll_alt = (
        n00 * math.log(1 - pi01)
        + n01 * math.log(pi01)
        + n10 * math.log(1 - pi11)
        + n11 * math.log(pi11)
    )
    assert independence(hits).statistic == pytest.approx(-2 * (ll_null - ll_alt), rel=1e-12)


def test_zero_exceptions():
    hits = [0] * 250
    r = pof(hits, 0.01)
    assert r.exceptions == 0
    assert r.statistic == pytest.approx(-2 * 250 * math.log(0.99))
    # The test is two-sided: LR at x = 0 lies between LR at x = 6 and x = 7,
    # so the exact tail is P(X = 0) + P(X >= 7).
    pmf = [math.comb(250, k) * 0.01**k * 0.99 ** (250 - k) for k in range(251)]
    assert lr_pof(250, 6, 0.01) < r.statistic < lr_pof(250, 7, 0.01)
    assert r.p_value_exact == pytest.approx(pmf[0] + math.fsum(pmf[7:]), rel=1e-10)
    ind = independence(hits)
    assert ind.statistic == 0.0
    assert ind.p_value_exact == 1.0
    assert ind.p_value_asymptotic == 1.0
    cc = conditional_coverage(hits, 0.01)
    assert cc.statistic == pytest.approx(r.statistic)
    assert 0.0 < cc.p_value_exact <= 1.0
    assert traffic_light(hits).zone == "green"


def test_all_exceptions():
    hits = [1] * 30
    r = pof(hits, 0.01)
    assert math.isfinite(r.statistic)
    assert r.p_value_exact == pytest.approx(0.01**30, rel=1e-9)
    assert independence(hits).statistic == 0.0
    assert independence(hits).p_value_exact == 1.0
    cc = conditional_coverage(hits, 0.01)
    assert cc.p_value_exact == pytest.approx(0.01**30, rel=1e-9)
    assert traffic_light(hits).zone == "red"


def test_no_consecutive_exceptions_gives_finite_independence_statistic():
    hits = [0] * 250
    for day in (10, 60, 111, 200):
        hits[day] = 1
    assert count_transitions(hits).n11 == 0
    r = independence(hits)
    assert math.isfinite(r.statistic) and r.statistic > 0
    # With 4 isolated exceptions the evidence of clustering is nil: nearly
    # every arrangement is at least as extreme.
    assert r.p_value_exact > 0.9
    cc = conditional_coverage(hits, 0.01)
    assert math.isfinite(cc.statistic)
    assert 0.0 < cc.p_value_exact <= 1.0


def test_adjacent_exceptions_match_enumeration_of_all_placements():
    n = 60
    hits = [0] * n
    hits[20] = hits[21] = hits[22] = 1
    r = independence(hits)
    stats = []
    for days in itertools.combinations(range(n), 3):
        seq = [0] * n
        for d in days:
            seq[d] = 1
        stats.append(lr_ind(*count_transitions(seq)))
    expected = sum(1 for s in stats if s >= r.statistic - 1e-9) / len(stats)
    assert r.p_value_exact == pytest.approx(expected, rel=1e-9)
    assert r.reject(0.01)


def test_two_adjacent_exceptions_are_significant_and_chi_square_misstates_p():
    hits = [0] * 250
    hits[100] = hits[101] = 1
    r = independence(hits)
    assert r.p_value_exact < 0.05
    assert r.p_value_exact != pytest.approx(r.p_value_asymptotic, rel=0.1)


@pytest.mark.parametrize("day", [0, 249])
def test_single_exception_on_first_or_last_day(day):
    n = 250
    hits = [0] * n
    hits[day] = 1
    r = independence(hits)
    assert math.isfinite(r.statistic)
    # Enumerate the n placements of one exception as an oracle.
    placements = []
    for d in range(n):
        seq = [0] * n
        seq[d] = 1
        placements.append(lr_ind(*count_transitions(seq)))
    expected = sum(1 for s in placements if s >= r.statistic - 1e-9) / n
    assert r.p_value_exact == pytest.approx(expected, rel=1e-9)
    assert pof(hits, 0.01).exceptions == 1
    assert traffic_light(hits).zone == "green"


def test_first_and_last_day_differ_from_interior():
    interior = [0] * 250
    interior[125] = 1
    first = [1] + [0] * 249
    last = [0] * 249 + [1]
    # On day 1 no transition into the exception is observed and on day n none out
    # of it, so both fit the i.i.d. model perfectly; an interior exception does not.
    assert independence(first).statistic == 0.0
    assert independence(last).statistic == 0.0
    assert independence(interior).statistic > 0.0
    assert independence(first).p_value_exact == 1.0
    assert independence(last).p_value_exact == 1.0
    assert independence(interior).p_value_exact == pytest.approx(248 / 250, rel=1e-12)


def test_single_observation():
    for hits in ([0], [1]):
        assert independence(hits).p_value_exact == 1.0
        assert 0.0 < pof(hits, 0.3).p_value_exact <= 1.0
        assert 0.0 < conditional_coverage(hits, 0.3).p_value_exact <= 1.0


BASEL_1996 = [8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99]


def test_traffic_light_reproduces_basel_1996_table():
    yellow, red, cdf = traffic_light_zones(250, 0.01)
    assert [round(100 * c, 2) for c in cdf[:11]] == BASEL_1996
    assert (yellow, red) == (5, 10)
    zones = [traffic_light([1] * k + [0] * (250 - k)).zone for k in range(13)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 3
    plus = [traffic_light([1] * k + [0] * (250 - k)).plus_factor for k in range(11)]
    assert plus == [0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85, 1.0]


def test_traffic_light_p_values():
    hits = [1] * 6 + [0] * 244
    r = traffic_light(hits)
    pmf = [math.comb(250, k) * 0.01**k * 0.99 ** (250 - k) for k in range(251)]
    assert r.p_value_exact == pytest.approx(math.fsum(pmf[6:]), rel=1e-10)
    assert r.cumulative_probability == pytest.approx(math.fsum(pmf[:7]), rel=1e-12)
    z = (6 - 2.5) / math.sqrt(2.5 * 0.99)
    assert r.p_value_asymptotic == pytest.approx(0.5 * math.erfc(z / math.sqrt(2)))
    other = traffic_light([1] * 6 + [0] * 494, 0.01)
    assert other.plus_factor is None
    assert other.zone == "green"


@pytest.mark.parametrize(
    "call",
    [
        lambda: pof([0, 1], 0.0),
        lambda: pof([0, 1], 1.0),
        lambda: conditional_coverage([0, 1], -0.2),
        lambda: pof([], 0.01),
        lambda: independence([0, 2, 1]),
        lambda: independence(["1", "0"]),
        lambda: traffic_light([0.5, 1], 0.01),
    ],
)
def test_invalid_input_raises(call):
    with pytest.raises(ValueError):
        call()


def test_booleans_and_float_bits_accepted():
    assert pof([True, False, False], 0.1) == pof([1, 0, 0], 0.1)
    assert independence([1.0, 0.0, 1.0]) == independence([1, 0, 1])
