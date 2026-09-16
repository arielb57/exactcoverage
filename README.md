# exactcoverage

VaR backtests (Kupiec POF, Christoffersen independence and conditional coverage, Basel traffic light) with exact finite-sample p-values instead of chi-square approximations.

## The problem

Banks and risk vendors backtest a 99% VaR on 250 trading days with Kupiec's proportion-of-failures (POF) test and Christoffersen's independence and conditional-coverage tests, and read the p-values off a chi-square distribution. With about 2.5 expected exceptions that approximation does not hold: the "5%" POF test actually rejects a correct model 9.5% of the time, and a desk with **zero** exceptions in a year gets rejected (chi-square p = 0.025). Christoffersen's statistic also contains log(π₁₁), which is log 0 whenever no two exceptions fall on consecutive days, which is the usual case. A direct implementation crashes or returns NaN, and a silent patch leaves you unsure what the p-value means.

exactcoverage computes the exact null distribution of each statistic by combinatorics, with no simulation. Every test returns the statistic, the exact p-value and the asymptotic p-value, so you can see where the two disagree.

## How it works

**Run signatures.** Under the null hypothesis the hit sequence I₁…Iₙ is i.i.d. Bernoulli(p). Summarise a sequence by its *run signature* `(x, r, first, last)`: the number of exceptions, the number of maximal runs of exceptions, and the first and last bits. Two facts make this enough:

1. **The transition counts follow from the signature.** Every run of ones is entered from a zero unless it starts on day 1, and left to a zero unless it ends on day n:

   ```
   n01 = r - first        n10 = r - last
   n11 = x - r            n00 = (n - x) - s,   where s = r + 1 - first - last zero runs
   ```

2. **The number of sequences with a given signature has a closed form.** The ones make up a composition of x into r positive parts and the zeros a composition of n−x into s parts:

   ```
   count(n, x, r, first, last) = C(x-1, r-1) · C(n-x-1, s-1)
   ```

   Every such sequence has probability pˣ(1−p)ⁿ⁻ˣ.

Worked example, n = 5, sequence `0 1 1 0 1`: x = 3, r = 2, first = 0, last = 1, so s = 2 zero runs and count = C(2,1)·C(1,1) = 2 (the sequences `01101` and `01011`). Both have n01 = 2, n11 = 1, n10 = 1, n00 = 0.

There are O(n²) signatures (x ≤ n, r ≤ min(x, n−x+1), four end-bit combinations), against 2ⁿ sequences. For n = 1000 that means about 10⁶ terms, each evaluated in log space from a table of log-factorials so the tails neither overflow nor underflow early.

**Exact p-values.** For each test, every signature is mapped to its LR statistic and probability, the outcomes are sorted, and the p-value is the total probability of outcomes whose statistic is **at least** the observed one:

- **POF**: LR over x ~ Binomial(n, p). The test is two-sided, so too few exceptions count as extreme too.
- **Independence**: the exact p-value is **conditional on x**. Given x, every placement of the exceptions is equally likely whatever the true i.i.d. rate is, so the null distribution is `count / C(n, x)` and has no nuisance parameter.
- **Conditional coverage**: LR_cc = LR_pof + LR_ind, with its joint distribution over all signatures at rate p.
- **Traffic light**: zones come from the exact binomial CDF. Green runs until P(X ≤ k) reaches 95%, red starts where it reaches 99.99%. The exact p-value is P(X ≥ x).

The log(0) problem is handled at its source. The maximised likelihood uses 0·log 0 = 0, which is the limit and the true supremum, so the statistic is always finite. Empty conditioning cells (for example, an exception only on day n, so no transition out of an exception is ever observed) add nothing to either likelihood. Because the exact distribution applies the same convention to every outcome, the p-value stays correct whatever the convention does to the statistic's scale.

**Ties.** Different transition counts can give mathematically equal statistics that differ in the 15th digit. Outcomes within a relative 1e-9 are merged and counted as "at least as extreme", which keeps the p-value conservative.

**True size.** `size` sums the probability of every outcome whose chi-square p-value is ≤ α. That gives the real rejection rate of the textbook test, computed exactly. It also reports the size of the exact test, which discreteness keeps at or below α.

## Install and usage

Requires Python 3.9 or newer. There are no runtime dependencies.

From a clone of this repository:

```
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
pytest                      # 146 tests, about 12 s
```

The input is a CSV of daily 0/1 exception indicators. A header is optional, and with several columns the one named `hit`, `hits`, `exception` and so on is used, or you pick one with `--column`. `-` reads stdin.

A desk with no exceptions in a year (`examples/zero_exceptions.csv`):

```
$ exactcoverage examples/zero_exceptions.csv --p 0.01
exactcoverage report: examples/zero_exceptions.csv
observations  250
exceptions    0  (expected 2.50 at p = 0.01)
transitions   n00=249 n01=0 n10=0 n11=0

test                            statistic    exact p  asympt. p    exact @5%   asympt. @5%
POF (Kupiec)                       5.0252     0.0948     0.0250       accept        reject
independence (Christoffersen)      0.0000     1.0000     1.0000       accept        accept
conditional coverage               5.0252     0.1106     0.0811       accept        accept

traffic light  green zone: 0 exceptions, P(X <= 0) = 8.1059%
               yellow from 5, red from 10; P(X >= 0) exact 1.0000, normal approx. 0.9440
               Basel plus factor 0.00
```

Three exceptions, two of them on consecutive days (`examples/adjacent_pair.csv`). Here the chi-square conditional-coverage test misses clustering that the exact test catches:

```
$ exactcoverage examples/adjacent_pair.csv --p 0.01
exactcoverage report: examples/adjacent_pair.csv
observations  250
exceptions    3  (expected 2.50 at p = 0.01)
transitions   n00=244 n01=2 n10=2 n11=1

test                            statistic    exact p  asympt. p    exact @5%   asympt. @5%
POF (Kupiec)                       0.0949     1.0000     0.7580       accept        accept
independence (Christoffersen)      5.4252     0.0239     0.0198       reject        reject
conditional coverage               5.5202     0.0246     0.0633       reject        accept

traffic light  green zone: 3 exceptions, P(X <= 3) = 75.8117%
               yellow from 5, red from 10; P(X >= 3) exact 0.4568, normal approx. 0.3753
               Basel plus factor 0.00
```

(The exact POF p-value is 1 because x = 3 has the smallest LR of any count: it is the value closest to the 2.5 you would expect.)

True size of the textbook tests:

```
$ exactcoverage size --n 250 --p 0.01
true size of nominal 0.05 tests, n = 250, p = 0.01 (exact computation, no simulation)

test                      asymptotic test  exact test
pof                                0.0948      0.0137
independence                       0.0140      0.0144
conditional_coverage               0.0082      0.0295
```

Synthetic data, for i.i.d. nulls or clustered Markov alternatives with P(exception | exception yesterday) = `--p11`, keeping the unconditional rate at `--p`. The command writes a `hit` column; leave out `--out` to print it to stdout.

```
$ exactcoverage generate --n 1000 --p 0.01 --p11 0.3 --seed 7 --out clustered.csv
$ exactcoverage clustered.csv --p 0.01
```

As a library:

```python
from exactcoverage import pof, independence, conditional_coverage, traffic_light, size_report
from exactcoverage import markov_hits, clustered_p01

hits = markov_hits(500, clustered_p01(0.01, 0.3), 0.3, seed=1)
r = conditional_coverage(hits, p=0.01)
print(r.statistic, r.p_value_exact, r.p_value_asymptotic, r.reject(0.05))
# 5.679078821012652 0.027428178737923612 0.05845258241064528 True
print(traffic_light(hits[:250]).zone)
# green
```

## Results

The size table for nominal α = 5%, where every number is an exact probability rather than a simulation estimate. `asymptotic` is how often the chi-square test rejects a correct model. `exact` is the same for the exact test.

| n | p | POF asymptotic | POF exact | independence asymptotic | independence exact | CC asymptotic | CC exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.01 | **0.0948** | 0.0137 | 0.0140 | 0.0144 | 0.0082 | 0.0295 |
| 250 | 0.05 | 0.0585 | 0.0462 | 0.0167 | 0.0283 | 0.0402 | 0.0495 |
| 500 | 0.01 | **0.0709** | 0.0198 | 0.0148 | 0.0156 | 0.0168 | 0.0304 |
| 500 | 0.05 | 0.0539 | 0.0395 | 0.0330 | 0.0251 | 0.0397 | 0.0494 |
| 1000 | 0.01 | 0.0551 | 0.0425 | 0.0179 | 0.0136 | 0.0265 | 0.0379 |
| 1000 | 0.05 | 0.0514 | 0.0419 | **0.0824** | 0.0230 | 0.0552 | 0.0492 |

What the table shows:

- **POF at the regulatory setting** (n = 250, p = 1%) rejects correct models almost twice as often as advertised.
- **The independence test's chi-square size moves around** as n and p change: 1.4% at n = 250, p = 1%, and 8.2% at n = 1000, p = 5%. You cannot correct it with a fixed adjustment.
- **The exact tests never exceed 5%.** They fall below it because the statistics are discrete. With about 2.5 expected exceptions, some sizes simply cannot be reached.

To reproduce, run `exactcoverage bench`. It rewrites `benchmarks/size_table.csv` and prints it. The run takes 7.4 s wall-clock on one core of an Apple Silicon Mac with Python 3.13, and n = 1000 at p = 1% is the slowest cell at about 3.5 s. `tests/test_cli_and_benchmark.py` recomputes the table and fails if any value drifts by more than 1e-9.

How correctness is checked (all in `tests/`):

1. **Closed form against brute force.** For every n from 1 to 16, the closed-form signature counts equal a brute-force enumeration of all 2ⁿ sequences. For p ∈ {1/100, 1/20, 2/7, 1/2}, the distributions agree exactly as `Fraction`s.
2. **Normalisation.** The log-space distribution sums to 1 within 1e-12 for n = 250, 500 and 1000, at p = 0.01, 0.05 and 0.5.
3. **p-values against enumeration.** Exact p-values for all three tests match full enumeration of every sequence for n = 9, 12 and 13. The independence p-value also matches enumeration of all 34,220 placements of 3 exceptions in 60 days, and of 1 exception in 250 days.
4. **Basel table.** For n = 250, p = 0.01, the cumulative probabilities round to the 1996 Basel table (8.11, 28.58, 54.32, 75.81, 89.22, 95.88, 98.63, 99.60, 99.89, 99.97, 99.99%). The zones are green 0–4, yellow 5–9 and red 10+, with the published plus factors.
5. **Validity by simulation.** A seeded Monte Carlo draws 200,000 null sequences (n = 250, p = 0.01) plus 50,000 at p = 0.05. No exact test rejects more than α beyond a 99.9% binomial margin. Each of the six simulated rejection rates (exact and asymptotic, three tests) matches the computed size within that margin, which independently checks the size table.
6. **Edge cases.** Tests cover zero exceptions, all exceptions, n11 = 0, a single exception on day 1, on day n or in the middle, n = 1, and invalid input.
7. **Power.** Against a Markov alternative with P(1|1) = 0.3 at a 1% rate, the mean conditional-coverage p-value and the rejection rate both improve as n goes from 250 to 500 to 1000.

## Design notes

**Enumerate signatures, not sequences, and never simulate.** A Monte Carlo p-value would have been quicker to write, but it has its own error, which is worst in the far tail where decisions are made, and two runs disagree. Counting signatures makes each p-value a deterministic sum with a proof behind it: the closed form is checked against 2ⁿ enumeration in exact rationals. It costs O(n²) work per (n, p). That is why null distributions are cached, so backtesting many desks with the same horizon pays once. Probabilities are computed in log space as floats, not `Fraction`s: rationals with denominators like 100¹⁰⁰⁰ are far too slow for n = 1000. The float version is checked against the rational one for small n.

**Condition the independence test on the number of exceptions.** Christoffersen's independence hypothesis does not fix the exception rate, so an unconditional "exact" test would need a rate plugged in, and its size would depend on that choice. Conditioning on x removes the nuisance parameter: given x, all C(n, x) placements are equally likely under any i.i.d. rate. This gives a test that is valid for every rate at once, and it is the reason `independence()` takes no `p`. The trade-off is some power, because the conditional test ignores information in x. The conditional-coverage test still uses p, as its hypothesis requires. The reported sizes are unconditional (averaged over x at the given p), so they stay comparable with the chi-square column.

## Limitations

- **Input.** The tool takes a 0/1 exception series. It does not compute VaR or compare P&L with VaR for you.
- **Test coverage.** It covers the first-order Markov alternative, as in Christoffersen (1998). There are no duration-based tests (Christoffersen–Pelletier), no multi-level or expected-shortfall backtests, and nothing for exceptions that depend on something other than yesterday.
- **LR_cc definition.** LR_cc is defined as LR_pof (on all n days) + LR_ind (on n−1 transitions), the common textbook form. Christoffersen's original likelihood conditions on the first observation, so results can differ slightly from software that uses that form.
- **Discreteness.** Exact tests are conservative: their true size is often well below α (1.4% for POF at n = 250, p = 1%). No randomised or mid-p variant is offered.
- **Speed.** It is pure Python. The first call for a new (n, p) takes about 0.2 s at n = 250 and about 3.5 s at n = 1000; later calls reuse the cached distribution. Beyond a few thousand observations, the O(n²) cost becomes noticeable.
- **Plus factors.** Basel plus factors are reported only for the 250-day, 99% setting they were defined for. Zones and cumulative probabilities work for any n and p.

## License

MIT. See [LICENSE](LICENSE).
