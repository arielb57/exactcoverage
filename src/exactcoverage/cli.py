"""Command-line interface: ``exactcoverage hits.csv --p 0.01`` and friends."""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from typing import List, Optional, Sequence, TextIO

from .backtests import conditional_coverage, independence, pof, size_report, traffic_light
from .generate import bernoulli_hits, clustered_p01, markov_hits
from .runs import count_transitions

SUBCOMMANDS = ("report", "size", "generate", "bench")
BENCH_NS = (250, 500, 1000)
BENCH_PS = (0.01, 0.05)
TRUE_TOKENS = {"1", "1.0", "true"}
FALSE_TOKENS = {"0", "0.0", "false"}
HEADER_HINTS = ("hit", "hits", "exception", "exceptions", "breach", "violation")


class CliError(Exception):
    pass


def _parse_bit(cell: str) -> Optional[int]:
    token = cell.strip().lower()
    if token in TRUE_TOKENS:
        return 1
    if token in FALSE_TOKENS:
        return 0
    return None


def read_hits(stream: TextIO, column: Optional[str] = None) -> List[int]:
    """Read a 0/1 hit column from CSV text, with or without a header row."""
    rows = [row for row in csv.reader(stream) if any(cell.strip() for cell in row)]
    if not rows:
        raise CliError("input contains no rows")
    header: Optional[List[str]] = None
    if all(_parse_bit(cell) is None for cell in rows[0]):
        header = [cell.strip() for cell in rows[0]]
        rows = rows[1:]
        if not rows:
            raise CliError("input has a header but no data rows")

    width = len(rows[0])
    if column is not None:
        if header is not None and column in header:
            index = header.index(column)
        elif column.isdigit() and int(column) < width:
            index = int(column)
        else:
            raise CliError(f"column {column!r} not found")
    elif width == 1:
        index = 0
    elif header is not None and any(h.lower() in HEADER_HINTS for h in header):
        index = next(i for i, h in enumerate(header) if h.lower() in HEADER_HINTS)
    else:
        raise CliError("input has several columns; choose one with --column")

    hits = []
    offset = 2 if header is not None else 1
    for line, row in enumerate(rows, start=offset):
        if index >= len(row):
            raise CliError(f"row {line} has no column {index}")
        bit = _parse_bit(row[index])
        if bit is None:
            raise CliError(f"row {line}: expected 0/1, got {row[index]!r}")
        hits.append(bit)
    return hits


def _fmt_p(value: float) -> str:
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.2e}"


def render_report(hits: Sequence[int], p: float, alpha: float, source: str) -> str:
    n, x = len(hits), sum(hits)
    t = count_transitions(hits)
    results = [
        ("POF (Kupiec)", pof(hits, p)),
        ("independence (Christoffersen)", independence(hits)),
        ("conditional coverage", conditional_coverage(hits, p)),
    ]
    light = traffic_light(hits, p)
    out = io.StringIO()
    out.write(f"exactcoverage report: {source}\n")
    out.write(f"observations  {n}\n")
    out.write(f"exceptions    {x}  (expected {n * p:.2f} at p = {p:g})\n")
    out.write(f"transitions   n00={t.n00} n01={t.n01} n10={t.n10} n11={t.n11}\n\n")
    pct = f"{alpha:.0%}" if abs(alpha * 100 - round(alpha * 100)) < 1e-9 else f"{alpha:g}"
    out.write(
        f"{'test':<31}{'statistic':>10}{'exact p':>11}{'asympt. p':>11}"
        f"{'exact @' + pct:>13}{'asympt. @' + pct:>14}\n"
    )
    for label, res in results:
        out.write(
            f"{label:<31}{res.statistic:>10.4f}{_fmt_p(res.p_value_exact):>11}"
            f"{_fmt_p(res.p_value_asymptotic):>11}"
            f"{'reject' if res.reject(alpha) else 'accept':>13}"
            f"{'reject' if res.reject(alpha, exact=False) else 'accept':>14}\n"
        )
    out.write("\n")
    out.write(
        f"traffic light  {light.zone} zone: {x} exceptions, P(X <= {x}) = "
        f"{light.cumulative_probability:.4%}\n"
    )
    out.write(
        f"               yellow from {light.yellow_from}, red from {light.red_from}; "
        f"P(X >= {x}) exact {_fmt_p(light.p_value_exact)}, "
        f"normal approx. {_fmt_p(light.p_value_asymptotic)}\n"
    )
    if light.plus_factor is not None:
        out.write(f"               Basel plus factor {light.plus_factor:.2f}\n")
    return out.getvalue()


def render_size(rows) -> str:
    out = io.StringIO()
    alpha = rows[0].alpha
    out.write(
        f"true size of nominal {alpha:g} tests, n = {rows[0].n}, p = {rows[0].p:g} "
        f"(exact computation, no simulation)\n\n"
    )
    out.write(f"{'test':<24}{'asymptotic test':>17}{'exact test':>12}\n")
    for row in rows:
        out.write(f"{row.test:<24}{row.asymptotic_size:>17.4f}{row.exact_size:>12.4f}\n")
    return out.getvalue()


def bench_rows(alpha: float = 0.05):
    rows = []
    for n in BENCH_NS:
        for p in BENCH_PS:
            rows.extend(size_report(n, p, alpha))
    return rows


def write_size_csv(rows, stream: TextIO) -> None:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["n", "p", "alpha", "test", "asymptotic_size", "exact_size"])
    for row in rows:
        writer.writerow(
            [
                row.n,
                f"{row.p:g}",
                f"{row.alpha:g}",
                row.test,
                f"{row.asymptotic_size:.10f}",
                f"{row.exact_size:.10f}",
            ]
        )


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _open_rate(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must lie strictly between 0 and 1")
    return value


def _build_parsers():
    report = argparse.ArgumentParser(
        prog="exactcoverage",
        description="Exact VaR backtests. Subcommands: size, generate, bench "
        "(run 'exactcoverage SUBCOMMAND --help').",
    )
    report.add_argument("file", help="CSV of daily 0/1 exception indicators ('-' for stdin)")
    report.add_argument(
        "--p", type=_open_rate, default=0.01, help="expected exception rate, e.g. 0.01 for 99%% VaR"
    )
    report.add_argument("--alpha", type=_open_rate, default=0.05)
    report.add_argument("--column", help="column name or 0-based index")

    size = argparse.ArgumentParser(
        prog="exactcoverage size", description="Exact size of asymptotic and exact tests."
    )
    size.add_argument("--n", type=_positive_int, required=True)
    size.add_argument("--p", type=_open_rate, required=True)
    size.add_argument("--alpha", type=_open_rate, default=0.05)

    gen = argparse.ArgumentParser(
        prog="exactcoverage generate", description="Write a synthetic hit sequence as CSV."
    )
    gen.add_argument("--n", type=_positive_int, required=True)
    gen.add_argument("--p", type=_open_rate, required=True, help="unconditional exception rate")
    gen.add_argument(
        "--p11",
        type=float,
        default=None,
        help="P(exception | exception yesterday); omit for i.i.d. hits",
    )
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--out", default="-")

    bench = argparse.ArgumentParser(
        prog="exactcoverage bench", description="Size table for n in 250/500/1000, p in 0.01/0.05."
    )
    bench.add_argument("--alpha", type=_open_rate, default=0.05)
    bench.add_argument("--out", default="benchmarks/size_table.csv")
    return report, size, gen, bench


def _write_file(path: str, text: str) -> None:
    try:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            fh.write(text)
    except OSError as exc:
        raise CliError(f"cannot write {path}: {exc.strerror}") from exc


def main(
    argv: Optional[Sequence[str]] = None,
    stdout: Optional[TextIO] = None,
    stdin: Optional[TextIO] = None,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stdout = stdout or sys.stdout
    stdin = stdin or sys.stdin
    report, size, gen, bench = _build_parsers()
    command = argv[0] if argv and argv[0] in SUBCOMMANDS else "report"
    if argv and argv[0] in SUBCOMMANDS:
        argv = argv[1:]
    try:
        if command == "report":
            args = report.parse_args(argv)
            if args.file == "-":
                hits = read_hits(stdin, args.column)
            else:
                try:
                    with open(args.file, newline="") as fh:
                        hits = read_hits(fh, args.column)
                except OSError as exc:
                    raise CliError(f"cannot read {args.file}: {exc.strerror}") from exc
            stdout.write(render_report(hits, args.p, args.alpha, args.file))
        elif command == "size":
            args = size.parse_args(argv)
            stdout.write(render_size(size_report(args.n, args.p, args.alpha)))
        elif command == "generate":
            args = gen.parse_args(argv)
            if args.p11 is None:
                hits = bernoulli_hits(args.n, args.p, seed=args.seed)
            else:
                hits = markov_hits(
                    args.n, clustered_p01(args.p, args.p11), args.p11, seed=args.seed
                )
            text = "hit\n" + "".join(f"{h}\n" for h in hits)
            if args.out == "-":
                stdout.write(text)
            else:
                _write_file(args.out, text)
        else:
            args = bench.parse_args(argv)
            buf = io.StringIO()
            write_size_csv(bench_rows(args.alpha), buf)
            _write_file(args.out, buf.getvalue())
            stdout.write(buf.getvalue())
            stdout.write(f"\nwrote {args.out}\n")
    except (CliError, ValueError) as exc:
        sys.stderr.write(f"exactcoverage: error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
