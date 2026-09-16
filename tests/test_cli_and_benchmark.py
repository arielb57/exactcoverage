import csv
import io
from pathlib import Path

import pytest

from exactcoverage import bernoulli_hits, conditional_coverage, pof
from exactcoverage.cli import BENCH_NS, BENCH_PS, bench_rows, main, read_hits

ROOT = Path(__file__).resolve().parents[1]


def run(argv, stdin_text=None):
    out = io.StringIO()
    code = main(argv, stdout=out, stdin=io.StringIO(stdin_text or ""))
    return code, out.getvalue()


def test_size_table_csv_is_reproduced_exactly():
    with open(ROOT / "benchmarks" / "size_table.csv", newline="") as fh:
        committed = list(csv.DictReader(fh))
    rows = bench_rows(0.05)
    assert len(committed) == len(rows) == len(BENCH_NS) * len(BENCH_PS) * 3
    for saved, row in zip(committed, rows):
        assert (int(saved["n"]), float(saved["p"]), saved["test"]) == (row.n, row.p, row.test)
        assert float(saved["asymptotic_size"]) == pytest.approx(row.asymptotic_size, abs=1e-9)
        assert float(saved["exact_size"]) == pytest.approx(row.exact_size, abs=1e-9)
        assert row.exact_size <= 0.05


def test_generate_then_report_round_trip(tmp_path):
    path = tmp_path / "hits.csv"
    code, _ = run(
        ["generate", "--n", "250", "--p", "0.01", "--p11", "0.3", "--seed", "4", "--out", str(path)]
    )
    assert code == 0
    with open(path) as fh:
        hits = read_hits(fh)
    assert len(hits) == 250
    code, text = run([str(path), "--p", "0.01"])
    assert code == 0
    expected = conditional_coverage(hits, 0.01)
    assert f"exceptions    {sum(hits)}" in text
    line = next(ln for ln in text.splitlines() if ln.startswith("conditional coverage"))
    assert f"{expected.statistic:.4f}" in line
    assert "traffic light" in text


def test_report_reads_stdin_and_prints_basel_plus_factor():
    hits = [0] * 250
    for d in (3, 4, 50, 90, 91, 200, 230):
        hits[d] = 1
    code, text = run(["-", "--p", "0.01"], "hit\n" + "\n".join(map(str, hits)) + "\n")
    assert code == 0
    assert "yellow zone: 7 exceptions" in text
    assert "Basel plus factor 0.65" in text
    assert "n00=237 n01=5 n10=5 n11=2" in text


def test_size_subcommand_prints_exact_sizes():
    code, text = run(["size", "--n", "250", "--p", "0.01"])
    assert code == 0
    pof_line = next(ln for ln in text.splitlines() if ln.startswith("pof"))
    assert pof_line.split()[1:] == ["0.0948", "0.0137"]


def test_bench_subcommand_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setattr("exactcoverage.cli.BENCH_NS", (40,))
    monkeypatch.setattr("exactcoverage.cli.BENCH_PS", (0.1,))
    out = tmp_path / "nested" / "size.csv"
    code, text = run(["bench", "--out", str(out)])
    assert code == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["test"] for r in rows] == ["pof", "independence", "conditional_coverage"]
    assert "wrote" in text


def test_read_hits_formats():
    assert read_hits(io.StringIO("0\n1\n\n1\n")) == [0, 1, 1]
    assert read_hits(io.StringIO("date,hit\n2024-01-02,0\n2024-01-03,true\n")) == [0, 1]
    assert read_hits(io.StringIO("a,b\n1,0\n0,0\n"), column="b") == [0, 0]
    assert read_hits(io.StringIO("1,0\n0,1\n"), column="1") == [0, 1]


@pytest.mark.parametrize(
    "text,column,message",
    [
        ("", None, "no rows"),
        ("hit\n", None, "no data rows"),
        ("a,b\n1,0\n", None, "--column"),
        ("hit\n0\n2\n", None, "row 3"),
        ("a,b\n1,0\n", "c", "not found"),
    ],
)
def test_bad_input_exits_with_error(text, column, message, capsys):
    argv = ["-"] + (["--column", column] if column else [])
    code, _ = run(argv, text)
    assert code == 2
    assert message in capsys.readouterr().err


def test_missing_file_is_reported(tmp_path, capsys):
    code, _ = run([str(tmp_path / "absent.csv")])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_unwritable_output_is_reported(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    code, _ = run(["generate", "--n", "10", "--p", "0.1", "--out", str(blocker / "hits.csv")])
    assert code == 2
    assert "cannot write" in capsys.readouterr().err


def test_generate_rejects_impossible_clustering(capsys):
    code, _ = run(["generate", "--n", "10", "--p", "0.6", "--p11", "0.0"])
    assert code == 2
    assert "no Markov chain" in capsys.readouterr().err


def test_headerless_stdin_report_matches_library():
    hits = bernoulli_hits(1000, 0.05, seed=12)
    code, text = run(["-", "--p", "0.05"], "\n".join(map(str, hits)))
    assert code == 0
    line = next(ln for ln in text.splitlines() if ln.startswith("POF (Kupiec)"))
    assert f"{pof(hits, 0.05).statistic:.4f}" in line
    assert "Basel plus factor" not in text
