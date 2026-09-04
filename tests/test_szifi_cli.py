"""CLI argument parsing for flamingo-szifi (no MMF I/O)."""

from flamingo_mock.szifi.cli import parse_args


def test_parse_benchmark():
    args = parse_args(["benchmark", "--q-th-obs", "6"])
    assert args.command == "benchmark"
    assert args.q_th_obs == 6.0


def test_parse_true_snr():
    args = parse_args(["true-snr", "--split", "B"])
    assert args.command == "true-snr"
    assert args.split == "B"


def test_parse_run_still_works():
    args = parse_args(["run", "--footprint", "--method", "immf"])
    assert args.command == "run"
    assert args.method == "immf"
    assert args.kind == "npipe"
