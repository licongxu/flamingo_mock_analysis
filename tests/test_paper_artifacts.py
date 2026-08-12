"""Regression for the completeness bar-count numerator (TP vs detectable)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_completeness_bar_counts_match_fraction():
    """Drive shipped completeness_bar_counts; numerator must match fraction.

    Regression for the 691/430 (TP/detectable) bug: the label must use
    n_truth_detectable - n_undetected, which equals completeness_detectable.
    """
    sys.path.insert(0, str(SCRIPTS))
    from regenerate_paper_figures import completeness_bar_counts

    # Synthetic case matching the real iMMF JSON structure
    b = {
        "n_true_positives": 691,  # detection-side (can exceed detectable)
        "n_truth_detectable": 430,
        "n_undetected": 97,
        "completeness_detectable": (430 - 97) / 430,
    }
    num, den = completeness_bar_counts(b)
    assert den == 430
    assert num == 333  # 430 - 97, NOT 691
    assert abs(num / den - b["completeness_detectable"]) < 1e-12
    # Must not use detection-side TP as numerator
    assert num != b["n_true_positives"]


def test_completeness_bar_counts_on_real_json_if_present():
    """If on-disk SZiFi benchmark JSON exists, exercise real file path."""
    sys.path.insert(0, str(SCRIPTS))
    from regenerate_paper_figures import completeness_bar_counts
    import json

    cat = Path(
        "/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi/catalogues"
    )
    for name in ("immf", "scimmf"):
        p = cat / f"footprint_splitA_{name}_q5_benchmark_szifi.json"
        if not p.is_file():
            pytest.skip(f"benchmark JSON not on disk: {p}")
        b = json.loads(p.read_text())
        num, den = completeness_bar_counts(b)
        assert den == int(b["n_truth_detectable"])
        assert num == den - int(b["n_undetected"])
        assert abs(num / den - float(b["completeness_detectable"])) < 1e-9
        # Detection-side TP may exceed detectable truth; do not use it
        assert num <= den
