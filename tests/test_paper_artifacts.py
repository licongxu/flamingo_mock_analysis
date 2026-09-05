"""Tests for the publication paper and figure-generation entry points.

These drive real shipped modules (spectral conversions, pub style, figure
regeneration helpers) rather than re-implementing them.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIG = ROOT / "figures"
SCRIPTS = ROOT / "scripts"


def test_paper_tex_exists_and_has_required_sections():
    tex_path = PAPER / "main.tex"
    if not tex_path.is_file():
        pytest.skip("paper/ is gitignored")
    tex = tex_path.read_text()
    assert r"\begin{abstract}" in tex
    assert "Construction of multi-frequency temperature maps" in tex
    assert "Discussion and conclusions" in tex
    # Component simulation keywords
    for key in ("tSZ", "kSZ", "CIB", "Compton", "lensed", "NPIPE", "greybody"):
        assert key in tex
    # Deprojection suite and cluster maps must be in the paper
    assert "deprojection" in tex.lower() or "Deprojection" in tex
    assert "szifi_footprint_immf_mollview" in tex
    assert "ilc_deproj_suite" in tex
    assert "Limitations" in tex
    # Not a software-release note
    assert "software paper" not in tex.lower()
    assert "orion_branch" not in tex


def test_paper_pdf_exists_multipage():
    pdf = PAPER / "main.pdf"
    if not pdf.is_file():
        pytest.skip("paper/ is gitignored")
    assert pdf.stat().st_size > 10_000
    # Prefer pdfinfo if available
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("pdfinfo not available")
    m = re.search(r"Pages:\s+(\d+)", out)
    assert m is not None
    assert int(m.group(1)) >= 5


def test_all_includegraphics_files_exist():
    tex_path = PAPER / "main.tex"
    if not tex_path.is_file():
        pytest.skip("paper/ is gitignored")
    tex = tex_path.read_text()
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    assert len(figs) >= 10
    missing = []
    for f in figs:
        name = Path(f).name
        hits = list(FIG.rglob(name))
        if not hits:
            missing.append(f)
    assert not missing, f"Missing figure files: {missing}"


def test_pub_style_usetex_no_grid():
    sys.path.insert(0, str(SCRIPTS))
    from pub_style import PUB_RCPARAMS, apply_pub_style, no_grid

    assert PUB_RCPARAMS["text.usetex"] is True
    assert PUB_RCPARAMS["axes.grid"] is False
    assert PUB_RCPARAMS["axes.labelsize"] >= 12
    apply_pub_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.grid(True)  # force on, then clear with shipped helper
    no_grid(ax)
    # After no_grid, major gridlines must be off
    assert ax.xaxis.get_gridlines()[0].get_visible() is False or (
        not any(g.get_visible() for g in ax.xaxis.get_gridlines())
    )
    assert plt.rcParams["axes.grid"] is False
    assert plt.rcParams["text.usetex"] is True
    assert plt.rcParams["axes.labelsize"] >= 12
    plt.close(fig)


def test_regenerate_script_has_no_grid_true():
    src = (SCRIPTS / "regenerate_paper_figures.py").read_text()
    assert "grid(True" not in src
    assert "from pub_style import" in src or "pub_style" in src
    style = (SCRIPTS / "pub_style.py").read_text()
    assert "text.usetex" in style
    assert "axes.grid" in style


def test_spectral_pipeline_real_functions():
    """Drive shipped spectral conversions used in the paper methods."""
    from flamingo_mock.spectral import (
        intensity_to_uK,
        ksz_response_uK,
        sed_ratio,
        tsz_f,
        tsz_response_uK,
        y_to_delta_T_uK,
    )

    # tSZ null near 217 GHz: f(x) changes sign across ~217
    f_100 = float(tsz_f(100.0))
    f_353 = float(tsz_f(353.0))
    assert f_100 < 0
    assert f_353 > 0
    # response scales y -> uK
    y = np.array([1e-6], dtype=np.float64)
    dt = y_to_delta_T_uK(y, 143.0)
    assert dt.shape == (1,)
    assert np.isfinite(dt[0])
    # kSZ frequency-independent response
    assert ksz_response_uK() < 0
    # CIB intensity conversion positive
    I = np.array([1.0], dtype=np.float64)  # Jy/sr
    t = intensity_to_uK(I, 353.0)
    assert t[0] > 0
    # SED ratio at z_eff=1.90: 100 GHz fainter than 217 GHz greybody
    r = sed_ratio(100.0, 217.0, z_eff=1.90)
    assert 0 < r < 1


def test_powerspectra_bin_cl_real():
    from flamingo_mock.powerspectra import bin_cl, dl_from_cl

    cl = np.ones(100)
    ell_b, cl_b = bin_cl(cl, delta_ell=7, lmin=2)
    assert ell_b.size > 0
    assert np.all(np.isfinite(cl_b))
    dl = dl_from_cl(ell_b, cl_b)
    assert dl.shape == ell_b.shape


def test_regenerated_line_plots_are_pdf():
    """Key regenerated paper plots should have vector PDF counterparts."""
    required = [
        "yang26_figs67_tsz_ksz_kappa_autos.pdf",
        "yang26_fig8left_cib_auto_spectra.pdf",
        "szifi_footprint_immf_benchmark_szifi.pdf",
        "szifi_footprint_scimmf_benchmark_szifi.pdf",
        "ilc_y_vs_truth_spectra_pub.pdf",
        "ilc_transfer_vs_truth_pub.pdf",
        "szifi_footprint_purity_immf_scimmf.pdf",
    ]
    missing = [n for n in required if not (FIG / n).is_file()]
    assert not missing, f"Missing regenerated PDFs: {missing}"


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


def test_paper_tex_flushes_floats_before_conclusions():
    tex_path = PAPER / "main.tex"
    if not tex_path.is_file():
        pytest.skip("paper/ is gitignored")
    tex = tex_path.read_text()
    # clearpage must appear before Discussion section
    i_disc = tex.index(r"\section{Discussion and conclusions}")
    i_clear = tex.rfind(r"\clearpage", 0, i_disc)
    assert i_clear != -1, "missing \\clearpage before Discussion"
    # Figures should prefer htbp (not bare [t] only)
    assert r"\begin{figure}[htbp]" in tex
    assert r"\begin{figure}[t]" not in tex
