"""Truth y NaMaster 18-bin matches flamingo_repo; CIB deproj lowers CIB residual."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import ILC  # noqa: E402

PS = ILC / "ps_namaster"
REF = Path(
    "/scratch/scratch-lxu/flamingo_repo/data_paper/binned_bandpowers"
    "/Dl_yy_L1_m9_fullsky_binned_18.txt"
)
ELL_EFF = np.array(
    [
        10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5,
        198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5,
    ]
)


def test_fullsky_truth_matches_flamingo_repo_18bin():
    path = PS / "L1_m9_truth_total.npz"
    assert path.is_file(), "run scripts/compute_hilc_residuals_truth.py"
    z = np.load(path)
    dl = np.asarray(z["dl"], dtype=np.float64) * 1.0e12
    ref = np.loadtxt(REF)
    assert dl.shape == (18,)
    rel = np.abs(dl - ref[:, 1]) / np.maximum(np.abs(ref[:, 1]), 1e-30)
    assert np.all(rel < 0.05), rel.max()


def test_deproj_cib_lowers_cib_residual_below_ell500():
    lo = ELL_EFF < 500.0
    for kind in ("total", "masked"):
        none = PS / f"L1_m9_nodeproj_{kind}_residuals.npz"
        cib = PS / f"L1_m9_deproj_cib_{kind}_residuals.npz"
        assert none.is_file() and cib.is_file(), "run scripts/compute_hilc_residuals_truth.py"
        a = np.abs(np.asarray(np.load(none)["dl_cib"]))
        b = np.abs(np.asarray(np.load(cib)["dl_cib"]))
        assert np.all(b[lo] < a[lo]), kind


def test_cib_freq_autos_positive_six_channels():
    sys.path.insert(0, str(_SCRIPTS))
    import plot_hilc_recovered_residuals_deproj as p

    dl = p._cib_freq_dl("L1_m9")
    assert tuple(dl) == p.FREQS_PLOT
    for nu, arr in dl.items():
        assert arr.shape == (18,)
        assert np.all(np.isfinite(arr) & (arr > 0.0)), nu
