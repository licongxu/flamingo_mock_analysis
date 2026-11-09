"""Truth y NaMaster 18-bin matches flamingo_repo; CIB deproj lowers CIB residual."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import ILC  # noqa: E402
import plot_hilc_truth_yy_ratio as truth_ratio_plot  # noqa: E402

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

TRUTH_RUNS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma")


def test_fullsky_truth_matches_flamingo_repo_18bin():
    path = PS / "L1_m9_truth_total.npz"
    assert path.is_file(), "run scripts/compute_hilc_residuals_truth.py"
    z = np.load(path)
    dl = np.asarray(z["dl"], dtype=np.float64) * 1.0e12
    ref = np.loadtxt(REF)
    assert dl.shape == (18,)
    rel = np.abs(dl - ref[:, 1]) / np.maximum(np.abs(ref[:, 1]), 1e-30)
    assert np.all(rel < 0.05), rel.max()


def test_variant_truth_yy_18bin_finite():
    for name in TRUTH_RUNS:
        for kind in ("total", "masked"):
            path = PS / f"{name}_truth_{kind}.npz"
            assert path.is_file(), "run scripts/compute_hilc_residuals_truth.py"
            z = np.load(path)
            dl = np.asarray(z["dl"], dtype=np.float64)
            assert dl.shape == (18,)
            assert np.all(np.isfinite(dl))
            if kind == "masked":
                assert float(z["fsky_eff"]) < 1.0


def test_truth_yy_ratios_finite():
    fid = {
        kind: np.asarray(np.load(PS / f"L1_m9_truth_{kind}.npz")["dl"], dtype=np.float64)
        for kind in ("total", "masked")
    }
    for name in ("fgas-8sigma", "Mstar-1sigma"):
        for kind in ("total", "masked"):
            dl = np.asarray(np.load(PS / f"{name}_truth_{kind}.npz")["dl"], dtype=np.float64)
            ratio = dl / fid[kind]
            assert np.all(np.isfinite(ratio) & (ratio > 0.0)), (name, kind)


def test_truth_yy_ratio_plot_import():
    assert truth_ratio_plot.STEM == "l1_m9_hilc_truth_yy_ratio"
    assert truth_ratio_plot.VARIANTS == ("fgas-8sigma", "Mstar-1sigma")


def test_tsz_cib_cross_negative_no_deproj_last_bin():
    for kind in ("total", "masked"):
        none = PS / f"L1_m9_nodeproj_{kind}_residuals.npz"
        cib = PS / f"L1_m9_deproj_cib_{kind}_residuals.npz"
        assert none.is_file() and cib.is_file(), "run scripts/compute_hilc_residuals_truth.py"
        if kind == "masked":
            assert bool(np.load(none)["apodized_sht"])
            assert bool(np.load(cib)["apodized_sht"])
        x0 = np.asarray(np.load(none)["dl_tsz_cib"])
        x1 = np.asarray(np.load(cib)["dl_tsz_cib"])
        assert x0[-1] < 0.0, (kind, float(x0[-1]))
        assert np.abs(x1[-1]) < np.abs(x0[-1]), (kind, float(x1[-1]), float(x0[-1]))


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


def test_budget_closes():
    from hilc_prescriptions import ALL_DEPROJ

    hi = ELL_EFF >= 20.0
    # Saved δβ/moment weights do not reproduce the pyILC y-map (recon ~ tSZ,
    # measured several times larger at ℓ=40). Algebra of the component maps
    # still closes. Match-to-y is required only where the operator is recovered.
    close_vs_y = {"nodeproj", "deproj_cib"}
    for name in ("L1_m9", "Mstar-1sigma"):
        for deproj in ALL_DEPROJ:
            path = PS / f"{name}_{deproj.key}_masked_budget.npz"
            assert path.is_file(), "run scripts/compute_hilc_component_budget.py"
            z = np.load(path)
            meas = np.asarray(z["dl_meas"])
            total = np.asarray(z["dl_sum"])
            recon = np.asarray(z["dl_recon"])
            alg = np.abs(total - recon) / np.maximum(np.abs(recon), 1e-30)
            rel = np.abs(total - meas) / np.maximum(np.abs(meas), 1e-30)
            assert np.all(alg < 1e-8), (name, deproj.key, float(alg.max()))
            if deproj.key in close_vs_y:
                assert np.all(rel[hi] < 0.05), (name, deproj.key, float(rel[hi].max()))


def test_noise_cross_consistent_with_zero():
    from hilc_prescriptions import ALL_DEPROJ

    for name in ("L1_m9", "Mstar-1sigma"):
        for deproj in ALL_DEPROJ:
            path = PS / f"{name}_{deproj.key}_masked_budget.npz"
            assert path.is_file(), "run scripts/compute_hilc_component_budget.py"
            z = np.load(path)
            dn = np.asarray(z["dl_n"])
            sig = np.asarray(z["dl_n_sigma"])
            ok = np.isfinite(sig) & (sig > 0.0)
            assert np.any(ok), (name, deproj.key)
            assert np.all(np.abs(dn[ok]) < 3.0 * sig[ok]), (
                name,
                deproj.key,
                float(np.max(np.abs(dn[ok]) / sig[ok])),
            )
