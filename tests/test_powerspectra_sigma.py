"""Gaussian bandpower errors: auto vs split-cross."""

from __future__ import annotations

import numpy as np

from flamingo_mock.powerspectra import (
    ilc_bias_fraction,
    n_modes_tophat_hilc,
    sigma_dl_auto_binned,
    sigma_dl_cross_binned,
)


def test_auto_matches_hand_sum():
    cl = np.ones(8, dtype=np.float64)
    ell_min = np.array([2])
    ell_max = np.array([5])  # ℓ = 2, 3, 4
    got = sigma_dl_auto_binned(cl, ell_min, ell_max, fsky=1.0)[0]

    ell = np.array([2.0, 3.0, 4.0])
    fac = ell * (ell + 1.0) / (2.0 * np.pi)
    var_c = 2.0 * 1.0 / (2.0 * ell + 1.0)
    want = np.sqrt(np.sum(fac**2 * var_c) / 9.0)
    assert np.isclose(got, want)


def test_auto_is_not_the_split_cross_formula():
    """Auto uses 2 C^2; a split-cross with C12=C, C11=C22=C+N is smaller."""
    c_s = 1.0
    c_n = 3.0
    cl_auto = np.full(20, c_s + c_n)
    cl_cross = np.full(20, c_s)
    ell_min = np.array([5])
    ell_max = np.array([10])
    sig_auto = sigma_dl_auto_binned(cl_auto, ell_min, ell_max)[0]
    sig_x = sigma_dl_cross_binned(cl_auto, cl_auto, cl_cross, ell_min, ell_max)[0]
    assert sig_auto > sig_x
    # Same C in auto and in a fake 'cross' with C11=C22=C12=C recovers the auto.
    sig_fake = sigma_dl_cross_binned(cl_auto, cl_auto, cl_auto, ell_min, ell_max)[0]
    assert np.isclose(sig_fake, sig_auto)


def test_auto_uses_total_cl_not_signal_only():
    """Error bars must grow with noise in the auto; tSZ-only C_l is too small."""
    cl_tot = np.full(30, 10.0)
    cl_tsz = np.full(30, 1.0)
    ell_min = np.array([8])
    ell_max = np.array([16])
    sig_tot = sigma_dl_auto_binned(cl_tot, ell_min, ell_max)[0]
    sig_tsz = sigma_dl_auto_binned(cl_tsz, ell_min, ell_max)[0]
    assert np.isclose(sig_tot / sig_tsz, 10.0)


def test_wider_bin_smaller_sigma_for_flat_dl():
    """For D_ℓ ≈ const, more modes in the bin shrink σ of the bin mean."""
    ell = np.arange(80, dtype=np.float64)
    cl = np.zeros(80)
    cl[2:] = 2.0 * np.pi / (ell[2:] * (ell[2:] + 1.0))
    sig_narrow = sigma_dl_auto_binned(cl, np.array([10]), np.array([20]))[0]
    sig_wide = sigma_dl_auto_binned(cl, np.array([10]), np.array([50]))[0]
    assert sig_wide < sig_narrow


def test_auto_sigma_matches_gaussian_simulations():
    """Empirical check: std of Ĉ over GRF realisations vs 2 C^2 / ((2ℓ+1) fsky)."""
    import healpy as hp

    nside, lmax, nsim = 64, 96, 80
    cl_true = np.zeros(lmax + 1)
    cl_true[2:] = 1.0
    cls = np.empty((nsim, lmax + 1))
    for i in range(nsim):
        m = hp.synfast(cl_true, nside=nside, lmax=lmax, pixwin=False)
        cls[i] = hp.anafast(m, lmax=lmax, iter=0)
    ell = np.arange(lmax + 1, dtype=np.float64)
    pred = np.sqrt(2.0) * cl_true / np.sqrt(2.0 * ell + 1.0)
    emp = cls.std(axis=0, ddof=1)
    band = (ell >= 10) & (ell <= 50)
    assert abs(float(np.median(emp[band] / pred[band])) - 1.0) < 0.15

    ell_min = np.array([10, 30])
    ell_max = np.array([20, 50])
    dl_sims = np.empty((nsim, 2))
    for i, cl in enumerate(cls):
        for j, (lo, hi) in enumerate(zip(ell_min, ell_max)):
            e = np.arange(lo, hi, dtype=np.float64)
            dl_sims[i, j] = np.mean(e * (e + 1.0) * cl[lo:hi] / (2.0 * np.pi))
    pred_b = sigma_dl_auto_binned(cl_true, ell_min, ell_max)
    emp_b = dl_sims.std(axis=0, ddof=1)
    assert np.all(np.abs(emp_b / pred_b - 1.0) < 0.25)


def test_tophat_hilc_n_modes_first_bin():
    """Σ_{ℓ=0}^{49} (2ℓ+1) = 50² for BinSize=50."""
    n = n_modes_tophat_hilc(lmax=200, bin_size=50, fsky=1.0)
    assert n[0] == 2500.0
    assert n[49] == 2500.0
    assert n[50] == 7500.0
    assert np.isclose(n[50], np.sum(2.0 * np.arange(50, 100) + 1.0))


def test_ilc_bias_fraction_decreases_with_deprojection():
    n_modes = np.array([2500.0])
    f0 = ilc_bias_fraction(0, 6, n_modes)[0]
    f3 = ilc_bias_fraction(3, 6, n_modes)[0]
    assert np.isclose(f0, 5.0 / 2500.0)
    assert np.isclose(f3, 2.0 / 2500.0)
    assert f3 < f0
    # Fully constrained: N_freq = 1 + N_deproj ⇒ bias vanishes.
    assert ilc_bias_fraction(5, 6, n_modes)[0] == 0.0


def test_cross_experimental_error_grows_with_noise():
    """Var_exp = (C11 C22 − C12²)/N_modes is the piece beyond tSZ cosmic variance."""
    c_s, n = 1.0, 3.0
    cl_auto = np.full(20, c_s + n)
    cl_cross = np.full(20, c_s)
    ell_min, ell_max = np.array([5]), np.array([10])
    sig_tot = sigma_dl_cross_binned(cl_auto, cl_auto, cl_cross, ell_min, ell_max)[0]
    sig_cv = sigma_dl_cross_binned(cl_cross, cl_cross, cl_cross, ell_min, ell_max)[0]
    assert sig_tot > sig_cv
    sig_exp = np.sqrt(sig_tot**2 - sig_cv**2)
    cl_loud = np.full(20, c_s + 10.0)
    sig_loud = sigma_dl_cross_binned(cl_loud, cl_loud, cl_cross, ell_min, ell_max)[0]
    sig_loud_exp = np.sqrt(sig_loud**2 - sig_cv**2)
    assert sig_loud_exp > sig_exp


def test_hilc_plot_scripts_use_l1_m9_not_test():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    files = (
        "plot_hilc_homog_r1xr2_split_diagnostics.py",
        "plot_hilc_homog_auto_fig9.py",
        "plot_hilc_homog_q5masked.py",
        "plot_hilc_homog_pyilc_convention.py",
    )
    for name in files:
        src = (root / name).read_text()
        assert "components/tsz/L1_m9" in src, name
        assert "components/tsz/test" not in src, name
        if "CIB_DIR" in src:
            assert "components/cib/L1_m9" in src, name
            assert "components/cib/test" not in src, name


def test_q5masked_r1xr2_fig9_writes_unbinned_yy_cache():
    """Ratio ILC errors read cl_11/22/12 from this cache; residuals-only npz is not enough."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_hilc_homog_q5masked_r1xr2_fig9.py"
    ).read_text()
    assert "np.savez" in src
    assert "case.cl_cache" in src
    assert 'cl_11=stored["cl_11"]' in src


def test_joint_fisher_matches_x_matrix_inverse():
    """σ(A_tSZ) = sqrt((Σ_A)_{tt}) for Σ_A = (X^T M^{-1} X)^{-1}, X = (t | f)."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_l1_m9_feedback_ratio_ilc_errors.py"
    )
    spec = importlib.util.spec_from_file_location("ilc_ratio_err", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t = np.array([1.0, 2.0, 0.5, 1.5])
    f = np.array([0.3, 0.8, 1.1, 0.2])
    m = np.diag([1.0, 2.0, 0.5, 1.2])
    j = mod.joint_fisher(t, f, m)
    x = np.column_stack((t, f))
    cov = np.linalg.inv(x.T @ np.linalg.solve(m, x))
    schur = 1.0 / (
        t @ np.linalg.solve(m, t)
        - (t @ np.linalg.solve(m, f)) ** 2 / (f @ np.linalg.solve(m, f))
    )
    assert np.allclose(j["cov_A"], cov)
    assert np.isclose(j["sig_yy"] ** 2, cov[0, 0])
    assert np.isclose(j["sig_yy"] ** 2, schur)
    assert np.isclose(j["sig_A"] ** 2, cov[1, 1])


def test_joint_fisher_orthogonal_templates_no_inflate():
    """d = A t + B f with M=I: orthogonal templates leave σ_A = 1/||f||."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_l1_m9_feedback_ratio_ilc_errors.py"
    )
    spec = importlib.util.spec_from_file_location("ilc_ratio_err", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t = np.ones(4)
    f = np.array([1.0, -1.0, 1.0, -1.0])
    j = mod.joint_fisher(t, f, np.eye(4))
    assert abs(j["cos"]) < 1e-12
    assert np.isclose(j["sig_A"], 0.5)
    assert np.isclose(j["inflate"], 1.0)
    j_par = mod.joint_fisher(t, t, np.eye(4))
    assert j_par["inflate"] > 1e3


def test_joint_model_band_keeps_amplitude_cross_covariance():
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "plot_l1_m9_feedback_ratio_ilc_errors.py"
    )
    spec = importlib.util.spec_from_file_location("ilc_ratio_err", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t = np.array([1.0, 2.0, 0.5, 1.5])
    f = np.array([0.3, 0.8, 1.1, 0.2])
    j = mod.joint_fisher(t, f, np.diag([1.0, 2.0, 0.5, 1.2]))
    lo, hi = mod._joint_model_lohi(j, t)
    response = np.column_stack((np.ones_like(t), f / t))
    expected = np.sqrt(np.diag(response @ j["cov_A"] @ response.T))
    without_cross = np.sqrt(j["sig_yy"] ** 2 + (j["sig_A"] * f / t) ** 2)

    assert np.allclose((hi - lo) / 2.0, expected)
    assert not np.allclose(expected, without_cross)
