"""Tests for the pyILC multi-backend ILC weight solves (numpy/numba/JAX/CuPy).

These exercise the installed pyilc package directly — the FLAMINGO pipeline
delegates all ILC linear algebra to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyilc.ilc_linalg import (
    available_backends,
    compute_ilc_weights_from_cov,
    resolve_backend,
)


def _random_spd_cov(rng, n_pix=64, n_freq=3):
    a = rng.normal(size=(n_pix, n_freq, n_freq))
    cov = np.einsum("pij,pkj->pik", a, a)
    cov += np.eye(n_freq)[None] * 1e-2
    return cov


def test_jax_backend_available_and_resolves():
    info = available_backends()
    assert info["jax"], "jax must be importable in the pipeline environment"
    assert resolve_backend("jax") == "jax"
    # On this GPU host 'auto' must pick JAX; elsewhere allow numba/numpy.
    if info["jax_gpu"]:
        assert info["auto"] == "jax"


def test_weights_jax_matches_numpy_and_preserves_response():
    rng = np.random.default_rng(0)
    cov = _random_spd_cov(rng)
    # tSZ-like preserved SED (any nonzero vector tests the constraint)
    a_mix = np.array([[-1.0], [0.0], [1.5]])

    w_np, used_np = compute_ilc_weights_from_cov(cov, a_mix, backend="numpy")
    assert used_np == "numpy"
    w_jax, used_jax = compute_ilc_weights_from_cov(cov, a_mix, backend="jax")
    assert used_jax == "jax"

    assert w_np.shape == (cov.shape[0], cov.shape[1])
    assert np.allclose(w_jax, w_np, rtol=1e-8, atol=1e-10)
    # ILC constraint: w . a = 1 for the preserved component
    response = w_jax @ a_mix[:, 0]
    assert np.allclose(response, 1.0, rtol=1e-8, atol=1e-10)


def test_weights_numba_matches_numpy():
    pytest.importorskip("numba")
    rng = np.random.default_rng(1)
    cov = _random_spd_cov(rng)
    a_mix = np.array([[-1.0], [0.0], [1.5]])
    w_np, _ = compute_ilc_weights_from_cov(cov, a_mix, backend="numpy")
    w_nb, used = compute_ilc_weights_from_cov(cov, a_mix, backend="numba")
    assert used == "numba"
    assert np.allclose(w_nb, w_np, rtol=1e-8, atol=1e-10)
