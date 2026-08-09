"""Unit tests for flamingo_mock.ilc.beams (Gaussian beam helpers)."""

from __future__ import annotations

import numpy as np
import pytest

from flamingo_mock.ilc.beams import deconvolve_cl_beam, gaussian_beam_bl


def test_beam_utils_deconvolution_recovers_flat_cl():
    """Unit test of shipped beam helper (not a reimplementation of ILC)."""
    lmax = 3000
    fwhm = 5.0
    bl = gaussian_beam_bl(fwhm, lmax)
    assert bl[0] == pytest.approx(1.0)
    assert bl[2000] < 0.5  # 5' beam suppresses high ell
    # fake beamed spectrum
    cl_true = np.ones(lmax + 1)
    cl_obs = cl_true * bl**2
    cl_dec = deconvolve_cl_beam(cl_obs, fwhm, bl_floor=1e-4)
    good = np.isfinite(cl_dec) & (np.arange(lmax + 1) < 1500)
    assert np.allclose(cl_dec[good], 1.0, rtol=1e-5)


def test_channel_beams_match_planck_hfi():
    from flamingo_mock.ilc.paths import channel_beams_arcmin

    beams = channel_beams_arcmin()
    assert beams == pytest.approx({100: 9.66, 143: 7.22, 353: 4.92})
