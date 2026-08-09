"""Validation of an existing ILC y-map against truth (skipped until a run exists)."""

from __future__ import annotations

import json

import pytest

from flamingo_mock.ilc.paths import ILCPaths
from flamingo_mock.ilc.validate import find_default_ymap, summary_ok, validate_ymap


def test_validate_existing_ymap_beamdec_if_present(tmp_path):
    paths = ILCPaths()
    ymap = find_default_ymap(paths)
    if ymap is None:
        pytest.skip("no full ILC y-map yet (flamingo-ilc run ...)")
    figdir = tmp_path / "figures"
    summary = validate_ymap(
        ymap,
        lmax=3000,
        ilc_beam_fwhm_arcmin=5.0,
        figures_dir=figdir,
    )
    body = json.dumps(summary)
    assert summary["beam_deconvolved"] is True
    assert summary["ok_amplitude"] is True
    assert summary["ok_beam_deconvolution"] is True
    assert summary_ok(summary)
    # ratio plot must use cross transfer, not noise-biased auto Cyy/Ctt
    assert "y x truth" in body or "transfer" in body
    assert (figdir / "ilc_y_vs_truth_spectra.png").is_file()
    assert (figdir / "ilc_y_beam_deconv_ratio.png").is_file()
    # sanity: mid-ell transfer should be O(1), not >>1 like noise-biased auto ratio
    t_mid = summary["transfer_mid_ell_200_800_dec"]
    assert t_mid is not None and 0.3 < t_mid < 3.0, t_mid
