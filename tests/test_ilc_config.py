"""Tests for flamingo_mock.ilc.config: YAML generation and pyILC parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyilc.input import ILCInfo

from flamingo_mock.ilc.config import GN_FWHM_ARCMIN, ILCConfig, pyilc_param_dict_file
from flamingo_mock.ilc.paths import ILC_BEAM_FWHM_ARCMIN, ILC_ELLMAX, ILC_FREQUENCIES_GHZ, ILCPaths

REPO = Path(__file__).resolve().parents[1]
HILC_SPLIT_A = REPO / "configs" / "hilc_y_flamingo_npipe_splitA.yml"
HILC_SPLIT_B = REPO / "configs" / "hilc_y_flamingo_npipe_splitB.yml"
NILC_SPLIT_A = REPO / "configs" / "nilc_y_flamingo_npipe_splitA.yml"


def test_param_dict_file_from_installed_pyilc():
    assert pyilc_param_dict_file().is_file()


def test_config_writer_roundtrip_hilc(tmp_path):
    cfg = ILCConfig(method="hilc", split="A", paths=ILCPaths())
    out = cfg.write(tmp_path / "hilc.yml")
    info = ILCInfo(str(out))
    assert info.ELLMAX == ILC_ELLMAX
    assert info.N_side >= 2048
    assert info.ILC_preserved_comp == "tSZ"
    assert info.N_deproj == 0
    assert info.wavelet_type == "TopHatHarmonic"
    assert info.perform_ILC_at_beam == pytest.approx(ILC_BEAM_FWHM_ARCMIN)
    assert info.perform_ILC_at_beam == pytest.approx(10.0)
    assert info.N_freqs == 6
    assert list(info.freqs_delta_ghz) == pytest.approx(list(ILC_FREQUENCIES_GHZ))
    assert list(info.beam_FWHM_arcmin) == pytest.approx(
        [9.66, 7.22, 4.92, 4.90, 4.67, 4.22]
    )
    assert info.ilc_backend == "jax"
    assert len(info.freq_map_files) == 6
    for f in info.freq_map_files:
        assert "npipe_splitA" in f
    # Planck GAL×PS mask on both cov and wavelet stages
    assert info.mask_before_covariance_computation is not None
    assert info.mask_before_wavelet_computation is not None
    fsky = float(info.mask_before_covariance_computation.mean())
    assert 0.4 < fsky < 0.8, fsky


def test_config_writer_roundtrip_nilc(tmp_path):
    cfg = ILCConfig(method="nilc", split="A", paths=ILCPaths())
    out = cfg.write(tmp_path / "nilc.yml")
    info = ILCInfo(str(out))
    assert info.wavelet_type == "GaussianNeedlets"
    assert info.N_scales == 10
    assert list(info.GN_FWHM_arcmin) == pytest.approx(GN_FWHM_ARCMIN)
    assert info.ELLMAX == ILC_ELLMAX
    assert info.ILC_preserved_comp == "tSZ"
    assert info.perform_ILC_at_beam == pytest.approx(10.0)
    assert info.N_freqs == 6
    assert info.mask_before_covariance_computation is not None
    assert info.mask_before_wavelet_computation is not None


def test_splits_differ_only_in_maps_and_output(tmp_path):
    a = ILCConfig(method="hilc", split="A", paths=ILCPaths()).to_dict()
    b = ILCConfig(method="hilc", split="B", paths=ILCPaths()).to_dict()
    assert a["freq_map_files"] != b["freq_map_files"]
    assert all("splitA" in f for f in a["freq_map_files"])
    assert all("splitB" in f for f in b["freq_map_files"])
    assert a["output_dir"] != b["output_dir"]
    for key in ("ELLMAX", "N_side", "beam_FWHM_arcmin", "ilc_backend", "N_freqs"):
        assert a[key] == b[key]
    assert a["N_freqs"] == 6


def test_tracked_repo_configs_parse_and_match_writer():
    """The tracked configs/ YAMLs must stay in sync with the package writer."""
    for path, method, split in (
        (HILC_SPLIT_A, "hilc", "A"),
        (HILC_SPLIT_B, "hilc", "B"),
        (NILC_SPLIT_A, "nilc", "A"),
    ):
        assert path.is_file(), f"{path} missing — run: flamingo-ilc config"
        on_disk = ILCInfo(str(path))
        fresh = ILCConfig(method=method, split=split, paths=ILCPaths())
        assert on_disk.ELLMAX == fresh.ellmax
        assert on_disk.ilc_backend == fresh.ilc_backend
        assert on_disk.N_freqs == 6
        assert list(on_disk.freq_map_files) == [
            str(p) for p in map(Path, fresh.to_dict()["freq_map_files"])
        ]
        assert on_disk.mask_before_covariance_computation is not None
        assert on_disk.mask_before_wavelet_computation is not None
        assert list(fresh.to_dict()["mask_before_covariance_computation"]) == [
            str(fresh.paths.combined_gal_ps_mask()),
            0,
        ]


def test_deproj_config_roundtrip_cib(tmp_path):
    """Constrained ILC: N_deproj + ILC_deproj_comps parse via shipped pyILC."""
    cfg = ILCConfig(
        method="hilc",
        split="A",
        paths=ILCPaths(),
        n_deproj=1,
        deproj_comps=("CIB",),
    )
    out = cfg.write(tmp_path / "hilc_deproj_cib.yml")
    info = ILCInfo(str(out))
    assert info.N_deproj == 1
    assert list(info.ILC_deproj_comps) == ["CIB"]
    assert info.ILC_preserved_comp == "tSZ"
    assert "deproj_CIB" in str(info.output_dir)


def test_deproj_config_rejects_too_many_components():
    with pytest.raises(ValueError, match="not enough channels"):
        ILCConfig(
            method="hilc",
            n_deproj=6,
            deproj_comps=("CMB", "kSZ", "CIB", "CIB_dbeta", "CIB_dT", "mu"),
        )


def test_tracked_configs_are_nodeproj_baseline():
    """Baseline tracked YAMLs intentionally use N_deproj: 0 (no deprojection)."""
    for path in (HILC_SPLIT_A, HILC_SPLIT_B, NILC_SPLIT_A):
        info = ILCInfo(str(path))
        assert info.N_deproj == 0
        assert list(getattr(info, "ILC_deproj_comps", []) or []) == []


def test_mask_product_is_gal_times_ps_not_nilc_only():
    """Combined mask must encode galactic AND point-source cuts."""
    import numpy as np
    import healpy as hp

    from flamingo_mock.ilc.masks import load_gal_ps_product
    from flamingo_mock.ilc.paths import ILCPaths, MASK_GAL_FIELD, MASK_PS_FIELD

    paths = ILCPaths()
    src = paths.planck_masks_fits()
    if not src.is_file():
        pytest.skip("PR4 masks not on disk")
    gal = hp.read_map(str(src), field=MASK_GAL_FIELD, dtype=np.float64)
    ps = hp.read_map(str(src), field=MASK_PS_FIELD, dtype=np.float64)
    product = load_gal_ps_product(src)
    expected = ((gal > 0.5) * (ps > 0.5)).astype(np.float64)
    assert np.array_equal(product, expected)
    assert product.mean() < ps.mean() - 0.1
    assert 0.4 < product.mean() < 0.8
