"""Tests for FLAMINGO mock pyILC path: ELLMAX≥3000, noise splits, beam deconv."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

HILC_SPLIT0 = REPO / "configs" / "hilc_y_flamingo_noise_split0.yml"
HILC_SPLIT1 = REPO / "configs" / "hilc_y_flamingo_noise_split1.yml"
NILC_SPLIT0 = REPO / "configs" / "nilc_y_flamingo_noise_split0.yml"
RUN_SCRIPT = REPO / "scripts" / "run_pyilc_y.py"
VAL_SCRIPT = REPO / "scripts" / "validate_ymap_vs_truth.py"
PREP_SCRIPT = REPO / "scripts" / "prepare_mock_ilc_inputs.py"
BEAM_UTILS = REPO / "scripts" / "beam_utils.py"
INPUT_DIR = Path(
    "/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/inputs_nside2048_noise"
)


def test_beam_utils_deconvolution_recovers_flat_cl():
    """Unit test of shipped beam helper (not a reimplementation of ILC)."""
    from beam_utils import deconvolve_cl_beam, gaussian_beam_bl

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


def test_hilc_noise_split_yaml_ellmax_beam_and_paths():
    assert HILC_SPLIT0.is_file()
    from pyilc.input import ILCInfo

    info = ILCInfo(str(HILC_SPLIT0))
    assert info.ELLMAX >= 3000
    assert info.N_side >= 2048
    assert info.ILC_preserved_comp == "tSZ"
    assert info.N_deproj == 0
    assert info.wavelet_type == "TopHatHarmonic"
    assert info.perform_ILC_at_beam == pytest.approx(5.0)
    assert list(info.beam_FWHM_arcmin) == pytest.approx([9.66, 7.22, 4.92])
    assert info.N_freqs == 3
    assert len(info.freq_map_files) == 3
    for f in info.freq_map_files:
        assert "noise_split" in f
        assert "100GHz" in f or "143GHz" in f or "353GHz" in f


def test_nilc_noise_split_yaml_paper_needlets_ellmax():
    assert NILC_SPLIT0.is_file()
    from pyilc.input import ILCInfo

    info = ILCInfo(str(NILC_SPLIT0))
    assert info.wavelet_type == "GaussianNeedlets"
    assert info.N_scales == 10
    assert info.ELLMAX >= 3000
    assert info.N_side >= 2048
    assert info.ILC_preserved_comp == "tSZ"
    assert info.perform_ILC_at_beam == pytest.approx(5.0)


def test_both_noise_splits_yaml_exist_and_differ():
    from pyilc.input import ILCInfo

    i0 = ILCInfo(str(HILC_SPLIT0))
    i1 = ILCInfo(str(HILC_SPLIT1))
    assert i0.ELLMAX == i1.ELLMAX >= 3000
    assert i0.freq_map_files != i1.freq_map_files
    assert all("split0" in f for f in i0.freq_map_files)
    assert all("split1" in f for f in i1.freq_map_files)


def test_noise_split_input_maps_exist_when_prepared():
    from pyilc.input import ILCInfo

    info = ILCInfo(str(HILC_SPLIT0))
    present = [Path(f) for f in info.freq_map_files if Path(f).is_file()]
    if len(present) < 3:
        pytest.skip("noise-split inputs not prepared yet")
    for f in present:
        assert f.stat().st_size > 1_000_000
    # split1 also
    info1 = ILCInfo(str(HILC_SPLIT1))
    for f in info1.freq_map_files:
        p = Path(f)
        if not p.is_file():
            pytest.skip("split1 maps missing")
        assert p.stat().st_size > 1_000_000


def test_prep_script_and_validate_use_beam_deconv():
    assert PREP_SCRIPT.is_file()
    assert BEAM_UTILS.is_file()
    assert VAL_SCRIPT.is_file()
    assert RUN_SCRIPT.is_file()
    val = VAL_SCRIPT.read_text()
    assert "deconvolve_cl_beam" in val
    assert "ilc-beam-fwhm-arcmin" in val
    assert "beam_deconvolved" in val
    prep = PREP_SCRIPT.read_text()
    assert "ffp10_noise" in prep
    assert "split" in prep


def test_noise_residual_order_of_magnitude_when_present():
    """map − signal should look like FFP10 noise, not zero (proves noise added)."""
    import healpy as hp

    sig = INPUT_DIR / "sky_CMB_tSZ_CIB_signal_100GHz_nside2048_K.fits"
    sk0 = INPUT_DIR / "sky_CMB_tSZ_CIB_noise_split0_100GHz_nside2048_K.fits"
    sk1 = INPUT_DIR / "sky_CMB_tSZ_CIB_noise_split1_100GHz_nside2048_K.fits"
    if not (sig.is_file() and sk0.is_file() and sk1.is_file()):
        pytest.skip("prepared inputs missing")
    s = hp.read_map(str(sig), dtype=np.float64)
    m0 = hp.read_map(str(sk0), dtype=np.float64)
    m1 = hp.read_map(str(sk1), dtype=np.float64)
    n0 = m0 - s
    n1 = m1 - s
    assert n0.std() > 1e-6
    assert n1.std() > 1e-6
    # independent splits: residual correlation should be low
    step = max(1, len(n0) // 500_000)
    c = float(np.corrcoef(n0[::step], n1[::step])[0, 1])
    assert abs(c) < 0.2
    # noise should be a non-negligible fraction of sky variance at 100 GHz
    assert n0.std() > 0.05 * m0.std()


def test_validate_existing_ymap_beamdec_if_present(tmp_path):
    candidates = list(
        Path("/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc").glob(
            "**/flamingo_*needletILCmap_component_tSZ*.fits"
        )
    )
    # exclude per-scale maps
    candidates = [c for c in candidates if "scale" not in c.name]
    if not candidates:
        pytest.skip("no full ILC y-map yet")
    ymap = max(candidates, key=lambda p: p.stat().st_mtime)
    log = tmp_path / "validate.log"
    figdir = tmp_path / "figures"
    r = subprocess.run(
        [
            sys.executable,
            str(VAL_SCRIPT),
            "--ymap",
            str(ymap),
            "--figures-dir",
            str(figdir),
            "--log",
            str(log),
            "--lmax",
            "3000",
            "--ilc-beam-fwhm-arcmin",
            "5.0",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
    body = log.read_text()
    assert '"beam_deconvolved": true' in body
    assert '"ok_amplitude": true' in body
    assert '"ok_beam_deconvolution": true' in body
    # ratio plot must use cross transfer, not noise-biased auto Cyy/Ctt
    assert "y x truth" in body or "C_ell^{y x truth}" in body or "transfer" in body
    assert (figdir / "ilc_y_vs_truth_spectra.png").is_file()
    assert (figdir / "ilc_y_beam_deconv_ratio.png").is_file()
    # sanity: mid-ell transfer should be O(1), not >>1 like noise-biased auto ratio
    import json

    summary = json.loads(body)
    t_mid = summary.get("transfer_mid_ell_200_800_dec")
    assert t_mid is not None and 0.3 < t_mid < 3.0, t_mid
