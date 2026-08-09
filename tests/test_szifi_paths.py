"""Unit tests for SZiFi path helpers."""

from flamingo_mock.config import BEAM_FWHM_ARCMIN
from flamingo_mock.szifi.paths import FREQS_GHZ, SZiFiPaths, beam_fwhm_vec_arcmin


def test_freqs_and_fwhm_order():
    assert FREQS_GHZ == (100, 143, 217, 353, 545, 857)
    assert list(beam_fwhm_vec_arcmin()) == [BEAM_FWHM_ARCMIN[f] for f in FREQS_GHZ]


def test_total_map_path_naming():
    p = SZiFiPaths()
    path = p.total_map_path("A", 143)
    assert path.name == "sky_CMB_tSZ_kSZ_CIB_npipe_splitA_143GHz_nside2048_K.fits"
