"""Nested nside=8 CIB tile permutation used for L1_m9_cibshuffle totals."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import healpy as hp
import numpy as np

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_l1_m9_cibshuffle_totals.py"
_SPEC = importlib.util.spec_from_file_location("cibshuffle_totals", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)

permute_nested_tiles = _MOD.permute_nested_tiles
tile_permutation = _MOD.tile_permutation


def test_permutation_preserves_values_not_identity_same_across_freq():
    nside_tile = 8
    nside = 32
    npix = hp.nside2npix(nside)
    perm = tile_permutation(seed=20260831, nside_tile=nside_tile)
    assert perm.shape == (hp.nside2npix(nside_tile),)
    assert not np.array_equal(perm, np.arange(perm.size))

    rng = np.random.default_rng(0)
    freq_a = rng.normal(size=npix)
    freq_b = rng.normal(size=npix)
    shuf_a = permute_nested_tiles(freq_a, perm, nside_tile=nside_tile)
    shuf_b = permute_nested_tiles(freq_b, perm, nside_tile=nside_tile)

    assert np.allclose(np.sort(shuf_a), np.sort(freq_a))
    assert np.allclose(np.sort(shuf_b), np.sort(freq_b))
    assert not np.allclose(shuf_a, freq_a)
    assert np.allclose(shuf_a.std(), freq_a.std())

    nested_a = hp.reorder(freq_a, r2n=True).reshape(perm.size, -1)
    nested_shuf_a = hp.reorder(shuf_a, r2n=True).reshape(perm.size, -1)
    nested_b = hp.reorder(freq_b, r2n=True).reshape(perm.size, -1)
    nested_shuf_b = hp.reorder(shuf_b, r2n=True).reshape(perm.size, -1)
    np.testing.assert_allclose(nested_shuf_a, nested_a[perm])
    np.testing.assert_allclose(nested_shuf_b, nested_b[perm])
    # Same permutation applied independently to both frequencies.
    assert perm.tolist() == tile_permutation(seed=20260831, nside_tile=nside_tile).tolist()
