"""Planck NPIPE (PR4) instrumental noise maps: paths and loading.

See ``noise_description.md`` for the full reference. Key facts used here:

* File template ``npipe6v20_noise_{freq}_{split}_mc_{mc:05d}.fits`` with
  detector-set splits ``A`` / ``B`` (independently destriped, so A×B
  cross-spectra are noise-decoupled) and realisations 00200–00299.
* HEALPix Nside=2048, RING, Galactic, float32, full sky.
* Units: K_CMB for 100–353 GHz; **MJy/sr for 545/857 GHz**, converted to
  K_CMB on load via dB_ν/dT at T_CMB (noise_description.md §7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spectral import dB_dT_Jy_per_sr_per_K
from .paths import NPIPE_MC_DEFAULT, NPIPE_SPLITS

# Channels whose NPIPE noise maps are stored in K_CMB (noise_description.md §7).
_K_CMB_FREQS = (30, 44, 70, 100, 143, 217, 353)
# Channels stored in MJy/sr — must convert to K_CMB before coadding with signal.
_MJY_SR_FREQS = (545, 857)


def noise_map_path(
    noise_dir: Path,
    freq_ghz: int,
    split: str,
    mc: int = NPIPE_MC_DEFAULT,
) -> Path:
    """Path to an NPIPE noise realisation, e.g. ``.../100GHz/A/npipe6v20_noise_100_A_mc_00200.fits``."""
    if split not in NPIPE_SPLITS + ("full",):
        raise ValueError(f"split must be one of {NPIPE_SPLITS + ('full',)}, got {split!r}")
    token = f"_{split}" if split in NPIPE_SPLITS else ""
    return (
        Path(noise_dir)
        / f"{freq_ghz}GHz"
        / split
        / f"npipe6v20_noise_{freq_ghz}{token}_mc_{mc:05d}.fits"
    )


def mjy_sr_to_K_cmb(freq_ghz: float) -> float:
    """Multiplicative factor: map[MJy/sr] × factor → map[K_CMB].

    ``1 MJy/sr = 1e6 Jy/sr`` and T[K] = I[Jy/sr] / (dB/dT [Jy/sr/K]).
    """
    return 1.0e6 / dB_dT_Jy_per_sr_per_K(float(freq_ghz))


def load_noise_map(
    noise_dir: Path,
    freq_ghz: int,
    split: str,
    mc: int = NPIPE_MC_DEFAULT,
    nside_out: int | None = None,
) -> np.ndarray:
    """Load an NPIPE noise temperature map [K_CMB], optionally ud-graded.

    545/857 GHz maps are converted from MJy/sr → K_CMB; lower HFI channels
    are already in K_CMB and are returned unchanged.
    """
    import healpy as hp

    if freq_ghz not in _K_CMB_FREQS and freq_ghz not in _MJY_SR_FREQS:
        raise ValueError(f"unsupported NPIPE noise frequency: {freq_ghz} GHz")

    path = noise_map_path(noise_dir, freq_ghz, split, mc)
    if not path.is_file():
        raise FileNotFoundError(path)
    m = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))
    # NPIPE residual maps are full-sky but occasionally contain a single
    # pathological pixel (~healpy UNSEEN magnitude); zero those so they do
    # not blow up the map after unit conversion.
    bad = ~np.isfinite(m) | (np.abs(m) > 1.0e20)
    if np.any(bad):
        m = m.copy()
        m[bad] = 0.0
    if freq_ghz in _MJY_SR_FREQS:
        m = m * mjy_sr_to_K_cmb(freq_ghz)
    if nside_out is not None and hp.npix2nside(m.size) != nside_out:
        m = hp.ud_grade(m, nside_out)
    return m
