"""Planck NPIPE (PR4) instrumental noise maps: paths and loading.

See ``noise_description.md`` for the full reference. Key facts used here:

* File template ``npipe6v20_noise_{freq}_{split}_mc_{mc:05d}.fits`` with
  detector-set splits ``A`` / ``B`` (independently destriped, so A×B
  cross-spectra are noise-decoupled) and realisations 00200–00299.
* HEALPix Nside=2048, RING, Galactic, float32, full sky.
* Units are K_CMB for 100–353 GHz (the channels used here); 545/857 GHz are
  MJy/sr and are *not* supported by this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import NPIPE_MC_DEFAULT, NPIPE_SPLITS

# Channels whose NPIPE noise maps are stored in K_CMB (noise_description.md §7).
_K_CMB_FREQS = (30, 44, 70, 100, 143, 217, 353)


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


def load_noise_map(
    noise_dir: Path,
    freq_ghz: int,
    split: str,
    mc: int = NPIPE_MC_DEFAULT,
    nside_out: int | None = None,
) -> np.ndarray:
    """Load an NPIPE noise temperature map [K_CMB], optionally ud-graded."""
    if freq_ghz not in _K_CMB_FREQS:
        raise ValueError(
            f"NPIPE noise at {freq_ghz} GHz is not in K_CMB; "
            "545/857 GHz need a MJy/sr conversion this module does not do"
        )
    import healpy as hp

    path = noise_map_path(noise_dir, freq_ghz, split, mc)
    if not path.is_file():
        raise FileNotFoundError(path)
    m = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))
    if nside_out is not None and hp.npix2nside(m.size) != nside_out:
        m = hp.ud_grade(m, nside_out)
    return m
