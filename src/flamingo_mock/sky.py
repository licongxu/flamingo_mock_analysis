"""Coadd sky components into per-frequency synthetic skies.

**Not part of the default pipeline.** Component maps are stored separately
under ``components/``; beam convolution and sky coaddition are deferred
until explicitly requested via ``--steps coadd``.
"""

from __future__ import annotations

import healpy as hp
import numpy as np

from .config import BEAM_FWHM_ARCMIN, MockConfig
from .io import write_map


def beam_fwhm_rad(nu_ghz: float) -> float | None:
    """Gaussian beam FWHM [radians] for a channel, or None if unknown."""
    fwhm = BEAM_FWHM_ARCMIN.get(int(round(nu_ghz)))
    return np.radians(fwhm / 60.0) if fwhm is not None else None


def make_coadd_maps(
    cfg: MockConfig,
    cmb_uK: np.ndarray,
    tsz_uK: dict[float, np.ndarray],
    ksz_uK: np.ndarray,
    cib_uK: dict[float, np.ndarray],
    smooth: bool = False,
) -> dict[float, np.ndarray]:
    """Sum components at each frequency; write uK (float32) and K (float64)."""
    skies: dict[float, np.ndarray] = {}
    for nu in cfg.frequencies:
        total = cmb_uK + tsz_uK[nu] + ksz_uK + cib_uK[nu]
        print(
            f"  {nu:g} GHz: CMB={cmb_uK.std():.2f} | tSZ={tsz_uK[nu].std():.2f} | "
            f"kSZ={ksz_uK.std():.2f} | CIB={cib_uK[nu].std():.2f} | "
            f"total={total.std():.2f} uK"
        )
        if smooth:
            fwhm = beam_fwhm_rad(nu)
            if fwhm is None:
                print(f"    no beam tabulated for {nu:g} GHz, skipping smoothing")
            else:
                print(f"    smoothing with FWHM={np.degrees(fwhm) * 60:.2f}'")
                total = hp.smoothing(total, fwhm=fwhm)

        tag = f"{nu:.0f}GHz_nside{cfg.nside}"
        extra = [("COMPS", "CMB+tSZ+kSZ+CIB"), ("SEED", cfg.seed)]
        if smooth:
            extra.append(("BEAM", f"Gaussian {np.degrees(fwhm) * 60:.2f} arcmin"))
        write_map(
            cfg.coadd_dir / f"sky_CMB_tSZ_kSZ_CIB_{tag}_uK.fits",
            total,
            unit="uK_CMB",
            freq=nu,
            extra=extra,
            dtype=np.float32,
        )
        write_map(
            cfg.coadd_dir / f"sky_CMB_tSZ_kSZ_CIB_{tag}_K.fits",
            total * 1.0e-6,
            unit="K_CMB",
            freq=nu,
            extra=extra,
            dtype=np.float64,
        )
        skies[nu] = total
        del total
    return skies
