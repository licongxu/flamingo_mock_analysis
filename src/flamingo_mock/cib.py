"""CIB component: released bandpass maps + approximate maps at other frequencies.

The released maps (217/353/545/857 GHz, bandpass-convolved, lensed) are
specific intensities in Jy/sr. Maps at other frequencies are approximated as
in the notebooks:

* exact released band        -> use that map;
* between two released bands -> log-frequency interpolation of I_nu;
* outside the released range -> scale the nearest band with the
  three-parameter greybody SED at an effective redshift z_eff.

The approximate maps ignore frequency decorrelation of small-scale structure
(Yang et al. 2026, Table 3); fine for 90/150 GHz where the CIB is subdominant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CIB_FILES, MockConfig
from .io import copy_or_link, load_flamingo_map, write_map
from .spectral import intensity_to_uK, sed_ratio


def _needed_bands(frequencies: tuple[float, ...]) -> set[int]:
    """Released bands that must be loaded to serve the requested frequencies."""
    released = sorted(CIB_FILES)
    needed: set[int] = set()
    for nu in frequencies:
        match = min(released, key=lambda b: abs(b - nu))
        if abs(nu - match) < 0.5:
            needed.add(match)
        elif released[0] < nu < released[-1]:
            i = int(np.searchsorted(released, nu))
            needed.update((released[i - 1], released[i]))
        else:
            needed.add(match)
    return needed


def load_cib_intensities(
    cfg: MockConfig, bands: set[int] | None = None
) -> dict[int, np.ndarray]:
    """Load released CIB intensity maps [Jy/sr] for the given bands."""
    bands = bands or set(CIB_FILES)
    out = {}
    for nu in sorted(bands):
        print(f"CIB: loading released {nu} GHz...")
        out[nu] = load_flamingo_map(cfg.data_dir / CIB_FILES[nu], cfg.nside)
    return out


def approximate_cib_intensity(
    nu_ghz: float, cib_I: dict[int, np.ndarray], z_eff: float
) -> tuple[np.ndarray, str]:
    """Return I_nu [Jy/sr] at nu_ghz plus a short method tag."""
    bands = np.array(sorted(cib_I), dtype=float)

    for b in bands:
        if abs(nu_ghz - b) < 0.5:
            return cib_I[int(b)].copy(), f"released {int(b)} GHz"

    if bands.min() < nu_ghz < bands.max():
        i = int(np.searchsorted(bands, nu_ghz))
        nu_a, nu_b = bands[i - 1], bands[i]
        w = np.log(nu_b / nu_ghz) / np.log(nu_b / nu_a)
        Ia, Ib = cib_I[int(nu_a)], cib_I[int(nu_b)]
        # Allow small negatives from lensing: floor before taking logs
        floor = 1e-3  # Jy/sr
        logI = w * np.log(np.clip(Ia, floor, None)) + (1.0 - w) * np.log(
            np.clip(Ib, floor, None)
        )
        return np.exp(logI), f"log-interp {nu_a:.0f}-{nu_b:.0f} GHz (w={w:.3f})"

    nu_ref = float(bands[np.argmin(np.abs(bands - nu_ghz))])
    ratio = sed_ratio(nu_ghz, nu_ref, z_eff=z_eff)
    return (
        cib_I[int(nu_ref)] * ratio,
        f"SED scale from {nu_ref:.0f} GHz at z_eff={z_eff} (x{ratio:.3f})",
    )


def copy_released_cib_intensity(
    cfg: MockConfig, out_dir: Path | None = None, *, use_symlink: bool = True
) -> dict[int, Path]:
    """Copy or link released bandpass-convolved CIB intensity maps [Jy/sr]."""
    out_dir = out_dir or cfg.raw_dir / "cib"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for nu in sorted(CIB_FILES):
        src = cfg.data_dir / CIB_FILES[nu]
        dst = out_dir / f"CIB_I_{nu}GHz_nside{cfg.nside}.fits"
        if not dst.exists():
            print(f"CIB: archiving released {nu} GHz intensity...")
            copy_or_link(src, dst, use_symlink=use_symlink)
        else:
            print(f"CIB: reusing {dst.name}")
        paths[nu] = dst
    return paths


def make_cib_maps(
    cfg: MockConfig, out_dir: Path | None = None
) -> dict[float, np.ndarray]:
    """Build CIB dT maps [uK_CMB] at every frequency in the config."""
    out_dir = out_dir or cfg.raw_dir / "cib"
    cib_I = load_cib_intensities(cfg, _needed_bands(cfg.frequencies))

    maps: dict[float, np.ndarray] = {}
    for nu in cfg.frequencies:
        I, method = approximate_cib_intensity(nu, cib_I, cfg.z_eff)
        T = intensity_to_uK(I, nu, t_cmb=cfg.t_cmb)
        del I
        maps[nu] = T
        approx = int("released" not in method)
        write_map(
            out_dir / f"CIB_deltaT_{nu:.0f}GHz_nside{cfg.nside}.fits",
            T,
            unit="uK_CMB",
            freq=nu,
            extra=[
                ("COMP", "CIB"),
                ("APPROX", approx, "1 if not an official released band"),
                ("METHOD", method[:60].encode("ascii", "replace").decode("ascii")),
            ],
        )
        print(f"  CIB {nu:g} GHz: std={T.std():.3e} uK | {method}")
    return maps
