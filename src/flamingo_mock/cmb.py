"""Lensed primary CMB temperature map.

Draws an unlensed Gaussian CMB realization from a CAMB C_ell^TT at the
FLAMINGO fiducial (D3A / DES Y3) cosmology, then deflects it with the
FLAMINGO convergence map using ``pixell.lensing.lens_map_curved``.

``camb`` and ``pixell`` are imported lazily so that the tSZ/kSZ/CIB steps
work without them installed.
"""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

from .config import MockConfig
from .io import load_flamingo_map, write_map


def camb_unlensed_cltt(lmax: int, cosmo: dict) -> np.ndarray:
    """Unlensed scalar C_ell^TT in uK^2, length lmax+1, from CAMB."""
    import camb

    h, Om, Ob = cosmo["h"], cosmo["Om"], cosmo["Ob"]
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100.0 * h,
        ombh2=Ob * h**2,
        omch2=(Om - Ob) * h**2,
        mnu=cosmo["mnu"],
        omk=0.0,
    )
    pars.InitPower.set_params(As=cosmo["As"], ns=cosmo["ns"])
    # Request slightly higher lmax so the spectrum is accurate up to lmax
    pars.set_for_lmax(max(lmax + 500, 3000), lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    return powers["unlensed_scalar"][: lmax + 1, 0]


def kappa_to_phi_alm(kappa: np.ndarray, lmax: int) -> np.ndarray:
    """Convergence map -> lensing potential alms via phi_lm = 2 kappa_lm / l(l+1)."""
    from pixell import lensing

    kappa_alm = hp.map2alm(kappa, lmax=lmax, iter=3)
    phi_alm = lensing.kappa_to_phi(kappa_alm)
    phi_alm[0:3] = 0.0  # monopole/dipole of phi are ill-defined
    return phi_alm


def lens_cmb_map(
    cltt: np.ndarray,
    phi_alm: np.ndarray,
    nside: int,
    lmax: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw an unlensed realization and lens it. Returns (lensed, unlensed) in uK."""
    from pixell import enmap, lensing, reproject

    np.random.seed(seed)
    cmb_alm = hp.synalm(cltt, lmax=lmax, new=True)
    t_unlensed = hp.alm2map(cmb_alm, nside=nside, pixwin=False)

    # CAR pixelization compatible with pixell SHTs: res = pi / (2 Nside)
    shape, wcs = enmap.fullsky_geometry(res=np.pi / (2 * nside))
    lensed_car, _ = lensing.lens_map_curved(
        shape,
        wcs,
        phi_alm,
        cmb_alm[None],
        output="lu",
        spin=0,
        verbose=True,
    )
    t_lensed = reproject.map2healpix(
        lensed_car, nside=nside, lmax=lmax, method="harm", spin=0
    )
    return np.asarray(t_lensed).reshape(-1), t_unlensed


def make_lensed_cmb(cfg: MockConfig, out_dir: Path | None = None) -> np.ndarray:
    """Full pipeline: CAMB spectrum -> realization -> lensing -> FITS.

    Returns the lensed CMB temperature map in uK_CMB at ``cfg.nside``.
    """
    out_dir = out_dir or cfg.raw_dir / "cmb"
    tag = f"nside{cfg.nside}_seed{cfg.seed}"
    out_path = out_dir / f"primary_CMB_T_lensed_{tag}.fits"
    if out_path.is_file():
        print(f"CMB: reusing cached {out_path}")
        return hp.read_map(str(out_path), dtype=np.float64)

    print(f"CMB: CAMB unlensed C_ell^TT (lmax={cfg.lmax})...")
    cltt = camb_unlensed_cltt(cfg.lmax, cfg.cosmology)

    print("CMB: loading FLAMINGO kappa...")
    kappa = load_flamingo_map(cfg.data_dir / "kappa_rot.fits", cfg.nside)
    kappa = kappa - np.mean(kappa)

    print("CMB: kappa -> phi_alm...")
    phi_alm = kappa_to_phi_alm(kappa, cfg.lmax)

    print("CMB: lensing realization (pixell.lens_map_curved)...")
    t_lensed, t_unlensed = lens_cmb_map(
        cltt, phi_alm, cfg.nside, cfg.lmax, cfg.seed
    )
    print(
        f"CMB: unlensed std={t_unlensed.std():.2f} uK, "
        f"lensed std={t_lensed.std():.2f} uK"
    )

    write_map(
        out_path,
        t_lensed,
        unit="uK_CMB",
        extra=[("COMP", "lensed primary CMB"), ("SEED", cfg.seed)],
        dtype=np.float64,
    )
    write_map(
        out_dir / f"primary_CMB_T_unlensed_{tag}.fits",
        t_unlensed,
        unit="uK_CMB",
        extra=[("COMP", "unlensed primary CMB"), ("SEED", cfg.seed)],
        dtype=np.float64,
    )
    np.savez(
        out_dir / f"camb_cltt_unlensed_{tag}.npz",
        ell=np.arange(cltt.size),
        cltt=cltt,
        cosmo=cfg.cosmology,
        lmax=cfg.lmax,
        nside=cfg.nside,
        seed=cfg.seed,
    )
    return t_lensed
