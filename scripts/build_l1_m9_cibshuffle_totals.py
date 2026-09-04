#!/usr/bin/env python3
"""L1_m9 totals with CIB shuffled among nside=8 SZiFi tiles.

CMB, tSZ, and homogeneous noise stay pixel-aligned with the fiducial L1_m9
sky. CIB is permuted among the 768 HEALPix nside=8 tiles (same permutation
on every HFI channel so the SED stays coherent), then beamed and coadded.

Writes total_maps/L1_m9_cibshuffle/. Refuses to write into total_maps/L1_m9/.
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

from flamingo_mock.config import BEAM_FWHM_ARCMIN, PLANCK_FREQUENCIES_GHZ
from flamingo_mock.io import write_map
from flamingo_mock.szifi.paths import TILE_NSIDE

COMP = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components")
NOISE = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")
OUT_ROOT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/total_maps")
FIDUCIAL = "L1_m9"
SHUFFLE_TAG = "L1_m9_cibshuffle"
SHUF_SEED = 20260831
NSIDE = 2048
NSIDE_IN = 4096
FREQS = tuple(int(f) for f in PLANCK_FREQUENCIES_GHZ)
SANITY_NU = 217


def permute_nested_tiles(
    m: np.ndarray, perm: np.ndarray, nside_tile: int = TILE_NSIDE
) -> np.ndarray:
    """Permute nested nside_tile superpixels of a RING HEALPix map."""
    nside = hp.get_nside(m)
    if nside % nside_tile != 0:
        raise ValueError(f"nside={nside} is not a multiple of nside_tile={nside_tile}")
    npix_tile = hp.nside2npix(nside_tile)
    if perm.shape != (npix_tile,):
        raise ValueError(f"perm length {perm.size} != {npix_tile}")
    nested = np.asarray(hp.reorder(m, r2n=True))
    n_sub = nested.size // npix_tile
    shuffled = nested.reshape(npix_tile, n_sub)[perm].ravel()
    return np.asarray(hp.reorder(shuffled, n2r=True))


def tile_permutation(seed: int = SHUF_SEED, nside_tile: int = TILE_NSIDE) -> np.ndarray:
    npix = hp.nside2npix(nside_tile)
    return np.random.default_rng(int(seed)).permutation(npix)


def load_uK(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))


def to_nside(m: np.ndarray, nside: int = NSIDE) -> np.ndarray:
    if hp.get_nside(m) == nside:
        return m
    return hp.ud_grade(m, nside)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def build_totals(out: Path, seed: int = SHUF_SEED) -> None:
    if out.resolve() == (OUT_ROOT / FIDUCIAL).resolve():
        raise SystemExit(f"refusing to overwrite fiducial totals at {out}")
    out.mkdir(parents=True, exist_ok=True)
    cmb_p = COMP / "cmb" / "primary_CMB_T_lensed_nside4096_seed42.fits"
    cib_dir = COMP / "cib" / FIDUCIAL
    tsz_dir = COMP / "tsz" / FIDUCIAL
    perm = tile_permutation(seed)
    print(f"prescription: {FIDUCIAL}  CIB shuffle seed={seed}  n_tiles={perm.size}")
    print("CMB:", cmb_p)
    print("CIB:", cib_dir)
    print("tSZ:", tsz_dir)
    print("out:", out)
    cmb = to_nside(load_uK(cmb_p))
    print(f"  CMB nside={hp.get_nside(cmb)}  rms={cmb.std():.2f} uK", flush=True)
    tsz_217 = None
    cib_217 = None
    cib_shuf_217 = None
    for nu in FREQS:
        tsz_m = to_nside(load_uK(tsz_dir / f"tSZ_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        cib = to_nside(load_uK(cib_dir / f"CIB_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        cib_shuf = permute_nested_tiles(cib, perm)
        if not np.allclose(cib.std(), cib_shuf.std(), rtol=1e-12, atol=0):
            raise RuntimeError(f"{nu} GHz: shuffled CIB rms {cib_shuf.std()} != {cib.std()}")
        noise = load_uK(NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside{NSIDE}_uK.fits")
        if hp.get_nside(noise) != NSIDE:
            raise ValueError(f"noise nside {hp.get_nside(noise)} != {NSIDE}")
        if nu == SANITY_NU:
            tsz_217, cib_217, cib_shuf_217 = tsz_m, cib, cib_shuf
        signal = cmb + tsz_m + cib_shuf
        fwhm = float(BEAM_FWHM_ARCMIN[nu])
        print(f"  {nu} GHz FWHM={fwhm:.2f}' ...", flush=True)
        signal_b = hp.smoothing(signal, fwhm=np.radians(fwhm / 60.0))
        total = signal_b + noise
        print(
            f"    rms CMB={cmb.std():.2f} tSZ={tsz_m.std():.2f} CIB={cib.std():.2f} "
            f"CIB_shuf={cib_shuf.std():.2f} beamed={signal_b.std():.2f} "
            f"noise={noise.std():.2f} total={total.std():.2f}"
        )
        write_map(
            out / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits",
            total,
            unit="uK_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB_shuffled+homog_noise"),
                ("PRESCRIP", FIDUCIAL),
                ("CIBSHUF", 1),
                ("SHUFSEED", int(seed)),
                ("FWHM", fwhm, "Gaussian beam [arcmin] on signal only"),
                ("BEAMON", 1),
                ("NOISE", "homogeneous white, not beamed"),
                ("NKSZ", 1, "kSZ not included"),
                ("PIXWIN", 0, "0=do NOT deconvolve HEALPix pixwin"),
                ("COMMENT", "CIB nside=8 tiles permuted; CMB/tSZ/noise unshuffled"),
            ],
            dtype=np.float32,
        )
        del tsz_m, cib, cib_shuf, noise, signal, signal_b, total
    if tsz_217 is not None:
        r_fid = pearson(tsz_217, cib_217)
        r_shuf = pearson(tsz_217, cib_shuf_217)
        lmax = 64
        tsz_lo = hp.ud_grade(tsz_217, 64)
        cl_fid = hp.anafast(tsz_lo, hp.ud_grade(cib_217, 64), lmax=lmax)
        cl_shuf = hp.anafast(tsz_lo, hp.ud_grade(cib_shuf_217, 64), lmax=lmax)
        print(
            f"  {SANITY_NU} GHz tSZ–CIB Pearson fid={r_fid:.4e} shuf={r_shuf:.4e}  "
            f"|r_shuf|/|r_fid|={abs(r_shuf)/max(abs(r_fid), 1e-30):.3f}"
        )
        print(
            f"  {SANITY_NU} GHz mean |C_ell| (l=2..{lmax}) fid={np.mean(np.abs(cl_fid[2:])):.4e} "
            f"shuf={np.mean(np.abs(cl_shuf[2:])):.4e}"
        )


def main() -> None:
    out = OUT_ROOT / SHUFFLE_TAG
    build_totals(out)
    print("Done.")


if __name__ == "__main__":
    main()
