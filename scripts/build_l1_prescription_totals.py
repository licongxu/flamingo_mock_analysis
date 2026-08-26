#!/usr/bin/env python3
"""tSZ ΔT + beamed CMB+tSZ+CIB+homog-noise totals for every L1_m9 prescription.

Writes
  components/tsz/{L1_m9,fgas-8sigma,Mstar-1sigma,LS8}/
  total_maps/{L1_m9,fgas-8sigma,Mstar-1sigma,LS8}/sky_CMB_tSZ_CIB_homog_*GHz_nside2048_uK.fits

Does not touch total_maps/test/ (L2p8 demo).
"""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np

from flamingo_mock import tsz
from flamingo_mock.config import BEAM_FWHM_ARCMIN, PLANCK_FREQUENCIES_GHZ, MockConfig
from flamingo_mock.io import write_map

COMP = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/components")
NOISE = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/homogeneous")
OUT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/total_maps")
PRESCRIPTIONS = ("L1_m9", "fgas-8sigma", "Mstar-1sigma", "LS8")
NSIDE = 2048
NSIDE_IN = 4096
FREQS = tuple(int(f) for f in PLANCK_FREQUENCIES_GHZ)
TSZ_NAMES = [
    "compton_y_nside4096.fits",
    *[f"tSZ_deltaT_{nu}GHz_nside4096.fits" for nu in FREQS],
]
TOTAL_NAMES = [f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits" for nu in FREQS]


def _park_unlabeled(src_dir: Path, dest_dir: Path, names: list[str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if not src.exists() and not src.is_symlink():
            continue
        target = dest_dir / name
        if target.exists() or target.is_symlink():
            if src.resolve() == target.resolve():
                continue
            print(f"skip park {src} (already in {dest_dir.name}/)")
            continue
        src.rename(target)
        print(f"moved {src_dir.name}/{name} -> {src_dir.name}/{dest_dir.name}/")


def _tsz_complete(run: str) -> bool:
    d = COMP / "tsz" / run
    return all((d / n).is_file() for n in TSZ_NAMES)


def _total_complete(run: str) -> bool:
    d = OUT / run
    return all((d / n).is_file() for n in TOTAL_NAMES)


def build_tsz(run: str) -> None:
    cfg = MockConfig(
        data_dir=COMP / "L1_m9" / run,
        frequencies=PLANCK_FREQUENCIES_GHZ,
        nside=NSIDE_IN,
        tsz_file="lensed_tSZ_rot_same_rot.hdf5",
    )
    print(f"\n=== tSZ {run} ===")
    tsz.make_tsz_maps(cfg, out_dir=COMP / "tsz" / run)


def load_uK(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))


def to_nside(m: np.ndarray, nside: int = NSIDE) -> np.ndarray:
    if hp.get_nside(m) == nside:
        return m
    return hp.ud_grade(m, nside)


def cmb_path(run: str) -> Path:
    tag = "_LS8" if run == "LS8" else ""
    return COMP / "cmb" / f"primary_CMB_T_lensed_nside4096_seed42{tag}.fits"


def build_totals(run: str) -> None:
    out = OUT / run
    out.mkdir(parents=True, exist_ok=True)
    cmb_p = cmb_path(run)
    cib_dir = COMP / "cib" / run
    tsz_dir = COMP / "tsz" / run
    print(f"\n=== totals {run} ===")
    print("CMB:", cmb_p)
    print("CIB:", cib_dir)
    print("tSZ:", tsz_dir)
    cmb = to_nside(load_uK(cmb_p))
    print(f"  CMB nside={hp.get_nside(cmb)}  rms={cmb.std():.2f} uK")
    for nu in FREQS:
        tsz_m = to_nside(load_uK(tsz_dir / f"tSZ_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        cib = to_nside(load_uK(cib_dir / f"CIB_deltaT_{nu}GHz_nside{NSIDE_IN}.fits"))
        noise = load_uK(NOISE / f"{nu}GHz" / f"white_noise_{nu}GHz_nside{NSIDE}_uK.fits")
        signal = cmb + tsz_m + cib
        fwhm = float(BEAM_FWHM_ARCMIN[nu])
        print(f"  {nu} GHz FWHM={fwhm:.2f}' ...", flush=True)
        signal_b = hp.smoothing(signal, fwhm=np.radians(fwhm / 60.0))
        total = signal_b + noise
        print(
            f"    rms CMB={cmb.std():.2f} tSZ={tsz_m.std():.2f} CIB={cib.std():.2f} "
            f"beamed={signal_b.std():.2f} noise={noise.std():.2f} total={total.std():.2f}"
        )
        write_map(
            out / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside{NSIDE}_uK.fits",
            total,
            unit="uK_CMB",
            freq=float(nu),
            extra=[
                ("COMPS", "CMB+tSZ+CIB+homog_noise"),
                ("PRESCRIP", run),
                ("FWHM", fwhm, "Gaussian beam [arcmin] on signal only"),
                ("BEAMON", 1),
                ("NOISE", "homogeneous white, not beamed"),
                ("NKSZ", 1, "kSZ not included"),
                ("PIXWIN", 0, "0=do NOT deconvolve HEALPix pixwin"),
                ("COMMENT", "Planck-like mock: beam on signal only; no pixwin inverse"),
            ],
            dtype=np.float32,
        )
        del tsz_m, cib, noise, signal, signal_b, total


def main() -> None:
    _park_unlabeled(COMP / "tsz", COMP / "tsz" / "L1_m9", TSZ_NAMES)
    _park_unlabeled(OUT, OUT / "L1_m9", TOTAL_NAMES)
    for run in PRESCRIPTIONS:
        if not _tsz_complete(run):
            build_tsz(run)
        else:
            print(f"{run}: tSZ already present")
        if not _total_complete(run):
            build_totals(run)
        else:
            print(f"{run}: totals already present")
    print("Done.")


if __name__ == "__main__":
    main()
