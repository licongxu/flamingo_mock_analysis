"""Execute pyILC (NILC or HILC) from a YAML config.

Thin orchestration layer over :mod:`pyilc` — the ILC itself (including the
multi-backend weight solves: numpy / numba / JAX / CuPy) lives in the
installed ``pyilc`` package. Backend priority: ``--backend`` /
``PYILC_BACKEND`` env > ``ilc_backend`` in the YAML > ``auto`` (JAX on GPU
hosts).
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def run_ilc(yaml_config: str | Path, backend: str | None = None) -> Path:
    """Run pyILC from ``yaml_config``; return the path of the written y-map."""
    cfg = Path(yaml_config).expanduser().resolve()
    if not cfg.is_file():
        raise FileNotFoundError(cfg)

    if backend:
        os.environ["PYILC_BACKEND"] = backend

    from pyilc.input import ILCInfo
    from pyilc.wavelets import Wavelets, _ILC_map_filename, harmonic_ILC, wavelet_ILC

    t0 = time.time()
    print(f"[run_ilc] config={cfg}")
    print(f"[run_ilc] PYILC_BACKEND={os.environ.get('PYILC_BACKEND')}")

    info = ILCInfo(str(cfg))
    print(
        f"[run_ilc] wavelet_type={info.wavelet_type} "
        f"N_freqs={info.N_freqs} preserved={info.ILC_preserved_comp} "
        f"N_deproj={info.N_deproj} N_side={info.N_side} ELLMAX={info.ELLMAX} "
        f"ilc_backend={info.ilc_backend}"
    )
    print(f"[run_ilc] output_dir={info.output_dir}")
    os.makedirs(info.output_dir, exist_ok=True)

    info.read_bandpasses()
    info.read_beams()
    if getattr(info, "work_in_car", False):
        info.read_geometries()

    # Report mask wiring (GAL×PS product via mask_before_covariance / wavelet).
    cov_mask = getattr(info, "mask_before_covariance_computation", None)
    wav_mask = getattr(info, "mask_before_wavelet_computation", None)
    if cov_mask is not None:
        import numpy as np

        fsky = float(np.asarray(cov_mask).mean())
        print(f"[run_ilc] mask_before_covariance fsky={fsky:.4f}")
    else:
        print("[run_ilc] WARNING: no mask_before_covariance_computation set")
    if wav_mask is not None:
        print("[run_ilc] mask_before_wavelet_computation: set")
    else:
        print("[run_ilc] mask_before_wavelet_computation: None")

    wv = Wavelets(
        N_scales=info.N_scales,
        ELLMAX=info.ELLMAX,
        tol=1.0e-6,
        taper_width=info.taper_width,
    )
    if info.wavelet_type == "GaussianNeedlets":
        wv.GaussianNeedlets(FWHM_arcmin=info.GN_FWHM_arcmin)
    elif info.wavelet_type == "CosineNeedlets":
        wv.CosineNeedlets(ellmin=info.ellmin, ellpeaks=info.ellpeaks)
    elif info.wavelet_type == "TopHatHarmonic":
        wv.TopHatHarmonic(info.ellbins)
    elif info.wavelet_type == "TaperedTopHats":
        wv.TaperedTopHats(ellboundaries=info.ellboundaries, taperwidths=info.taperwidths)
    else:
        raise TypeError(f"unsupported wavelet type: {info.wavelet_type}")

    if info.wavelet_type == "TopHatHarmonic":
        info.read_maps()
        # HILC harmonic path ignores mask_before_covariance for C_ell; zero
        # masked pixels on the maps before map2alm so the cut is applied.
        if cov_mask is not None:
            import numpy as np

            m = np.asarray(cov_mask, dtype=np.float64)
            for i in range(len(info.maps)):
                info.maps[i] = np.asarray(info.maps[i], dtype=np.float64) * m
            print("[run_ilc] applied GAL×PS mask to HILC input maps before SHTs")
        # healpix path: pix_size (arcmin) is only set by read_geometries for CAR.
        if not hasattr(info, "pix_size") or info.pix_size is None:
            import healpy as hp

            info.pix_size = float(hp.nside2resol(info.N_side, arcmin=True))
        if not info.weights_exist:
            info.maps2alms()
            info.alms2cls()
        # pyILC main.py checks info.maps_to_apply_weights, but ILCInfo only
        # sets freq_map_files_for_weights when YAML has maps_to_apply_weights.
        if getattr(info, "freq_map_files_for_weights", None) is not None:
            info.maps_to_apply_weights2alms()
        harmonic_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)
    else:
        # NILC: pyILC applies mask_before_wavelet and mask_before_covariance.
        wavelet_ILC(wv, info, resp_tol=info.resp_tol, map_images=False)

    ypath = Path(_ILC_map_filename(info))
    dt = time.time() - t0
    print(f"[run_ilc] finished in {dt:.1f}s")
    print(f"[run_ilc] y_map={ypath}")
    if not ypath.is_file():
        raise RuntimeError(f"expected y-map missing: {ypath}")
    print(f"[run_ilc] y_map_bytes={ypath.stat().st_size}")
    return ypath
