"""Build pyILC input maps: beam-smoothed coadded skies + NPIPE noise splits.

Pipeline per frequency channel:

1. Coadd lensed CMB + tSZ + kSZ + CIB component maps (µK_CMB, Nside=4096)
   and convert to K_CMB.
2. Smooth with the Planck HFI Gaussian beam for the channel.
3. Downgrade to the working Nside (2048, the native NPIPE resolution).
4. Add NPIPE detector-set noise splits A/B (already K_CMB at Nside=2048).
5. Write per-split sky maps, the signal-only companion, and the beam-free
   truth Compton-y at the working Nside.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .beams import apply_beam_to_map
from .noise import load_noise_map
from .paths import (
    ILC_FREQUENCIES_GHZ,
    NPIPE_MC_DEFAULT,
    NPIPE_SPLITS,
    ILCPaths,
    channel_beams_arcmin,
)

UK_TO_K = 1e-6


def load_coadd_K(paths: ILCPaths, freq_ghz: int) -> np.ndarray:
    """Coadded signal sky (CMB + tSZ + kSZ + CIB) at one frequency [K_CMB]."""
    import healpy as hp

    def read(p: Path) -> np.ndarray:
        if not p.is_file():
            raise FileNotFoundError(p)
        return np.asarray(hp.read_map(str(p), dtype=np.float64))

    coadd_uK = (
        read(paths.cmb_map())
        + read(paths.tsz_deltaT_map(freq_ghz))
        + read(paths.ksz_deltaT_map())
        + read(paths.cib_deltaT_map(freq_ghz))
    )
    return coadd_uK * UK_TO_K


def build_split_maps(
    paths: ILCPaths,
    *,
    freqs: tuple[int, ...] = ILC_FREQUENCIES_GHZ,
    splits: tuple[str, ...] = NPIPE_SPLITS,
    mc: int = NPIPE_MC_DEFAULT,
    ellmax_smooth: int = 6000,
    overwrite: bool = False,
) -> dict:
    """Write noise-added split skies and the truth y-map; return a manifest."""
    import healpy as hp

    nside = paths.nside
    beams = channel_beams_arcmin(freqs)
    paths.inputs_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "nside": nside,
        "units": "K_CMB",
        "components": "CMB + tSZ + kSZ + CIB",
        "beams_fwhm_arcmin": beams,
        "noise": f"NPIPE mc_{mc:05d} detector-set splits {list(splits)}",
        "freqs_ghz": list(freqs),
        "splits": list(splits),
        "files": {},
        "noise_residual_std": {},
    }

    for freq in freqs:
        print(f"[{freq} GHz] coadding components ...")
        signal = load_coadd_K(paths, freq)
        print(f"[{freq} GHz] beam-smoothing FWHM={beams[freq]}' ...")
        signal = apply_beam_to_map(signal, beams[freq], lmax=ellmax_smooth)
        if hp.npix2nside(signal.size) != nside:
            signal = hp.ud_grade(signal, nside)

        sig_out = paths.signal_map(freq)
        if not sig_out.exists() or overwrite:
            hp.write_map(str(sig_out), signal, overwrite=True, dtype=np.float64)
            print(f"  wrote {sig_out}")
        meta["files"][f"signal_{freq}"] = str(sig_out)

        for split in splits:
            noise = load_noise_map(paths.noise_dir, freq, split, mc, nside_out=nside)
            sky = signal + noise
            out = paths.split_map(freq, split)
            if out.exists() and out.stat().st_size > 1_000_000 and not overwrite:
                print(f"  exists {out}")
            else:
                hp.write_map(str(out), sky, overwrite=True, dtype=np.float64)
                print(
                    f"  wrote {out} std={sky.std():.4e} "
                    f"(sig={signal.std():.4e} noise={noise.std():.4e})"
                )
            meta["files"][f"split{split}_{freq}"] = str(out)
            meta["noise_residual_std"][f"split{split}_{freq}"] = float(noise.std())

    truth_out = paths.truth_map()
    if not truth_out.exists() or overwrite:
        yt = np.asarray(hp.read_map(str(paths.compton_y_map()), dtype=np.float64))
        yt = hp.ud_grade(yt, nside)
        hp.write_map(str(truth_out), yt, overwrite=True, dtype=np.float64)
        print(f"wrote {truth_out} median|y|={np.median(np.abs(yt)):.4e}")
    else:
        print(f"exists {truth_out}")
    meta["truth"] = str(truth_out)

    meta_path = paths.inputs_dir / "input_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")
    return meta
