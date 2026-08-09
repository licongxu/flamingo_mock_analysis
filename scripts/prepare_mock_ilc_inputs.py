#!/usr/bin/env python3
"""Build FLAMINGO mock multi-frequency ILC inputs with Planck FFP10 noise splits.

Pipeline per frequency channel:
  1. Load CMB+tSZ+CIB coadd in K_CMB (nside=4096).
  2. Optionally smooth with a Gaussian beam matching Planck HFI FWHM.
  3. Ud_grade to working nside.
  4. Add FFP10 noise realization (mc_00000 / mc_00001), already in K_CMB.
  5. Write per-split sky maps used by pyILC.

Also writes a beam-free truth Compton-y at the working nside.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np

# Planck HFI approximate FWHMs (arcmin) used in McCarthy & Hill pyILC samples
DEFAULT_BEAMS = {100: 9.66, 143: 7.22, 353: 4.92}
FREQS = (100, 143, 353)
SPLITS = (0, 1)


def _noise_path(noise_dir: Path, freq: int, split: int) -> Path:
    return noise_dir / f"ffp10_noise_{freq}_full_map_mc_{split:05d}.fits"


def build_split_maps(
    *,
    nside: int,
    src_coadd: Path,
    noise_dir: Path,
    dst: Path,
    beams: dict[int, float],
    apply_channel_beams: bool,
    ellmax_smooth: int | None,
    overwrite: bool,
) -> dict:
    from beam_utils import apply_beam_to_map

    dst.mkdir(parents=True, exist_ok=True)
    meta: dict = {
        "nside": nside,
        "apply_channel_beams": apply_channel_beams,
        "beams_fwhm_arcmin": beams,
        "units": "K_CMB",
        "freqs_ghz": list(FREQS),
        "splits": list(SPLITS),
        "files": {},
        "noise_residual_std": {},
    }

    for freq in FREQS:
        coadd_path = src_coadd / f"sky_CMB_tSZ_CIB_{freq}GHz_nside4096_K.fits"
        if not coadd_path.is_file():
            raise FileNotFoundError(coadd_path)
        signal = np.asarray(hp.read_map(str(coadd_path), dtype=np.float64))
        if apply_channel_beams:
            fwhm = float(beams[freq])
            lmax = ellmax_smooth if ellmax_smooth is not None else min(3 * 4096 - 1, 6000)
            print(f"[{freq} GHz] beam-smoothing FWHM={fwhm}' ...")
            signal = apply_beam_to_map(signal, fwhm, lmax=lmax)
        if hp.npix2nside(len(signal)) != nside:
            signal = hp.ud_grade(signal, nside)

        for split in SPLITS:
            npath = _noise_path(noise_dir, freq, split)
            if not npath.is_file():
                raise FileNotFoundError(npath)
            noise = np.asarray(hp.read_map(str(npath), field=0, dtype=np.float64))
            if hp.npix2nside(len(noise)) != nside:
                noise = hp.ud_grade(noise, nside)
            sky = signal + noise
            out = dst / f"sky_CMB_tSZ_CIB_noise_split{split}_{freq}GHz_nside{nside}_K.fits"
            if out.exists() and out.stat().st_size > 1_000_000 and not overwrite:
                print(f"exists {out}")
            else:
                hp.write_map(str(out), sky, overwrite=True, dtype=np.float64)
                print(f"wrote {out} std={sky.std():.4e} (sig={signal.std():.4e} noise={noise.std():.4e})")
            meta["files"][f"split{split}_{freq}"] = str(out)
            meta["noise_residual_std"][f"split{split}_{freq}"] = float(noise.std())

            # signal-only companion (for noise residual checks)
            sig_out = dst / f"sky_CMB_tSZ_CIB_signal_{freq}GHz_nside{nside}_K.fits"
            if not sig_out.exists() or overwrite:
                hp.write_map(str(sig_out), signal, overwrite=True, dtype=np.float64)
            meta["files"][f"signal_{freq}"] = str(sig_out)

    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nside", type=int, default=2048)
    p.add_argument(
        "--src-coadd",
        type=str,
        default="/home/ext_andyxlcnb_gmail_com/flamingo_mock_analysis/maps_100_143_353/coadd",
    )
    p.add_argument(
        "--src-truth",
        type=str,
        default="/home/ext_andyxlcnb_gmail_com/flamingo_mock_analysis/maps_100_143_353/raw/compton_y_nside4096.fits",
    )
    p.add_argument(
        "--noise-dir",
        type=str,
        default="/home/ext_andyxlcnb_gmail_com/cosmology_data/planck_noise",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="/home/ext_andyxlcnb_gmail_com/cosmology_data/flamingo_ilc/inputs_nside2048_noise",
    )
    p.add_argument(
        "--no-channel-beams",
        action="store_true",
        help="Do not smooth coadds with Planck HFI Gaussians (not recommended)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--ellmax-smooth", type=int, default=6000)
    args = p.parse_args(argv)

    # allow `python scripts/prepare_...` and `%run -i` from repo root
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    dst = Path(args.dst)
    meta = build_split_maps(
        nside=args.nside,
        src_coadd=Path(args.src_coadd),
        noise_dir=Path(args.noise_dir),
        dst=dst,
        beams=DEFAULT_BEAMS,
        apply_channel_beams=not args.no_channel_beams,
        ellmax_smooth=args.ellmax_smooth,
        overwrite=args.overwrite,
    )

    truth_src = Path(args.src_truth)
    truth_out = dst / f"compton_y_nside{args.nside}.fits"
    if not truth_out.exists() or args.overwrite:
        yt = hp.ud_grade(hp.read_map(str(truth_src), dtype=np.float64), args.nside)
        hp.write_map(str(truth_out), yt, overwrite=True, dtype=np.float64)
        print(f"wrote {truth_out} median|y|={np.median(np.abs(yt)):.4e}")
    else:
        print(f"exists {truth_out}")
    meta["truth"] = str(truth_out)

    meta_path = dst / "input_manifest.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
