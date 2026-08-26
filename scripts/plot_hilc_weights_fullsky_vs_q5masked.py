"""Compare HILC tSZ weights: full-sky vs q>5 cluster-masked (same homog maps)."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from flamingo_mock.config import BEAM_FWHM_ARCMIN

LMAX = 4096
ELL_MAX_PLOT = 1500
BINSIZE, BEAM_CRIT = 50, 1.0e-3
FREQS = (100, 143, 353, 217, 545, 857)
WDIR_FULL = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog")
WDIR_Q5 = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/hilc_output_homog_q5masked")
FIG = Path("/scratch/scratch-lxu/flamingo_mock_analysis/figures/hilc_homog_weights_fullsky_vs_q5masked.png")


def hilc_weights(wdir: Path, lmax: int) -> np.ndarray:
    fwhm_ch = [BEAM_FWHM_ARCMIN[int(f)] for f in FREQS]
    ellbins = np.arange(0, lmax + 1, BINSIZE)
    n_scales = len(ellbins) - 1
    filts = np.zeros((n_scales, lmax + 1))
    for i in range(n_scales - 1):
        filts[i, ellbins[i] : ellbins[i + 1]] = 1.0
    filts[-1, ellbins[-1] :] = 1.0
    inp_beams = [hp.gauss_beam(np.deg2rad(f / 60.0), lmax=lmax) for f in fwhm_ch]
    ell_B = np.array([int(np.argmin(np.abs(b - BEAM_CRIT))) for b in inp_beams])
    ell_F = np.zeros(n_scales)
    for i in range(n_scales - 1):
        peak = int(np.argmax(filts[i]))
        ell_F[i] = min(lmax, peak + int(np.argmin(np.abs(filts[i][peak:] - BEAM_CRIT))))
    ell_F[-1] = ell_F[-2]
    w_ell = np.zeros((len(FREQS), lmax + 1))
    for j in range(n_scales):
        use = [ell_F[j] <= ell_B[a] for a in range(len(FREQS))]
        wraw = np.atleast_1d(
            np.loadtxt(wdir / f"flamingo_weightvector_scale{j}_component_tSZ.txt")
        ).ravel()
        sl = filts[j] > 0
        count = 0
        for a, ok in enumerate(use):
            if not ok:
                continue
            w_ell[a, sl] = wraw[count]
            count += 1
    return w_ell


def main() -> None:
    w_full = hilc_weights(WDIR_FULL, LMAX)
    w_q5 = hilc_weights(WDIR_Q5, LMAX)
    ell = np.arange(LMAX + 1)
    dw = w_q5 - w_full
    print("max |dw| by channel:")
    for i, f in enumerate(FREQS):
        m = (ell >= 2) & (ell <= ELL_MAX_PLOT)
        abs_full = np.max(np.abs(w_full[i, m]))
        abs_dw = np.max(np.abs(dw[i, m]))
        print(f"  {f:3d} GHz  max|w|={abs_full:.4g}  max|dw|={abs_dw:.4g}  rel={abs_dw/max(abs_full,1e-30):.3f}")

    fig, axes = plt.subplots(3, 2, figsize=(9.2, 8.4), sharex=True)
    for ax, i, f in zip(axes.ravel(), range(len(FREQS)), FREQS):
        ax.plot(ell, w_full[i], color="0.35", lw=1.6, label="full sky")
        ax.plot(ell, w_q5[i], color="C0", lw=1.4, ls="--", label=r"$q>5$ masked")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlim(2, ELL_MAX_PLOT)
        ax.set_ylabel(rf"$w_{{{f}}}$")
        ax.set_title(f"{f} GHz")
        if i == 0:
            ax.legend(frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\ell$")
    fig.suptitle(r"HILC tSZ weights: full sky vs $q>5$ cluster mask", y=1.01)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, f in enumerate(FREQS):
        ax.plot(ell, dw[i], lw=1.3, label=f"{f} GHz")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_xlim(2, ELL_MAX_PLOT)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$w^{q>5}-w^{\mathrm{full}}$")
    ax.set_title(r"HILC weight difference after $q>5$ mask")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    out2 = FIG.with_name("hilc_homog_weights_q5masked_minus_fullsky.png")
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
