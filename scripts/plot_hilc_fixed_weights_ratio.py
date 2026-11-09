#!/usr/bin/env python3
"""Fixed fiducial CIB+δβ weights on M*-1σ maps (and the reverse)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from pub_style import apply_pub_style, savefig
from hilc_prescriptions import ILC

ROOT = Path(__file__).resolve().parents[1]
PS = ILC / "ps_namaster"
FIG_DIR = ROOT / "figures" / "l1_m9"
STEM = "l1_m9_hilc_feedback_ratio_fixed_weights"


def _p18(run: str, deproj: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(PS / f"{run}_{deproj}_masked_p18.npz")
    return np.asarray(z["ell"]), np.asarray(z["dl"])


def main() -> None:
    apply_pub_style()
    z = np.load(PS / "fixedw_inputs_deproj_cib_dbeta_masked.npz")
    ell = np.asarray(z["ell"])
    fid = np.asarray(z["fid_fidw_dl"])
    own = np.asarray(z["mstar_mstarw_dl"]) / np.maximum(fid, 1e-30)
    fix = np.asarray(z["mstar_fidw_dl"]) / np.maximum(fid, 1e-30)
    swap = np.asarray(z["fid_mstarw_dl"]) / np.maximum(fid, 1e-30)
    ell_p, dl_fid = _p18("L1_m9", "deproj_cib_dbeta")
    _, dl_ms = _p18("Mstar-1sigma", "deproj_cib_dbeta")
    _, dl_none_f = _p18("L1_m9", "nodeproj")
    _, dl_none_m = _p18("Mstar-1sigma", "nodeproj")
    fig, ax = plt.subplots(figsize=(4.6, 3.6), layout="constrained")
    ax.axhline(1.0, color="k", lw=1.0)
    ax.semilogx(
        ell_p, dl_none_m / dl_none_f, color="#0055A4", marker="o", ms=4, lw=1.8,
        label=r"$q>5$, no deproj.",
    )
    ax.semilogx(
        ell_p, dl_ms / dl_fid, color="#6A1B9A", marker="^", ms=4.5, lw=1.8, ls="--",
        label=r"own weights, CIB+$\delta\beta$",
    )
    ax.semilogx(
        ell, own, color="#6A1B9A", marker="^", ms=4.5, lw=1.0, ls=":", alpha=0.7,
        label=r"own weights (fid.\ mask)",
    )
    ax.semilogx(
        ell, fix, color="#C44E00", marker="s", ms=4, lw=1.8,
        label=r"$M_\ast$ maps, fid.\ $\delta\beta$ weights",
    )
    ax.semilogx(
        ell, swap, color="#00796B", marker="v", ms=4, lw=1.6, ls="-.",
        label=r"fid.\ maps, $M_\ast$ $\delta\beta$ weights",
    )
    ax.set_xlim(10.0, 959.5)
    ax.set_ylim(0.55, 1.35)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell / D_\ell^{\mathrm{fid},\,\delta\beta}$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(False)
    savefig(fig, STEM, fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"ell10 own={own[0]:.3f} fid-w={fix[0]:.3f} mstar-w-on-fid={swap[0]:.3f}")


if __name__ == "__main__":
    main()
