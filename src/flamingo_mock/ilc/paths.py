"""Filesystem layout for the FLAMINGO ILC pipeline.

Defaults follow the on-disk layout on this machine (see
``noise_description.md`` and ``data_description.md``):

* Synthetic components: ``<synth_root>/components/{cmb,tsz,ksz,cib}``
  (µK_CMB, Nside=4096), produced by the ``flamingo-mock-maps`` package.
* Planck NPIPE noise: ``<synth_root>/planck_noise/npipe/<freq>GHz/{A,B,full}``
  (K_CMB for 100–353 GHz; MJy/sr for 545/857 → converted to K_CMB on load).
* ILC working products: ``<synth_root>/ilc``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import BEAM_FWHM_ARCMIN, NSIDE_NATIVE, MockConfig

# All six Planck HFI channels (McCarthy & Hill style tSZ ILC uses HFI; we
# omit LFI 30/44/70 which are not in the FLAMINGO synthetic sky model).
# Ordered lowest-res → highest-res (paper convention on beam FWHM).
ILC_FREQUENCIES_GHZ = (100, 143, 353, 217, 545, 857)

# NPIPE detector-set splits; A and B are processed independently, so their
# cross-spectrum is noise-decoupled.
NPIPE_SPLITS = ("A", "B")

# Default NPIPE Monte-Carlo realisation on disk (mc_00200).
NPIPE_MC_DEFAULT = 200

# Common ILC beam [arcmin] — McCarthy & Hill (2024) sample YAMLs use 10'.
# Validation deconvolves this beam from the reported spectra.
ILC_BEAM_FWHM_ARCMIN = 10.0

# pyILC multipole ceiling matching sample_run_Planck_tsz_*.yml.
ILC_ELLMAX = 4096

# Planck PR4 NILC mask archive (Masks.fits fields: NILC-MASK, GAL-MASK, PS-MASK).
# Pipeline uses the product of GAL-MASK × PS-MASK for realistic sky cuts.
MASK_FITS_REL = Path("masks") / "pr4_nilc" / "Masks.fits"
MASK_GAL_FIELD = 1  # GAL-MASK
MASK_PS_FIELD = 2  # PS-MASK
# Combined product written under ilc/ for pyILC's single-field mask API.
COMBINED_MASK_NAME = "gal_ps_mask_nside2048.fits"


@dataclass
class ILCPaths:
    """Locations of ILC inputs and outputs."""

    synth_root: Path = MockConfig.out_dir
    ilc_root: Path | None = None  # default: synth_root / "ilc"
    nside: int = 2048
    seed: int = 42

    def __post_init__(self) -> None:
        self.synth_root = Path(self.synth_root)
        self.ilc_root = Path(self.ilc_root) if self.ilc_root else self.synth_root / "ilc"

    # -- upstream products -------------------------------------------------
    @property
    def components_dir(self) -> Path:
        return self.synth_root / "components"

    @property
    def noise_dir(self) -> Path:
        return self.synth_root / "planck_noise" / "npipe"

    @property
    def masks_dir(self) -> Path:
        return self.synth_root / "masks"

    def planck_masks_fits(self) -> Path:
        """PR4 NILC multi-field mask FITS (NILC / GAL / PS)."""
        return self.synth_root / MASK_FITS_REL

    def combined_gal_ps_mask(self) -> Path:
        """Binary GAL×PS product used by pyILC and validation."""
        return self.ilc_root / COMBINED_MASK_NAME

    def cmb_map(self) -> Path:
        return (
            self.components_dir
            / "cmb"
            / f"primary_CMB_T_lensed_nside{NSIDE_NATIVE}_seed{self.seed}.fits"
        )

    def tsz_deltaT_map(self, freq_ghz: int) -> Path:
        return (
            self.components_dir
            / "tsz"
            / f"tSZ_deltaT_{freq_ghz}GHz_nside{NSIDE_NATIVE}.fits"
        )

    def ksz_deltaT_map(self) -> Path:
        return self.components_dir / "ksz" / f"kSZ_deltaT_nside{NSIDE_NATIVE}.fits"

    def cib_deltaT_map(self, freq_ghz: int) -> Path:
        return (
            self.components_dir
            / "cib"
            / f"CIB_deltaT_{freq_ghz}GHz_nside{NSIDE_NATIVE}.fits"
        )

    def compton_y_map(self) -> Path:
        return self.components_dir / "tsz" / f"compton_y_nside{NSIDE_NATIVE}.fits"

    # -- ILC products --------------------------------------------------------
    @property
    def inputs_dir(self) -> Path:
        return self.ilc_root / f"inputs_nside{self.nside}_npipe"

    def signal_map(self, freq_ghz: int) -> Path:
        return (
            self.inputs_dir
            / f"sky_CMB_tSZ_kSZ_CIB_signal_{freq_ghz}GHz_nside{self.nside}_K.fits"
        )

    def split_map(self, freq_ghz: int, split: str) -> Path:
        return (
            self.inputs_dir
            / f"sky_CMB_tSZ_kSZ_CIB_npipe_split{split}_{freq_ghz}GHz_nside{self.nside}_K.fits"
        )

    def truth_map(self) -> Path:
        return self.inputs_dir / f"compton_y_nside{self.nside}.fits"

    def output_dir(self, method: str, split: str) -> Path:
        return self.ilc_root / f"{method}_output_npipe_split{split}"


def channel_beams_arcmin(freqs: tuple[int, ...] = ILC_FREQUENCIES_GHZ) -> dict[int, float]:
    """Planck HFI Gaussian beam FWHMs [arcmin] for the ILC channels."""
    return {f: BEAM_FWHM_ARCMIN[f] for f in freqs}
