"""Configuration for FLAMINGO synthetic map making.

Default paths follow the on-disk layout documented in ``data_description.md``:

* Input integrated maps: ``/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells``
  (FLAMINGO L2p8_m9, Yang et al. 2026, HEALPix Nside=4096 RING FITS).
* Output synthetic maps: ``/rds/rds-lxu/flamingo/integrated_maps_synthetic``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# COBE/FIRAS CMB temperature [K]. (Yang et al. use 2.73 K for their kSZ
# normalisation; we use one consistent value everywhere — the difference is
# 0.02 % and irrelevant for mocks.)
T_CMB = 2.7255

# Native resolution of the FLAMINGO integrated maps.
NSIDE_NATIVE = 4096

# FLAMINGO fiducial (D3A / DES Y3) cosmology, Yang et al. 2026 Table 1.
COSMOLOGY_D3A = dict(
    h=0.681,
    Om=0.306,
    Ob=0.0486,
    mnu=0.06,
    As=2.099e-9,
    ns=0.967,
)

# Fiducial three-parameter CIB SED (Yang et al. 2026, Section 3.5.1).
CIB_BETA_D = 1.65
CIB_T0 = 35.14  # K
CIB_ALPHA = 0.0
# Effective redshift used when scaling CIB intensity out of the released bands.
CIB_Z_EFF = 1.5

# Input files (relative to the lightcone directory). All are lensed, share the
# same shell rotations, Nside=4096 RING float32.
TSZ_FILE = "lensed_tSZ_rot.fits"  # Compton y, dimensionless
KSZ_FILE = "lensed_kSZ_rot.fits"  # Doppler b, dT/T_CMB = -b
KAPPA_FILE = "kappa_rot.fits"  # CMB lensing convergence
CIB_FILES = {  # CIB specific intensity [Jy/sr], bandpass-convolved
    217: "lensed_CIB_rot_BANDPASS_F217_three_params.fits",
    353: "lensed_CIB_rot_BANDPASS_F353_three_params.fits",
    545: "lensed_CIB_rot_BANDPASS_F545_three_params.fits",
    857: "lensed_CIB_rot_BANDPASS_F857_three_params.fits",
}

# Default channel set: SO-LAT-like 90/150/217 + Planck-HFI 353/545/857.
DEFAULT_FREQUENCIES_GHZ = (90.0, 150.0, 217.0, 353.0, 545.0, 857.0)

# Per-frequency Gaussian beams [arcmin FWHM] used in the compsep pipeline
# (data_description.md Section 7). Only applied when smoothing is requested.
BEAM_FWHM_ARCMIN = {
    90: 2.2,
    150: 1.4,
    217: 1.0,
    353: 4.5,
    545: 4.72,
    857: 4.42,
}


@dataclass
class MockConfig:
    """Runtime configuration for synthetic map making."""

    data_dir: Path = Path(
        "/rds/flamingo/L2800N5040/HYDRO_FIDUCIAL/lightcone0_shells"
    )
    out_dir: Path = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic")
    frequencies: tuple[float, ...] = DEFAULT_FREQUENCIES_GHZ
    nside: int = NSIDE_NATIVE  # set below 4096 to downgrade inputs (testing)
    t_cmb: float = T_CMB
    seed: int = 42
    z_eff: float = CIB_Z_EFF
    cosmology: dict = field(default_factory=lambda: dict(COSMOLOGY_D3A))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir)
        self.frequencies = tuple(float(f) for f in self.frequencies)
        if NSIDE_NATIVE % self.nside != 0:
            raise ValueError(f"nside must divide {NSIDE_NATIVE}, got {self.nside}")

    @property
    def lmax(self) -> int:
        return 3 * self.nside - 1

    @property
    def raw_dir(self) -> Path:
        """Directory for individual component maps."""
        return self.out_dir / "components"

    @property
    def coadd_dir(self) -> Path:
        """Directory for coadded per-frequency skies."""
        return self.out_dir / "coadd"

    def make_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.coadd_dir.mkdir(parents=True, exist_ok=True)
