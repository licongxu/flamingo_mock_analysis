"""Paths and geometry for the FLAMINGO × SZiFi pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flamingo_mock.config import BEAM_FWHM_ARCMIN, PLANCK_FREQUENCIES_GHZ

# Six HFI channels in SZiFi / Planck_simple order.
FREQS_GHZ: tuple[int, ...] = tuple(int(f) for f in PLANCK_FREQUENCIES_GHZ)

# SZiFi Planck flat-sky tile geometry (surveys/data_planck.py).
TILE_NSIDE = 8
TILE_NX = 1024
TILE_L_DEG = 14.8

DEFAULT_OUT_ROOT = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic/szifi")
DEFAULT_COMPONENTS = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/components"
)
DEFAULT_NOISE = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/planck_noise/npipe"
)
DEFAULT_MASKS = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/masks/pr4_nilc/Masks.fits"
)
# Existing total (coadd+beam+noise) maps — storage only, not ILC products.
DEFAULT_TOTAL_MAPS = Path(
    "/rds/rds-lxu/flamingo/integrated_maps_synthetic/ilc/inputs_nside2048_npipe"
)


def beam_fwhm_vec_arcmin(freqs: tuple[int, ...] = FREQS_GHZ) -> list[float]:
    """Table I Gaussian FWHMs in channel order (arcmin)."""
    return [float(BEAM_FWHM_ARCMIN[int(f)]) for f in freqs]


@dataclass
class SZiFiPaths:
    """On-disk roots for SZiFi prepare / run / catalogues."""

    out_root: Path = DEFAULT_OUT_ROOT
    components_dir: Path = DEFAULT_COMPONENTS
    noise_dir: Path = DEFAULT_NOISE
    masks_fits: Path = DEFAULT_MASKS
    total_maps_dir: Path = DEFAULT_TOTAL_MAPS
    nside: int = 2048

    def __post_init__(self) -> None:
        self.out_root = Path(self.out_root)
        self.components_dir = Path(self.components_dir)
        self.noise_dir = Path(self.noise_dir)
        self.masks_fits = Path(self.masks_fits)
        self.total_maps_dir = Path(self.total_maps_dir)

    def tiles_dir(self, split: str) -> Path:
        return self.out_root / "tiles" / f"split{split}"

    def coupling_dir(self, split: str) -> Path:
        return self.out_root / "coupling" / f"split{split}"

    def pilot_dir(self) -> Path:
        return self.out_root / "pilot"

    def catalogues_dir(self) -> Path:
        return self.out_root / "catalogues"

    def make_dirs(self, split: str = "A") -> None:
        for d in (
            self.tiles_dir(split),
            self.coupling_dir(split),
            self.pilot_dir(),
            self.catalogues_dir(),
        ):
            d.mkdir(parents=True, exist_ok=True)

    def total_map_path(self, split: str, freq_ghz: int) -> Path:
        """Path to a multi-frequency total sky map [K_CMB]."""
        return (
            self.total_maps_dir
            / f"sky_CMB_tSZ_kSZ_CIB_npipe_split{split}_{freq_ghz}GHz_nside{self.nside}_K.fits"
        )

    def tmap_path(self, split: str, field_id: int) -> Path:
        return self.tiles_dir(split) / f"flamingo_field_{field_id}_tmap.npy"

    def mask_path(self, split: str, field_id: int) -> Path:
        return self.tiles_dir(split) / f"flamingo_field_{field_id}_mask.npy"

    def coupling_path(self, split: str, field_id: int) -> Path:
        return self.coupling_dir(split) / f"apod_smooth_{field_id}.fits"
