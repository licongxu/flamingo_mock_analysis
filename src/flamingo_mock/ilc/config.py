"""pyILC YAML config generation for FLAMINGO mock ILC runs.

Produces the YAML consumed by :class:`pyilc.input.ILCInfo` for the two
methods used in McCarthy & Hill (2024):

* ``hilc`` — harmonic ILC (``TopHatHarmonic`` needlets, fast; primary).
* ``nilc`` — needlet ILC (``GaussianNeedlets``; paper-style, heavier).

The ILC weight-solve backend defaults to JAX (GPU); override with the
``ilc_backend`` field, the ``--backend`` CLI flag, or ``PYILC_BACKEND``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .paths import (
    ILC_BEAM_FWHM_ARCMIN,
    ILC_FREQUENCIES_GHZ,
    NPIPE_MC_DEFAULT,
    ILCPaths,
    channel_beams_arcmin,
)

# Gaussian needlet FWHMs [arcmin] for the NILC (McCarthy & Hill 2024 style).
GN_FWHM_ARCMIN = [600.0, 300.0, 120.0, 60.0, 30.0, 15.0, 10.0, 7.5, 5.0]

# HILC harmonic-bin width.
HILC_BIN_SIZE = 50

METHODS = ("hilc", "nilc")


def pyilc_param_dict_file() -> Path:
    """``fg_SEDs_default_params.yml`` shipped with the installed pyILC."""
    import pyilc

    path = Path(pyilc.__file__).resolve().parent.parent / "input" / "fg_SEDs_default_params.yml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass
class ILCConfig:
    """One pyILC run: a method (hilc/nilc) on one noise split."""

    method: str = "hilc"
    split: str = "A"
    paths: ILCPaths = field(default_factory=ILCPaths)
    freqs: tuple[int, ...] = ILC_FREQUENCIES_GHZ
    ellmax: int = 3000
    taper_width: int = 150
    ilc_beam_fwhm_arcmin: float = ILC_BEAM_FWHM_ARCMIN
    preserved_comp: str = "tSZ"
    ilc_backend: str = "jax"
    mc: int = NPIPE_MC_DEFAULT

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir(self.method, self.split)

    @property
    def output_suffix(self) -> str:
        return f"_{self.method}_y_npipe_split{self.split}"

    def to_dict(self) -> dict:
        """pyILC YAML keys (mirrors pyilc/input.py parsing)."""
        beams = channel_beams_arcmin(self.freqs)
        d: dict = {
            "work_in_healpix": "yes",
            "output_dir": str(self.output_dir) + "/",
            "output_prefix": "flamingo_",
            "output_suffix": self.output_suffix,
            "save_weights": "yes",
            "save_as": "fits",
            "param_dict_file": str(pyilc_param_dict_file()),
            "ELLMAX": self.ellmax,
            "taper_width": self.taper_width,
            "N_side": self.paths.nside,
            "N_freqs": len(self.freqs),
            "bandpass_type": "DeltaBandpasses",
            "freqs_delta_ghz": [float(f) for f in self.freqs],
            "freq_map_files": [
                str(self.paths.split_map(f, self.split)) for f in self.freqs
            ],
            "beam_type": "Gaussians",
            # Must match the channel beams applied by ilc.prepare.
            "beam_FWHM_arcmin": [beams[f] for f in self.freqs],
            # Common resolution for the ILC; validation deconvolves this B_ell.
            "perform_ILC_at_beam": self.ilc_beam_fwhm_arcmin,
            "ILC_preserved_comp": self.preserved_comp,
            "N_deproj": 0,
            "ILC_deproj_comps": [],
            "N_maps_xcorr": 0,
            "print_timing": "true",
            "ilc_backend": self.ilc_backend,
        }
        if self.method == "hilc":
            d["wavelet_type"] = "TopHatHarmonic"
            d["BinSize"] = HILC_BIN_SIZE
        else:
            d["wavelet_type"] = "GaussianNeedlets"
            d["N_scales"] = len(GN_FWHM_ARCMIN) + 1
            d["GN_FWHM_arcmin"] = GN_FWHM_ARCMIN
            d["N_SED_params"] = 0
            d["SED_params"] = []
            d["SED_params_vals"] = []
            d["SED_params_priors"] = []
            d["SED_params_priors_params"] = []
            d["ILC_bias_tol"] = 0.01
        return d

    def write(self, path: Path) -> Path:
        """Write the pyILC YAML config to ``path``."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# pyILC {self.method.upper()} Compton-y on FLAMINGO mock skies "
            f"+ Planck NPIPE noise (split {self.split}, mc_{self.mc:05d}).\n"
            f"# ELLMAX={self.ellmax}, N_side={self.paths.nside}. Channel beams = Planck HFI\n"
            f"# Gaussians applied by flamingo_mock.ilc.prepare; common ILC beam "
            f"{self.ilc_beam_fwhm_arcmin:g}'.\n"
            f"# ILC weight backend: {self.ilc_backend} (override with PYILC_BACKEND).\n"
        )
        with open(path, "w") as f:
            f.write(header)
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=None)
        print(f"wrote {path}")
        return path


def write_config_set(
    out_dir: Path,
    *,
    methods: tuple[str, ...] = METHODS,
    splits: tuple[str, ...] = ("A", "B"),
    paths: ILCPaths | None = None,
    ilc_backend: str = "jax",
) -> list[Path]:
    """Write YAMLs for each method/split combination; return the paths.

    NILC is only written for the first split (it is expensive; the HILC
    split pair is what the cross-split validation uses).
    """
    paths = paths or ILCPaths()
    written = []
    for method in methods:
        for split in splits:
            if method == "nilc" and split != splits[0]:
                continue
            cfg = ILCConfig(
                method=method, split=split, paths=paths, ilc_backend=ilc_backend
            )
            written.append(
                cfg.write(Path(out_dir) / f"{method}_y_flamingo_npipe_split{split}.yml")
            )
    return written
