"""On-disk HILC paths for L1_m9 feedback / cosmology variants.

L1_m9 keeps the original unlabeled homog folders. The baryonic and LS8
runs get tagged input/output dirs so they cannot overwrite the fiducial.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SYNTH = Path("/rds/rds-lxu/flamingo/integrated_maps_synthetic")
ILC = SYNTH / "ilc"
REPO = Path("/scratch/scratch-lxu/flamingo_mock_analysis")
# pyILC channel order (353 before 217), same as hilc_y_flamingo_homog.yml
FREQS_ILC = (100, 143, 353, 217, 545, 857)
PRESCRIPTIONS = ("fgas-8sigma", "Mstar-1sigma", "LS8")
ALL_RUNS = ("L1_m9",) + PRESCRIPTIONS
LABELS = {
    "L1_m9": r"L1\_m9 fiducial",
    "fgas-8sigma": r"$f_{\mathrm{gas}}-8\sigma$",
    "Mstar-1sigma": r"$M_\ast-1\sigma$",
    "LS8": "LS8",
}
SED_YML = (
    "/scratch/scratch-lxu/agent_dev/auto_research_agent/pyilc/"
    "input/fg_SEDs_default_params.yml"
)


@dataclass(frozen=True)
class DeprojCase:
    key: str
    dir_tag: str
    wtag: str
    yaml_slug: str
    n_deproj: int
    comps: tuple[str, ...]
    label: str


DEPROJ_NONE = DeprojCase("nodeproj", "", "", "", 0, (), "no deprojection")
DEPROJ_CIB = DeprojCase(
    "deproj_cib", "deproj_CIB", "_deproject_CIB", "deproj_cib", 1, ("CIB",), "CIB"
)
DEPROJ_CIB_DBETA = DeprojCase(
    "deproj_cib_dbeta",
    "deproj_CIB_CIB_dbeta",
    "_deproject_CIB_CIB_dbeta",
    "deproj_cib_dbeta",
    2,
    ("CIB", "CIB_dbeta"),
    r"CIB + $\delta\beta$",
)
DEPROJ_CIB_DBETA_CMB = DeprojCase(
    "deproj_cib_dbeta_cmb",
    "deproj_CIB_CIB_dbeta_CMB",
    "_deproject_CIB_CIB_dbeta_CMB",
    "deproj_cib_dbeta_cmb",
    3,
    ("CIB", "CIB_dbeta", "CMB"),
    r"CIB + $\delta\beta$ + CMB",
)
DEPROJ_MOMENTS = DeprojCase(
    "deproj_moments",
    "deproj_CIB_CIB_dbeta_CIB_dT",
    "_deproject_CIB_CIB_dbeta_CIB_dT",
    "deproj_cib_moments",
    3,
    ("CIB", "CIB_dbeta", "CIB_dT"),
    r"CIB + $\delta\beta$ + $\delta T$",
)

ALL_DEPROJ = (
    DEPROJ_NONE,
    DEPROJ_CIB,
    DEPROJ_CIB_DBETA,
    DEPROJ_CIB_DBETA_CMB,
    DEPROJ_MOMENTS,
)
DEPROJ_BY_KEY = {d.key: d for d in ALL_DEPROJ}


def total_maps_dir(name: str) -> Path:
    return SYNTH / "total_maps" / name


def catalogue_path(name: str) -> Path:
    if name == "L1_m9":
        return (
            SYNTH / "szifi_homog" / "catalogues"
            / "homog_immf_fullsky_splitA_immf_q5.npz"
        )
    return (
        SYNTH / "szifi_homog" / name / "catalogues" / "fullsky_splitA_immf_q5.npz"
    )


def tsz_dir(name: str) -> Path:
    return SYNTH / "components" / "tsz" / name


def cib_dir(name: str) -> Path:
    return SYNTH / "components" / "cib" / name


def cmb_path(name: str) -> Path:
    tag = "_LS8" if name == "LS8" else ""
    return (
        SYNTH / "components" / "cmb" / f"primary_CMB_T_lensed_nside4096_seed42{tag}.fits"
    )


def ilc_input_dir(name: str, real: int) -> Path:
    if name == "L1_m9":
        return ILC / ("inputs_nside2048_homog" if real == 1 else "inputs_nside2048_homog_r2")
    return ILC / (
        f"inputs_nside2048_homog_{name}"
        if real == 1
        else f"inputs_nside2048_homog_{name}_r2"
    )


def _sky_real_tags(*, masked: bool, real: int) -> tuple[str, str]:
    return ("_q5masked" if masked else "", "_r2" if real == 2 else "")


def hilc_output_dir(
    name: str, *, masked: bool, real: int, deproj: DeprojCase = DEPROJ_NONE
) -> Path:
    q, r = _sky_real_tags(masked=masked, real=real)
    dtag = f"_{deproj.dir_tag}" if deproj.dir_tag else ""
    if name == "L1_m9":
        return ILC / f"hilc_output_homog{q}{r}{dtag}"
    return ILC / f"hilc_output_homog_{name}{q}{r}{dtag}"


def hilc_suffix(
    name: str, *, masked: bool, real: int, deproj: DeprojCase = DEPROJ_NONE
) -> str:
    sky = "q5masked" if masked else "fullsky"
    rtag = "_r2" if real == 2 else ""
    suf = f"_hilc_y_homog_{sky}{rtag}"
    if deproj.dir_tag:
        suf += f"_{deproj.dir_tag}"
    if name != "L1_m9":
        suf += f"_{name}"
    return suf


def hilc_ymap(
    name: str, *, masked: bool, real: int, deproj: DeprojCase = DEPROJ_NONE
) -> Path:
    d = hilc_output_dir(name, masked=masked, real=real, deproj=deproj)
    suf = hilc_suffix(name, masked=masked, real=real, deproj=deproj)
    comp = f"component_tSZ{deproj.wtag}"
    return d / f"flamingo_needletILCmap_{comp}{suf}.fits"


def cluster_mask_binary(name: str) -> Path:
    if name == "L1_m9":
        return ILC / "szifi_immf_q5_cluster_mask_nside2048.fits"
    return ILC / f"szifi_immf_q5_cluster_mask_{name}_nside2048.fits"


def cluster_mask_apo(name: str) -> Path:
    if name == "L1_m9":
        return ILC / "szifi_immf_q5_cluster_mask_c2_025deg_nside2048.fits"
    return ILC / f"szifi_immf_q5_cluster_mask_{name}_c2_025deg_nside2048.fits"


def freq_map_files(name: str, real: int) -> list[str]:
    d = ilc_input_dir(name, real)
    return [
        str(d / f"sky_CMB_tSZ_CIB_homog_{nu}GHz_nside2048_K.fits")
        for nu in FREQS_ILC
    ]


def yaml_path(
    name: str, *, masked: bool, real: int, deproj: DeprojCase = DEPROJ_NONE
) -> Path:
    q, r = _sky_real_tags(masked=masked, real=real)
    slug = f"_{deproj.yaml_slug}" if deproj.yaml_slug else ""
    if name == "L1_m9":
        # Match legacy order: homog[_q5masked][_r2][_deproj_*]
        return REPO / "configs" / f"hilc_y_flamingo_homog{q}{r}{slug}.yml"
    return REPO / "configs" / f"hilc_y_flamingo_homog_{name}{slug}{q}{r}.yml"


def fig_dir(name: str, deproj: DeprojCase = DEPROJ_NONE) -> Path:
    return REPO / "figures" / "hilc" / name / deproj.key


def fig_combined_dir(deproj: DeprojCase = DEPROJ_NONE) -> Path:
    return REPO / "figures" / "hilc" / "combined" / deproj.key


def write_hilc_yaml(
    name: str, *, masked: bool, real: int, deproj: DeprojCase = DEPROJ_NONE
) -> Path:
    """Write an HILC config; identical ILC settings except paths and deprojection."""
    deproj_note = "" if deproj.n_deproj == 0 else f" ({deproj.label})"
    lines = [
        f"# HILC y on {name} homog skies{deproj_note}"
        + (" with q>5 iMMF holes." if masked else ".")
        + (" Independent white-noise r2." if real == 2 else " Noise realisation r1."),
        "work_in_healpix: 'yes'",
        f"output_dir: {hilc_output_dir(name, masked=masked, real=real, deproj=deproj)}/",
        "output_prefix: flamingo_",
        f"output_suffix: {hilc_suffix(name, masked=masked, real=real, deproj=deproj)}",
        "save_weights: 'yes'",
        "save_as: fits",
        f"param_dict_file: {SED_YML}",
        "ELLMAX: 4096",
        "taper_width: 200",
        "N_side: 2048",
        "N_freqs: 6",
        "bandpass_type: DeltaBandpasses",
        "freqs_delta_ghz: [100.0, 143.0, 353.0, 217.0, 545.0, 857.0]",
        "freq_map_files: ["
        + ",\n  ".join(freq_map_files(name, real))
        + "]",
        "beam_type: Gaussians",
        "beam_FWHM_arcmin: [9.66, 7.22, 4.92, 4.9, 4.67, 4.22]",
        "perform_ILC_at_beam: 10.0",
        "ILC_preserved_comp: tSZ",
        f"N_deproj: {deproj.n_deproj}",
        f"ILC_deproj_comps: [{', '.join(deproj.comps)}]",
        "N_maps_xcorr: 0",
    ]
    if masked:
        mask = cluster_mask_binary(name)
        lines += [
            f"mask_before_covariance_computation: [{mask},",
            "  0]",
            f"mask_before_wavelet_computation: [{mask},",
            "  0]",
        ]
    lines += [
        "print_timing: 'true'",
        "ilc_backend: jax",
        "wavelet_type: TopHatHarmonic",
        "BinSize: 50",
        "",
    ]
    out = yaml_path(name, masked=masked, real=real, deproj=deproj)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def iter_hilc_configs(
    *,
    names: tuple[str, ...] = ALL_RUNS,
    deprojs: tuple[DeprojCase, ...] = ALL_DEPROJ,
) -> list[Path]:
    """All YAML paths for the prescription × deprojection grid."""
    out: list[Path] = []
    for name in names:
        for deproj in deprojs:
            for masked in (False, True):
                for real in (1, 2):
                    out.append(yaml_path(name, masked=masked, real=real, deproj=deproj))
    return out
