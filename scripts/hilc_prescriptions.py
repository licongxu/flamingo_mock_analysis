"""On-disk HILC paths for L1_m9 feedback / cosmology variants.

L1_m9 keeps the original unlabeled homog folders. The baryonic and LS8
runs get tagged input/output dirs so they cannot overwrite the fiducial.
"""
from __future__ import annotations

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


def total_maps_dir(name: str) -> Path:
    return SYNTH / "total_maps" / name


def catalogue_path(name: str) -> Path:
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


def hilc_output_dir(name: str, *, masked: bool, real: int) -> Path:
    if name == "L1_m9":
        if masked:
            return ILC / ("hilc_output_homog_q5masked" if real == 1 else "hilc_output_homog_q5masked_r2")
        return ILC / ("hilc_output_homog" if real == 1 else "hilc_output_homog_r2")
    tag = "_q5masked" if masked else ""
    rtag = "" if real == 1 else "_r2"
    return ILC / f"hilc_output_homog_{name}{tag}{rtag}"


def hilc_suffix(name: str, *, masked: bool, real: int) -> str:
    sky = "q5masked" if masked else "fullsky"
    rtag = "" if real == 1 else "_r2"
    if name == "L1_m9":
        return f"_hilc_y_homog_{sky}{rtag}"
    return f"_hilc_y_homog_{sky}{rtag}_{name}"


def hilc_ymap(name: str, *, masked: bool, real: int) -> Path:
    d = hilc_output_dir(name, masked=masked, real=real)
    return d / f"flamingo_needletILCmap_component_tSZ{hilc_suffix(name, masked=masked, real=real)}.fits"


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


def yaml_path(name: str, *, masked: bool, real: int) -> Path:
    sky = "_q5masked" if masked else ""
    rtag = "" if real == 1 else "_r2"
    if name == "L1_m9":
        return REPO / "configs" / f"hilc_y_flamingo_homog{sky}{rtag}.yml"
    return REPO / "configs" / f"hilc_y_flamingo_homog_{name}{sky}{rtag}.yml"


def write_hilc_yaml(name: str, *, masked: bool, real: int) -> Path:
    """Write a no-deprojection HILC config; identical to L1_m9 except paths."""
    lines = [
        f"# HILC y on {name} homog skies"
        + (" with q>5 iMMF holes." if masked else ".")
        + (" Independent white-noise r2." if real == 2 else " Noise realisation r1."),
        "work_in_healpix: 'yes'",
        f"output_dir: {hilc_output_dir(name, masked=masked, real=real)}/",
        "output_prefix: flamingo_",
        f"output_suffix: {hilc_suffix(name, masked=masked, real=real)}",
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
        "N_deproj: 0",
        "ILC_deproj_comps: []",
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
    out = yaml_path(name, masked=masked, real=real)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out
