"""HILC YAML for baryonic / LS8 variants must not reuse the L1_m9 mask or totals."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    DEPROJ_CIB,
    DEPROJ_MOMENTS,
    DEPROJ_NONE,
    PRESCRIPTIONS,
    catalogue_path,
    cluster_mask_binary,
    cmb_path,
    hilc_output_dir,
    hilc_suffix,
    hilc_ymap,
    ilc_input_dir,
    tsz_dir,
    write_hilc_yaml,
    yaml_path,
)


def test_catalogues_and_masks_are_per_prescription():
    for name in PRESCRIPTIONS:
        cat = catalogue_path(name)
        assert f"szifi_homog/{name}/catalogues/fullsky_splitA_immf_q5.npz" in str(cat)
        assert name in cluster_mask_binary(name).name
        assert "L1_m9" not in cluster_mask_binary(name).name
        assert tsz_dir(name).as_posix().endswith(f"components/tsz/{name}")
    assert cmb_path("LS8").name.endswith("_LS8.fits")
    assert "_LS8" not in cmb_path("fgas-8sigma").name


def test_r1_r2_ilc_dirs_differ():
    for name in PRESCRIPTIONS:
        d1, d2 = ilc_input_dir(name, 1), ilc_input_dir(name, 2)
        assert d1 != d2
        assert name in d1.name and name in d2.name
        assert d1 != ilc_input_dir("L1_m9", 1)
        assert hilc_output_dir(name, masked=True, real=1) != hilc_output_dir(
            "L1_m9", masked=True, real=1
        )


def test_deproj_output_dirs_differ_by_prescription():
    for name in PRESCRIPTIONS:
        d = hilc_output_dir(name, masked=False, real=1, deproj=DEPROJ_CIB)
        assert name in d.name
        assert "deproj_CIB" in d.name
        assert d != hilc_output_dir("L1_m9", masked=False, real=1, deproj=DEPROJ_CIB)


def test_l1_m9_deproj_suffix_matches_existing_yaml():
    assert hilc_suffix("L1_m9", masked=False, real=1, deproj=DEPROJ_NONE) == (
        "_hilc_y_homog_fullsky"
    )
    assert hilc_suffix("L1_m9", masked=False, real=1, deproj=DEPROJ_CIB) == (
        "_hilc_y_homog_fullsky_deproj_CIB"
    )
    assert hilc_suffix("fgas-8sigma", masked=False, real=1, deproj=DEPROJ_CIB) == (
        "_hilc_y_homog_fullsky_deproj_CIB_fgas-8sigma"
    )


def test_written_yaml_uses_matching_mask_and_noise_split(tmp_path, monkeypatch):
    import hilc_prescriptions as hpmod

    monkeypatch.setattr(hpmod, "REPO", tmp_path)
    for name in PRESCRIPTIONS:
        y1 = hpmod.write_hilc_yaml(name, masked=True, real=1, deproj=DEPROJ_CIB)
        y2 = hpmod.write_hilc_yaml(name, masked=True, real=2, deproj=DEPROJ_CIB)
        t1, t2 = y1.read_text(), y2.read_text()
        mask = str(cluster_mask_binary(name))
        assert mask in t1 and mask in t2
        assert "/szifi_immf_q5_cluster_mask_nside2048.fits" not in t1
        assert str(ilc_input_dir(name, 1)) in t1
        assert str(ilc_input_dir(name, 2)) in t2
        assert "N_deproj: 1" in t1
        assert "ILC_deproj_comps: [CIB]" in t1
        assert "TopHatHarmonic" in t1
        assert yaml_path("L1_m9", masked=False, real=1).name == "hilc_y_flamingo_homog.yml"


def test_l1_m9_yaml_paths_keep_unlabeled_homog_dirs():
    src = yaml_path("L1_m9", masked=False, real=1).read_text()
    assert "inputs_nside2048_homog/" in src
    assert "inputs_nside2048_homog_fgas" not in src
    assert yaml_path("L1_m9", masked=True, real=2, deproj=DEPROJ_MOMENTS).name == (
        "hilc_y_flamingo_homog_q5masked_r2_deproj_cib_moments.yml"
    )


def test_ymap_path_includes_deproject_tag():
    p = hilc_ymap("L1_m9", masked=False, real=1, deproj=DEPROJ_CIB)
    assert "deproject_CIB" in p.name
    assert "deproj_CIB" in p.name


def test_build_mask_defaults_to_l1_m9_fiducial():
    src = (_SCRIPTS / "build_szifi_q5_cluster_mask.py").read_text()
    assert 'default="L1_m9"' in src
    assert "L2P8_TEST_CAT" in src
    assert "LEGACY_CAT" not in src


def test_l1_m9_fiducial_catalogue_has_2509_detections():
    import numpy as np

    cat = np.load(catalogue_path("L1_m9"))
    assert int(cat["q_opt"].size) == 2509


def test_l1_m9_written_yaml_uses_fiducial_mask(tmp_path, monkeypatch):
    import hilc_prescriptions as hpmod

    monkeypatch.setattr(hpmod, "REPO", tmp_path)
    y = hpmod.write_hilc_yaml("L1_m9", masked=True, real=1, deproj=DEPROJ_NONE)
    mask = str(cluster_mask_binary("L1_m9"))
    assert mask in y.read_text()


def test_plot_script_uses_each_catalogue():
    src = (_SCRIPTS / "plot_hilc_homog_prescriptions.py").read_text()
    assert "catalogue_path" in src
    assert "cluster_mask_apo" in src
    assert "ALL_DEPROJ" in src
    assert "components/tsz/test" not in src
