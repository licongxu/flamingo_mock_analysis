"""HILC YAML for baryonic / LS8 variants must not reuse the L1_m9 mask or totals."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from hilc_prescriptions import (  # noqa: E402
    PRESCRIPTIONS,
    catalogue_path,
    cluster_mask_binary,
    cmb_path,
    hilc_output_dir,
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


def test_written_yaml_uses_matching_mask_and_noise_split(tmp_path, monkeypatch):
    import hilc_prescriptions as hpmod

    monkeypatch.setattr(hpmod, "REPO", tmp_path)
    for name in PRESCRIPTIONS:
        y1 = hpmod.write_hilc_yaml(name, masked=True, real=1)
        y2 = hpmod.write_hilc_yaml(name, masked=True, real=2)
        t1, t2 = y1.read_text(), y2.read_text()
        mask = str(cluster_mask_binary(name))
        assert mask in t1 and mask in t2
        assert "/szifi_immf_q5_cluster_mask_nside2048.fits" not in t1
        assert str(ilc_input_dir(name, 1)) in t1
        assert str(ilc_input_dir(name, 2)) in t2
        assert str(ilc_input_dir(name, 2)) not in t1
        assert "N_deproj: 0" in t1
        assert "TopHatHarmonic" in t1
        assert yaml_path("L1_m9", masked=False, real=1).name == "hilc_y_flamingo_homog.yml"


def test_l1_m9_yaml_paths_keep_unlabeled_homog_dirs():
    src = yaml_path("L1_m9", masked=False, real=1).read_text()
    assert "inputs_nside2048_homog/" in src
    assert "inputs_nside2048_homog_fgas" not in src
    assert yaml_path("L1_m9", masked=True, real=2).name == (
        "hilc_y_flamingo_homog_q5masked_r2.yml"
    )


def test_plot_script_uses_each_catalogue():
    src = (_SCRIPTS / "plot_hilc_homog_prescriptions.py").read_text()
    assert "catalogue_path" in src
    assert "cluster_mask_apo" in src
    assert "fgas-8sigma" in src or "PRESCRIPTIONS" in src or "ALL_RUNS" in src
    assert "components/tsz/test" not in src
