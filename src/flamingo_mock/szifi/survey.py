"""SZiFi survey adapter for FLAMINGO mock tiles (Planck-like flat sky)."""

from __future__ import annotations

import numpy as np
import healpy as hp
from szifi import maps
from szifi import expt

from flamingo_mock.szifi.paths import (
    TILE_L_DEG,
    TILE_NSIDE,
    TILE_NX,
    SZiFiPaths,
    beam_fwhm_vec_arcmin,
)


class input_data_survey:
    """Drop-in survey class for SZiFi ``survey_file`` import.

    Expects tile products written by ``flamingo_mock.szifi.tiles.prepare_tiles``.
    Temperature maps are already in µK_CMB (no 545/857 Jy conversion).
    """

    def __init__(self, params_szifi=None, params_data=None):
        paths = SZiFiPaths(out_root=params_szifi.get("flamingo_out_root", SZiFiPaths().out_root))
        split = params_data.get("other_params", {}).get("npipe_split", "A")
        field_ids = params_data["field_ids"]

        self.data = {}
        self.data["params_data"] = params_data

        self.data["nx"] = {}
        self.data["ny"] = {}
        self.data["dx_arcmin"] = {}
        self.data["dy_arcmin"] = {}
        self.data["pix"] = {}

        self.data["t_obs"] = {}
        self.data["t_noi"] = {}

        self.data["mask_map"] = {}
        self.data["mask_point"] = {}
        self.data["mask_select"] = {}
        self.data["mask_select_no_tile"] = {}
        self.data["mask_select_buffer"] = {}
        self.data["mask_ps"] = {}
        self.data["mask_peak_finding"] = {}
        self.data["mask_peak_finding_no_tile"] = {}
        self.data["mask_tile"] = {}
        self.data["coupling_matrix_name"] = {}

        self.nside_tile = TILE_NSIDE
        self.n_tile = hp.nside2npix(self.nside_tile)

        self.nx = TILE_NX
        self.l = TILE_L_DEG
        self.dx_arcmin = self.l / self.nx * 60.0
        self.dx = self.dx_arcmin / 180.0 / 60.0 * np.pi
        self.pix = maps.pixel(self.nx, self.dx)

        self.data["nside_tile"] = self.nside_tile

        buffer_arcmin = 10.0

        for field_id in field_ids:
            self.data["nx"][field_id] = TILE_NX
            self.data["ny"][field_id] = TILE_NX
            self.data["dx_arcmin"][field_id] = self.dx_arcmin
            self.data["dy_arcmin"][field_id] = self.dx_arcmin
            self.data["pix"][field_id] = self.pix

            tmap = np.load(paths.tmap_path(split, field_id), allow_pickle=True)[0]
            tmap = np.asarray(tmap, dtype=np.float32)
            self.data["t_obs"][field_id] = tmap
            self.data["t_noi"][field_id] = tmap

            masks = np.load(paths.mask_path(split, field_id), allow_pickle=True)
            mask_galaxy = np.asarray(masks[0], dtype=np.float64)
            mask_point = np.asarray(masks[1], dtype=np.float64)
            mask_tile = np.asarray(masks[2], dtype=np.float64)

            mask_ps = maps.get_apodised_mask(
                self.pix, mask_galaxy, apotype="Smooth", aposcale=0.2
            )

            mask_peak_finding_no_tile = mask_galaxy * mask_point
            mask_select_no_tile = maps.get_buffered_mask(
                self.pix, mask_peak_finding_no_tile, buffer_arcmin, type="fft"
            )
            mask_peak_finding = mask_peak_finding_no_tile * mask_tile
            mask_select = mask_select_no_tile * mask_tile
            mask_select = maps.get_fsky_criterion_mask(
                self.pix, mask_select, self.nside_tile, criterion=params_szifi["min_ftile"]
            )

            if params_szifi["tilemask_mode"] == "catalogue_buffer":
                mask_select_buffer = mask_select_no_tile * maps.get_buffer_region(
                    self.pix, mask_tile, params_szifi["tilemask_buffer_arcmin"]
                )
            else:
                mask_select_buffer = 0

            self.data["mask_point"][field_id] = mask_point
            self.data["mask_select"][field_id] = mask_select
            self.data["mask_select_no_tile"][field_id] = mask_select_no_tile
            self.data["mask_select_buffer"][field_id] = mask_select_buffer
            self.data["mask_map"][field_id] = mask_ps
            self.data["mask_ps"][field_id] = mask_ps
            self.data["mask_peak_finding_no_tile"][field_id] = mask_peak_finding_no_tile
            self.data["mask_peak_finding"][field_id] = mask_peak_finding
            self.data["mask_tile"][field_id] = mask_tile

            cm_name = str(paths.coupling_path(split, field_id))
            self.data["coupling_matrix_name"][field_id] = cm_name

        exp = expt.experiment(experiment_name="Planck_simple", params_szifi=params_szifi)
        # Match mock Gaussian Table I beams used in the total maps.
        exp.FWHM = np.asarray(beam_fwhm_vec_arcmin(), dtype=np.float64)
        self.data["experiment"] = exp
