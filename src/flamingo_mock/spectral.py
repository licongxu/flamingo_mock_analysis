"""Spectral energy distributions and unit conversions.

All functions are monochromatic; the released CIB maps are already
bandpass-convolved, so no further bandpass integration is applied.

Conventions (matching Yang et al. 2026 and the compsep pipeline):

* tSZ: non-relativistic spectral function f(x) = x coth(x/2) - 4 with
  x = h nu / (k_B T_CMB);  dT_tSZ = T_CMB * y * f(x).
* kSZ: dT_kSZ / T_CMB = -b, frequency independent in thermodynamic units.
* CIB: specific intensity I_nu [Jy/sr], converted to thermodynamic
  temperature via dB_nu/dT at T_CMB.
"""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u

from .config import CIB_ALPHA, CIB_BETA_D, CIB_T0, T_CMB

# ---------------------------------------------------------------------------
# tSZ
# ---------------------------------------------------------------------------


def tsz_x(nu_ghz: float | np.ndarray, t_cmb: float = T_CMB) -> np.ndarray:
    """Dimensionless frequency x = h nu / (k_B T_CMB)."""
    nu = np.asarray(nu_ghz, dtype=np.float64) * u.GHz
    return (const.h * nu / (const.k_B * t_cmb * u.K)).to_value(
        u.dimensionless_unscaled
    )


def tsz_f(nu_ghz: float | np.ndarray, t_cmb: float = T_CMB) -> np.ndarray:
    """Non-relativistic tSZ spectral function f(x) = x coth(x/2) - 4."""
    x = tsz_x(nu_ghz, t_cmb=t_cmb)
    return x / np.tanh(0.5 * x) - 4.0


def tsz_response_uK(nu_ghz: float, t_cmb: float = T_CMB) -> float:
    """tSZ response in uK_CMB per unit Compton y."""
    return float(t_cmb * 1.0e6 * tsz_f(nu_ghz, t_cmb=t_cmb))


def y_to_delta_T_uK(
    y: np.ndarray, nu_ghz: float, t_cmb: float = T_CMB
) -> np.ndarray:
    """Convert a Compton-y map to dT_tSZ [uK_CMB] at frequency nu_ghz."""
    return tsz_response_uK(nu_ghz, t_cmb=t_cmb) * y


# ---------------------------------------------------------------------------
# kSZ
# ---------------------------------------------------------------------------


def ksz_response_uK(t_cmb: float = T_CMB) -> float:
    """kSZ response in uK_CMB per unit Doppler b (frequency independent)."""
    return -t_cmb * 1.0e6


def b_to_delta_T_uK(b: np.ndarray, t_cmb: float = T_CMB) -> np.ndarray:
    """Convert a Doppler-b map to dT_kSZ [uK_CMB] (valid at all frequencies)."""
    return ksz_response_uK(t_cmb=t_cmb) * b


# ---------------------------------------------------------------------------
# Intensity <-> thermodynamic temperature
# ---------------------------------------------------------------------------


def dB_dT_Jy_per_sr_per_K(nu_ghz: float, t_cmb: float = T_CMB) -> float:
    """dB_nu/dT at T_CMB in Jy/sr/K for monochromatic frequency nu_ghz."""
    nu = nu_ghz * u.GHz
    T = t_cmb * u.K
    x = (const.h * nu / (const.k_B * T)).to_value(u.dimensionless_unscaled)
    prefactor_si = (2.0 * const.k_B * nu**2 / const.c**2).to_value(
        u.W / u.m**2 / u.Hz / u.K
    )
    dBdT_si = prefactor_si * (x**2 * np.exp(x) / np.expm1(x) ** 2)
    return dBdT_si / 1e-26  # 1 Jy = 1e-26 W/m^2/Hz


def intensity_to_uK(
    I_Jy_sr: np.ndarray, nu_ghz: float, t_cmb: float = T_CMB
) -> np.ndarray:
    """Convert specific intensity [Jy/sr] to thermodynamic uK_CMB."""
    return I_Jy_sr / dB_dT_Jy_per_sr_per_K(nu_ghz, t_cmb=t_cmb) * 1.0e6


def uK_to_intensity(
    T_uK: np.ndarray, nu_ghz: float, t_cmb: float = T_CMB
) -> np.ndarray:
    """Convert thermodynamic uK_CMB to specific intensity [Jy/sr]."""
    return T_uK * 1.0e-6 * dB_dT_Jy_per_sr_per_K(nu_ghz, t_cmb=t_cmb)


# ---------------------------------------------------------------------------
# CIB greybody SED (three-parameter model, Yang et al. 2026 Section 3.5.1)
# ---------------------------------------------------------------------------


def tdust(z: float, t0: float = CIB_T0, alpha: float = CIB_ALPHA) -> float:
    """Dust temperature T_dust(z) = T0 (1+z)^alpha."""
    return t0 * (1.0 + z) ** alpha


def theta_nu(
    nu_ghz: float | np.ndarray,
    t_dust: float,
    beta_d: float = CIB_BETA_D,
) -> np.ndarray:
    """Greybody SED shape Theta(nu, T) ~ nu^(beta+3) / (e^{h nu / kT} - 1)."""
    nu = np.asarray(nu_ghz, dtype=np.float64)
    x = (const.h * nu * u.GHz / (const.k_B * t_dust * u.K)).to_value(
        u.dimensionless_unscaled
    )
    x = np.clip(x, 1e-8, 100.0)  # avoid overflow in expm1
    return (nu ** (beta_d + 3.0)) / np.expm1(x)


def sed_shape_observed(
    nu_obs_ghz: float | np.ndarray,
    z: float,
    t0: float = CIB_T0,
    alpha: float = CIB_ALPHA,
    beta_d: float = CIB_BETA_D,
) -> np.ndarray:
    """Observed-frame SED shape at nu_obs for a source at redshift z."""
    t = tdust(z, t0=t0, alpha=alpha)
    nu_rest = np.asarray(nu_obs_ghz, dtype=np.float64) * (1.0 + z)
    return theta_nu(nu_rest, t, beta_d=beta_d)


def sed_ratio(
    nu_tgt: float, nu_ref: float, z_eff: float
) -> float:
    """Theta(nu_tgt)/Theta(nu_ref) at effective redshift z_eff."""
    return float(
        sed_shape_observed(nu_tgt, z_eff) / sed_shape_observed(nu_ref, z_eff)
    )
