"""Planck galactic × point-source mask product for pyILC.

The PR4 NILC ``Masks.fits`` archive under ``masks/pr4_nilc/`` carries three
fields (healpy field indices):

* 0 — ``NILC-MASK`` (near full-sky; *not* used alone here)
* 1 — ``GAL-MASK`` (Galactic plane cut; continuous / soft edges)
* 2 — ``PS-MASK`` (point-source holes; binary)

The pipeline builds a single **binary** product ``(GAL > 0.5) × (PS > 0.5)``
for both of pyILC's mask hooks:

* ``mask_before_covariance_computation`` — NILC real-space cov (and HILC
  map zeroing in :mod:`flamingo_mock.ilc.run` before SHTs);
* ``mask_before_wavelet_computation`` — zeros maps before needlet transforms.

pyILC only accepts a single ``[file, field]`` pair, so the multi-field PR4
archive is reduced to one FITS map under ``ilc/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import MASK_GAL_FIELD, MASK_PS_FIELD, ILCPaths

# Threshold for binarising soft GAL-MASK edges before multiplying by PS-MASK.
# Values near 0/1 with soft edges; 0.5 is the usual mid-point cut.
_GAL_THRESHOLD = 0.5


def load_gal_ps_product(
    mask_fits: str | Path,
    *,
    nside_out: int | None = None,
    gal_threshold: float = _GAL_THRESHOLD,
) -> np.ndarray:
    """Return binary GAL×PS mask (float 0/1) at ``nside_out`` (or native).

    Uses the product of the **galactic** and **point-source** fields only —
    never the near-full-sky ``NILC-MASK`` field alone.
    """
    import healpy as hp

    path = Path(mask_fits)
    if not path.is_file():
        raise FileNotFoundError(path)
    gal = np.asarray(hp.read_map(str(path), field=MASK_GAL_FIELD, dtype=np.float64))
    ps = np.asarray(hp.read_map(str(path), field=MASK_PS_FIELD, dtype=np.float64))
    # Explicit product of both cuts (soft GAL thresholded, binary PS).
    product = ((gal > gal_threshold).astype(np.float64) * (ps > 0.5).astype(np.float64))
    if nside_out is not None and hp.npix2nside(product.size) != nside_out:
        # Conservative: keep a pixel only if the majority of children are unmasked.
        product = hp.ud_grade(product, nside_out)
        product = (product > 0.5).astype(np.float64)
    return product


def ensure_combined_gal_ps_mask(
    paths: ILCPaths | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Write binary GAL×PS FITS under ``ilc/`` if missing; return its path."""
    import healpy as hp

    paths = paths or ILCPaths()
    out = paths.combined_gal_ps_mask()
    if out.is_file() and out.stat().st_size > 1_000_000 and not overwrite:
        return out

    product = load_gal_ps_product(paths.planck_masks_fits(), nside_out=paths.nside)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Explicit RING ordering so pyILC/healpy field-0 reads match N_side.
    hp.write_map(
        str(out),
        product,
        nest=False,
        overwrite=True,
        dtype=np.float64,
        column_names=["GAL_PS_MASK"],
    )
    fsky = float(product.mean())
    print(f"wrote {out} fsky={fsky:.4f} nside={paths.nside} (GAL×PS binary)")
    return out


def load_combined_mask(
    paths: ILCPaths | None = None,
    *,
    nside: int | None = None,
) -> np.ndarray:
    """Load the on-disk GAL×PS product (build it if needed)."""
    import healpy as hp

    paths = paths or ILCPaths()
    path = ensure_combined_gal_ps_mask(paths)
    m = np.asarray(hp.read_map(str(path), field=0, dtype=np.float64))
    if nside is not None and hp.npix2nside(m.size) != nside:
        m = hp.ud_grade(m, nside)
        m = (m > 0.5).astype(np.float64)
    return m


def pyilc_mask_yaml_entry(paths: ILCPaths | None = None) -> list:
    """``[fits_path, field_index]`` list for pyILC YAML mask keys."""
    paths = paths or ILCPaths()
    mask_path = ensure_combined_gal_ps_mask(paths)
    return [str(mask_path), 0]
