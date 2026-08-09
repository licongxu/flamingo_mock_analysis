"""flamingo_mock: synthetic multi-frequency CMB skies from FLAMINGO maps.

Sky model: lensed primary CMB + tSZ + kSZ + CIB, in uK_CMB, built from the
Yang et al. (2026) integrated lightcone maps (FLAMINGO L2p8_m9).
"""

from .config import MockConfig

__all__ = ["MockConfig"]
__version__ = "0.1.0"
