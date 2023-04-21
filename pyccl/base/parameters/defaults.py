__all__ = ("DEFAULT_POWER_SPECTRUM", "DefaultParams",)

import warnings

from ... import CCLDeprecationWarning

DEFAULT_POWER_SPECTRUM = "delta_matter:delta_matter"


class DefaultParams:
    """Default cosmological parameters used throughout the library."""
    T_CMB = 2.725
    T_ncdm = 0.71611

    warnings.warn(
        "The default CMB temperaHaloProfileCIBture (T_CMB) will change in "
        "CCLv3.0.0, from 2.725 to 2.7255 (Kelvin).", CCLDeprecationWarning)
