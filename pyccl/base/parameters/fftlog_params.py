__all__ = ("FFTLogParams",)

from dataclasses import dataclass
from enum import Enum

from ... import lib


class ExtrapolationMethods(Enum):
    NONE = "none"
    CONSTANT = "constant"
    LINX_LINY = "linx_liny"
    LINX_LOGY = "linx_logy"
    LOGX_LINY = "logx_liny"
    LOGX_LOGY = "logx_logy"


extrap_types = {'none': lib.f1d_extrap_0,
                'constant': lib.f1d_extrap_const,
                'linx_liny': lib.f1d_extrap_linx_liny,
                'linx_logy': lib.f1d_extrap_linx_logy,
                'logx_liny': lib.f1d_extrap_logx_liny,
                'logx_logy': lib.f1d_extrap_logx_logy}


# TODO: py310+ add `kw_only=True` argument to dataclass decorator.
@dataclass(unsafe_hash=True, frozen=True)
class FFTLogParams:
    """Objects of this class store the FFTLog accuracy parameters."""
    padding_lo_fftlog: float = 0.1  # | Anti-aliasing: lower boundary factor.
    padding_hi_fftlog: float = 10.  # |                upper boundary factor.

    n_per_decade: int = 100         # Hankel transforms samples per dex.
    extrapol: str = "linx_liny"     # Extrapolation type.

    padding_lo_extra: float = 0.1   # Padding for the middle step of a double
    padding_hi_extra: float = 10.   # transform. Doesn't have to be as precise.
    large_padding_2D: bool = False  # Flag for high precision middle transform.

    plaw_fourier: float = -1.5      # Real <--> Fourier transforms.
    plaw_projected: float = -1.0    # 2D proj & cumul density profiles.

    def __post_init__(self):
        if self.extrapol not in extrap_types:
            raise ValueError("Invalid FFTLog extrapolation type.")

    def __getitem__(self, name):
        return getattr(self, name)

    # TODO: docs_v3 - This entire docstring as explanatory in HaloProfile base
    # and remove from here.
    def update_parameters(self, **kwargs):
        """Update the precision of FFTLog for the Hankel transforms.

        Arguments
        ---------
        padding_lo_fftlog, padding_hi_fftlog : float
            Multiply the lower and upper boundary of the input range
            to avoid aliasing. The defaults are 0.1 and 10.0, respectively.
        n_per_decade : float
            Samples per decade for the Hankel transforms.
            The default is 100.
        extrapol : {'linx_liny', 'linx_logy'}
            Extrapolation type when FFTLog has narrower output support.
            The default is 'linx_liny'.
        padding_lo_extra, padding_hi_extra : float
            Padding for the intermediate step of a double Hankel transform.
            Used to compute the 2D projected profile and the 2D cumulative
            density, where the first transform goes from 3D real space to
            Fourier, then from Fourier to 2D real space. Usually, it doesn't
            have to be as precise as ``padding_xx_fftlog``.
            The defaults are 0.1 and 10.0, respectively.
        large_padding_2D : bool
            Override ``padding_xx_extra`` in the intermediate transform,
            and use ``padding_xx_fftlog``. The default is False.
        plaw_fourier, plaw_projected : float
            FFTLog pre-whitens its arguments (makes them flatter) to avoid
            aliasing. The ``plaw`` parameters describe the tilt of the profile,
            :math:`P(r) \\sim r^{\\mathrm{tilt}}`, between real and Fourier
            transforms, and between 2D projected and cumulative density,
            respectively. Subclasses of ``HaloProfile`` may obtain finer
            control via ``_get_plaw_[fourier | projected]``, and some level of
            experimentation with these parameters is recommended.
            The defaults are -1.5 and -1.0, respectively.
        """
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise AttributeError(f"Parameter {name} does not exist.")
            object.__setattr__(self, name, value)
