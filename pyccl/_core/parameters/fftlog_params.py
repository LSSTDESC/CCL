__all__ = ("ExtrapolationMethods", "FFTLogParams",)

from enum import Enum
from numbers import Real

from . import Parameters


class ExtrapolationMethods(Enum):
    """Extrapolation methods for FFTLog transforms."""
    NONE = "none"
    CONSTANT = "constant"
    LINX_LINY = "linx_liny"
    LINX_LOGY = "linx_logy"
    LOGX_LINY = "logx_liny"
    LOGX_LOGY = "logx_logy"


class FFTLogParams(Parameters):
    """Objects of this class store the FFTLog accuracy parameters.

    Stored in instances of :class:`~pyccl.halos.HaloProfile`.
    """
    padding_lo_fftlog: Real = 0.1
    """Multiply the lower boundary of the input range to avoid aliasing."""

    padding_hi_fftlog: Real = 10.
    """Multiply the lower boundary of the input range to avoid aliasing."""

    n_per_decade: int = 100
    """Samples per decade for the Hankel transforms."""

    extrapol: str = "linx_liny"
    """Extrapolation type when FFTLog has narrower output support.
    Available extrapolation types are listed in :class:`~ExtrapolationMethods`
    """

    padding_lo_extra: Real = 0.1
    """Padding for the intermediate step of a double Hankel transform.
    Used to compute the 2D projected profile and the 2D cumulative
    density, where the first transform goes from 3D real space to
    Fourier, then from Fourier to 2D real space. Usually, it doesn't
    have to be as precise as `padding_lo_fftlog`.
    """

    padding_hi_extra: Real = 10.
    """Padding for the intermediate step of a double Hankel transform.
    Used to compute the 2D projected profile and the 2D cumulative
    density, where the first transform goes from 3D real space to
    Fourier, then from Fourier to 2D real space. Usually, it doesn't
    have to be as precise as `padding_lo_fftlog`.
    """

    large_padding_2D: bool = False
    """Override `padding_xx_extra` in the intermediate transform,
    and use `padding_xx_fftlog`.
    """

    plaw_fourier: Real = -1.5
    r"""FFTLog pre-whitens its arguments (makes them flatter) to avoid
    aliasing. The `plaw` parameters describe the tilt of the profile,
    :math:`P(r) \sim r^{\rm tilt}`, between real and Fourier transforms,
    and between 2D projected and cumulative density, respectively.
    Some level of experimentation with these parameters is recommended.

    .. note::

        Finer control of these parameters can be achieved by overriding
        ``~pyccl.halos.HaloProfile._get_plaw_fourier()``.
    """

    plaw_projected = -1.0
    r"""FFTLog pre-whitens its arguments (makes them flatter) to avoid
    aliasing. The `plaw` parameters describe the tilt of the profile,
    :math:`P(r) \sim r^{\rm tilt}`, between real and Fourier transforms,
    and between 2D projected and cumulative density, respectively.
    Some level of experimentation with these parameters is recommended.

    .. note::

        Finer control of these parameters can be achieved by overriding
        ``~pyccl.halos.HaloProfile._get_plaw_projected()``.
    """

    def update_parameters(self, **kwargs):
        """Update the precision of FFTLog for the Hankel transforms."""
        for name, value in kwargs.items():
            setattr(self, name, value)
