"""
.. warning:: The functionality contained in this module has been deprecated
             in favour of the newer halo model implementation
             :py:mod:`~pyccl.halos`.
"""

__all__ = ("massfunc", "halo_bias", "massfunc_m2r",)

from . import CCLError, deprecated
from . import halos as hal
from .power import sigmaM  # noqa


@deprecated(hal.MassFunc)
def massfunc(cosmo, halo_mass, a, overdensity=200):
    """Halo mass function, dn/dlog10M.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.
        overdensity (float): overdensity parameter (default: 200)

    Returns:
        float or array_like: Halo mass function; dn/dlog10M.
    """
    mdef = hal.MassDef(overdensity, 'matter')
    mf_par = cosmo._config_init_kwargs['mass_function']
    if mf_par == 'tinker10':
        mf = hal.MassFuncTinker10(cosmo, mdef)
    elif mf_par == 'tinker':
        mf = hal.MassFuncTinker08(cosmo, mdef)
    elif mf_par == 'watson':
        mf = hal.MassFuncWatson13(cosmo, mdef)
    elif mf_par == 'shethtormen':
        mf = hal.MassFuncSheth99(cosmo)
    elif mf_par == 'angulo':
        mf = hal.MassFuncAngulo12(cosmo)

    return mf(cosmo, halo_mass, a)


@deprecated(hal.HaloBias)
def halo_bias(cosmo, halo_mass, a, overdensity=200):
    """Halo bias

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): Scale factor.
        overdensity (float): Overdensity parameter (default: 200).

    Returns:
        float or array_like: Halo bias.
    """
    mdef = hal.MassDef(overdensity, 'matter')
    mf_par = cosmo._config_init_kwargs['mass_function']
    if mf_par == 'tinker10':
        bf = hal.HaloBiasTinker10(cosmo, mdef)
    elif mf_par == 'shethtormen':
        bf = hal.HaloBiasSheth99(cosmo)
    else:
        raise CCLError("No b(M) fitting function implemented for "
                       "mass_function_method: "+mf_par)
    return bf(cosmo, halo_mass, a)


@deprecated(hal.mass2radius_lagrangian)
def massfunc_m2r(cosmo, halo_mass):
    """Converts smoothing halo mass into smoothing halo radius.

    .. note:: This is :math:`R=(3M/(4\\pi\\rho_M))^{1/3}``, where
              :math:`\\rho_M` is the mean comoving matter density.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.

    Returns:
        float or array_like: Smoothing halo radius; Mpc.
    """
    return hal.mass2radius_lagrangian(cosmo, halo_mass)
