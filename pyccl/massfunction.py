from . import halos as hal
from .pyutils import deprecated
from .errors import CCLError
from .power import sigmaM  # noqa


@deprecated(hal.MassFunc)
def massfunc(cosmo, halo_mass, a, overdensity=200):
    """Halo mass function, dn/dlog10M.

    .. note:: Note that this function is deprecated. Please use the
              functionality in the :mod:`~pyccl.halos.hmfunc` module.

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

    return mf.get_mass_function(cosmo,
                                halo_mass,
                                a)


@deprecated(hal.HaloBias)
def halo_bias(cosmo, halo_mass, a, overdensity=200):
    """Halo bias

    .. note:: Note that this function is deprecated. Please use the
              functionality in the :mod:`~pyccl.halos.hbias` module.

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
    return bf.get_halo_bias(cosmo,
                            halo_mass,
                            a)


@deprecated(hal.mass2radius_lagrangian)
def massfunc_m2r(cosmo, halo_mass):
    """Converts smoothing halo mass into smoothing halo radius.

    .. note:: This is :math:`R=(3M/(4\\pi\\rho_M))^{1/3}``, where
              :math:`\\rho_M` is the mean comoving matter density.

    .. note:: Note that this function is deprecated. Please use
              :meth:`~pyccl.halos.massdef.mass2radius_lagrangian`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.

    Returns:
        float or array_like: Smoothing halo radius; Mpc.
    """
    return hal.mass2radius_lagrangian(cosmo, halo_mass)
