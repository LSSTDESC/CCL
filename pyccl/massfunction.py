from . import ccllib as lib
from .pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn4


def massfunc(cosmo, halo_mass, a, overdensity=200):
    """Tinker et al. (2010) halo mass function, dn/dlog10M.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.
        overdensity (float): overdensity parameter (default: 200)

    Returns:
        float or array_like: Halo mass function; dn/dlog10M.
    """
    return _vectorize_fn4(lib.massfunc,
                          lib.massfunc_vec, cosmo, halo_mass, a, overdensity)


def massfunc_m2r(cosmo, halo_mass):
    """Converts smoothing halo mass into smoothing halo radius.

    .. note:: This is R=(3M/(4*pi*rho_m))^(1/3), where rho_m is the mean
              matter density.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.

    Returns:
        float or array_like: Smoothing halo radius; Mpc.
    """
    return _vectorize_fn(lib.massfunc_m2r,
                         lib.massfunc_m2r_vec, cosmo, halo_mass)


def sigmaM(cosmo, halo_mass, a):
    """Root mean squared variance for the given halo mass of the linear power
    spectrum; Msun.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.

    Returns:
        float or array_like: RMS variance of halo mass.
    """
    return _vectorize_fn2(lib.sigmaM,
                          lib.sigmaM_vec, cosmo, halo_mass, a)


def halo_bias(cosmo, halo_mass, a, overdensity=200):
    """Tinker et al. (2010) halo bias

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): Scale factor.
        overdensity (float): Overdensity parameter (default: 200).

    Returns:
        float or array_like: Halo bias.
    """
    return _vectorize_fn4(lib.halo_bias,
                          lib.halo_bias_vec, cosmo, halo_mass, a, overdensity)
