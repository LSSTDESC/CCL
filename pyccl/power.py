from . import ccllib as lib
from .pyutils import _vectorize_fn2


def linear_matter_power(cosmo, k, a):
    """The linear matter power spectrum; Mpc^3.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Linear matter power spectrum; Mpc^3.
    """
    return _vectorize_fn2(lib.linear_matter_power,
                          lib.linear_matter_power_vec, cosmo, k, a)


def nonlin_matter_power(cosmo, k, a):
    """The nonlinear matter power spectrum; Mpc^3.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Nonlinear matter power spectrum; Mpc^3.
    """
    return _vectorize_fn2(lib.nonlin_matter_power,
                          lib.nonlin_matter_power_vec, cosmo, k, a)


def sigmaR(cosmo, R, a=1.):
    """RMS variance in a top-hat sphere of radius R in Mpc.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.
        a (float): optional scale factor; defaults to a=1

    Returns:
        float or array_like: RMS variance in the density field in top-hat
                             sphere; Mpc.
    """
    return _vectorize_fn2(lib.sigmaR, lib.sigmaR_vec, cosmo, R, a)


def sigmaV(cosmo, R, a=1.):
    """RMS variance in the displacement field in a top-hat sphere of radius R.
    The linear displacement field is the gradient of the linear density field.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.
        a (float): optional scale factor; defaults to a=1

    Returns:
        sigmaV (float or array_like): RMS variance in the displacement field in
                                      top-hat sphere.
    """
    return _vectorize_fn2(lib.sigmaV, lib.sigmaV_vec, cosmo, R, a)


def sigma8(cosmo):
    """RMS variance in a top-hat sphere of radius 8 Mpc/h.

    .. note:: 8 Mpc/h is rescaled based on the Hubble constant.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.

    Returns:
        float: RMS variance in top-hat sphere of radius 8 Mpc/h.
    """
    return sigmaR(cosmo, 8.0 / cosmo['h'])
