from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2

def linear_matter_power(cosmo, k, a):
    """The linear matter power spectrum; Mpc^3.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
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
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Nonlinear matter power spectrum; Mpc^3.

    """
    return _vectorize_fn2(lib.nonlin_matter_power, 
                          lib.nonlin_matter_power_vec, cosmo, k, a)

def sigmaR(cosmo, R):
    """RMS variance in a top-hat sphere of radius R in Mpc.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.

    Returns:
        float or array_like: RMS variance in top-hat sphere; Mpc.

    """
    return _vectorize_fn(lib.sigmaR, 
                         lib.sigmaR_vec, cosmo, R)

def sigma8(cosmo):
    """RMS variance in a top-hat sphere of radius 8 Mpc/h.

    .. note:: 8 Mpc/h is rescaled based on the Hubble constant.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.

    Returns:
        float: RMS variance in top-hat sphere of radius 8 Mpc/h.

    """
    return sigmaR(cosmo,8./cosmo['h'])

