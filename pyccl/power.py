
from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2

def linear_matter_power(cosmo, k, a):
    """The linear matter power spectrum; Mpc^3.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        linear_matter_power (float or array_like): Linear matter power spectrum; Mpc^3.

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
        nonlin_matter_power (float or array_like): Nonlinear matter power spectrum; Mpc^3.

    """
    return _vectorize_fn2(lib.nonlin_matter_power, 
                          lib.nonlin_matter_power_vec, cosmo, k, a)

def sigmaR(cosmo, R):
    """RMS variance in a top-hat sphere of radius R.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.

    Returns:
        sigmaR (float or array_like): RMS variance in top-hat sphere.

    """
    return _vectorize_fn(lib.sigmaR, 
                         lib.sigmaR_vec, cosmo, R)

def sigma8(cosmo):
    """RMS variance in a top-hat sphere of radius 8 Mpc.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.

    Returns:
        sigma8 (float): RMS variance in top-hat sphere of radius 8 Mpc.

    """
    return sigmaR(cosmo,8./cosmo['h'])

