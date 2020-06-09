from . import ccllib as lib
from .pyutils import _vectorize_fn2
import numpy as np
from .core import check


def linear_matter_power(cosmo, k, a):
    """The linear matter power spectrum; Mpc^3.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Linear matter power spectrum; Mpc^3.
    """
    cosmo.compute_linear_power()
    return _vectorize_fn2(lib.linear_matter_power,
                          lib.linear_matter_power_vec, cosmo, k, a)


def nonlin_matter_power(cosmo, k, a):
    """The nonlinear matter power spectrum; Mpc^3.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Nonlinear matter power spectrum; Mpc^3.
    """
    cosmo.compute_nonlin_power()
    return _vectorize_fn2(lib.nonlin_matter_power,
                          lib.nonlin_matter_power_vec, cosmo, k, a)


def sigmaM(cosmo, M, a):
    """Root mean squared variance for the given halo mass of the linear power
    spectrum; Msun.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        M (float or array_like): Halo masses; Msun.
        a (float): scale factor.

    Returns:
        float or array_like: RMS variance of halo mass.
    """
    cosmo.compute_sigma()

    # sigma(M)
    logM = np.log10(np.atleast_1d(M))
    status = 0
    sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                len(logM), status)
    check(status)
    if np.ndim(M) == 0:
        sigM = sigM[0]
    return sigM


def sigmaR(cosmo, R, a=1.):
    """RMS variance in a top-hat sphere of radius R in Mpc.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.
        a (float): optional scale factor; defaults to a=1

    Returns:
        float or array_like: RMS variance in the density field in top-hat \
            sphere; Mpc.
    """
    cosmo.compute_linear_power()
    return _vectorize_fn2(lib.sigmaR, lib.sigmaR_vec, cosmo, R, a)


def sigmaV(cosmo, R, a=1.):
    """RMS variance in the displacement field in a top-hat sphere of radius R.
    The linear displacement field is the gradient of the linear density field.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        R (float or array_like): Radius; Mpc.
        a (float): optional scale factor; defaults to a=1

    Returns:
        sigmaV (float or array_like): RMS variance in the displacement field \
            in top-hat sphere.
    """
    cosmo.compute_linear_power()
    return _vectorize_fn2(lib.sigmaV, lib.sigmaV_vec, cosmo, R, a)


def sigma8(cosmo):
    """RMS variance in a top-hat sphere of radius 8 Mpc/h.

    .. note:: 8 Mpc/h is rescaled based on the chosen value of the Hubble
              constant within `cosmo`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.

    Returns:
        float: RMS variance in top-hat sphere of radius 8 Mpc/h.
    """
    cosmo.compute_linear_power()
    return sigmaR(cosmo, 8.0 / cosmo['h'])
