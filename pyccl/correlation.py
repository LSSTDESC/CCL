from pyccl import ccllib as lib
import pyccl.constants as const
from pyccl.pyutils import _cosmology_obj, check
import numpy as np

correlation_methods = {
    'fftlog':   const.CCL_CORR_FFTLOG,
    'bessel':   const.CCL_CORR_BESSEL,
    'legendre': const.CCL_CORR_LGNDRE,
}

correlation_types = {
    'gg': const.CCL_CORR_GG,
    'gl': const.CCL_CORR_GL,
    'l+': const.CCL_CORR_LP,
    'l-': const.CCL_CORR_LM,
}

def correlation(cosmo, ell, C_ell, theta, corr_type='gg', method='fftlog'):
    """
    Compute the angular correlation function.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        ell (array_like): Multipoles corresponding to the input angular power spectrum
        C_ell (array_like): Input angular power spectrum.
        theta (float or array_like): Angular separation(s) at which to calculate the angular correlation function (in degrees).
        corr_type (string): Type of correlation function. Choices: 'GG' (galaxy-galaxy), 'GL' (galaxy-shear), 'L+' (shear-shear, xi+), 'L-' (shear-shear, xi-).
        method (string, optional): Method to compute the correlation function. Choices: 'Bessel' (direct integration over Bessel function), 'FFTLog' (fast integration with FFTLog), 'Legendre' (brute-force sum over Legendre polynomials).
    Returns:
        Value(s) of the correlation function at the input angular separation(s).
    """
    cosmo_in = cosmo
    cosmo = _cosmology_obj(cosmo)
    status = 0

    # Convert to lower case
    corr_type = corr_type.lower()
    method = method.lower()

    if corr_type not in correlation_types.keys():
        raise KeyError("'%s' is not a valid correlation type." % corr_type)

    if method.lower() not in correlation_methods.keys():
        raise KeyError("'%s' is not a valid correlation method." % method)

    # Convert scalar input into an array
    scalar = False
    if isinstance(theta, float):
        scalar = True
        theta = np.array([theta,])

    # Call correlation function
    wth, status = lib.correlation_vec(cosmo, ell, C_ell, theta,
                                      correlation_types[corr_type],
                                      correlation_methods[method],
                                      len(theta), status)
    check(status, cosmo_in)
    if scalar: return wth[0]
    return wth
