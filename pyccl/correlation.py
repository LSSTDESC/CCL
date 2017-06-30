import ccllib as lib
import constants as const
from pyutils import _cosmology_obj, check
import numpy as np

correlation_methods = {
    'FFTLog': const.CCL_CORR_FFTLOG,
    'Bessel': const.CCL_CORR_BESSEL,
    'Legendre': const.CCL_CORR_LGNDRE,
}

correlation_types = {
    'GG': const.CCL_CORR_GG,
    'GL': const.CCL_CORR_GL,
    'L+': const.CCL_CORR_LP,
    'L-': const.CCL_CORR_LM,
}

def correlation(cosmo,ell,cell,theta,corr_type='GG',method='FFTLog') :
    """
    Compute the angular correlation function.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        ell (array_like): multipoles corresponding input angular power spectrum
        cell (array_like): input angular power spectrum
        theta (float or array_like): angular separation(s) at which the angular correlation function is requested (in degrees).
        corr_type (string): type of correlation function. Choices: 'GG' (galaxy-galaxy), 'GL' (galaxy-shear), 'LP' (shear-shear, xi+), 'LM' (shear-shear, xi-).
        method (string): method to compute the correlation function. Choices: 'Bessel' (direct integration over Bessel function), 'FFTLog' (fast integration with FFTLog), 'Legendre' (brute-force sum over Legendre polynomials).
    Returns:
        Value(s) of the correlation function at the input angular separation(s).
    """

    cosmo = _cosmology_obj(cosmo)
    status = 0

    if corr_type not in correlation_types.keys():
        raise KeyError("'%s' is not a valid correlation type."%corr_type)

    if method not in correlation_methods.keys():
        raise KeyError("'%s' is not a valid correlation method."%method)

    scalar=False
    if isinstance(theta, float):
        scalar = True
        theta=np.array([theta,])

    wth,status=lib.correlation_vec(cosmo,ell,cell,theta,correlation_types[corr_type],
                                   correlation_methods[method],len(theta),status)
    check(status)

    if scalar :
        return wth[0]
    else :
        return wth
