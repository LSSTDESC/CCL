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

#STATUS
#Comments
#strings
def correlation(cosmo,ell,cell,theta,corr_type='GG',method='FFTLog') :
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
