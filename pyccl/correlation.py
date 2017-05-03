import ccllib as lib
import numpy as np
from pyutils import _cosmology_obj, check
from cls import _cltracer_obj

def correlation(t,cosmo,cltracer1,cltracer2,i_bessel):

    """The correlation function, w/o FFTlog implementation.

    Args:
        t: the angle in radians where to sample the correlation (vector or scalar)
        cosmo (:obj:`Cosmology`): Either a ccl_cosmology or a Cosmology object
        cltracer1 (:obj:): A Cl tracer.
        cltracer2 (:obj:): A Cl tracer.
        index of bessel function
    Returns:
        correlation function (array_like)

    """
    # Access ccl_cosmology object
    cosmo = _cosmology_obj(cosmo)
    status = 0
    scalar = False
    
    clt1 = _cltracer_obj(cltracer1)
    clt2 = _cltracer_obj(cltracer2)
    
    if isinstance(t, float):
        scalar = True
        t=np.array([t,])

    if isinstance(t, np.ndarray):
        return lib.correlation_vec(cosmo, clt1,clt2,i_bessel, t, t.size)
    else:
        return lib.single_tracer_corr(t,cosmo,clt1,clt2,i_bessel)
  
