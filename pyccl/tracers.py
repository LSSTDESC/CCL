from . import ccllib as lib
from .core import check
import numpy as np
import collections

NoneArr = np.array([])

class Tracer(object):
    def __init__(self, cosmo, kernel, transfer, der_bessel, der_angles):
        self.trc=[]

    def __del__(self):
        if hasattr(self, 'trc'):
            for t in self.trc:
                lib.cl_tracer_t_free(t)

class NumberCountsTracer(Tracer):
    def __init__(self, cosmo, has_rsd, dndz, bias, mag_bias=None):
        self.trc=[]
        z_n, n = _check_array_params(dndz)
        z_b, b = _check_array_params(bias)
        z_s, s = _check_array_params(mag_bias)
        if bias is not None:  # Has density term
            status = 0
            ret = lib.tracer_get_nc_dens(cosmo.cosmo, z_n, n, z_b, b, status)
            self.trc.append(check_returned_tracer(ret))
        if has_rsd:  # Has RSDs
            status = 0
            ret = lib.tracer_get_nc_rsd(cosmo.cosmo, z_n, n, status)
            self.trc.append(check_returned_tracer(ret))
        if mag_bias is not None:  # Has magnification bias
            status = 0
            ret = lib.tracer_get_nc_mag(cosmo.cosmo, z_n, n, z_s, s, status)
            self.trc.append(check_returned_tracer(ret))

class WeakLensingTracer(Tracer):
    def __init__(self, cosmo, dndz, has_shear=True, ia_bias=None):
        self.trc=[]
        z_n, n = _check_array_params(dndz)
        z_a, a = _check_array_params(ia_bias)
        if has_shear:  # Has RSDs 
            status = 0
            ret = lib.tracer_get_wl_shear(cosmo.cosmo, z_n, n, status)
            self.trc.append(check_returned_tracer(ret))
        if ia_bias is not None:  # Has magnification bias
            status = 0
            ret = lib.tracer_get_wl_ia(cosmo.cosmo, z_n, n, z_a, a, status)
            self.trc.append(check_returned_tracer(ret))

class CMBLensingTracer(Tracer):
    def __init__(self, cosmo, z_source):
        self.trc=[]
        status = 0
        ret = lib.tracer_get_kappa(cosmo.cosmo, z_source, status)
        self.trc.append(check_returned_tracer(ret))

def check_returned_tracer(return_val):
    if (isinstance(return_val, int)):
        check(return_val)
        tr = None
    else:
        tr, _ = return_val
    return tr
        
def _check_array_params(f_arg):
    """Check whether an argument `f_arg` passed into the constructor of
    Tracer() is valid.

    If the argument is set to `None`, it will be replaced with a special array
    that signals to the CCL wrapper that this argument is NULL.
    """
    if f_arg is None:
        # Return empty array if argument is None
        f = NoneArr
        z_f = NoneArr
    else:
        z_f = np.atleast_1d(np.array(f_arg[0], dtype=float))
        f = np.atleast_1d(np.array(f_arg[1], dtype=float))
    return z_f, f
                                
