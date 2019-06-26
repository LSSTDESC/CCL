from . import ccllib as lib
from .core import check
import numpy as np
import collections

NoneArr = np.array([])

class Tracer(object):
    def __init__(self):
        # Do nothing, just initialize list of tracers
        self.trc=[]

    def add_tracer(self, cosmo, kernel=None,
                   transfer_ka=None, transfer_k=None, transfer_a=None,
                   der_bessel=0, der_angles=0,
                   is_logt=False, extrap_order_lok=0, extrap_order_hik=2):
        is_factorizable = transfer_ka is None
        is_k_constant = (transfer_ka is None) and (transfer_k is None)
        is_a_constant = (transfer_ka is None) and (transfer_a is None)
        is_kernel_constant = kernel is None

        chi_s, wchi_s = _check_array_params(kernel)
        if is_factorizable:
            a_s, ta_s = _check_array_params(transfer_a)
            lk_s, tk_s = _check_array_params(transfer_k)
            tka_s = NoneArr
            if (not is_a_constant) and (a_s.shape != ta_s.shape):
                raise ValueError("Time-dependent transfer arrays should have the same shape")
            if (not is_k_constant) and (lk_s.shape != tk_s.shape):
                raise ValueError("Scale-dependent transfer arrays should have the same shape") 
        else:
            a_s, lk_s, tka_s = _check_array_params(transfer_ka)
            if tka_s.shape != (len(a_s),len(lk_s)):
                raise ValueError("2D transfer array has inconsistent shape. Should be (na,nk)")
            tka_s = tka_s.flatten()
            ta_s = NoneArr
            tk_s = NoneArr

        status = 0
        ret = lib.cl_tracer_t_new_wrapper(cosmo.cosmo,
                                          int(der_bessel),
                                          int(der_angles),
                                          chi_s, wchi_s,
                                          a_s, lk_s,
                                          tka_s, tk_s, ta_s,
                                          int(is_logt),
                                          int(is_factorizable),
                                          int(is_k_constant),
                                          int(is_a_constant),
                                          int(is_kernel_constant),
                                          int(extrap_order_lok),
                                          int(extrap_order_hik),
                                          status)
        self.trc.append(check_returned_tracer(ret))
                                                    
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
        
def _check_array_params(f_arg, arr3=False):
    """Check whether an argument `f_arg` passed into the constructor of
    Tracer() is valid.

    If the argument is set to `None`, it will be replaced with a special array
    that signals to the CCL wrapper that this argument is NULL.
    """
    if f_arg is None:
        # Return empty array if argument is None
        f1 = NoneArr
        f2 = NoneArr
        f3 = NoneArr
    else:
        f1 = np.atleast_1d(np.array(f_arg[0], dtype=float))
        f2 = np.atleast_1d(np.array(f_arg[1], dtype=float))
        if arr3:
            f3 = np.atleast_1d(np.array(f_arg[2], dtype=float))
    if arr3:
        return f1, f2, f3
    else:
        return f1, f2
