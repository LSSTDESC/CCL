from . import ccllib as lib
from .core import check
from .background import comoving_radial_distance, growth_rate
import numpy as np
import collections

NoneArr = np.array([])

def get_density_kernel(cosmo, dndz):
    z_n, n = _check_array_params(dndz)
    chi = comoving_radial_distance(cosmo, 1./(1.+z_n))
    status = 0
    wchi, status = lib.get_number_counts_kernel_wrapper(cosmo.cosmo, z_n, n, len(z_n), status)
    check(status)
    return chi, wchi

def get_lensing_kernel(cosmo, dndz, mag_bias=None):
    z_n, n = _check_array_params(dndz)
    has_magbias = mag_bias is not None
    z_s, s = _check_array_params(mag_bias)

    # Calculate number of samples in chi
    nchi = lib.get_nchi_lensing_kernel_wrapper(z_n)
    # Compute array of chis
    status = 0
    chi, status = lib.get_chis_lensing_kernel_wrapper(cosmo.cosmo, z_n[-1],
                                                      nchi, status)
    # Compute kernel
    wchi, status = lib.get_lensing_kernel_wrapper(cosmo.cosmo,
                                                  z_n, n, z_n[-1],
                                                  int(has_magbias), z_s, s,
                                                  chi, nchi, status)
    check(status)
    return chi, wchi

def get_kappa_kernel(cosmo, z_source, nsamples):
    chi_source = comoving_radial_distance(cosmo, 1./(1.+z_source))
    chi = np.linspace(0, chi_source, nsamples)
    
    status = 0
    wchi, status = lib.get_kappa_kernel_wrapper(cosmo.cosmo, chi_source,
                                                chi, nsamples, status)
    check(status)
    return chi, wchi

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
            a_s, lk_s, tka_s = _check_array_params(transfer_ka, arr3=True)
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

        kernel_d = None
        if bias is not None:  # Has density term
            # Kernel
            if kernel_d is None:
                kernel_d = get_density_kernel(cosmo, dndz)
            # Transfer
            z_b, b = _check_array_params(bias)
            # Reverse order for increasing a
            t_a = (1./(1+z_b[::-1]), b[::-1]) 
            self.add_tracer(cosmo, kernel=kernel_d, transfer_a=t_a)
        if has_rsd:  # Has RSDs
            # Kernel
            if kernel_d is None:
                kernel_d = get_density_kernel(cosmo, dndz)
            # Transfer (growth rate)
            a_s = 1./(1+z_b[::-1])
            t_a = (a_s, -growth_rate(cosmo, a_s))
            self.add_tracer(cosmo, kernel=kernel_d, transfer_a=t_a, der_bessel=2)
        if mag_bias is not None:  # Has magnification bias
            # Kernel
            chi,w = get_lensing_kernel(cosmo, dndz, mag_bias=mag_bias)
            # Multiply by -2 for magnification
            kernel_m=(chi, -2 * w)
            self.add_tracer(cosmo, kernel=kernel_m, der_bessel=-1, der_angles=1)

class WeakLensingTracer(Tracer):
    def __init__(self, cosmo, dndz, has_shear=True, ia_bias=None):
        self.trc=[]
        if has_shear:
            # Kernel
            kernel_l = get_lensing_kernel(cosmo, dndz)
            self.add_tracer(cosmo, kernel=kernel_l, der_bessel=-1, der_angles=2)
        if ia_bias is not None:  # Has magnification bias
            status = 0
            # Kernel
            kernel_i = get_density_kernel(cosmo, dndz)
            # Transfer
            z_a, a = _check_array_params(ia_bias)
            # Reverse order for increasing a
            t_a = (1./(1+z_a[::-1]), a[::-1])
            self.add_tracer(cosmo, kernel=kernel_i, transfer_a=t_a,
                            der_bessel=-1, der_angles=2)

class CMBLensingTracer(Tracer):
    def __init__(self, cosmo, z_source, n_samples=100):
        self.trc=[]
        kernel = get_kappa_kernel(cosmo, z_source, n_samples)
        self.add_tracer(cosmo, kernel=kernel, der_bessel=-1, der_angles=1)

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
