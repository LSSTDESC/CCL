
import ccllib as lib
from pyutils import _vectorize_fn, _vectorize_fn2

def massfunc(cosmo, halo_mass, a):
    return _vectorize_fn2(lib.massfunc, 
                          lib.massfunc_vec, cosmo, halo_mass, a)

def massfunc_m2r(cosmo, halo_mass):
    return _vectorize_fn(lib.massfunc_m2r, 
                         lib.massfunc_m2r_vec, cosmo, halo_mass)

def sigmaM(cosmo, halo_mass, a):
    return _vectorize_fn2(lib.sigmaM, 
                          lib.sigmaM_vec, cosmo, halo_mass, a)

def halo_bias(cosmo, halo_mass, a):
    return _vectorize_fn2(lib.halo_bias, 
                          lib.halo_bias_vec, cosmo, halo_mass, a)
