
import ccllib as lib
from pyutils import _vectorize_fn

def linear_matter_power(cosmo, a, k):
    return _vectorize_fn(lib.linear_matter_power, 
                         lib.linear_matter_power_vec, cosmo, a, k)

def nonlin_matter_power(cosmo, a, k):
    return _vectorize_fn(lib.nonlin_matter_power, 
                         lib.nonlin_matter_power_vec, cosmo, a, k)

def sigmaR(cosmo, R):
    return _vectorize_fn(lib.sigmaR, 
                         lib.sigmaR_vec, cosmo, R)

def sigma8(cosmo):
    return lib.sigma8(cosmo)

