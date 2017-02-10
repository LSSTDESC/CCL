
import ccllib as lib
from pyutils import _vectorize_fn

def growth_factor(cosmo, a):
    """
    growth factor
    
    Parameters
    ----------
    cosmo : ccl.cosmology
        Input cosmological parameters.
    
    a : float or array_like
        Scale factor, normalized to 1 today.
    
    Returns
    -------
    growth_factor : float or array_like
        Growth factor.
    """
    return _vectorize_fn(lib.growth_factor, 
                         lib.growth_factor_vec, cosmo, a)

def growth_factor_unnorm(cosmo, a):
    return _vectorize_fn(lib.growth_factor_unnorm, 
                         lib.growth_factor_unnorm_vec, cosmo, a)

def growth_rate(cosmo, a):
    return _vectorize_fn(lib.growth_rate, 
                         lib.growth_rate_vec, cosmo, a)

def comoving_radial_distance(cosmo, a):
    return _vectorize_fn(lib.comoving_radial_distance, 
                         lib.comoving_radial_distance_vec, cosmo, a)

def h_over_h0(cosmo, a):
    return _vectorize_fn(lib.h_over_h0, 
                         lib.h_over_h0_vec, cosmo, a)

def luminosity_distance(cosmo, a):
    return _vectorize_fn(lib.luminosity_distance, 
                         lib.luminosity_distance_vec, cosmo, a)

def scale_factor_of_chi(cosmo, a):
    return _vectorize_fn(lib.scale_factor_of_chi, 
                         lib.scale_factor_of_chi_vec, cosmo, a)

def omega_m_z(cosmo, a):
    return _vectorize_fn(lib.omega_m_z, 
                         lib.omega_m_z_vec, cosmo, a)

