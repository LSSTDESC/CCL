
import ccllib as lib
from pyutils import _vectorize_fn

def growth_factor(cosmo, a):
    """Growth factor.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        growth_factor (float or array_like): Growth factor.

    """
    return _vectorize_fn(lib.growth_factor, 
                         lib.growth_factor_vec, cosmo, a)

def growth_factor_unnorm(cosmo, a):
    """Unnormalized growth factor.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        growth_factor_unnorm (float or array_like): Unnormalized growth factor.

    """
    return _vectorize_fn(lib.growth_factor_unnorm, 
                         lib.growth_factor_unnorm_vec, cosmo, a)

def growth_rate(cosmo, a):
    """Growth rate.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        growth_rate (float or array_like): Growth rate; .

    """
    return _vectorize_fn(lib.growth_rate, 
                         lib.growth_rate_vec, cosmo, a)

def comoving_radial_distance(cosmo, a):
    """Comoving radial distance.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        comoving_radial_distance (float or array_like): Comoving radial distance; Mpc.

    """
    return _vectorize_fn(lib.comoving_radial_distance, 
                         lib.comoving_radial_distance_vec, cosmo, a)

def comoving_angular_distance(cosmo, a):
    """Comoving angular distance.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        comoving_angular_distance (float or array_like): Comoving angular distance; Mpc.

    """
    return _vectorize_fn(lib.comoving_angular_distance, 
                         lib.comoving_angular_distance_vec, cosmo, a)

def h_over_h0(cosmo, a):
    """Ratio of Hubble constant at `a` over Hubble constant today.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        h_over_h0 (float or array_like): H(a)/H0.

    """
    return _vectorize_fn(lib.h_over_h0, 
                         lib.h_over_h0_vec, cosmo, a)

def luminosity_distance(cosmo, a):
    """Luminosity distance.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        luminosity_distance (float or array_like): Luminosity distance; Mpc.

    """
    return _vectorize_fn(lib.luminosity_distance, 
                         lib.luminosity_distance_vec, cosmo, a)

def scale_factor_of_chi(cosmo, chi):
    """Scale factor, a, at a comoving distance chi.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        chi (float or array_like): Comoving distance(s); Mpc.

    Returns:
        scale_factor_of_chi (float or array_like): Scale factor(s), normalized to 1 today.

    """
    return _vectorize_fn(lib.scale_factor_of_chi, 
                         lib.scale_factor_of_chi_vec, cosmo, chi)

def omega_m_z(cosmo, a):
    """Matter density fraction at a redshift different than z=0.

    Note: the name of this function (_z) is inconsistent with its 
    input names (a; scale factor) and same for the C-code.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        omega_m_z (float or array_like): Matter density fraction at a scale factor.

    """
    return _vectorize_fn(lib.omega_m_z, 
                         lib.omega_m_z_vec, cosmo, a,
                         returns_status=False)

