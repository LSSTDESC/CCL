
from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn3

species_types = {
    'matter':      lib.omega_m_label,
    'dark_energy': lib.omega_l_label,
    'radiation':   lib.omega_g_label,
    'curvature':   lib.omega_k_label,
    'neutrinos_rel': lib.omega_ur_label,
    'neutrinos_massive': lib.omega_nu_label,
}

species_types_rho = {
    'critical':                   lib.rho_crit_label,
    'critical_comoving':          lib.rho_crit_comoving_label,
    'matter':                     lib.rho_m_label,
    'matter_comoving':            lib.rho_m_comoving_label,
    'dark_energy':                lib.rho_l_label,
    'dark_energy_comoving':       lib.rho_l_comoving_label,
    'radiation':                  lib.rho_g_label,
    'radiation_comoving':         lib.rho_g_comoving_label,
    'curvature':                  lib.rho_k_label,
    'curvature_comoving':         lib.rho_k_comoving_label,
    'neutrinos_rel':              lib.rho_ur_label,
    'neutrinos_rel_comoving':     lib.rho_ur_comoving_label,
    'neutrinos_massive':          lib.rho_nu_label,
    'neutrinos_massive_comoving': lib.rho_nu_comoving_label,
}

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
        growth_factor_unnorm (float or array_like): Unnormalized growth factor, normalized to the scale factor at early times.

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
    NOTE this quantity is otherwise known as the transverse comoving distance, and is NOT angular diameter
    distance or angular separation
    
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

def distance_modulus(cosmo, a):
    """Distance Modulus
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        distance_modulus (float or array_like): Distance modulus at a.  
    """
    return _vectorize_fn(lib.distance_modulus,
                         lib.distance_modulus_vec, cosmo, a)

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

def omega_x(cosmo, a, label):
    """Density fraction of a given species at a redshift different than z=0.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        label (string): species type. Available: 'matter', 'dark_energy',
                        'radiation', 'curvature', 'neutrinos_rel', and 'neutrinos_massive'

    Returns:
        omega_x (float or array_like): Density fraction of a given species
        at a scale factor.

    """
    if label not in species_types.keys() :
        raise ValueError( "'%s' is not a valid species type. "
                          "Available options are: %s" \
                         % (label,species_types.keys()) )

    return _vectorize_fn3(lib.omega_x, 
                          lib.omega_x_vec, cosmo, a, species_types[label])

def rho_x(cosmo, a, label):
    """Physical density as a function of scale factor.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        label (string): species type. Available: 'critical', 'critical_comoving', 'matter', 'matter_comoving', 
                        'dark_energy', 'dark_energy_comoving', 'radiation', 'radiation_comoving',
                        'curvature', 'curvature_comoving', 'neutrinos_rel', 'neutrinos_rel_comoving', 
                        and 'neutrinos_massive', 'neutrinos_massive_comoving'

    Returns:
        rho_x (float or array_like): Physical density of a given species
        at a scale factor.

    """
    if label not in species_types_rho.keys() :
        raise ValueError( "'%s' is not a valid species type. "
                          "Available options are: %s" \
                         % (label,species_types_rho.keys()) )

    return _vectorize_fn3(lib.rho_x, 
                          lib.rho_x_vec, cosmo, a, species_types_rho[label])
