"""Smooth background quantities

CCL defines seven species types:

- 'matter': cold dark matter and baryons
- 'dark_energy': cosmological constant or otherwise
- 'radiation': relativistic species besides massless neutrinos (i.e.,
  only photons)
- 'curvature': curvature density
- 'neutrinos_rel': relativistic neutrinos
- 'neutrinos_massive': massive neutrinos

These strings define the `species` inputs to the functions below.
"""
from . import ccllib as lib
from .pyutils import _vectorize_fn, _vectorize_fn3, _vectorize_fn4

species_types = {
    'critical':                   lib.species_crit_label,
    'matter':                     lib.species_m_label,
    'dark_energy':                lib.species_l_label,
    'radiation':                  lib.species_g_label,
    'curvature':                  lib.species_k_label,
    'neutrinos_rel':              lib.species_ur_label,
    'neutrinos_massive':          lib.species_nu_label,
}


def growth_factor(cosmo, a):
    """Growth factor.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Growth factor, normalized to unity today.
    """
    return _vectorize_fn(lib.growth_factor,
                         lib.growth_factor_vec, cosmo, a)


def growth_factor_unnorm(cosmo, a):
    """Unnormalized growth factor.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Unnormalized growth factor, normalized to the
                             scale factor at early times.
    """
    return _vectorize_fn(lib.growth_factor_unnorm,
                         lib.growth_factor_unnorm_vec, cosmo, a)


def growth_rate(cosmo, a):
    """Growth rate defined as the logarithmic derivative of the
    growth factor, dlnD/dlna.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Growth rate.

    """
    return _vectorize_fn(lib.growth_rate,
                         lib.growth_rate_vec, cosmo, a)


def comoving_radial_distance(cosmo, a):
    """Comoving radial distance.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Comoving radial distance; Mpc.
    """
    return _vectorize_fn(lib.comoving_radial_distance,
                         lib.comoving_radial_distance_vec, cosmo, a)


def comoving_angular_distance(cosmo, a):
    """Comoving angular distance.

    .. note:: This quantity is otherwise known as the transverse
              comoving distance, and is NOT angular diameter
              distance or angular separation. The comovoing angular distance
              is defined such that the comoving distance between
              two objects at a fixed scale factor separated by an angle
              :math:`\theta` is :math:`\theta D_{T}(a)` where :math:`D_{T}(a)`
              is the comoving angular distance.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Comoving angular distance; Mpc.
    """
    return _vectorize_fn(lib.comoving_angular_distance,
                         lib.comoving_angular_distance_vec, cosmo, a)


def h_over_h0(cosmo, a):
    """Ratio of Hubble constant at `a` over Hubble constant today.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: H(a)/H0.
    """
    return _vectorize_fn(lib.h_over_h0,
                         lib.h_over_h0_vec, cosmo, a)


def luminosity_distance(cosmo, a):
    """Luminosity distance.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Luminosity distance; Mpc.
    """
    return _vectorize_fn(lib.luminosity_distance,
                         lib.luminosity_distance_vec, cosmo, a)


def distance_modulus(cosmo, a):
    """Distance Modulus, defined as 5 * log(luminosity distance / 10 pc).

    .. note :: The distance modulus can be used to convert between apparent
               and absolute magnitudes via m = M + distance modulus, where m
               is the apparent magnitude and M is the absolute magnitude.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Distance modulus at a.
    """
    return _vectorize_fn(lib.distance_modulus,
                         lib.distance_modulus_vec, cosmo, a)


def scale_factor_of_chi(cosmo, chi):
    """Scale factor, a, at a comoving radial distance chi.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        chi (float or array_like): Comoving radial distance(s); Mpc.

    Returns:
        float or array_like: Scale factor(s), normalized to 1 today.
    """
    return _vectorize_fn(lib.scale_factor_of_chi,
                         lib.scale_factor_of_chi_vec, cosmo, chi)


def omega_x(cosmo, a, species):
    """Density fraction of a given species at a redshift different than z=0.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        species (string): species type. Available:
            'matter': cold dark matter and baryons
            'dark_energy': cosmological constant or otherwise
            'radiation': relativistic species besides massless neutrinos (i.e.,
                only photons)
            'curvature': curvature density
            'neutrinos_rel': relativistic neutrinos
            'neutrinos_massive': massive neutrinos

    Returns:
        float or array_like: Density fraction of a given species at a
                             scale factor.
    """
    if species not in species_types.keys():
        raise ValueError("'%s' is not a valid species type. "
                         "Available options are: %s"
                         % (species, species_types.keys()))

    return _vectorize_fn3(lib.omega_x,
                          lib.omega_x_vec, cosmo, a, species_types[species])


def rho_x(cosmo, a, species, is_comoving=False):
    """Physical or comoving density as a function of scale factor.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        species (string): species type. Available:
            'matter': cold dark matter and baryons
            'dark_energy': cosmological constant or otherwise
            'radiation': relativistic species besides massless neutrinos
            'curvature': curvature density
            'neutrinos_rel': relativistic neutrinos
            'neutrinos_massive': massive neutrinos
        is_comoving (bool): either physical (False, default) or comoving (True)

    Returns:
        rho_x (float or array_like): Physical density of a given species
        at a scale factor.
    """
    if species not in species_types.keys():
        raise ValueError("'%s' is not a valid species type. "
                         "Available options are: %s"
                         % (species, species_types.keys()))

    return _vectorize_fn4(
        lib.rho_x, lib.rho_x_vec, cosmo, a,
        species_types[species], int(is_comoving))
