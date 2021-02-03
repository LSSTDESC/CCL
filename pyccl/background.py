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
import numpy as np
from . import ccllib as lib
from .pyutils import _vectorize_fn, _vectorize_fn3
from .pyutils import _vectorize_fn4, _vectorize_fn5

species_types = {
    'critical': lib.species_crit_label,
    'matter': lib.species_m_label,
    'dark_energy': lib.species_l_label,
    'radiation': lib.species_g_label,
    'curvature': lib.species_k_label,
    'neutrinos_rel': lib.species_ur_label,
    'neutrinos_massive': lib.species_nu_label,
}


def growth_factor(cosmo, a):
    """Growth factor.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Growth factor, normalized to unity today.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor,
                         lib.growth_factor_vec, cosmo, a)


def growth_factor_unnorm(cosmo, a):
    """Unnormalized growth factor.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Unnormalized growth factor, normalized to \
            the scale factor at early times.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor_unnorm,
                         lib.growth_factor_unnorm_vec, cosmo, a)


def growth_rate(cosmo, a):
    """Growth rate defined as the logarithmic derivative of the
    growth factor, dlnD/dlna.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Growth rate.

    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_rate,
                         lib.growth_rate_vec, cosmo, a)


def comoving_radial_distance(cosmo, a):
    """Comoving radial distance.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Comoving radial distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_radial_distance,
                         lib.comoving_radial_distance_vec, cosmo, a)


def comoving_angular_distance(cosmo, a):
    """Comoving angular distance.

    .. note:: This quantity is otherwise known as the transverse
              comoving distance, and is NOT angular diameter
              distance or angular separation. The comoving angular distance
              is defined such that the comoving distance between
              two objects at a fixed scale factor separated by an angle
              :math:`\\theta` is :math:`\\theta r_{A}(a)` where
              :math:`r_{A}(a)` is the comoving angular distance.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Comoving angular distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_angular_distance,
                         lib.comoving_angular_distance_vec, cosmo, a)


def angular_diameter_distance(cosmo, a1, a2=None):
    """Angular diameter distance.

    .. note:: The angular diameter distance in Mpc from scale factor
              a1 to scale factor a2. If a2 is not provided, it is assumed that
              the distance will be calculated between 1 and a1. Note that a2
              has to be smaller than a1.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a1 (float or array_like): Scale factor(s), normalized to 1 today.
        a2 (float or array_like): Scale factor(s), normalized to 1 today,
        optional.

    Returns:
        float or array_like: angular diameter distance; Mpc.
    """
    cosmo.compute_distances()
    if(a2 is not None):
        return _vectorize_fn5(lib.angular_diameter_distance,
                              lib.angular_diameter_distance_vec,
                              cosmo, a1, a2)
    else:
        if(isinstance(a1, float) or isinstance(a1, int)):
            return _vectorize_fn5(lib.angular_diameter_distance,
                                  lib.angular_diameter_distance_vec,
                                  cosmo, 1., a1)
        else:
            return _vectorize_fn5(lib.angular_diameter_distance,
                                  lib.angular_diameter_distance_vec,
                                  cosmo, np.ones(len(a1)), a1)


def h_over_h0(cosmo, a):
    """Ratio of Hubble constant at `a` over Hubble constant today.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: H(a)/H0.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.h_over_h0,
                         lib.h_over_h0_vec, cosmo, a)


def luminosity_distance(cosmo, a):
    """Luminosity distance.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Luminosity distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.luminosity_distance,
                         lib.luminosity_distance_vec, cosmo, a)


def distance_modulus(cosmo, a):
    """Distance Modulus, defined as 5 * log10(luminosity distance / 10 pc).

    .. note :: The distance modulus can be used to convert between apparent
               and absolute magnitudes via m = M + distance modulus, where m
               is the apparent magnitude and M is the absolute magnitude.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        float or array_like: Distance modulus at a.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.distance_modulus,
                         lib.distance_modulus_vec, cosmo, a)


def scale_factor_of_chi(cosmo, chi):
    """Scale factor, a, at a comoving radial distance chi.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        chi (float or array_like): Comoving radial distance(s); Mpc.

    Returns:
        float or array_like: Scale factor(s), normalized to 1 today.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.scale_factor_of_chi,
                         lib.scale_factor_of_chi_vec, cosmo, chi)


def omega_x(cosmo, a, species):
    """Density fraction of a given species at a redshift different than z=0.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        species (string): species type. Should be one of

            - 'matter': cold dark matter, massive neutrinos, and baryons
            - 'dark_energy': cosmological constant or otherwise
            - 'radiation': relativistic species besides massless neutrinos
            - 'curvature': curvature density
            - 'neutrinos_rel': relativistic neutrinos
            - 'neutrinos_massive': massive neutrinos

    Returns:
        float or array_like: Density fraction of a given species at a \
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
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        species (string): species type. Should be one of

            - 'matter': cold dark matter, massive neutrinos, and baryons
            - 'dark_energy': cosmological constant or otherwise
            - 'radiation': relativistic species besides massless neutrinos
            - 'curvature': curvature density
            - 'neutrinos_rel': relativistic neutrinos
            - 'neutrinos_massive': massive neutrinos

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
