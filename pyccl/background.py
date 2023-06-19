"""Smooth background quantities

CCL defines seven species types:

* 'matter': cold dark matter and baryons
* 'dark_energy': cosmological constant or otherwise
* 'radiation': relativistic species besides massless neutrinos (i.e., only photons)
* 'curvature': curvature density
* 'neutrinos_rel': relativistic neutrinos
* 'neutrinos_massive': massive neutrinos

These strings define the `species` inputs to the functions below.
"""  # noqa
__all__ = (
    "Species",
    "h_over_h0",
    "comoving_radial_distance",
    "scale_factor_of_chi",
    "comoving_angular_distance",
    "angular_diameter_distance",
    "luminosity_distance",
    "distance_modulus",
    "sigma_critical",
    "omega_x",
    "rho_x",
    "growth_factor",
    "growth_factor_unnorm",
    "growth_rate",
)

from enum import Enum

import numpy as np

from . import lib, physical_constants
from .pyutils import (
    _vectorize_fn,
    _vectorize_fn3,
    _vectorize_fn4,
    _vectorize_fn5,
)


class Species(Enum):
    CRITICAL = "critical"
    MATTER = "matter"
    DARK_ENERGY = "dark_energy"
    RADIATION = "radiation"
    CURVATURE = "curvature"
    NEUTRINOS_REL = "neutrinos_rel"
    NEUTRINOS_MASSIVE = "neutrinos_massive"


species_types = {
    "critical": lib.species_crit_label,
    "matter": lib.species_m_label,
    "dark_energy": lib.species_l_label,
    "radiation": lib.species_g_label,
    "curvature": lib.species_k_label,
    "neutrinos_rel": lib.species_ur_label,
    "neutrinos_massive": lib.species_nu_label,
}


def h_over_h0(cosmo, a):
    """Ratio of Hubble constant at `a` over Hubble constant today.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): H(a)/H0.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.h_over_h0, lib.h_over_h0_vec, cosmo, a)


def comoving_radial_distance(cosmo, a):
    """Comoving radial distance.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Comoving radial distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(
        lib.comoving_radial_distance,
        lib.comoving_radial_distance_vec,
        cosmo,
        a,
    )


def scale_factor_of_chi(cosmo, chi):
    """Scale factor, a, at a comoving radial distance chi.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        chi (:obj:`float` or `array`): Comoving radial distance(s); Mpc.

    Returns:
        (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
    """
    cosmo.compute_distances()
    return _vectorize_fn(
        lib.scale_factor_of_chi, lib.scale_factor_of_chi_vec, cosmo, chi
    )


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
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Comoving angular distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(
        lib.comoving_angular_distance,
        lib.comoving_angular_distance_vec,
        cosmo,
        a,
    )


def angular_diameter_distance(cosmo, a1, a2=None):
    """Angular diameter distance.

    The angular diameter distance in Mpc from scale factor
    `a1` to scale factor `a2`. If `a2` is not provided, it is
    assumed that the distance will be calculated between 1 and
    `a1`.

    .. note:: `a2` has to be smaller than `a1` (i.e. a source at
              `a2` is behind one at `a1`). You can compute the
              distance between a single lens at `a1` and multiple
              sources at `a2` by passing a scalar `a1`.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a1 (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
        a2 (:obj:`float` or `array`): Scale factor(s), normalized to 1 today,
            optional.

    Returns:
        (:obj:`float` or `array`): angular diameter distance; Mpc.
    """
    cosmo.compute_distances()
    if a2 is not None:
        # One lens, multiple sources
        if (np.ndim(a1) == 0) and (np.ndim(a2) != 0):
            a1 = np.full(len(a2), a1)
        return _vectorize_fn5(
            lib.angular_diameter_distance,
            lib.angular_diameter_distance_vec,
            cosmo,
            a1,
            a2,
        )
    else:
        if isinstance(a1, (int, float)):
            return _vectorize_fn5(
                lib.angular_diameter_distance,
                lib.angular_diameter_distance_vec,
                cosmo,
                1.0,
                a1,
            )
        else:
            return _vectorize_fn5(
                lib.angular_diameter_distance,
                lib.angular_diameter_distance_vec,
                cosmo,
                np.ones(len(a1)),
                a1,
            )


def luminosity_distance(cosmo, a):
    """Luminosity distance.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Luminosity distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(
        lib.luminosity_distance, lib.luminosity_distance_vec, cosmo, a
    )


def distance_modulus(cosmo, a):
    """Distance Modulus, defined as 5 * log10(luminosity distance / 10 pc).

    .. note :: The distance modulus can be used to convert between apparent
               and absolute magnitudes via m = M + distance modulus, where m
               is the apparent magnitude and M is the absolute magnitude.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Distance modulus at a.
    """
    cosmo.compute_distances()
    return _vectorize_fn(
        lib.distance_modulus, lib.distance_modulus_vec, cosmo, a
    )


def sigma_critical(cosmo, *, a_lens, a_source):
    """Returns the critical surface mass density.

    .. math::
         \\Sigma_{\\mathrm{crit}} = \\frac{c^2}{4\\pi G}
         \\frac{D_{\\rm{s}}}{D_{\\rm{l}}D_{\\rm{ls}}},

    where :math:`c` is the speed of light, :math:`G` is the
    gravitational constant, and :math:`D_i` is the angular diameter
    distance. The labels :math:`i = \\{s,\\,l,\\,ls\\}` denote the distances
    to the source, lens, and between source and lens, respectively.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        a_lens (:obj:`float`): lens' scale factor.
        a_source (:obj:`float` or `array`): source's scale factor.

    Returns:
        (:obj:`float` or `array`): :math:`\\Sigma_{\\mathrm{crit}}` in units
        of :math:`M_{\\odot}/{\\rm Mpc}^2`
    """
    Ds = angular_diameter_distance(cosmo, a_source, a2=None)
    Dl = angular_diameter_distance(cosmo, a_lens, a2=None)
    Dls = angular_diameter_distance(cosmo, a_lens, a_source)
    A = (
        physical_constants.CLIGHT**2
        * physical_constants.MPC_TO_METER
        / (
            4.0
            * np.pi
            * physical_constants.GNEWT
            * physical_constants.SOLAR_MASS
        )
    )

    Sigma_crit = A * Ds / (Dl * Dls)
    return Sigma_crit


def omega_x(cosmo, a, species):
    """Density fraction of a given species at a redshift different than z=0.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
        species (:obj:`str`): species type. Should be one of

            * 'matter': cold dark matter, massive neutrinos, and baryons
            * 'dark_energy': cosmological constant or otherwise
            * 'radiation': relativistic species besides massless neutrinos
            * 'curvature': curvature density
            * 'neutrinos_rel': relativistic neutrinos
            * 'neutrinos_massive': massive neutrinos
            * 'critical'

    Returns:
        (:obj:`float` or `array`): Density fraction of a given species at a \
                             scale factor.
    """
    # TODO: Replace docstring enum with ref to Species.
    if species not in species_types:
        raise ValueError(f"Unknown species {species}.")

    return _vectorize_fn3(
        lib.omega_x, lib.omega_x_vec, cosmo, a, species_types[species]
    )


def rho_x(cosmo, a, species, *, is_comoving=False):
    """Physical or comoving density as a function of scale factor.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
        species (:obj:`str`): species type. Should be one of

            - 'matter': cold dark matter, massive neutrinos, and baryons
            - 'dark_energy': cosmological constant or otherwise
            - 'radiation': relativistic species besides massless neutrinos
            - 'curvature': curvature density
            - 'neutrinos_rel': relativistic neutrinos
            - 'neutrinos_massive': massive neutrinos
            - 'critical'

        is_comoving (:obj:`bool`): either physical (False, default) or
            comoving (True)

    Returns:
        (:obj:`float` or `array`): Physical density of a given species
        at a scale factor, in units of :math:`M_\\odot / {\\rm Mpc}^3`.
    """
    # TODO: Replace docstring enum with ref to Species.
    if species not in species_types:
        raise ValueError(f"Unknown species {species}.")

    return _vectorize_fn4(
        lib.rho_x,
        lib.rho_x_vec,
        cosmo,
        a,
        species_types[species],
        int(is_comoving),
    )


def growth_factor(cosmo, a):
    """Growth factor.

    .. warning:: CCL is not able to compute the scale-dependent growth
                 factor for cosmologies with massive neutrinos.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Growth factor, normalized to unity today.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor, lib.growth_factor_vec, cosmo, a)


def growth_factor_unnorm(cosmo, a):
    """Unnormalized growth factor.

    .. warning:: CCL is not able to compute the scale-dependent growth
                 factor for cosmologies with massive neutrinos.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Unnormalized growth factor, normalized to \
            the scale factor at early times.
    """
    cosmo.compute_growth()
    return _vectorize_fn(
        lib.growth_factor_unnorm, lib.growth_factor_unnorm_vec, cosmo, a
    )


def growth_rate(cosmo, a):
    """Growth rate defined as the logarithmic derivative of the
    growth factor, dlnD/dlna.

    .. warning:: CCL is not able to compute the scale-dependent growth
                 rate for cosmologies with massive neutrinos.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Growth rate.

    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_rate, lib.growth_rate_vec, cosmo, a)
