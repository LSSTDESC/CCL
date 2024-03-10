"""Smooth background quantities

CCL defines seven species types:

* 'matter': cold dark matter and baryons
* 'dark_energy': cosmological constant or otherwise
* 'radiation': relativistic species besides massless neutrinos (i.e., only photons)
* 'curvature': curvature density
* 'neutrinos_rel': relativistic neutrinos
* 'neutrinos_massive': massive neutrinos

These strings define the `species` inputs to the functions below.
""" # noqa
__all__ = (
    "Species", "compute_distances",
    "h_over_h0", "comoving_radial_distance", "scale_factor_of_chi",
    "comoving_angular_distance", "angular_diameter_distance",
    "luminosity_distance", "distance_modulus",
    "hubble_distance", "comoving_volume_element", "comoving_volume",
    "lookback_time", "age_of_universe",
    "sigma_critical", "omega_x", "rho_x",
    "growth_factor", "growth_factor_unnorm", "growth_rate",)

from enum import Enum

import numpy as np
from scipy.interpolate import Akima1DInterpolator as interp

from . import lib, physical_constants
from .pyutils import (_vectorize_fn, _vectorize_fn3,
                      _vectorize_fn4, _vectorize_fn5,
                      check, loglin_spacing)


class Species(Enum):
    CRITICAL = "critical"
    MATTER = "matter"
    DARK_ENERGY = "dark_energy"
    RADIATION = "radiation"
    CURVATURE = "curvature"
    NEUTRINOS_REL = "neutrinos_rel"
    NEUTRINOS_MASSIVE = "neutrinos_massive"


species_types = {
    'critical': lib.species_crit_label,
    'matter': lib.species_m_label,
    'dark_energy': lib.species_l_label,
    'radiation': lib.species_g_label,
    'curvature': lib.species_k_label,
    'neutrinos_rel': lib.species_ur_label,
    'neutrinos_massive': lib.species_nu_label,
}


def compute_distances(cosmo):
    """Compute the distance splines."""
    if cosmo.has_distances:
        return
    status = 0
    status = lib.cosmology_compute_distances(cosmo.cosmo, status)
    check(status, cosmo)

    # lookback time
    spl = cosmo.cosmo.spline_params
    a = loglin_spacing(spl.A_SPLINE_MINLOG, spl.A_SPLINE_MIN, spl.A_SPLINE_MAX,
                       spl.A_SPLINE_NLOG, spl.A_SPLINE_NA)
    t_H = (physical_constants.MPC_TO_METER / 1e14
           / physical_constants.YEAR / cosmo["h"])
    hoh0 = cosmo.h_over_h0(a)
    integral = interp(a, 1/(a*hoh0)).antiderivative()
    a_eval = np.r_[1.0, a]  # make a single call to the spline
    vals = integral(a_eval)
    t_arr = t_H * (vals[0] - vals[1:])

    cosmo.data.lookback = interp(a, t_arr)
    cosmo.data.age0 = cosmo.data.lookback(0, extrapolate=True)[()]


def h_over_h0(cosmo, a):
    """Ratio of Hubble constant at `a` over Hubble constant today.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): H(a)/H0.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.h_over_h0,
                         lib.h_over_h0_vec, cosmo, a)


def comoving_radial_distance(cosmo, a):
    """Comoving radial distance.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Comoving radial distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_radial_distance,
                         lib.comoving_radial_distance_vec, cosmo, a)


def scale_factor_of_chi(cosmo, chi):
    """Scale factor, a, at a comoving radial distance chi.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        chi (:obj:`float` or `array`): Comoving radial distance(s); Mpc.

    Returns:
        (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.scale_factor_of_chi,
                         lib.scale_factor_of_chi_vec, cosmo, chi)


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
    return _vectorize_fn(lib.comoving_angular_distance,
                         lib.comoving_angular_distance_vec, cosmo, a)


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
    if (a2 is not None):
        # One lens, multiple sources
        if (np.ndim(a1) == 0) and (np.ndim(a2) != 0):
            a1 = np.full(len(a2), a1)
        return _vectorize_fn5(lib.angular_diameter_distance,
                              lib.angular_diameter_distance_vec,
                              cosmo, a1, a2)
    else:
        if isinstance(a1, (int, float)):
            return _vectorize_fn5(lib.angular_diameter_distance,
                                  lib.angular_diameter_distance_vec,
                                  cosmo, 1., a1)
        else:
            return _vectorize_fn5(lib.angular_diameter_distance,
                                  lib.angular_diameter_distance_vec,
                                  cosmo, np.ones(len(a1)), a1)


def luminosity_distance(cosmo, a):
    """Luminosity distance.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Luminosity distance; Mpc.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.luminosity_distance,
                         lib.luminosity_distance_vec, cosmo, a)


def distance_modulus(cosmo, a):
    """Distance Modulus, defined as

    .. math::
        \\mu = 5\\,\\log_{10}(d_L/10\\,{\\rm pc})

    where :math:`d_L` is the luminosity distance.

    .. note :: The distance modulus can be used to convert between apparent
               and absolute magnitudes via :math:`m = M + \\mu`, where
               :math:`m` is the apparent magnitude and :math:`M` is the
               absolute magnitude.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Distance modulus at a.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.distance_modulus,
                         lib.distance_modulus_vec, cosmo, a)


def hubble_distance(cosmo, a):
    r"""Hubble distance in :math:`\rm Mpc`.

    .. math::

        D_{\rm H} = \frac{cz}{H_0}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_H : float or (na,) ``numpy.ndarray``
        Hubble distance.
    """
    return (1/a - 1) * physical_constants.CLIGHT_HMPC / cosmo["h"]


def comoving_volume_element(cosmo, a):
    r"""Comoving volume element in :math:`\rm Mpc^3 \, sr^{-1}`.

    .. math::

        \frac{\mathrm{d}V}{\mathrm{d}a \, \mathrm{d} \Omega}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    dV : float or (na,) ``numpy.ndarray``
        Comoving volume per unit scale factor per unit solid angle.

    See Also
    --------
    comoving_volume : integral of the comoving volume element
    """
    Dm = comoving_angular_distance(cosmo, a)
    Ez = h_over_h0(cosmo, a)
    Dh = physical_constants.CLIGHT_HMPC / cosmo["h"]
    return Dh * Dm**2 / (Ez * a**2)


def comoving_volume(cosmo, a, *, solid_angle=4*np.pi):
    r"""Comoving volume, in :math:`\rm Mpc^3`.

    .. math::

        V_{\rm C} = \int_{\Omega} \mathrm{{d}}\Omega \int_z \mathrm{d}z
        D_{\rm H} \frac{(1+z)^2 D_{\mathrm{A}}^2}{E(z)}


    See Eq. 29 in `Hogg 2000 <https://arxiv.org/abs/astro-ph/9905116>`_.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    solid_angle : float
        Solid angle subtended in the sky for which
        the comoving volume is calculated.

    Returns
    -------
    V_C : float or (na,) ndarray
        Comoving volume at ``a``.

    See Also
    --------
    comoving_volume_element : comoving volume element
    """
    Omk = cosmo["Omega_k"]
    Dm = comoving_angular_distance(cosmo, a)
    if Omk == 0:
        return solid_angle/3 * Dm**3

    Dh = hubble_distance(cosmo, a)
    sqrt = np.sqrt(np.abs(Omk))
    DmDh = Dm / Dh
    arcsinn = np.arcsin if Omk < 0 else np.arcsinh
    return ((solid_angle * Dh**3 / (2 * Omk))
            * (DmDh * np.sqrt(1 + Omk * DmDh**2)
               - arcsinn(sqrt * DmDh)/sqrt))


def lookback_time(cosmo, a):
    r"""Difference of the age of the Universe between some scale factor
    and today, in :math:`\rm Gyr`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    t_L : float or (na,) ndarray
        Lookback time at ``a``. ``nan`` if ``a`` is out of bounds of the spline
        parametets stored in ``cosmo``.
    """
    cosmo.compute_distances()
    out = cosmo.data.lookback(a)
    return out[()]


def age_of_universe(cosmo, a):
    r"""Age of the Universe at some scale factor, in :math:`\rm Gyr`.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    t_age : float or (na,) ndarray
        Age of the Universe at ``a``. ``nan`` if ``a`` is out of bounds of the
        spline parametets stored in ``cosmo``.
    """
    cosmo.compute_distances()
    out = cosmo.data.age0 - cosmo.lookback_time(a)
    return out[()]


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
        / (4.0 * np.pi * physical_constants.GNEWT
           * physical_constants.SOLAR_MASS)
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

    return _vectorize_fn3(lib.omega_x,
                          lib.omega_x_vec, cosmo, a, species_types[species])


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
        lib.rho_x, lib.rho_x_vec, cosmo, a,
        species_types[species], int(is_comoving))


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
    return _vectorize_fn(lib.growth_factor,
                         lib.growth_factor_vec, cosmo, a)


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
    return _vectorize_fn(lib.growth_factor_unnorm,
                         lib.growth_factor_unnorm_vec, cosmo, a)


def growth_rate(cosmo, a):
    """Growth rate defined as the logarithmic derivative of the
    growth factor, :math:`f\\equiv d\\log D/d\\log a`.

    .. warning:: CCL is not able to compute the scale-dependent growth
                 rate for cosmologies with massive neutrinos.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float` or `array`): Scale factor(s), normalized to 1 today.

    Returns:
        (:obj:`float` or `array`): Growth rate.

    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_rate,
                         lib.growth_rate_vec, cosmo, a)
