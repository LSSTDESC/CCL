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
from scipy.interpolate import Akima1DInterpolator as interp
from . import ccllib as lib
from .pyutils import (_vectorize_fn, _vectorize_fn3,
                      _vectorize_fn4, _vectorize_fn5, check, loglin_spacing)
from .base.parameters import physical_constants as const
from .base import warn_api

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
    spl = cosmo.cosmo.spline_params  # Replace for CCLv3.
    a = loglin_spacing(spl.A_SPLINE_MINLOG, spl.A_SPLINE_MIN, spl.A_SPLINE_MAX,
                       spl.A_SPLINE_NLOG, spl.A_SPLINE_NA)
    t_H = const.MPC_TO_METER / 1e14 / const.YEAR / cosmo["h"]
    hoh0 = cosmo.h_over_h0(a)
    integral = interp(a, 1/(a*hoh0)).antiderivative()
    a_eval = np.r_[1.0, a]  # make a single call to the spline
    vals = integral(a_eval)
    t_arr = t_H * (vals[0] - vals[1:])

    cosmo.data.lookback = interp(a, t_arr)
    cosmo.data.age0 = cosmo.data.lookback(0, extrapolate=True)[()]


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

    The angular diameter distance in Mpc from scale factor
    `a1` to scale factor `a2`. If `a2` is not provided, it is
    assumed that the distance will be calculated between 1 and
    `a1`.

    .. note:: `a2` has to be smaller than `a1` (i.e. a source at
              `a2` is behind one at `a1`). You can compute the
              distance between a single lens at `a1` and multiple
              sources at `a2` by passing a scalar `a1`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        a1 (float or array_like): Scale factor(s), normalized to 1 today.
        a2 (float or array_like): Scale factor(s), normalized to 1 today,
        optional.

    Returns:
        float or array_like: angular diameter distance; Mpc.
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


@warn_api
def rho_x(cosmo, a, species, *, is_comoving=False):
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
        at a scale factor, in units of Msun / Mpc^3.
    """

    if species not in species_types.keys():
        raise ValueError("'%s' is not a valid species type. "
                         "Available options are: %s"
                         % (species, species_types.keys()))

    return _vectorize_fn4(
        lib.rho_x, lib.rho_x_vec, cosmo, a,
        species_types[species], int(is_comoving))


@warn_api
def sigma_critical(cosmo, *, a_lens, a_source):
    """Returns the critical surface mass density.

    .. math::
         \\Sigma_{\\mathrm{crit}} = \\frac{c^2}{4\\pi G}
          \\frac{D_{\\rm{s}}}{D_{\\rm{l}}D_{\\rm{ls}}},
           where :math:`c` is the speed of light, :math:`G` is the
           gravitational constant, and :math:`D_i` is the angular diameter
           distance. The labels :math:`i =` s, l and ls denotes the distances
           to the source, lens, and between source and lens, respectively.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        a_lens (float): lens' scale factor.
        a_source (float or array_like): source's scale factor.

    Returns:
        float or array_like: :math:`\\Sigma_{\\mathrm{crit}}` in units
        of :math:`\\M_{\\odot}/Mpc^2`
    """
    Ds = angular_diameter_distance(cosmo, a_source, a2=None)
    Dl = angular_diameter_distance(cosmo, a_lens, a2=None)
    Dls = angular_diameter_distance(cosmo, a_lens, a_source)
    A = (const.CLIGHT**2 * const.MPC_TO_METER
         / (4.0 * np.pi * const.GNEWT * const.SOLAR_MASS))

    Sigma_crit = A * Ds / (Dl * Dls)
    return Sigma_crit


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
    return (1/a - 1) * const.CLIGHT_HMPC / cosmo["h"]


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
    Dh = const.CLIGHT_HMPC / cosmo["h"]
    return Dh * Dm**2 / (Ez * a**2)


def comoving_volume(cosmo, a, *, solid_angle=4*np.pi):
    r"""Comoving volume, in :math:`\rm Mpc^3`.

    .. math::

        V_{\rm C} = \int_{\Omega} \mathrm{{d}}\Omega \int_z \mathrm{d}z
        D_{\rm H} \frac{(1+z)^2 D_{\mathrm{A}}^2}{E(z)}

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
