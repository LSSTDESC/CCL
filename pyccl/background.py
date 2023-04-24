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
__all__ = (
    "Species", "h_over_h0", "comoving_radial_distance", "scale_factor_of_chi",
    "comoving_angular_distance", "angular_diameter_distance",
    "luminosity_distance", "distance_modulus",
    "sigma_critical", "omega_x", "rho_x",
    "growth_factor", "growth_factor_unnorm", "growth_rate",)

from enum import Enum

import numpy as np

from . import lib, warn_api
from . import physical_constants as const
from .pyutils import (_vectorize_fn, _vectorize_fn3,
                      _vectorize_fn4, _vectorize_fn5)


class Species(Enum):
    CRITICAL = "critical"  # critical density
    MATTER = "matter"  # cold dark matter, massive neutrinos, baryons
    DARK_ENERGY = "dark_energy"  # cosmological constant or otherwise
    RADIATION = "radiation"  # relativistic species besides massless neutrinos
    CURVATURE = "curvature"  # curvature
    NEUTRINOS_REL = "neutrinos_rel"  # relativistic neutrinos
    NEUTRINOS_MASSIVE = "neutrinos_massive"  # massive neutrinos


species_types = {
    'critical': lib.species_crit_label,
    'matter': lib.species_m_label,
    'dark_energy': lib.species_l_label,
    'radiation': lib.species_g_label,
    'curvature': lib.species_k_label,
    'neutrinos_rel': lib.species_ur_label,
    'neutrinos_massive': lib.species_nu_label,
}


def h_over_h0(cosmo, a):
    r"""Ratio of Hubble constant at ``a`` over Hubble constant today.

    .. math::

        E(a) = \frac{H(a)}{H_0}.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    Ez : float or (na,) ndarray
        Value of the fraction.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.h_over_h0,
                         lib.h_over_h0_vec, cosmo, a)


def comoving_radial_distance(cosmo, a):
    r"""Comoving radial distance (in :math:`\rm Mpc`).

    .. math::

        D_{\rm c} = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_C : float or (na,) ``numpy.ndarray``
        Comoving radial distance at ``a``.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_radial_distance,
                         lib.comoving_radial_distance_vec, cosmo, a)


def scale_factor_of_chi(cosmo, chi):
    r"""Scale factor at some comoving radial distance, :math:`a(\chi)`.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    chi : int, float or (nchi,) array_like
        Comoving radial distance :math:`\chi` in :math:`\rm Mpc`.

    Returns
    -------
    a_chi : float or (nchi,) ndarray
        Scale factor at ``chi``.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.scale_factor_of_chi,
                         lib.scale_factor_of_chi_vec, cosmo, chi)


def comoving_angular_distance(cosmo, a):
    r"""Comoving angular distance (in :math:`\rm Mpc`).

    .. math::
        D_{\rm M} = \mathrm{sinn}(\chi(a)).

    .. note::

        This quantity is otherwise known as the transverse comoving distance,
        and is **not** the angular diameter distance or the angular separation.
        The comoving angular distance is defined such that the comoving
        distance between two objects at a fixed scale factor separated
        by an angle :math:`\theta` is :math:`\theta r_{A}(a)` where
        :math:`r_{A}(a)` is the comoving angular distance.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_M : float or (na,) ``numpy.ndarray``
        Comoving angular distance at ``a``.

    See also
    --------
    transverse_comoving_distance : alias of comoving_angular_distance
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_angular_distance,
                         lib.comoving_angular_distance_vec, cosmo, a)


def angular_diameter_distance(cosmo, a1, a2=None):
    r"""Angular diameter distance (in :math:`\rm Mpc `).

    Defined as the ratio of an object's physical transverse size to its
    angular size. It is related to the comoving angular distance as:

    .. math::

        D_{\rm A} = \frac{D_{\rm M}}{1 + z}

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
           Cosmological parameters.
    a1 : float or (na1,) array_like
        Scale factor(s), normalized to 1 today.
    a2 : float, (na1,) or (na2,) array_like, optional
        Scale factor(s) **smaller** than ``a1``, normalized to 1 today.

        - If nothing is passed, the distance is calculated from ``a1`` to 1.
        - If a float or is passed, ``a1`` must also be a float.
        - If an array of shape (na1,) is passed, the pairwise distances are
          computed.
        - If an array of shape (na2,) is passed, ``a1`` must be a float.

        The default is ``None``.

    Returns
    -------
    D_a : int or float, (na1,) or (na2,) ``numpy.ndarray``
        Angular diameter distance.

        - If ``a2`` is ``None`` the output shape is (na1,).
        - If ``shape(a1) == shape(a2)`` the pairwise distances are computed
          and the output shape is the common (na,).
        - If ``a1`` is a float and ``a2`` is an array the output shape is
          (na2,).

    Raises
    ------
    CCLError
        Shape mismatch of input arrays.
    """
    cosmo.compute_distances()
    if a2 is not None:
        # One lens, multiple sources
        if (np.ndim(a1) == 0) and (np.ndim(a2) != 0):
            a1 = np.full(len(a2), a1)
        return _vectorize_fn5(lib.angular_diameter_distance,
                              lib.angular_diameter_distance_vec,
                              cosmo, a1, a2)
    if isinstance(a1, (int, float)):
        return _vectorize_fn5(lib.angular_diameter_distance,
                              lib.angular_diameter_distance_vec,
                              cosmo, 1., a1)
    return _vectorize_fn5(lib.angular_diameter_distance,
                          lib.angular_diameter_distance_vec,
                          cosmo, np.ones(len(a1)), a1)


def luminosity_distance(cosmo, a):
    r"""Luminosity distance.

    Defined by the relationship between bolometric flux :math:`S` and
    bolometric luminosity :math:`L`.

    .. math::
        D_{\rm L} = \sqrt{\frac{L}{4 \pi S}}

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_L : float or (na,) ``numpy.ndarray``
        Luminosity distance at ``a``.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.luminosity_distance,
                         lib.luminosity_distance_vec, cosmo, a)


def distance_modulus(cosmo, a):
    r"""Distance modulus.

    Used to convert between apparent and absolute magnitudes
    via :math:`m = M + (\rm dist. \, mod.)` where :math:`m` is the
    apparent magnitude and :math:`M` is the absolute magnitude.

    .. math::

        m - M = 5 * \log_{10}(D_{\rm L} / 10 \, {\rm pc}).

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_M : float or (na,) ``numpy.ndarray``
        Distance modulus at ``a``.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.distance_modulus,
                         lib.distance_modulus_vec, cosmo, a)


def omega_x(cosmo, a, species):
    r"""Density fraction of a given species at a particular scale factor.

    .. math::

        \Omega_x(a) = \frac{\rho_x(a)}{\rho_{\rm c}(a)}

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor, normalized to 1 today.
    species : str
        Species type. Available options are enumerated in
        :class:`~pyccl.Species`.

    Returns
    -------
    Omega_x : float or (na,) ndarray
        Density fraction of a given species at ``a``.

    Raises
    ------
    ValueError
        Wrong species type.
    """
    # TODO: Replace docstring enum with ref to Species.
    if species not in species_types:
        raise ValueError(f"Unknown species {species}.")

    return _vectorize_fn3(lib.omega_x,
                          lib.omega_x_vec, cosmo, a, species_types[species])


@warn_api
def rho_x(cosmo, a, species, *, is_comoving=False):
    r"""Physical or comoving density as a function of scale factor.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    species : str
        Species type. Available options are enumerated in
        :class:`~pyccl.Species`.
    is_comoving : bool, optional
        Either physical or comoving. Default is ``False`` for physical.

    Returns
    -------
    rho_x : float or (na,) ndarray
        Physical density of a given species at a scale factor,
        in units of :math:`\rm M_\odot / Mpc^3`.

    Raises
    ------
    ValueError
        Wrong species type.
    """
    # TODO: Replace docstring enum with ref to Species.
    if species not in species_types:
        raise ValueError(f"Unknown species {species}.")

    return _vectorize_fn4(
        lib.rho_x, lib.rho_x_vec, cosmo, a,
        species_types[species], int(is_comoving))


def growth_factor(cosmo, a):
    """Growth factor.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D : float or (na,) ``numpy.ndarray``
        Growth factor at ``a``.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor, lib.growth_factor_vec, cosmo, a)


def growth_factor_unnorm(cosmo, a):
    """Unnormalized growth factor.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_unnorm : float or (na,) ``numpy.ndarray``
        Unnormalized growth factor at ``a``.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor_unnorm,
                         lib.growth_factor_unnorm_vec, cosmo, a)


def growth_rate(cosmo, a):
    r"""Growth rate defined as the logarithmic derivative of the
    growth factor,

    .. math::

        \frac{\mathrm{d}\ln{D}}{\mathrm{d}\ln{a}}.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    a : int, float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    dlnD_dlna : float or (na,) ``numpy.ndarray``
        Growth rate at ``a``.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_rate, lib.growth_rate_vec, cosmo, a)


@warn_api
def sigma_critical(cosmo, *, a_lens, a_source):
    r"""Returns the critical surface mass density.

    .. math::

         \Sigma_{{\rm c}} = \frac{c^2}{4 \pi G}
         \frac{D_{\rm s}}{D_{\rm l}D_{\rm ls}},

    where :math:`c` is the speed of light, :math:`G` is the
    gravitational constant, and :math:`D_i` is the angular diameter distance
    The labels :math:`\rm (s, l, ls)` denote the distances to the source, lens,
    and between source and lens, respectively.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        A Cosmology object.
    a_lens : float or (na_lens,) array_like
        Scale factor of lens.
    a_source : float, (na_lens,) or (na_source,) array_like
        Scale factor of source.

    Returns
    -------
    sigma_critical : float, (na_lens,) or (na_source,) ``numpy.ndarray``
        :math:`\Sigma_{\rm c} in units of :math:`\rm M_\odot / Mpc^2`.

    See also
    --------
    angular_diameter_distance : description of input array shape options
    """
    Ds = angular_diameter_distance(cosmo, a_source, a2=None)
    Dl = angular_diameter_distance(cosmo, a_lens, a2=None)
    Dls = angular_diameter_distance(cosmo, a_lens, a_source)
    A = (const.CLIGHT**2 * const.MPC_TO_METER
         / (4.0 * np.pi * const.GNEWT * const.SOLAR_MASS))
    return A * Ds / (Dl * Dls)
