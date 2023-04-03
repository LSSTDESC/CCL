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
from .pyutils import (_vectorize_fn, _vectorize_fn3,
                      _vectorize_fn4, _vectorize_fn5)
from .parameters import physical_constants as const
from .base import warn_api


__all__ = ("h_over_h0", "comoving_radial_distance", "scale_factor_of_chi",
           "comoving_angular_distance", "transverse_comoving_distance",
           "angular_diameter_distance", "luminosity_distance",
           "distance_modulus", "hubble_distance", "comoving_volume_element",
           "comoving_volume", "omega_x", "rho_x", "growth_factor",
           "growth_factor_unnorm", "growth_rate", "sigma_critical",)


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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    chi : float or (nchi,) array_like
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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


transverse_comoving_distance = comoving_angular_distance  # alias


def angular_diameter_distance(cosmo, a1, a2=None):
    r"""Angular diameter distance (in :math:`\rm Mpc `).

    Defined as the ratio of an object's physical transverse size to its
    angular size. It is related to the comoving angular distance as:

    .. math::

        D_{\rm A} = \frac{D_{\rm M}}{1 + z}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
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
    D_A : float, (na1,) or (na2,) ``numpy.ndarray``
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.

    Returns
    -------
    D_M : float or (na,) ``numpy.ndarray``
        Distance modulus at ``a``.
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


def comoving_volume(cosmo, a, *, solid_angle=4*np.pi, squeeze=True):
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
    Omk, sqrtk = cosmo["Omega_k"], cosmo["sqrtk"]
    Dm = comoving_angular_distance(cosmo, a)
    if Omk == 0:
        return solid_angle/3 * Dm**3

    Dh = hubble_distance(cosmo, a)
    DmDh = Dm / Dh
    arcsinn = np.arcsin if Omk < 0 else np.arcsinh
    return ((solid_angle * Dh**3 / (2 * Omk))
            * (DmDh * np.sqrt(1 + Omk * DmDh**2)
               - arcsinn(sqrtk * DmDh)/sqrtk))


def omega_x(cosmo, a, species):
    r"""Density fraction of a given species at a particular scale factor.

    .. math::

        \Omega_x(a) = \frac{\rho_x(a)}{\rho_{\rm c}(a)}

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor, normalized to 1 today.
    species : str
        Species type. Should be one of:
            - 'matter': cold dark matter, massive neutrinos, and baryons
            - 'dark_energy': cosmological constant or otherwise
            - 'radiation': relativistic species besides massless neutrinos
            - 'neutrinos_rel': relativistic neutrinos
            - 'neutrinos_massive': massive neutrinos

    Returns
    -------
    Omega_x : float or (na,) ndarray
        Density fraction of a given species at ``a``.

    Raises
    ------
    ValueError
        Wrong species type.
    """
    if species not in species_types.keys():
        raise ValueError(f"{species} is not a valid species type. "
                         f"Available options are: {species_types.keys()}.")
    return _vectorize_fn3(lib.omega_x, lib.omega_x_vec,
                          cosmo, a, species_types[species])


@warn_api
def rho_x(cosmo, a, species, *, is_comoving=False):
    r"""Physical or comoving density as a function of scale factor.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
        Scale factor(s) normalized to 1 today.
    species : str
        Species type. Should be one of

        - 'matter': cold dark matter, massive neutrinos, and baryons
        - 'dark_energy': cosmological constant or otherwise
        - 'radiation': relativistic species besides massless neutrinos
        - 'curvature': curvature density
        - 'neutrinos_rel': relativistic neutrinos
        - 'neutrinos_massive': massive neutrinos

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
    if species not in species_types.keys():
        raise ValueError(f"{species} is not a valid species type. "
                         f"Available options are: {species_types.keys()}.")
    return _vectorize_fn4(lib.rho_x, lib.rho_x_vec, cosmo, a,
                          species_types[species], int(is_comoving))


def growth_factor(cosmo, a):
    """Growth factor.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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
    cosmo : :class:`~pyccl.core.Cosmology`
        Cosmological parameters.
    a : float or (na,) array_like
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

         \Sigma_{\mathrm{crit}} = \frac{c^2}{4 \pi G}
         \frac{D_{\rm s}}{D_{\rm l}D_{\rm ls}},

    where :math:`c` is the speed of light, :math:`G` is the
    gravitational constant, and :math:`D_i` is the angular diameter distance
    The labels :math:`\rm (s, l, ls)` denote the distances to the source, lens,
    and between source and lens, respectively.

    Arguments
    ---------
    cosmo : :class:`~pyccl.core.Cosmology`
        A Cosmology object.
    a_lens : float or (na_lens,) array_like
        Scale factor of lens.
    a_source : float, (na_lens,) or (na_source,) array_like
        Scale factor of source.

    Returns
    -------
    sigma_critical : float, (na_lens,) or (na_source,) ``numpy.ndarray``
        :math:`\Sigma_{\rm crit} in units of :math:`\rm M_\odot / Mpc^2`.

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
