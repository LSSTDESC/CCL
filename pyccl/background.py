"""
====================================
Background (:mod:`pyccl.background`)
====================================

Functions to compute background quantities: distances, energies, growth.
"""

from __future__ import annotations

__all__ = (
    "Species", "h_over_h0", "comoving_radial_distance", "scale_factor_of_chi",
    "comoving_angular_distance", "angular_diameter_distance",
    "luminosity_distance", "distance_modulus",
    "sigma_critical", "omega_x", "rho_x",
    "growth_factor", "growth_factor_unnorm", "growth_rate",)

from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from . import lib, warn_api
from . import physical_constants as const
from .pyutils import _vectorize_fn

if TYPE_CHECKING:
    from . import Cosmology


class Species(Enum):
    """Energy species types defined in CCL.

    * 'critical' - critical density
    * 'matter' - cold dark matter, baryons, massive neutrinos
    * 'dark_energy' - cosmological constant or otherwise
    * 'radiation' - relativistic species besides massless neutrinos
    * 'curvature' - curvature
    * 'neutrinos_rel' - relativistic neutrinos
    * 'neutrinos_massive' - massive neutrinos
    """
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


def h_over_h0(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Ratio of Hubble constant at `a` over Hubble constant today.

    .. math::

        E(a) = \frac{H(a)}{H_0}.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        :math:`E(a)`.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.h_over_h0, lib.h_over_h0_vec, cosmo, x=a)


def comoving_radial_distance(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Comoving radial distance (in :math:`\rm Mpc`).

    .. math::

        D_{\rm c} = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Comoving radial distance.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_radial_distance,
                         lib.comoving_radial_distance_vec, cosmo, x=a)


def scale_factor_of_chi(
        cosmo: Cosmology,
        chi: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Scale factor at some comoving radial distance, :math:`a(\chi)`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    chi : array_like (nchi,)
        Comoving radial distance :math:`\chi` in :math:`\rm Mpc`.

    Returns
    -------
    array_like (nchi)
        Scale factor at `chi`.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.scale_factor_of_chi,
                         lib.scale_factor_of_chi_vec, cosmo, x=chi)


def comoving_angular_distance(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
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
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Comoving angular distance.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.comoving_angular_distance,
                         lib.comoving_angular_distance_vec, cosmo, x=a)


def angular_diameter_distance(
        cosmo: Cosmology,
        a1: Union[Real, NDArray[Real]],
        a2: Optional[Union[Real, NDArray[Real]]] = None
) -> Union[float, NDArray[float]]:
    r"""Angular diameter distance (in :math:`\rm Mpc`).

    Defined as the ratio of an object's physical transverse size to its
    angular size. It is related to the comoving angular distance as:

    .. math::

        D_{\rm A} = \frac{D_{\rm M}}{1 + z}

    ``angular_diameter_distance`` can be called with a varying number of
    positional arguments:

        * ``angular_diameter_distance(cosmo, a1)``: Distances are calculated
          between `a1` and :math:`1`.
        * ``angular_diameter_distance(cosmo, a1, a2)``: If
          `a1.shape == a2.shape`, the pairwise distances are computed.
          Otherwise, `a1` must be scalar and the distances are computed between
          one object at `a1` and multiple objects at `a2`.

    Arguments
    ---------
    cosmo
           Cosmological parameters.
    a1 : array_like (na1,)
        Scale factor.
    a2 : array_like (na1,) or (na2,)
        Scale factor **smaller** than `a1`.

    Returns
    -------
    array_like (na1,) or (na2,)
        Angular diameter distance. If `a2` is provided and
        `a1.shape != a2.shape`, the output has shape `(a2,)`. Otherwise, it has
        shape `(a1,)`.

    Raises
    ------
    CCLError
        Shape mismatch of input arrays.
    CCLError
        `CCL_ERROR_COMPUTECHI`: Distances are not pairwise, and `a2` is larger
        than `a1`.
    """
    cosmo.compute_distances()
    if a2 is None:
        a1, a2 = np.ones_like(a1)[()], a1
    else:
        a1 = np.broadcast_to(a1, np.shape(a2))[()]

    return _vectorize_fn(lib.angular_diameter_distance,
                         lib.angular_diameter_distance_vec,
                         cosmo, x=a1, x2=a2, pairwise=True)


def luminosity_distance(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Luminosity distance.

    Defined by the relationship between bolometric flux :math:`S` and
    bolometric luminosity :math:`L`.

    .. math::

        D_{\rm L} = \sqrt{\frac{L}{4 \pi S}}

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Luminosity distance at `a`.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.luminosity_distance,
                         lib.luminosity_distance_vec, cosmo, x=a)


def distance_modulus(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Distance modulus.

    Used to convert between apparent and absolute magnitudes
    via :math:`m = M + (\rm dist. \, mod.)` where :math:`m` is the
    apparent magnitude and :math:`M` is the absolute magnitude.

    .. math::

        m - M = 5 * \log_{10}(D_{\rm L} / 10 \, {\rm pc}).

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Distance modulus at `a`.
    """
    cosmo.compute_distances()
    return _vectorize_fn(lib.distance_modulus,
                         lib.distance_modulus_vec, cosmo, x=a)


@warn_api
def sigma_critical(
        cosmo: Cosmology,
        *,
        a_lens: Union[Real, NDArray[Real]],
        a_source: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Compute the critical surface mass density.

    .. math::

         \Sigma_{{\rm c}} = \frac{c^2}{4 \pi G}
         \frac{D_{\rm s}}{D_{\rm l}D_{\rm ls}},

    where :math:`c` is the speed of light, :math:`G` is the gravitational
    constant, and :math:`D_i` is the angular diameter distance. Labels
    :math:`\rm {s, l, ls}` denote the distances to the source, lens, and
    between source and lens, respectively.

    .. note::

        See :func:`~angular_diameter_distance` for accepted input and output
        shapes of `a_lens` and `a_source`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a_lens : array_like (na_lens,)
        Scale factor of lens.
    a_source : array_like (na_lens,) or (na_source)
        Scale factor of source.

    Returns
    -------
    array_like (na_lens,) or (na_source,)
        :math:`\Sigma_{\rm c}` in units of :math:`\rm M_\odot / Mpc^2`.
    """
    Ds = angular_diameter_distance(cosmo, a_source, a2=None)
    Dl = angular_diameter_distance(cosmo, a_lens, a2=None)
    Dls = angular_diameter_distance(cosmo, a_lens, a_source)
    A = (const.CLIGHT**2 * const.MPC_TO_METER
         / (4.0 * np.pi * const.GNEWT * const.SOLAR_MASS))
    return A * Ds / (Dl * Dls)


def omega_x(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]],
        species: str
) -> Union[float, NDArray[float]]:
    r"""Density fraction of a given species at a particular scale factor.

    .. math::

        \Omega_{\rm x}(a) \equiv \frac{\rho_{\rm x}(a)}{\rho_{\rm c}(a)}

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.
    species
        Species type. Available options are enumerated in :class:`~Species`.

    Returns
    -------
    array_like (na,)
        Density fraction of a given species at `a`.

    Raises
    ------
    ValueError
        Wrong species type.
    """
    if species not in species_types:
        raise ValueError(f"Unknown species {species}.")

    return _vectorize_fn(lib.omega_x, lib.omega_x_vec,
                         cosmo, species_types[species], x=a)


@warn_api
def rho_x(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]],
        species: str,
        *,
        is_comoving: bool = False
) -> Union[float, NDArray[float]]:
    r"""Physical or comoving density, :math:`\rho_{\rm x}`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.
    species
        Species type. Available options are enumerated in :class:`~Species`.
    is_comoving
        Either physical or comoving.

    Returns
    -------
    array_like (na,)
        Physical density of a given species at a scale factor,
        in units of :math:`\rm M_\odot / Mpc^3`.

    Raises
    ------
    ValueError
        Wrong species type.
    """
    if species not in species_types:
        raise ValueError(f"Unknown species {species}.")

    return _vectorize_fn(lib.rho_x, lib.rho_x_vec,
                         cosmo, species_types[species], int(is_comoving), x=a)


def growth_factor(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]],
) -> Union[float, NDArray[float]]:
    """Growth factor, :math:`D(a)`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Growth factor at `a`.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor, lib.growth_factor_vec, cosmo, x=a)


def growth_factor_unnorm(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]],
) -> Union[float, NDArray[float]]:
    """Unnormalized growth factor.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Unnormalized growth factor at `a`.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_factor_unnorm,
                         lib.growth_factor_unnorm_vec, cosmo, x=a)


def growth_rate(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]],
) -> Union[float, NDArray[float]]:
    r"""Growth rate defined as the logarithmic derivative of the
    growth factor,

    .. math::

        \frac{{\rm d}\ln{D(a)}}{{\rm d}\ln{a}}.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na,)
        Growth rate at `a`.
    """
    cosmo.compute_growth()
    return _vectorize_fn(lib.growth_rate, lib.growth_rate_vec, cosmo, x=a)
