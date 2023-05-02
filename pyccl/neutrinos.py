"""
==================================
Neutrinos (:mod:`pyccl.neutrinos`)
==================================

Functionality related to neutrinos:
    * Omeganuh2 - (Deprecated) Compute OmNuh2.
    * nu_masses - Compute neutrino masses, according to a mass hierarchy.
"""

__all__ = ("nu_masses", "Omeganuh2",)

import warnings
from numbers import Real
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from . import CosmologyParams, NeutrinoMassSplits, lib, omega_x
from . import CCLDeprecationWarning, deprecated, warn_api
from . import physical_constants as const
from .pyutils import check


@deprecated(new_api=omega_x)
def Omeganuh2(
        a: Union[Real, NDArray[Real]],
        m_nu: Union[Real, Sequence[Real]],
        T_CMB: float = CosmologyParams.T_CMB,
        T_ncdm: float = CosmologyParams.T_ncdm
) -> Union[float, NDArray[float]]:
    r"""Calculate :math:`\Omega_{\nu} \, h^2` at a given scale factor given
    the neutrino masses.

    Arguments
    ---------
    a : array_like (na,)
        Scale factor.
    m_nu
        Neutrino mass in :math:`\rm eV`.
    T_CMB
        Temperature of the CMB in :`\rm K`.
    T_ncdm
        Non-CDM temperature in units of photon temperature.

    Returns
    -------
    array_like (na,)
        :math:`\Omega_{\nu} \, h^2`
    """
    status = 0
    scalar = True if np.ndim(a) == 0 else False

    # Convert to array if it's not already an array
    if not isinstance(a, np.ndarray):
        a = np.array([a, ]).flatten()
    if not isinstance(m_nu, np.ndarray):
        m_nu = np.array([m_nu, ]).flatten()

    # Keep only massive neutrinos
    m_nu = m_nu[m_nu > 0.]
    N_nu_mass = len(m_nu)

    OmNuh2, status = lib.Omeganuh2_vec(N_nu_mass, T_CMB, T_ncdm,
                                       a, m_nu, a.size, status)

    # Check status and return
    check(status)
    if scalar:
        return OmNuh2[0]
    return OmNuh2


@warn_api(pairs=[("OmNuh2", "Omega_nu_h2")])
def nu_masses(
        *,
        Omega_nu_h2: Optional[Real] = None,
        mass_split: str,
        T_CMB: Optional[Real] = None,
        m_nu: Optional[Union[Real, Sequence[Real]]] = None  # TODO: v3 2nd arg
) -> NDArray[Real]:
    r"""Compute the neutrinos mass(es) given a mass hierarchy.

    Arguments
    ---------
    Omega_nu_h2
        Neutrino energy density today, times :math:`h^2`.
        Either this or `m_nu` have to be specified.
    mass_split
        Mass hierarchy. Available options are enumerated in
        :class:`~NeutrinoMassSplits`.
    T_CMB
        Temperature of the CMB in :math:`\rm K`.

        .. deprecated:: 2.8.0

            `T_CMB` is not used internally and will be removed in the next
            major release.

    m_nu : array_like (nm,)
        Mass in :math:`\rm eV` of the massive neutrinos present. If a sequence,
        it is assumed that the elements represent the individualneutrino
        masses, and `mass_split` is ignored. Either this or `Omega_nu_h2` have
        to be provided.

    Returns
    -------
    array_like (nm,)
        Neutrino mass(es) according to the specified mass hierarchy.
    """
    if T_CMB is not None:
        warnings.warn("T_CMB is deprecated as an argument of `nu_masses.",
                      CCLDeprecationWarning)
    if m_nu is None:
        m_nu = 93.14 * Omega_nu_h2
    return _get_neutrino_masses(m_nu=m_nu, mass_split=mass_split)


def _get_neutrino_masses(*, m_nu, mass_split):
    # Split the neutrino masses according to a mass hierarchy.
    if isinstance(m_nu, Real) and m_nu == 0:  # no massive neutrinos
        return np.array([])
    if isinstance(m_nu, Iterable):  # input was list
        return np.asarray(m_nu).copy()

    split = NeutrinoMassSplits

    if split(mass_split) == split.SUM:
        return m_nu
    if split(mass_split) == split.SINGLE:
        return np.atleast_1d(m_nu)
    if split(mass_split) == split.EQUAL:
        return np.full(3, m_nu/3)

    c = const
    D12, D13p, D13n = c.DELTAM12_sq, c.DELTAM13_sq_pos, c.DELTAM13_sq_neg

    def M_nu(m, D13):
        m2 = m * m
        return np.array([m.sum()-m_nu, m2[1]-m2[0]-D12, m2[2]-m2[0]-D13])

    def check_mnu(val):
        if m_nu < val:
            raise ValueError(f"m_nu < {val} incompatible with mass hierarchy")

    if split(mass_split) == split.NORMAL:
        check_mnu(np.sqrt(D12) + np.sqrt(D13p))
        x0 = [0, np.sqrt(D12), np.sqrt(D13p)]
        return root(M_nu, x0, args=(D13p,)).x
    if split(mass_split) == split.INVERTED:
        check_mnu(np.sqrt(-(D13n + D12)) + np.sqrt(-D13n))
        x0 = [0, np.sqrt(-(D13n + D12)), np.sqrt(-D13n)]
        return root(M_nu, x0, args=(D13n,)).x
