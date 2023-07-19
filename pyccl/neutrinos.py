__all__ = ("NeutrinoMassSplits", "nu_masses",)

from enum import Enum
from numbers import Real
from typing import Iterable

import numpy as np
from scipy.optimize import root

from . import physical_constants as const


class NeutrinoMassSplits(Enum):
    """Enumeration listing all allowed neutrino mass
    split types.

    - 'sum': sum of masses.
    - 'single': single massive neutrino.
    - 'equal': total mass distributed equally among 3 species.
    - 'normal': normal hierarchy.
    - 'inverted': inverted hierarchy.
    - 'list': a list of 3 different masses is passed.
    """
    SUM = 'sum'
    SINGLE = 'single'
    EQUAL = 'equal'
    NORMAL = 'normal'
    INVERTED = 'inverted'
    LIST = 'list'  # placeholder for backwards-compatibility


def nu_masses(*, Omega_nu_h2=None, mass_split, m_nu=None):
    """Returns the neutrinos mass(es) for a given value of
    :math:`\\Omega_\\nu h^2`, according to the splitting convention
    specified by the user.

    Args:
        Omega_nu_h2 (:obj:`float`): Neutrino energy density at z=0 times
            :math:`h^2`.
        mass_split (:obj:`str`): indicates how the masses should be split up
            Should be one of 'normal', 'inverted', 'equal' or 'sum'.
        m_nu (:obj:`float` or array_like):
            Mass in eV of the massive neutrinos present.
            If a sequence is passed, it is assumed that the elements of the
            sequence represent the individual neutrino masses.

    Returns:
        :obj:`float` or `array`: Neutrino mass(es) corresponding to this
        :math:`\\Omega_\\nu h^2`.
    """
    if m_nu is None:
        m_nu = 93.14 * Omega_nu_h2
    return _get_neutrino_masses(m_nu=m_nu, mass_split=mass_split)


def _get_neutrino_masses(*, m_nu, mass_split):
    """
    """
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
