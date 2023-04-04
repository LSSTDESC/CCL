from . import ccllib as lib
from .pyutils import check
from .base import deprecated, warn_api
from .errors import CCLDeprecationWarning
from .core import _Defaults
import numpy as np
import warnings

neutrino_mass_splits = {
    'normal': lib.nu_normal,
    'inverted': lib.nu_inverted,
    'equal': lib.nu_equal,
    'sum': lib.nu_sum,
    'single': lib.nu_single,
}


def Omega_nu_h2(a, *, m_nu, T_CMB=_Defaults.T_CMB, T_ncdm=_Defaults.T_ncdm):
    """Calculate :math:`\\Omega_\\nu\\,h^2` at a given scale factor given
    the neutrino masses.

    Args:
        a (float or array-like): Scale factor, normalized to 1 today.
        m_nu (float or array-like): Neutrino mass(es) (in eV)
        T_CMB (float, optional): Temperature of the CMB (K).
            The default is the same as the Cosmology default.
        T_ncdm (float, optional): Non-CDM temperature in units of photon
            temperature. The default is the same as the Cosmology default.

    Returns:
        float or array_like: :math:`\\Omega_\\nu\\,h^2` at a given
        scale factor given the neutrino masses
    """
    status = 0
    scalar = True if np.ndim(a) == 0 else False

    # Convert to array if it's not already an array
    if not isinstance(a, np.ndarray):
        a = np.array([a, ]).flatten()
    if not isinstance(m_nu, np.ndarray):
        m_nu = np.array([m_nu, ]).flatten()

    N_nu_mass = len(m_nu)

    OmNuh2, status = lib.Omeganuh2_vec(N_nu_mass, T_CMB, T_ncdm,
                                       a, m_nu, a.size, status)

    # Check status and return
    check(status)
    if scalar:
        return OmNuh2[0]
    return OmNuh2


def OmNuh2(cosmo, a):
    """Like Omega_nu_h2 but it uses the parameters from a Cosmology object."""
    return Omega_nu_h2(a, m_nu=cosmo["m_nu"],
                       T_CMB=cosmo["T_CMB"], T_ncdm=cosmo["T_ncdm"])


@deprecated(Omega_nu_h2)
def Omeganuh2(a, m_nu, T_CMB=None):
    return Omega_nu_h2(a, m_nu=m_nu, T_CMB=T_CMB)


@warn_api(pairs=[("OmNuh2", "Omega_nu_h2")])
def nu_masses(*, Omega_nu_h2, mass_split, T_CMB=None):
    """Returns the neutrinos mass(es) for a given Omega_nu_h2, according to the
    splitting convention specified by the user.

    Args:
        Omega_nu_h2 (float): Neutrino energy density at z=0 times h^2
        mass_split (str): indicates how the masses should be split up
            Should be one of 'normal', 'inverted', 'equal' or 'sum'.
        T_CMB (float, optional): Deprecated - do not use.
            Temperature of the CMB (K). Default: 2.725.

    Returns:
        float or array-like: Neutrino mass(es) corresponding to this Omeganuh2
    """
    status = 0

    if T_CMB is not None:
        warnings.warn("T_CMB is deprecated as an argument of `nu_masses.",
                      CCLDeprecationWarning)

    if mass_split not in neutrino_mass_splits.keys():
        raise ValueError(
            "'%s' is not a valid species type. "
            "Available options are: %s"
            % (mass_split, neutrino_mass_splits.keys()))

    # Call function
    if mass_split in ['normal', 'inverted', 'equal']:
        mnu, status = lib.nu_masses_vec(
            Omega_nu_h2, neutrino_mass_splits[mass_split], 3, status)
    elif mass_split in ['sum', 'single']:
        mnu, status = lib.nu_masses_vec(
            Omega_nu_h2, neutrino_mass_splits[mass_split], 1, status)
        mnu = mnu[0]

    # Check status and return
    check(status)
    return mnu
