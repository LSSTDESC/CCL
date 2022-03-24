import numpy as np
from . import ccllib as lib
from .core import check
from .parameters import physical_constants

neutrino_mass_splits = {
    'normal': lib.nu_normal,
    'inverted': lib.nu_inverted,
    'equal': lib.nu_equal,
    'sum': lib.nu_sum,
    'single': lib.nu_single,
}


def Omeganuh2(a, m_nu, T_CMB=None):
    """Calculate :math:`\\Omega_\\nu\\,h^2` at a given scale factor given
    the neutrino masses.

    Args:
        a (float or array-like): Scale factor, normalized to 1 today.
        m_nu (float or array-like): Neutrino mass(es) (in eV)
        T_CMB (float, optional): Temperature of the CMB (K). Default: 2.725.

    Returns:
        float or array_like: :math:`\\Omega_\\nu\\,h^2` at a given
        scale factor given the neutrino masses
    """
    status = 0
    scalar = True if np.ndim(a) == 0 else False

    if T_CMB is None:
        T_CMB = physical_constants.T_CMB

    # Convert to array if it's not already an array
    if not isinstance(a, np.ndarray):
        a = np.array([a, ]).flatten()
    if not isinstance(m_nu, np.ndarray):
        m_nu = np.array([m_nu, ]).flatten()

    N_nu_mass = len(m_nu)

    # Call function
    OmNuh2, status = lib.Omeganuh2_vec(N_nu_mass, T_CMB,
                                       a, m_nu, a.size, status)

    # Check status and return
    check(status)
    if scalar:
        return OmNuh2[0]
    return OmNuh2


def nu_masses(OmNuh2, mass_split, T_CMB=None):
    """Returns the neutrinos mass(es) for a given OmNuh2, according to the
    splitting convention specified by the user.

    Args:
        OmNuh2 (float): Neutrino energy density at z=0 times h^2
        mass_split (str): indicates how the masses should be split up
            Should be one of 'normal', 'inverted', 'equal' or 'sum'.
        T_CMB (float, optional): Temperature of the CMB (K). Default: 2.725.

    Returns:
        float or array-like: Neutrino mass(es) corresponding to this Omeganuh2
    """
    status = 0

    if T_CMB is None:
        T_CMB = physical_constants.T_CMB

    if mass_split not in neutrino_mass_splits.keys():
        raise ValueError(
            "'%s' is not a valid species type. "
            "Available options are: %s"
            % (mass_split, neutrino_mass_splits.keys()))

    # Call function
    if mass_split in ['normal', 'inverted', 'equal']:
        mnu, status = lib.nu_masses_vec(
            OmNuh2, neutrino_mass_splits[mass_split], T_CMB, 3, status)
    elif mass_split in ['sum', 'single']:
        mnu, status = lib.nu_masses_vec(
            OmNuh2, neutrino_mass_splits[mass_split], T_CMB, 1, status)
        mnu = mnu[0]

    # Check status and return
    check(status)
    return mnu
