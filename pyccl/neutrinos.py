import numpy as np
from pyccl import ccllib as lib
from pyccl.pyutils import check


def Omeganuh2(a, Neff, mnu, TCMB=2.725):
    """The Omeganuh2 value for a given number of massive neutrino 
    species with mass Mnu.

    Note: for all practical purposes, Neff is simply N_nu_mass.

    Args:
        a (float): Scale factor, normalized to 1 today.
        Neff (float): Number of relativistic neutrino species.
        mnu (float): Neutrino mass (in eV).
        TCMB (float, optional): Temperature of the CMB (K). Default: 2.725.

    Returns:
        float or array_like: corresponding to a given neutrino mass.

    """
    status = 0
    scalar = True if isinstance(a, float) else False
    
    # Convert to array if it's not already an array
    if not isinstance(a, np.ndarray):
        a = np.array([a,]).flatten()
    
    # Call function
    OmNuh2, status = lib.Omeganuh2_vec(Neff, mnu, TCMB, a, a.size, status)
    
    # Check status and return
    check(status)
    if scalar: return OmNuh2[0]
    return OmNuh2


def Omeganuh2_to_Mnu(a, Neff, OmNuh2, TCMB=2.725):
    """Convert Omeganuh2 to neutrino mass Mnu.

    Note: for all practical purposes, Neff is simply N_nu_mass.

    Args:
        a (float or array_like): Scale factor(s), normalized to 1 today.
        Neff (float): Number of relativistic neutrino species.
        OmNuh2 (float): Neutrino energy density times h^2
        TCMB (float, optional): Temperature of the CMB (K). Default: 2.725.

    Returns:
        float or array_like: Neutrino mass corresponding to a given Omeganuh2.
    
    """
    status = 0
    scalar = True if isinstance(a, float) else False
    
    # Convert to array if it's not already an array
    if not isinstance(a, np.ndarray):
        a = np.array([a,]).flatten()
    
    # Call function
    mnu, status = lib.Omeganuh2_to_Mnu_vec(Neff, OmNuh2, TCMB, a, a.size, status)
    
    # Check status and return
    check(status)
    if scalar: return mnu[0]
    return mnu

