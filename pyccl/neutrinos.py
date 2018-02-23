import numpy as np
from pyccl import ccllib as lib
from pyccl.pyutils import check

neutrino_mass_splits = {
    'normal':      lib.nu_masses_normal_label,
    'inverted': lib.nu_masses_inverted_label,
    'equal':   lib.nu_masses_equal_label,
    'sum':   lib.nu_masses_sum_label,
}


def Omeganuh2(a, Neff, mnu, TCMB=2.725):
    """Omeganuh2

    Returns the Omehanuh2 value for a given
    number of massive neutrino species with mass
    mnu. 

    Args:
        a (float): Scale factor, normalized to 1 today.
        Neff (float): Number of massive neutrino species (NB: to all practical purposes, Neff is simply N_nu_mass)
        mnu (float or array_like): Neutrino mass (in eV)
        TCMB (float, optional): Temperature of the CMB (K). Default: 2.725.
    Returns:
        Omeganuh2 (float or array_like) corresponding to a given neutrino mass

    """
    status = 0
    scalar = True if isinstance(a, float) else False
    
    # Convert to array if it's not already an array
    if not isinstance(a, np.ndarray):
        a = np.array([a,]).flatten()
    if not isinstance(mnu, np.ndarray):
        mnu = np.array([mnu,]).flatten()
    
    # FIXME: Implement length checking of mnu (should it be a certain length?)
    
    # Call function
    OmNuh2, status = lib.Omeganuh2_vec(Neff, TCMB, a, mnu, a.size, status)
    
    # Check status and return
    check(status)
    if scalar: return OmNuh2[0]
    return OmNuh2


def nu_masses(OmNuh2, label, ccl_TCMB=2.725):
    """Omeganu2h_to_Mnu

    Args:
        OmNuh2 (float): Neutrino energy density at z=0 times h^2
        label: indicates how the masses should be split up
        TCMB (float, optional): Temperature of the CMB (K). Default: 2.725.

    Returns:
        Neutrino mass(es) corresponding to this Omeganuh2
    
    """
    status = 0
    
    if label not in neutrino_mass_splits.keys() :
        raise ValueError( "'%s' is not a valid species type. "
                          "Available options are: %s" \
                         % (label,neutrino_mass_splits.keys()) )
    
    # Call function
    if ((label=='normal') or (label=='inverted') or (label=='equal')):
        mnu, status = lib.nu_masses_vec(OmNuh2, neutrino_mass_splits[label], ccl_TCMB, 3, status)
    elif label=='sum':
        mnu, status = lib.nu_masses_vec(OmNuh2, neutrino_mass_splits[label], ccl_TCMB, 1, status)
        mnu = mnu[0]
        
	
    # Check status and return
    check(status)
    #if scalar: return mnu[0]
    return mnu

