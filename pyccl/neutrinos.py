import numpy as np
from pyccl import ccllib as lib
from pyccl.pyutils import check

neutrino_mass_splits = {
    'normal':      lib.nu_normal,
    'inverted': lib.nu_inverted,
    'equal':   lib.nu_equal,
    'sum':   lib.nu_sum,
}


def Omeganuh2(a, mnu, TCMB=2.725):
    """Omeganuh2

    Returns the Omehanuh2 value for a given
    number of massive neutrino species with mass
    mnu. 

    Args:
        a (float): Scale factor, normalized to 1 today.
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
        
    N_nu_mass = len(mnu)
    
    # Call function
    OmNuh2, status = lib.Omeganuh2_vec(N_nu_mass, TCMB, a, mnu, a.size, status)
    
    # Check status and return
    check(status)
    if scalar: return OmNuh2[0]
    return OmNuh2


def nu_masses(OmNuh2, mass_split, ccl_TCMB=2.725):
    """nu_masses
		Returns the neutrinos mass(es) for a given OmNuh2, according to the 
		splitting convention specified by the user.

    Args:
        OmNuh2 (float): Neutrino energy density at z=0 times h^2
        mass_split: indicates how the masses should be split up
        TCMB (float, optional): Temperature of the CMB (K). Default: 2.725.

    Returns:
        Neutrino mass(es) corresponding to this Omeganuh2
    
    """
    status = 0
    
    if mass_split not in neutrino_mass_splits.keys() :
        raise ValueError( "'%s' is not a valid species type. "
                          "Available options are: %s" \
                         % (mass_split,neutrino_mass_splits.keys()) )
    
    # Call function
    if ((mass_split=='normal') or (mass_split=='inverted') or (mass_split=='equal')):
        mnu, status = lib.nu_masses_vec(OmNuh2, neutrino_mass_splits[mass_split], ccl_TCMB, 3, status)
    elif mass_split=='sum':
        mnu, status = lib.nu_masses_vec(OmNuh2, neutrino_mass_splits[mass_split], ccl_TCMB, 1, status)
        mnu = mnu[0]
        
	
    # Check status and return
    check(status)
    return mnu

