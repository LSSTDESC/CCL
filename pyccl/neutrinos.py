from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn_simple

def Omeganuh2(a,Neff,mnu,TCMB):
    """Omeganuh2
    
    Args:
        a (float): Scale factor, normalized to 1 today.
        Neff (float): Number of relativistic neutrino species
        mnu (float): Neutrino mass ()
        TCMB (float): Temperature of the CMB (K)
    Returns:
        Omeganuh2 (float or array_like)

    """
    status=0
    return lib.Omeganuh2(a,Neff,mnu,TCMB,None,status)


def Omeganuh2_to_Mnu(a,Neff,OmNuh2,TCMB):
    """Omeganu2h_to_Mnu

    Args:
        a (float or array_like): Scale factor(s), normalized to 1 today.
        Neff (float): Number of relativistic neutrino species
        OmNuh2 (float): Neutrino energy density times h^2
        TCMB (float): Temperature of the CMB (K)

    Returns:
        Neutrino mass corresponding to this Omeganuh2
    
    """
    status=0
    return lib.Omeganuh2_to_Mnu(a,Neff,OmNuh2,TCMB,None,status)
