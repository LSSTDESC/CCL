from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn3, _vectorize_fn4

def p_1h(cosmo, a, k):
    """1halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        p_1h (float or array_like): 1 halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.p_1h, 
                          lib.p_1h_vec, cosmo, a, k)

def p_2h(cosmo, a, k):
    """2halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        p_2h (float or array_Like): 2 halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.p_2h,
			  lib.p_2h_vec, cosmo, a, k)

def p_halomod(cosmo, a, k):
    """matter power spectrum from halo model
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor.
        k (float or array_like): wavenumber

    Returns:
        p_halomod (float or array_like): matter power spectrum from halo model
    """
    return _vectorize_fn2(lib.p_halomod,
			  lib.p_halomod_vec, cosmo, a, k)

def halo_concentration(cosmo, a, halo_mass):
    """halo concentration
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        a (float): scale factor
        halo_mass (float or array_like): mass of halo in Msun
    
    Returns:
        halo_concentration: measure of halo concentration
    """
    return _vectorize_fn2(lib.halo_concentration,
			  lib.halo_concentration_vec, cosmo, a, halo_mass)

