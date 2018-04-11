from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn3, _vectorize_fn4

def p_1h(cosmo, k, a):
    """1halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        p_1h (float or array_like): 1 halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.p_1h, 
                          lib.p_1h_vec, cosmo, k, a)

def p_2h(cosmo, k, a):
    """2halo term for matter power spectrum
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        p_2h (float or array_Like): 2 halo term for matter power spectrum
    """
    return _vectorize_fn2(lib.p_2h,
			  lib.p_2h_vec, cosmo, k, a)

def p_halomod(cosmo, k, a):
    """matter power spectrum from halo model
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        k (float or array_like): wavenumber
        a (float): scale factor.

    Returns:
        p_halomod (float or array_like): matter power spectrum from halo model
    """
    return _vectorize_fn2(lib.p_halomod,
			  lib.p_halomod_vec, cosmo, k, a)

def halo_concentration(cosmo, halo_mass, a):
    """halo concentration
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): mass of halo in Msun
        a (float): scale factor.
    
    Returns:
        halo_concentration: measure of halo concentration
    """
    return _vectorize_fn2(lib.halo_concentration,
			  lib.halo_concentration_vec, cosmo, halo_mass, a)

