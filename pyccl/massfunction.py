
import ccllib as lib
from pyutils import _vectorize_fn, _vectorize_fn2

def massfunc(cosmo, halo_mass, a, odelta):
    """Halo mass function.

    Note: only Tinker (2010) is implemented right now.

    TODO: verify 2010 vs 2008 mass function.
    TODO: check that output is dndM or dnd_lnM
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.
        odelta (float): overdensity parameter

    Returns:
        massfunc (float or array_like): Halo mass function; dn/dlogM.

    """
    return _vectorize_fn2(lib.massfunc, 
                          lib.massfunc_vec, cosmo, halo_mass, a, odelta)

def massfunc_m2r(cosmo, halo_mass):
    """Converts smoothing halo mass into smoothing halo radius.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.

    Returns:
        massfunc_m2r (float or array_like): Smoothing halo radius; Mpc. 

    """
    return _vectorize_fn(lib.massfunc_m2r, 
                         lib.massfunc_m2r_vec, cosmo, halo_mass)

def sigmaM(cosmo, halo_mass, a):
    """RMS variance for the given halo mass of the linear power spectrum; Msun.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
<<<<<<< HEAD
        a (float): Scale factor.
=======
        a (float): scale factor.
>>>>>>> master

    Returns:
        sigmaM (float or array_like): RMS variance of halo mass.

    """
    return _vectorize_fn2(lib.sigmaM, 
                          lib.sigmaM_vec, cosmo, halo_mass, a)

def halo_bias(cosmo, halo_mass, a, odelta):
    """Halo bias.

    Note: only Tinker (2010) halo bias is implemented right now.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): Scale factor.
        odelta (float): overdensity parameter

    Returns:
        halo_bias (float or array_like): Halo bias.

    """
    return _vectorize_fn2(lib.halo_bias, 
                          lib.halo_bias_vec, cosmo, halo_mass, a, odelta)
