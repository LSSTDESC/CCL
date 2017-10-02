
from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn4

def massfunc(cosmo, halo_mass, a, odelta=200):
    """Halo mass function.

    Note: only Tinker (2010) is implemented right now.

    TODO: verify 2010 vs 2008 mass function.
    
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): scale factor.
        odelta (float): overdensity parameter (default: 200)

    Returns:
        massfunc (float or array_like): Halo mass function; dn/dlog10M.

    """
    return _vectorize_fn4(lib.massfunc, 
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
        a (float): scale factor.

    Returns:
        sigmaM (float or array_like): RMS variance of halo mass.

    """
    return _vectorize_fn2(lib.sigmaM, 
                          lib.sigmaM_vec, cosmo, halo_mass, a)

def halo_bias(cosmo, halo_mass, a, odelta=200):
    """Halo bias.

    Note: only Tinker (2010) halo bias is implemented right now.

    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        halo_mass (float or array_like): Halo masses; Msun.
        a (float): Scale factor.
        odelta (float): overdensity parameter (default: 200)

    Returns:
        halo_bias (float or array_like): Halo bias.

    """
    return _vectorize_fn4(lib.halo_bias, 
                          lib.halo_bias_vec, cosmo, halo_mass, a, odelta)
