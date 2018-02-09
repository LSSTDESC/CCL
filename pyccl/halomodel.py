from pyccl import ccllib as lib
from pyccl.pyutils import _vectorize_fn, _vectorize_fn2, _vectorize_fn4

def u_nfw_c(cosmo, c, halo_mass, k, a):
    """fourier transform of NFW profile for testing purposes!
    Args:
        cosmo (:obj:`ccl.cosmology`): Cosmological parameters.
        c (float): concentration
        halo_mass (float or array_like): Halo masses; Msun.
        k (float): wavenumber
        a (float): scale factor.

    Returns:
        u_nfw_c (float or array_like): fourier transform thingy
    """
    return _vectorize_fn4(lib.u_nfw_c, 
                          lib.u_nfw_c_vec, cosmo, c, halo_mass, k, a)
